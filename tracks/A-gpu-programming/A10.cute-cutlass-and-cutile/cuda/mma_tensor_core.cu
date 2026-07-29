// mma_tensor_core.cu - A10 Lab 1: a tensor-core GEMM with NO external dependencies.
//
// Why this file exists
// --------------------
// A06's shared-memory tiled GEMM runs its inner loop in the SIMT pipeline (`acc += a*b`),
// so it is capped by the FP32 roof no matter how well it is tiled. Peak requires the
// *matrix* hardware: a fixed-shape, mixed-precision MMA instruction. This file issues one
// and measures the difference, using only the documented CUDA C++ warp-level matrix API
// (`nvcuda::wmma`) so it builds with plain nvcc - no CUTLASS, no CuTe.
//
//   CUDA C++ Programming Guide, "Warp Matrix Functions":
//   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions
//
// What it is NOT: a fast GEMM. Both kernels read A and B straight from global memory.
// There is no shared-memory staging, no cp.async pipelining, and no swizzled layout - the
// three things that separate "uses tensor cores" from "uses tensor cores well", and the
// reason production code reaches for CUTLASS (see README section 5).
//
// Requires compute capability 7.0+ (Volta) for half-precision WMMA. The kernel body is
// guarded on __CUDA_ARCH__ and the host refuses to launch it on older hardware rather than
// producing garbage.
//
// Build/run:  make run-cuda SM_ARCH=sm_90        (from the module directory)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>
#include <mma.h>

#include "check.cuh"

using namespace nvcuda;

// The MMA shape. 16x16x16 with half inputs and float accumulate is the shape the Warp
// Matrix Functions API documents for compute capability 7.0+; it is what `wmma::fragment`
// is instantiated on below. The hardware instruction underneath is architecture-specific
// (Volta m8n8k4 class, Ampere m16n8k16 class) - WMMA deliberately hides that.
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// One warp computes one 16x16 output tile. WARPS_PER_BLOCK warps are stacked along y, so a
// block covers WMMA_M * WARPS_PER_BLOCK rows and WMMA_N columns of C.
static constexpr int WARPS_PER_BLOCK = 4;

static constexpr int MATRIX_DIM = 1024;   // M = N = K; divisible by 16*WARPS_PER_BLOCK
static constexpr int TIMED_ITERS = 20;

// ---------------------------------------------------------------------------
// Kernel 1: naive SIMT baseline. One thread per output element, fp32 accumulate.
// This is A06's arithmetic without the tiling, on purpose: it isolates "am I using the
// matrix hardware?" from "is my tiling any good?".
// ---------------------------------------------------------------------------
__global__ void simt_gemm(const half* __restrict__ A,
                          const half* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2: tensor-core GEMM. A row-major (lda = K), B row-major (ldb = N),
// C row-major fp32 (ldc = N).
// ---------------------------------------------------------------------------
__global__ void wmma_gemm(const half* __restrict__ A,
                          const half* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
    // Tile coordinates, in units of the MMA shape. Both are warp-uniform: rowTile depends
    // on threadIdx.y (the warp index within the block) and colTile only on blockIdx.
    // That is load-bearing - the guard below must not split a warp, because mma_sync is
    // warp-collective and all 32 lanes must reach it.
    const int rowTile = blockIdx.y * WARPS_PER_BLOCK + threadIdx.y;
    const int colTile = blockIdx.x;
    if (rowTile * WMMA_M >= M || colTile * WMMA_N >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;

    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // Fragments are opaque: the API decides which lane holds which element. All we
        // supply is the address of the tile's first element and the leading dimension.
        wmma::load_matrix_sync(aFrag, A + rowTile * WMMA_M * K + k, K);
        wmma::load_matrix_sync(bFrag, B + k * N + colTile * WMMA_N, N);
        wmma::mma_sync(acc, aFrag, bFrag, acc);
    }

    wmma::store_matrix_sync(C + rowTile * WMMA_M * N + colTile * WMMA_N,
                            acc, N, wmma::mem_row_major);
#else
    // No half-precision WMMA before compute capability 7.0. The host checks the device and
    // never launches this, so reaching here means a build/launch mismatch.
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// Inputs are drawn from {-0.5, -0.25, 0, 0.25, 0.5}: every value is exactly representable
// in fp16, every product is a multiple of 1/16, and every partial sum over K = 1024 fits
// exactly in an fp32 mantissa. So CPU and GPU results are exact, a tight tolerance is
// meaningful, and any mismatch is a real bug rather than rounding noise. Precision proper
// is Track B04's subject; here we want an unambiguous PASS/FAIL.
static void fill_exact_half(std::vector<half>& v, unsigned seed)
{
    unsigned state = seed;
    for (auto& x : v) {
        state = state * 1103515245u + 12345u;
        const int bucket = static_cast<int>((state >> 16) % 5u);   // 0..4
        x = __float2half(static_cast<float>(bucket - 2) * 0.25f);  // -0.5 .. 0.5
    }
}

static void cpu_gemm(const std::vector<half>& A, const std::vector<half>& B,
                     std::vector<float>& C, int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += __half2float(A[static_cast<size_t>(m) * K + k]) *
                       __half2float(B[static_cast<size_t>(k) * N + n]);
            }
            C[static_cast<size_t>(m) * N + n] = acc;
        }
    }
}

static int count_mismatches(const std::vector<float>& got, const std::vector<float>& want,
                            float tol)
{
    int bad = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const float denom = std::fabs(want[i]) > 1.0f ? std::fabs(want[i]) : 1.0f;
        if (std::fabs(got[i] - want[i]) / denom > tol) {
            ++bad;
        }
    }
    return bad;
}

// GFLOP/s for a GEMM: two FLOP (one multiply, one add) per multiply-accumulate.
static double gflops(int M, int N, int K, float ms)
{
    const double flop = 2.0 * M * N * K;
    return flop / (static_cast<double>(ms) * 1.0e6);
}

int main()
{
    const int M = MATRIX_DIM, N = MATRIX_DIM, K = MATRIX_DIM;

    static_assert(MATRIX_DIM % (WMMA_M * WARPS_PER_BLOCK) == 0,
                  "M must be a multiple of WMMA_M * WARPS_PER_BLOCK");
    static_assert(MATRIX_DIM % WMMA_N == 0, "N must be a multiple of WMMA_N");
    static_assert(MATRIX_DIM % WMMA_K == 0, "K must be a multiple of WMMA_K");

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    const bool has_tensor_cores = (prop.major >= 7);

    std::printf("device           : %s (compute capability %d.%d)\n",
                prop.name, prop.major, prop.minor);
    std::printf("problem          : C[%d x %d] = A[%d x %d] * B[%d x %d], fp16 in / fp32 out\n",
                M, N, M, K, K, N);
    std::printf("MMA shape        : %dx%dx%d (nvcuda::wmma)\n", WMMA_M, WMMA_N, WMMA_K);
    if (!has_tensor_cores) {
        std::printf("\nNOTE: compute capability %d.%d has no half-precision tensor cores "
                    "(7.0+ required).\n      Running the SIMT baseline only; the WMMA kernel "
                    "is skipped.\n", prop.major, prop.minor);
    }
    std::printf("\n");

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    std::vector<half> h_A(a_elems), h_B(b_elems);
    fill_exact_half(h_A, 1u);
    fill_exact_half(h_B, 7u);

    std::printf("computing CPU reference (this takes a second or two)...\n");
    std::vector<float> h_ref(c_elems);
    cpu_gemm(h_A, h_B, h_ref, M, N, K);

    half* d_A = nullptr;
    half* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(half), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> h_C(c_elems);
    // Inputs are exact in fp16 and all partial sums are exact in fp32, so this tolerance is
    // ~5 orders of magnitude looser than needed. A failure here is a layout bug, not rounding.
    const float tol = 1.0e-3f;

    std::printf("\n%-34s %14s %10s\n", "kernel", "GFLOP/s", "result");
    std::printf("%-34s %14s %10s\n", "----------------------------------",
                "--------------", "----------");

    // ---- SIMT baseline ----
    {
        const dim3 block(16, 16);
        const dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));
        simt_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);   // warm-up
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < TIMED_ITERS; ++i) {
            simt_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= TIMED_ITERS;

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
        const int bad = count_mismatches(h_C, h_ref, tol);
        std::printf("%-34s %14.1f %10s\n", "simt_gemm (scalar FFMA pipeline)",
                    gflops(M, N, K, ms), bad == 0 ? "PASSED" : "FAILED");
        if (bad != 0) {
            std::printf("   %d mismatching elements\n", bad);
        }
    }

    // ---- Tensor-core kernel ----
    if (has_tensor_cores) {
        const dim3 block(32, WARPS_PER_BLOCK);                       // one warp per y
        const dim3 grid(N / WMMA_N, M / (WMMA_M * WARPS_PER_BLOCK));

        CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));
        wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);          // warm-up
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < TIMED_ITERS; ++i) {
            wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= TIMED_ITERS;

        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));
        const int bad = count_mismatches(h_C, h_ref, tol);
        std::printf("%-34s %14.1f %10s\n", "wmma_gemm (tensor cores)",
                    gflops(M, N, K, ms), bad == 0 ? "PASSED" : "FAILED");
        if (bad != 0) {
            std::printf("   %d mismatching elements\n", bad);
        }
    }

    std::printf("\nArithmetic intensity: 2*N^3 / (8*N^2) = N/4 = %d FLOP/byte -> compute-bound.\n",
                MATRIX_DIM / 4);
    std::printf("Expect wmma_gemm >> simt_gemm (different functional unit), and both far below\n"
                "peak: neither stages tiles in shared memory or pipelines the loads. That\n"
                "remaining gap is what CUTLASS/CuTe exists to close.\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}
