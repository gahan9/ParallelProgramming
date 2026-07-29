// 01_wmma_gemm.cu - EXERCISE: drive the tensor cores yourself.
//
// Goal: implement one 16x16 output tile per warp using the documented nvcuda::wmma API,
// and make the CPU verification PASS.
//
// The harness (allocation, exact-in-fp16 input generation, CPU reference, event timing,
// GFLOP/s, verification) is written for you. Four TODOs stand between you and a PASS.
//
// This compiles as-is and prints FAILED, because C is never written. That is the starting
// state, not a bug.
//
// API reference (read this, do not guess):
//   CUDA C++ Programming Guide, "Warp Matrix Functions"
//   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions
//
// Build/run:
//   make 01_wmma_gemm.exe SM_ARCH=sm_80 && ./01_wmma_gemm.exe

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>
#include <mma.h>

#include "check.cuh"

using namespace nvcuda;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int WARPS_PER_BLOCK = 4;   // warps stacked along blockDim.y
static constexpr int MATRIX_DIM      = 512; // M = N = K
static constexpr int TIMED_ITERS     = 20;

// ===========================================================================
// YOUR WORK IS IN THIS KERNEL.
//
// Layouts you must respect:
//   A is MxK row-major half   -> leading dimension (lda) = K
//   B is KxN row-major half   -> leading dimension (ldb) = N
//   C is MxN row-major float  -> leading dimension (ldc) = N
//
// The tile this warp owns starts at row (rowTile * WMMA_M) and column
// (colTile * WMMA_N) of C.
// ===========================================================================
__global__ void wmma_gemm(const half* __restrict__ A,
                          const half* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
    // Both tile coordinates are warp-uniform, so the guard below cannot split a warp.
    // Keep it that way: mma_sync is warp-collective and every one of the 32 lanes must
    // reach it. A guard derived from threadIdx.x would be undefined behaviour.
    const int rowTile = blockIdx.y * WARPS_PER_BLOCK + threadIdx.y;
    const int colTile = blockIdx.x;
    if (rowTile * WMMA_M >= M || colTile * WMMA_N >= N) {
        return;
    }

    // Fragment types are given. Note what the template parameters mean:
    //   matrix_a / matrix_b / accumulator  - which operand this fragment holds
    //   WMMA_M, WMMA_N, WMMA_K             - the shape of the whole MMA operation
    //   half / float                       - element type (inputs half, accumulator float)
    //   row_major / col_major              - only for A and B; must match your data
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;

    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        // TODO 1: load this warp's WMMA_M x WMMA_K tile of A into aFrag.
        //         wmma::load_matrix_sync(fragment, pointer_to_tile_origin, leading_dim)
        //         The tile origin is row (rowTile * WMMA_M), column k of A.

        // TODO 2: load this warp's WMMA_K x WMMA_N tile of B into bFrag.
        //         The tile origin is row k, column (colTile * WMMA_N) of B.

        // TODO 3: accumulate. acc = aFrag * bFrag + acc
        //         wmma::mma_sync(d, a, b, c) computes d = a*b + c; passing acc as both
        //         c and d is how you accumulate across the K loop.
    }

    // TODO 4: store acc into C at row (rowTile * WMMA_M), column (colTile * WMMA_N).
    //         wmma::store_matrix_sync(pointer, fragment, leading_dim, layout)
    //         C is row-major, so the layout argument is wmma::mem_row_major.
    //
    // Until this is done the harness sees the zero-filled C it memset, and reports FAILED.

#else
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

// ===========================================================================
// Harness below - no changes needed.
// ===========================================================================

// Inputs come from {-0.5, -0.25, 0, 0.25, 0.5}: exact in fp16, and every partial sum over
// K = 512 is exact in fp32. So the tolerance below is not papering over rounding - a
// mismatch means a real layout or accumulation bug.
static void fill_exact_half(std::vector<half>& v, unsigned seed)
{
    unsigned state = seed;
    for (auto& x : v) {
        state = state * 1103515245u + 12345u;
        const int bucket = static_cast<int>((state >> 16) % 5u);
        x = __float2half(static_cast<float>(bucket - 2) * 0.25f);
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

int main()
{
    const int M = MATRIX_DIM, N = MATRIX_DIM, K = MATRIX_DIM;

    static_assert(MATRIX_DIM % (WMMA_M * WARPS_PER_BLOCK) == 0, "M divisibility");
    static_assert(MATRIX_DIM % WMMA_N == 0, "N divisibility");
    static_assert(MATRIX_DIM % WMMA_K == 0, "K divisibility");

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 7) {
        std::printf("device %s has compute capability %d.%d; half-precision WMMA needs 7.0+.\n",
                    prop.name, prop.major, prop.minor);
        return 1;
    }

    const size_t a_elems = static_cast<size_t>(M) * K;
    const size_t b_elems = static_cast<size_t>(K) * N;
    const size_t c_elems = static_cast<size_t>(M) * N;

    std::vector<half> h_A(a_elems), h_B(b_elems);
    fill_exact_half(h_A, 1u);
    fill_exact_half(h_B, 7u);

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
    CUDA_CHECK(cudaMemset(d_C, 0, c_elems * sizeof(float)));

    const dim3 block(32, WARPS_PER_BLOCK);
    const dim3 grid(N / WMMA_N, M / (WMMA_M * WARPS_PER_BLOCK));

    wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);   // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERS; ++i) {
        wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= TIMED_ITERS;

    std::vector<float> h_C(c_elems);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    const float tol = 1.0e-3f;
    int bad = 0;
    for (size_t i = 0; i < c_elems; ++i) {
        const float denom = std::fabs(h_ref[i]) > 1.0f ? std::fabs(h_ref[i]) : 1.0f;
        if (std::fabs(h_C[i] - h_ref[i]) / denom > tol) {
            ++bad;
        }
    }

    const double gflops = 2.0 * M * N * K / (static_cast<double>(ms) * 1.0e6);
    std::printf("problem      : %dx%dx%d fp16 in / fp32 out\n", M, N, K);
    std::printf("kernel time  : %.3f ms\n", ms);
    std::printf("throughput   : %.1f GFLOP/s\n", gflops);
    std::printf("result       : %s (%d mismatching elements)\n",
                bad == 0 ? "PASSED" : "FAILED", bad);
    if (bad != 0) {
        std::printf("\nStill FAILING? Work the TODOs in order and check:\n"
                    "  - leading dimensions: lda = K for A, ldb = N for B, ldc = N for C\n"
                    "  - the tile origins: A is (rowTile*16, k), B is (k, colTile*16)\n"
                    "  - mma_sync(acc, aFrag, bFrag, acc) - acc is both input and output\n"
                    "  - store_matrix_sync needs wmma::mem_row_major for row-major C\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return bad == 0 ? 0 : 1;
}
