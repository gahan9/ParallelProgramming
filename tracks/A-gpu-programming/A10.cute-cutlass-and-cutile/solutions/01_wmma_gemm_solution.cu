// 01_wmma_gemm_solution.cu - reference answer for exercises/01_wmma_gemm.cu.
//
// The four TODOs, solved and explained. The harness is identical to the exercise so you can
// diff the two files and see exactly what the answer adds.
//
// The whole lesson is in the four API calls and the two address expressions. Everything hard
// about a tensor-core kernel that ISN'T in this file - shared-memory staging, swizzled
// layouts, multi-stage cp.async pipelining, epilogue fusion - is what CUTLASS exists to
// provide. See ../README.md section 5.
//
// API reference: CUDA C++ Programming Guide, "Warp Matrix Functions"
//   https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions

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

static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int MATRIX_DIM      = 512;
static constexpr int TIMED_ITERS     = 20;

__global__ void wmma_gemm(const half* __restrict__ A,
                          const half* __restrict__ B,
                          float* __restrict__ C,
                          int M, int N, int K)
{
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
    // Warp-uniform tile coordinates: rowTile varies with threadIdx.y (the warp index),
    // colTile only with blockIdx.x. Nothing here depends on threadIdx.x, so the guard below
    // never splits a warp - required, because mma_sync is warp-collective.
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
        // TODO 1 + 2. The address is the tile ORIGIN; the last argument is the leading
        // dimension of the *full* matrix, not of the tile. That distinction is the single
        // most common bug here: passing WMMA_K or WMMA_N instead of K or N gives a
        // plausible-looking wrong answer, because the fragment loader happily strides
        // through the wrong memory.
        //
        //   A is MxK row-major: element (row, col) lives at row*K + col, so the tile
        //   starting at (rowTile*WMMA_M, k) begins at rowTile*WMMA_M*K + k, stride K.
        wmma::load_matrix_sync(aFrag, A + rowTile * WMMA_M * K + k, K);
        //   B is KxN row-major: the tile starting at (k, colTile*WMMA_N) begins at
        //   k*N + colTile*WMMA_N, stride N.
        wmma::load_matrix_sync(bFrag, B + k * N + colTile * WMMA_N, N);

        // TODO 3. mma_sync(d, a, b, c) computes d = a*b + c. Passing acc as both c and d
        // accumulates in place across the K loop, which is why the accumulator fragment is
        // declared outside it.
        wmma::mma_sync(acc, aFrag, bFrag, acc);
    }

    // TODO 4. Same origin/leading-dimension rule as the loads, plus the layout of the
    // destination: C is row-major, so mem_row_major. Getting this wrong transposes each
    // 16x16 output tile in place - a bug that looks like "mostly right".
    wmma::store_matrix_sync(C + rowTile * WMMA_M * N + colTile * WMMA_N,
                            acc, N, wmma::mem_row_major);
#else
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
#endif
}

// ===========================================================================
// Harness - identical to the exercise.
// ===========================================================================

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

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return bad == 0 ? 0 : 1;
}
