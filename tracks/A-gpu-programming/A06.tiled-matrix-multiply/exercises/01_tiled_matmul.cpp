// Exercise 1 - Write the tiled GEMM yourself.
//
// The naive kernel below is complete and correct: it is your correctness and
// performance baseline. The tiled kernel is a skeleton. Fill in the four TODOs
// and you should see the same PASSED result at several times the GFLOP/s.
//
// The four TODOs are exactly the four moving parts of every tiled kernel:
//   1. cooperatively load a TILE x TILE slice of A into shared memory
//   2. cooperatively load a TILE x TILE slice of B into shared memory
//   3. the barrier that separates "tile is being written" from "tile is read"
//   4. the inner product over the staged tile
// A fifth barrier is already written for you at the bottom of the k-loop -
// work out why it is there before you look at the solution.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_tiled_matmul.cpp -o 01_tiled_matmul.exe
// Run:   ./01_tiled_matmul.exe
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

constexpr int TILE = 32;
constexpr int DIM  = 512;      // square problem, an exact multiple of TILE

__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    const int col = blockIdx.x * TILE + threadIdx.x;
    const int row = blockIdx.y * TILE + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    }
    C[static_cast<size_t>(row) * N + col] = acc;
}

__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    const int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        // The operand indices this thread is responsible for staging:
        //   column of A = t * TILE + tx        row of B = t * TILE + ty
        // Note which one moves with threadIdx.x - that is what keeps the tile
        // load itself coalesced.

        // TODO 1: stage A[row][t*TILE + tx] into As. Every slot must be written
        //         on every iteration - use 0.0f when the index is out of range.
        As[ty][tx] = 0.0f;

        // TODO 2: stage B[t*TILE + ty][col] into Bs, same all-slots rule.
        Bs[ty][tx] = 0.0f;

        // TODO 3: add the barrier that makes the staged tile visible to the
        //         whole block before anyone reads it.

        // TODO 4: accumulate the TILE-long partial dot product from As and Bs
        //         into acc.

        __syncthreads();   // why is THIS one needed as well? (see solution)
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}

static void cpu_reference(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<double>& C, int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            const float* brow = &B[static_cast<size_t>(k) * N];
            double* crow = &C[static_cast<size_t>(i) * N];
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
}

// Relative check, not equality: the GPU sums in a different order than the CPU,
// so the two correct answers differ in the last fp32 bits.
static bool verify(const std::vector<double>& ref, const float* dC, int M, int N,
                   double* max_rel_out) {
    std::vector<float> host(static_cast<size_t>(M) * N);
    HIP_CHECK(hipMemcpy(host.data(), dC, sizeof(float) * host.size(),
                        hipMemcpyDeviceToHost));
    double max_rel = 0.0;
    for (size_t i = 0; i < host.size(); ++i)
        max_rel = std::max(max_rel,
                           std::abs(static_cast<double>(host[i]) - ref[i]) /
                               std::max(std::abs(ref[i]), 1e-6));
    *max_rel_out = max_rel;
    return max_rel <= 1e-4;
}

int main() {
    const int M = DIM, N = DIM, K = DIM;
    const int iters = 20;

    std::vector<float> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    std::vector<double> ref(static_cast<size_t>(M) * N);
    cpu_reference(hA, hB, ref, M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, sizeof(float) * hA.size()));
    HIP_CHECK(hipMalloc(&dB, sizeof(float) * hB.size()));
    HIP_CHECK(hipMalloc(&dC, sizeof(float) * ref.size()));
    HIP_CHECK(hipMemcpy(dA, hA.data(), sizeof(float) * hA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(float) * hB.size(), hipMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    const double flops = 2.0 * M * N * K;

    printf("%-12s %10s %12s %12s %-8s\n", "kernel", "ms/iter", "GFLOP/s", "max rel err",
           "result");
    printf("------------ ---------- ------------ ------------ --------\n");

    int failures = 0;
    for (int variant = 0; variant < 2; ++variant) {
        const char* name = variant == 0 ? "naive" : "tiled";
        HIP_CHECK(hipMemset(dC, 0, sizeof(float) * ref.size()));

        auto run = [&]() {
            if (variant == 0) matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
            else              matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
        };

        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        run();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipEventRecord(start));
        for (int i = 0; i < iters; ++i) run();
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        const double per_iter = static_cast<double>(ms) / iters;

        double max_rel = 0.0;
        const bool ok = verify(ref, dC, M, N, &max_rel);
        if (!ok) ++failures;
        printf("%-12s %10.3f %12.1f %12.2e %-8s\n", name, per_iter,
               flops / (per_iter / 1e3) / 1e9, max_rel, ok ? "PASSED" : "FAILED");
    }

    if (failures) {
        printf("\nThe tiled kernel is still a skeleton - finish TODO 1..4.\n");
    } else {
        printf("\nBoth PASSED. Note the GFLOP/s ratio: same arithmetic, %dx more\n"
               "reuse per loaded element.\n", TILE);
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
