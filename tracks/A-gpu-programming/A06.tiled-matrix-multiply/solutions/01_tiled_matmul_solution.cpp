// Solution 1 - Tiled GEMM, with the reasoning behind each of the four pieces.
//
// TODO 1 / TODO 2 (the cooperative tile loads)
//   The block owns a TILE x TILE window of C. To produce it, it must sweep the
//   full K dimension in TILE-wide slices, staging A[row][t*TILE + tx] and
//   B[t*TILE + ty][col]. Two properties matter:
//     * The index that varies fastest with threadIdx.x is the *column* in both
//       loads, so consecutive lanes touch consecutive addresses - the tile load
//       itself is coalesced (A02 section 2).
//     * EVERY slot is written on EVERY iteration, using 0.0f when the source
//       index is out of range. A slot that is merely skipped keeps the previous
//       iteration's value and silently poisons the dot product (exercise 2).
//
// TODO 3 (the first barrier)
//   Between "the block wrote the tile" and "the block reads the tile". Without
//   it, a fast wavefront can read As/Bs slots that a slower wavefront has not
//   written yet. The result is nondeterministic and often *mostly* right, which
//   is the worst kind of bug.
//
// TODO 4 (the inner product)
//   TILE multiply-adds out of shared memory. This is where the reuse is
//   collected: each element loaded from global feeds TILE FMAs instead of one.
//
// The second barrier (already provided in the exercise)
//   Between "the block finished reading tile t" and "the block overwrites the
//   tile for iteration t+1". Drop it and a fast wavefront racing ahead into the
//   next iteration clobbers As/Bs while a slow one is still accumulating from
//   them. Both barriers are required; they guard opposite directions of the
//   same write-after-read / read-after-write pair.
//
// Bank conflicts, for completeness: with TILE = 32 this kernel is already
// conflict-free. `As[ty][k]` is the same address for all lanes of a warp (same
// ty) -> broadcast; `Bs[k][tx]` walks 32 consecutive words -> 32 distinct
// banks. The padding trick A02 needed for transpose is not needed here.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_tiled_matmul_solution.cpp -o 01_tiled_matmul_solution.exe
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

constexpr int TILE = 32;
constexpr int DIM  = 512;

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
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        As[ty][tx] = (row < M && aCol < K) ? A[static_cast<size_t>(row) * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0f;

        __syncthreads();   // tile is complete before it is read

        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // tile is fully consumed before it is overwritten
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

    printf("\nBoth must PASS. The tiled kernel loads each element once per tile\n"
           "instead of once per output, raising global-memory intensity from\n"
           "0.25 to %.1f FLOP/byte.\n", TILE / 4.0);

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
