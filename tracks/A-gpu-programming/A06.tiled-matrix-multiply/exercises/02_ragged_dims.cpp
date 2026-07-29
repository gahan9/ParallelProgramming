// Exercise 2 - Make the tiled GEMM correct for dimensions that are not
//              multiples of the tile size. The classic boundary-guard drill.
//
// The kernel below already has bounds CHECKS on both tile loads and on the
// store, so it never touches memory it does not own. It is still wrong, and the
// harness proves it by running three shapes:
//
//   A) 512 x 512 x 512   every dimension is a multiple of TILE  -> PASSES
//   B) 500 x 260 x 512   M and N ragged, K a multiple of TILE   -> PASSES
//   C) 500 x 260 x 377   K ragged                               -> FAILS
//
// YOUR TASK
//   TODO 1 / TODO 2: fix the two tile loads so the kernel passes all three.
//   Then answer, in one sentence each:
//     Q1. Why does case B pass even though M and N are ragged?
//     Q2. Why does case C fail on essentially EVERY element rather than only on
//         the boundary tiles?
//     Q3. The store at the bottom is guarded. What exactly would go wrong
//         without that guard, and would the harness necessarily notice?
//   Answers in ../solutions/02_ragged_dims_solution.cpp.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_ragged_dims.cpp -o 02_ragged_dims.exe
// Run:   ./02_ragged_dims.exe
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

constexpr int TILE = 32;

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

        // TODO 1: the check is right but the kernel is not. Ask yourself what
        //         As[ty][tx] contains when this condition is false.
        if (row < M && aCol < K) {
            As[ty][tx] = A[static_cast<size_t>(row) * K + aCol];
        }

        // TODO 2: same defect, mirrored onto the B tile.
        if (bRow < K && col < N) {
            Bs[ty][tx] = B[static_cast<size_t>(bRow) * N + col];
        }

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // This guard is correct and deliberately left in place: without it the last
    // partial tile writes outside C. See Q3 in the header.
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

// Returns true when the case passes. Also reports WHERE the mismatches are:
// confined to the last partial tile, or spread across the whole matrix. Those
// two failure signatures point at different bugs.
static bool run_case(const char* label, int M, int N, int K) {
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
    HIP_CHECK(hipMemset(dC, 0, sizeof(float) * ref.size()));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> host(ref.size());
    HIP_CHECK(hipMemcpy(host.data(), dC, sizeof(float) * host.size(),
                        hipMemcpyDeviceToHost));

    const int interiorRows = (M / TILE) * TILE;   // rows covered by full tiles
    const int interiorCols = (N / TILE) * TILE;
    long long bad = 0, badFringe = 0;
    int firstR = -1, firstC = -1;
    double maxRel = 0.0;

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            const size_t i = static_cast<size_t>(r) * N + c;
            const double rel = std::abs(static_cast<double>(host[i]) - ref[i]) /
                               std::max(std::abs(ref[i]), 1e-6);
            maxRel = std::max(maxRel, rel);
            if (rel > 1e-4) {
                ++bad;
                if (r >= interiorRows || c >= interiorCols) ++badFringe;
                if (firstR < 0) { firstR = r; firstC = c; }
            }
        }
    }

    const bool ok = bad == 0;
    printf("%-28s %5d x %-5d k=%-5d  max rel %8.2e  ", label, M, N, K, maxRel);
    if (ok) {
        printf("PASSED\n");
    } else {
        printf("FAILED (%lld/%lld wrong, %lld of them in the boundary tiles, "
               "first at [%d,%d])\n",
               bad, static_cast<long long>(ref.size()), badFringe, firstR, firstC);
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return ok;
}

int main() {
    printf("TILE = %d. A case only exercises the guards when a dimension is not\n"
           "a multiple of TILE.\n\n", TILE);

    int failures = 0;
    failures += !run_case("A) all dims multiple", 512, 512, 512);
    failures += !run_case("B) M,N ragged  K aligned", 500, 260, 512);
    failures += !run_case("C) K ragged", 500, 260, 377);

    if (failures) {
        printf("\n%d case(s) failed. Fix TODO 1 and TODO 2, then answer Q1-Q3.\n",
               failures);
    } else {
        printf("\nAll shapes correct. Now re-read Q1-Q3 - passing is not the same\n"
               "as understanding which guard saved you.\n");
    }
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
