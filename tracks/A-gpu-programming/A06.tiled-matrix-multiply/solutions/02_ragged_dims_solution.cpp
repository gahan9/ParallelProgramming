// Solution 2 - Boundary-correct tiled GEMM, plus the answers to Q1-Q3.
//
// THE FIX
//   Turn each "if (in range) write" into an unconditional write of either the
//   value or 0.0f. Zero is the identity for the accumulation, so padding the
//   tile with zeros makes the fringe iterations contribute nothing while every
//   thread still executes the same straight-line code.
//
// Q1. Why does case B (M and N ragged, K a multiple of TILE) pass even with the
//     broken loads?
//     Because the load indexing is *thread-matched* to the read indexing.
//     Thread (ty, tx) writes As[ty][tx] and later reads As[ty][k]; the slot it
//     reads for its own row was written by a thread with the SAME ty, hence the
//     same `row`. If our thread's row is in range, so is that writer's. The
//     mirror argument holds for Bs and `col`. Out-of-range rows/columns are
//     therefore only ever consumed by threads that are themselves out of range,
//     and the store guard discards their results. M/N raggedness is contained;
//     no zero-fill is needed for it.
//
// Q2. Why does case C (K ragged) fail nearly everywhere rather than only on the
//     boundary tiles?
//     The K edge is different: `aCol < K` and `bRow < K` fail for the same k
//     slots in EVERY block, and those slots are read by every in-range thread.
//     On the last tile iteration the high-k slots keep whatever the previous
//     iteration left in LDS (garbage on the very first iteration), and
//     `acc += As[ty][k] * Bs[k][tx]` folds that stale product into every output
//     element. So a K-edge bug corrupts the whole matrix; only an M/N *store*
//     bug produces the classic fringe-only signature.
//
// Q3. What would go wrong without the store guard?
//     `C[row * N + col]` with `col >= N` does not fall outside the matrix - it
//     aliases element [row + 1][col - N]. So an unguarded store silently
//     overwrites *valid* results in the next row, and for the last block it
//     writes past the end of the allocation entirely (out-of-bounds write:
//     undefined behaviour, possibly a corrupted neighbouring allocation rather
//     than a clean fault). The harness would notice the aliasing here because
//     it checks every element, but a spot-check of the matrix interior might
//     not - which is exactly how this bug reaches production. Run under
//     `compute-sanitizer` (NVIDIA) or with `-fsanitize=address`-style ROCm
//     tooling to catch it deterministically.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_ragged_dims_solution.cpp -o 02_ragged_dims_solution.exe
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

        // Unconditional write: value or zero. Zero is the additive identity, so
        // padded slots contribute nothing to the dot product.
        As[ty][tx] = (row < M && aCol < K) ? A[static_cast<size_t>(row) * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
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

    const int interiorRows = (M / TILE) * TILE;
    const int interiorCols = (N / TILE) * TILE;
    long long bad = 0, badFringe = 0;
    int firstR = -1, firstC = -1;
    double maxRel = 0.0;

    for (int r = 0; r < M; ++r)
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
    printf("TILE = %d. All three shapes must PASS.\n\n", TILE);

    int failures = 0;
    failures += !run_case("A) all dims multiple", 512, 512, 512);
    failures += !run_case("B) M,N ragged  K aligned", 500, 260, 512);
    failures += !run_case("C) K ragged", 500, 260, 377);
    failures += !run_case("D) tall-skinny, all ragged", 1001, 37, 129);
    failures += !run_case("E) K smaller than TILE", 65, 65, 7);

    printf("\nCases D and E are the shapes that break most hand-written GEMMs:\n"
           "a K shorter than one tile means the very FIRST tile iteration is\n"
           "already a fringe iteration.\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
