// A06 Lab (CUDA) - Tiled matrix multiply: the GEMM optimization ladder.
//
// Line-for-line counterpart of hip/matmul.cpp; only the runtime prefix differs
// (cuda* vs hip*). Read the HIP file's header comment for the full narrative.
//
//   1) naive     - one thread per output element, all operands from global.
//   2) tiled     - TILE x TILE shared-memory staging, each element reused TILE times.
//   3) regblock  - TM x TN micro-tile of C per thread, held in registers.
//
// Verified against a double-precision CPU reference, timed with CUDA events.
// This is a teaching kernel: cuBLAS / CUTLASS use Tensor Cores and autotuned
// schedules and will be substantially faster. See README section 6.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda -DMDIM=1024 matmul.cu -o matmul.exe
// Run:   ./matmul.exe
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

#ifndef MDIM
#define MDIM 1024
#endif

constexpr int TILE = 32;
constexpr int TILE_THREADS = TILE * TILE;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;

constexpr int RB_THREADS = (BM / TM) * (BN / TN);   // 16 x 16 = 256
static_assert(RB_THREADS % BK == 0, "A-tile loader needs whole rows per pass");
static_assert(RB_THREADS % BN == 0, "B-tile loader needs whole rows per pass");
static_assert(BM % (RB_THREADS / BK) == 0, "A-tile rows must divide evenly");
static_assert(BK % (RB_THREADS / BN) == 0, "B-tile rows must divide evenly");

// ---------------------------------------------------------------------------
// 1) Naive
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
    matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
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

// ---------------------------------------------------------------------------
// 2) Shared-memory tiled
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
    matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
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

        // Out-of-range lanes must write 0.0f, not skip: a skipped slot leaves
        // the previous iteration's value behind and corrupts the dot product.
        As[ty][tx] = (row < M && aCol < K) ? A[static_cast<size_t>(row) * K + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N) ? B[static_cast<size_t>(bRow) * N + col] : 0.0f;

        __syncthreads();   // barrier 1: tile fully written before anyone reads it

        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // barrier 2: everyone done reading before we overwrite
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// 3) Register-blocked (thread coarsening)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(RB_THREADS)
    matmul_regblock(const float* __restrict__ A, const float* __restrict__ B,
                    float* __restrict__ C, int M, int N, int K) {
    // As is K-major so the inner loop walks m contiguously; the +1 pad keeps the
    // transposing store off a single shared-memory bank (A02 padding trick).
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN];

    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    const int aRow = tid / BK, aCol = tid % BK;
    const int bRow = tid / BN, bCol = tid % BN;
    constexpr int aRowStep = RB_THREADS / BK;
    constexpr int bRowStep = RB_THREADS / BN;

    float acc[TM][TN];
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n) acc[m][n] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = 0; i < BM; i += aRowStep) {
            const int gr = row0 + aRow + i, gc = k0 + aCol;
            As[aCol][aRow + i] =
                (gr < M && gc < K) ? A[static_cast<size_t>(gr) * K + gc] : 0.0f;
        }
        for (int j = 0; j < BK; j += bRowStep) {
            const int gr = k0 + bRow + j, gc = col0 + bCol;
            Bs[bRow + j][bCol] =
                (gr < K && gc < N) ? B[static_cast<size_t>(gr) * N + gc] : 0.0f;
        }

        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            float a[TM], b[TN];
            for (int m = 0; m < TM; ++m) a[m] = As[kk][threadIdx.y * TM + m];
            for (int n = 0; n < TN; ++n) b[n] = Bs[kk][threadIdx.x * TN + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n) acc[m][n] += a[m] * b[n];
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; ++m) {
        const int gr = row0 + threadIdx.y * TM + m;
        if (gr >= M) continue;
        for (int n = 0; n < TN; ++n) {
            const int gc = col0 + threadIdx.x * TN + n;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[m][n];
        }
    }
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------
enum Variant { NAIVE, TILED, REGBLOCK };

static void launch(Variant v, const float* dA, const float* dB, float* dC,
                   int M, int N, int K) {
    switch (v) {
        case NAIVE: {
            dim3 block(TILE, TILE);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
            break;
        }
        case TILED: {
            dim3 block(TILE, TILE);
            dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            matmul_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
            break;
        }
        case REGBLOCK: {
            dim3 block(BN / TN, BM / TM);
            dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
            matmul_regblock<<<grid, block>>>(dA, dB, dC, M, N, K);
            break;
        }
    }
}

static double time_variant(Variant v, const float* dA, const float* dB, float* dC,
                           int M, int N, int K, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch(v, dA, dB, dC, M, N, K);      // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) launch(v, dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

// i-k-j order so the inner loop walks a row of B contiguously.
static void cpu_reference(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<double>& C, int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            const float* brow = &B[static_cast<size_t>(k) * N];
            double* crow = &C[static_cast<size_t>(i) * N];
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
}

// Relative tolerance, not equality: the GPU sums in a different ORDER than the
// CPU reference, so the two correct answers differ in the last fp32 bits.
// fp32 eps ~1.2e-7; K = 1024 random-rounding accumulation gives ~4e-6 relative
// error, so 1e-4 leaves headroom while still catching real bugs, which produce
// O(1) relative error. Inputs are in [0, 1) so partial sums never cancel.
static bool verify(const std::vector<double>& ref, const float* dC,
                   int M, int N, double tol, double* max_rel_out) {
    std::vector<float> host(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(host.data(), dC, sizeof(float) * host.size(),
                          cudaMemcpyDeviceToHost));

    double max_rel = 0.0;
    for (size_t i = 0; i < host.size(); ++i) {
        const double r = ref[i];
        const double denom = std::max(std::abs(r), 1e-6);
        max_rel = std::max(max_rel, std::abs(static_cast<double>(host[i]) - r) / denom);
    }
    *max_rel_out = max_rel;
    return max_rel <= tol;
}

int main() {
    const int M = MDIM, N = MDIM, K = MDIM;
    const int iters = 10;
    const double tol = 1e-4;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const size_t elemsA = static_cast<size_t>(M) * K;
    const size_t elemsB = static_cast<size_t>(K) * N;
    const size_t elemsC = static_cast<size_t>(M) * N;

    std::vector<float> hA(elemsA), hB(elemsB);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    printf("device : %s\n", prop.name);
    printf("problem: C[%d x %d] = A[%d x %d] * B[%d x %d]   (%.2f GFLOP per call)\n",
           M, N, M, K, K, N, 2.0 * M * N * K / 1e9);
    printf("computing CPU reference in double (O(M*N*K), this takes a moment)...\n");
    fflush(stdout);

    std::vector<double> ref(elemsC);
    cpu_reference(hA, hB, ref, M, N, K);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * elemsA));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * elemsB));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * elemsC));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * elemsA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * elemsB, cudaMemcpyHostToDevice));

    struct Case { Variant v; const char* name; } cases[] = {
        {NAIVE,    "naive (global reads only)"},
        {TILED,    "tiled 32x32 (shared memory)"},
        {REGBLOCK, "regblock 64x64x16, 4x4/thread"},
    };

    const double flops = 2.0 * M * N * K;
    double baseline_ms = 0.0;

    printf("\n%-32s %10s %12s %9s %12s %-8s\n",
           "variant", "ms/iter", "GFLOP/s", "speedup", "max rel err", "result");
    printf("-------------------------------- ---------- ------------ --------- ------------ --------\n");

    for (const auto& c : cases) {
        CUDA_CHECK(cudaMemset(dC, 0, sizeof(float) * elemsC));
        const double ms = time_variant(c.v, dA, dB, dC, M, N, K, iters);

        double max_rel = 0.0;
        const bool ok = verify(ref, dC, M, N, tol, &max_rel);

        if (baseline_ms == 0.0) baseline_ms = ms;
        printf("%-32s %10.3f %12.1f %8.2fx %12.2e %-8s\n",
               c.name, ms, flops / (ms / 1e3) / 1e9, baseline_ms / ms,
               max_rel, ok ? "PASSED" : "FAILED");
    }

    printf("\nAnalytic global-memory intensity (fp32, ignoring cache hits):\n");
    printf("  naive    : 2*K FLOP per 8*K bytes            = %.2f FLOP/byte\n", 0.25);
    printf("  tiled    : T/4 with T = %-3d                  = %.2f FLOP/byte\n",
           TILE, TILE / 4.0);
    printf("  regblock : BM*BN / (2*(BM+BN)) = %d*%d/(2*%d) = %.2f FLOP/byte\n",
           BM, BN, BM + BN, (1.0 * BM * BN) / (2.0 * (BM + BN)));
    printf("\nExpect naive << tiled < regblock. All three do identical arithmetic;\n"
           "the delta is data reuse (see README section 5). A production GEMM\n"
           "(cuBLAS / CUTLASS) will still beat regblock by a wide margin because\n"
           "it uses Tensor Cores - that is expected, not a bug.\n");

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}
