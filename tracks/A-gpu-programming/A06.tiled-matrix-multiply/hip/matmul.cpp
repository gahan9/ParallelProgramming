// A06 Lab (HIP) - Tiled matrix multiply: the GEMM optimization ladder.
//
// One program, three kernels, identical results, very different GFLOP/s:
//
//   1) naive     - one thread per output element; every one of the 2K operand
//                  reads goes to global memory. Arithmetic intensity ~0.25
//                  FLOP/byte -> hopelessly memory-bound.
//   2) tiled     - a TILE x TILE block cooperatively stages A and B tiles in
//                  shared memory (LDS), so each loaded element is reused TILE
//                  times. Intensity rises to ~TILE/4 FLOP/byte.
//   3) regblock  - each thread computes a TM x TN micro-tile of C held in
//                  REGISTERS. That decouples "tile size" from "thread count",
//                  so a 64x64 block tile needs only 256 threads instead of an
//                  impossible 4096, and it cuts shared-memory traffic by ~TM x.
//
// Every variant is verified against a CPU reference computed in DOUBLE (see
// verify() for the tolerance argument) and timed with GPU events.
//
// This is a TEACHING kernel. It will not match rocBLAS / hipBLASLt / cuBLAS,
// which use matrix cores (MFMA / Tensor Cores), hand-scheduled assembly and
// autotuned tile shapes. Reaching a modest fraction of vendor-library
// throughput here is the expected outcome - see README section 6.
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip -DMDIM=1024 matmul.cpp -o matmul.exe
// Run:   ./matmul.exe
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

// Problem size. Square by default; the kernels take independent M, N, K so a
// rectangular or non-multiple-of-tile shape exercises the boundary guards.
#ifndef MDIM
#define MDIM 1024
#endif

// ---- Variant 2 geometry: one thread per output element ----
// 32 x 32 = 1024 threads is the hardware maximum block size on both vendors,
// which is exactly why this variant cannot grow its tile any further.
constexpr int TILE = 32;
constexpr int TILE_THREADS = TILE * TILE;

// ---- Variant 3 geometry: one thread per TM x TN micro-tile of C ----
constexpr int BM = 64;   // rows of C per block
constexpr int BN = 64;   // cols of C per block
constexpr int BK = 16;   // K-slice staged per iteration
constexpr int TM = 4;    // rows of C per thread
constexpr int TN = 4;    // cols of C per thread

constexpr int RB_THREADS = (BM / TM) * (BN / TN);   // 16 x 16 = 256
static_assert(RB_THREADS % BK == 0, "A-tile loader needs whole rows per pass");
static_assert(RB_THREADS % BN == 0, "B-tile loader needs whole rows per pass");
static_assert(BM % (RB_THREADS / BK) == 0, "A-tile rows must divide evenly");
static_assert(BK % (RB_THREADS / BN) == 0, "B-tile rows must divide evenly");

// ---------------------------------------------------------------------------
// 1) Naive: C[row][col] = sum_k A[row][k] * B[k][col], all reads from global.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE_THREADS)
    matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                 float* __restrict__ C, int M, int N, int K) {
    const int col = blockIdx.x * TILE + threadIdx.x;
    const int row = blockIdx.y * TILE + threadIdx.y;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        // A read is a broadcast within the warp (same row); B read is coalesced
        // (consecutive cols). Neither is reused - that is the whole problem.
        acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    }
    C[static_cast<size_t>(row) * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 2) Shared-memory tiled: stage TILE x TILE sub-blocks of A and B, reuse each
//    loaded element TILE times. Note the TWO barriers - README section 3
//    explains what breaks if you drop either one.
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
        const int aCol = t * TILE + tx;   // column of A this thread stages
        const int bRow = t * TILE + ty;   // row of B this thread stages

        // Out-of-range lanes must write 0.0f, not skip: the tile is read by
        // ALL threads, so a skipped slot leaves the previous iteration's value
        // behind and silently corrupts the dot product.
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
// 3) Register-blocked (thread coarsening): each thread owns a TM x TN patch of
//    C in registers. Operands are read once from shared memory into TM + TN
//    registers and reused for TM * TN multiply-adds, so shared traffic per FLOP
//    drops by roughly a factor of TM.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(RB_THREADS)
    matmul_regblock(const float* __restrict__ A, const float* __restrict__ B,
                    float* __restrict__ C, int M, int N, int K) {
    // As is stored K-major (As[k][m]) so the inner loop walks m contiguously.
    // The +1 pad is the A02 bank-conflict trick: without it the transposing
    // store below has all 16 loader lanes hitting one LDS bank.
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN];

    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;   // 0 .. RB_THREADS-1
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    // Loader maps: consecutive tid must walk consecutive columns of the source
    // row, or the global tile load is itself uncoalesced (A02 section 2).
    const int aRow = tid / BK, aCol = tid % BK;          // A tile is BM x BK
    const int bRow = tid / BN, bCol = tid % BN;          // B tile is BK x BN
    constexpr int aRowStep = RB_THREADS / BK;            // A rows filled per pass
    constexpr int bRowStep = RB_THREADS / BN;            // B rows filled per pass

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

// Milliseconds per iteration, measured with GPU events after a warmup launch.
static double time_variant(Variant v, const float* dA, const float* dB, float* dC,
                           int M, int N, int K, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    launch(v, dA, dB, dC, M, N, K);      // warmup: pays lazy-init and cache costs
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch(v, dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

// CPU reference in double precision. Loop order is i-k-j so the inner loop walks
// a row of B contiguously; the textbook i-j-k order is several times slower here.
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

// Why a RELATIVE tolerance, and why 1e-4:
//   The GPU accumulates in fp32 in a different ORDER than the CPU reference
//   (per-tile partial sums vs a single running sum), so bit-exact equality is
//   the wrong test - both answers are "correct", they just round differently.
//   fp32 has eps ~1.2e-7; summing K terms accumulates roughly sqrt(K) * eps of
//   relative error under random rounding, ~4e-6 at K = 1024. 1e-4 leaves ~25x
//   headroom for a larger K while staying far below the O(1) relative error a
//   real bug (missing guard, wrong barrier, transposed index) produces.
//   Inputs are drawn from [0, 1) so every partial sum is positive: that avoids
//   catastrophic cancellation, which would make a relative check meaningless
//   near zero.
static bool verify(const std::vector<double>& ref, const float* dC,
                   int M, int N, double tol, double* max_rel_out) {
    std::vector<float> host(static_cast<size_t>(M) * N);
    HIP_CHECK(hipMemcpy(host.data(), dC, sizeof(float) * host.size(), hipMemcpyDeviceToHost));

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

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

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
    HIP_CHECK(hipMalloc(&dA, sizeof(float) * elemsA));
    HIP_CHECK(hipMalloc(&dB, sizeof(float) * elemsB));
    HIP_CHECK(hipMalloc(&dC, sizeof(float) * elemsC));
    HIP_CHECK(hipMemcpy(dA, hA.data(), sizeof(float) * elemsA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), sizeof(float) * elemsB, hipMemcpyHostToDevice));

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
        HIP_CHECK(hipMemset(dC, 0, sizeof(float) * elemsC));
        const double ms = time_variant(c.v, dA, dB, dC, M, N, K, iters);

        double max_rel = 0.0;
        const bool ok = verify(ref, dC, M, N, tol, &max_rel);

        if (baseline_ms == 0.0) baseline_ms = ms;
        printf("%-32s %10.3f %12.1f %8.2fx %12.2e %-8s\n",
               c.name, ms, flops / (ms / 1e3) / 1e9, baseline_ms / ms,
               max_rel, ok ? "PASSED" : "FAILED");
    }

    // Analytic global-memory arithmetic intensity, the number that explains the
    // table above. Derivation in README section 2.
    printf("\nAnalytic global-memory intensity (fp32, ignoring cache hits):\n");
    printf("  naive    : 2*K FLOP per 8*K bytes            = %.2f FLOP/byte\n", 0.25);
    printf("  tiled    : T/4 with T = %-3d                  = %.2f FLOP/byte\n",
           TILE, TILE / 4.0);
    printf("  regblock : BM*BN / (2*(BM+BN)) = %d*%d/(2*%d) = %.2f FLOP/byte\n",
           BM, BN, BM + BN, (1.0 * BM * BN) / (2.0 * (BM + BN)));
    printf("\nExpect naive << tiled < regblock. All three do identical arithmetic;\n"
           "the delta is data reuse (see README section 5). A production GEMM\n"
           "(rocBLAS / hipBLASLt / cuBLAS) will still beat regblock by a wide\n"
           "margin because it uses matrix cores - that is expected, not a bug.\n");

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return 0;
}
