// Lab 1 (HIP) - Branch divergence: what it costs when lanes of ONE wavefront disagree.
//
// Three kernels, IDENTICAL arithmetic per element (`iters` fused multiply-adds each).
// The only difference is which lanes are active when each op issues:
//
//   1) lane_divergent - `if (i & 1)`: even and odd lanes of the same wavefront take
//                       different paths. The hardware runs BOTH bodies, masking the
//                       inactive lanes each time -> the wavefront pays A + B.
//   2) warp_uniform   - `if ((i / warpSize) & 1)`: the SAME branch, but every lane of
//                       a wavefront agrees, so each wavefront runs ONE body -> A or B.
//   3) branchless     - the per-lane choice becomes two selected coefficients
//                       (v_cndmask / predication), then one loop -> no divergence at
//                       all, and bit-identical results to (1).
//
// Lesson: divergence is not "branches are slow"; it is "disagreement WITHIN a
// wavefront is slow". The identical source-level branch is nearly free when it is
// wavefront-uniform.
//
// Two contracting affine maps are used so results stay bounded and the two paths are
// distinguishable (fixed points 1.0 and 2.0) - correctness proves each element really
// took the intended path, not just that the arithmetic ran.
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip divergence.cpp -o divergence.exe
// Run:   ./divergence.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

// x <- a*x + b converges to b/(1-a): path A -> 1.0, path B -> 2.0.
#define A_MUL 0.9999f
#define A_ADD 0.0001f
#define B_MUL 0.9997f
#define B_ADD 0.0006f

// `iters` is a runtime argument on purpose: with a compile-time trip count the
// compiler could unroll and merge the two bodies, silently removing the very
// divergence this lab measures.
__global__ void lane_divergent(const float* __restrict__ in, float* __restrict__ out,
                               int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = in[i];
    if (i & 1) {
        for (int k = 0; k < iters; ++k) x = fmaf(x, A_MUL, A_ADD);
    } else {
        for (int k = 0; k < iters; ++k) x = fmaf(x, B_MUL, B_ADD);
    }
    out[i] = x;
}

// Same branch, wavefront-aligned. blockDim must be a multiple of warpSize for
// `i / warpSize` to be uniform across a wavefront (main() launches 256 threads).
__global__ void warp_uniform(const float* __restrict__ in, float* __restrict__ out,
                             int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = in[i];
    if ((i / warpSize) & 1) {
        for (int k = 0; k < iters; ++k) x = fmaf(x, A_MUL, A_ADD);
    } else {
        for (int k = 0; k < iters; ++k) x = fmaf(x, B_MUL, B_ADD);
    }
    out[i] = x;
}

// Same per-lane choice as lane_divergent, expressed as data instead of control flow.
// The ternaries lower to predicated moves, so all lanes stay active in one loop.
__global__ void branchless(const float* __restrict__ in, float* __restrict__ out,
                           int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const bool  odd = (i & 1) != 0;
    const float mul = odd ? A_MUL : B_MUL;
    const float add = odd ? A_ADD : B_ADD;

    float x = in[i];
    for (int k = 0; k < iters; ++k) x = fmaf(x, mul, add);
    out[i] = x;
}

enum Kind { LANE, WARP, BRANCHLESS };

#define NUM_SEEDS 8           // inputs are 1.0 .. 8.0, so the whole reference is 2 x 8 values

// The chain result depends only on (seed, path), so the reference is a tiny table computed
// once with std::fmaf (same single-rounding fused op as the device). Every one of the n
// outputs is then checked against it - full verification, negligible cost.
static void build_reference(float table[2][NUM_SEEDS], int iters) {
    for (int p = 0; p < 2; ++p) {
        const float mul = p ? A_MUL : B_MUL;
        const float add = p ? A_ADD : B_ADD;
        for (int s = 0; s < NUM_SEEDS; ++s) {
            float x = 1.0f + static_cast<float>(s);
            for (int k = 0; k < iters; ++k) x = std::fmaf(x, mul, add);
            table[p][s] = x;
        }
    }
}

// Mirrors each kernel's branch predicate, so a kernel that silently took the wrong path
// shows up as FAILED rather than as a plausible number.
static int expected_path(int i, int warp_size, Kind kind) {
    return (kind == WARP) ? ((i / warp_size) & 1) : (i & 1);
}

static float time_kernel(Kind kind, const float* d_in, float* d_out,
                         int n, int iters, int block, int timed_reps) {
    const int grid = (n + block - 1) / block;

    auto launch = [&]() {
        switch (kind) {
            case LANE:       lane_divergent<<<grid, block>>>(d_in, d_out, n, iters); break;
            case WARP:       warp_uniform  <<<grid, block>>>(d_in, d_out, n, iters); break;
            case BRANCHLESS: branchless    <<<grid, block>>>(d_in, d_out, n, iters); break;
        }
    };

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    launch();                                  // warmup: absorb lazy module load / first-touch
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int r = 0; r < timed_reps; ++r) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / timed_reps;
}

int main() {
    const int    n           = 1 << 22;                 // 4.2M elements
    const int    iters       = 512;                     // FMAs per element (per path)
    const int    block       = 256;                     // multiple of 32 AND 64
    const int    timed_reps  = 20;
    const size_t bytes       = static_cast<size_t>(n) * sizeof(float);

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    std::vector<float> h_in(n), h_out(n);
    for (int i = 0; i < n; ++i) h_in[i] = 1.0f + static_cast<float>(i % NUM_SEEDS);

    float ref_table[2][NUM_SEEDS];
    build_reference(ref_table, iters);

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    printf("device      : %s (%s)\n", prop.name, prop.gcnArchName);
    printf("warpSize    : %d lanes per wavefront\n", prop.warpSize);
    printf("workload    : n = %d, %d FMAs per element, block = %d, %d timed reps\n\n",
           n, iters, block, timed_reps);

    // Branchless is the baseline: it does the same useful work with zero divergence.
    const float base_ms = time_kernel(BRANCHLESS, d_in, d_out, n, iters, block, timed_reps);

    struct Case { Kind kind; const char* name; } cases[] = {
        {LANE,       "lane-divergent  if (i % 2)"},
        {WARP,       "warp-uniform    if ((i / warpSize) % 2)"},
        {BRANCHLESS, "branchless      arithmetic select"},
    };

    printf("%-42s %8s %10s %9s %s\n", "kernel", "ms", "GFLOP/s", "vs base", "result");
    printf("------------------------------------------ -------- ---------- --------- ------\n");

    for (const Case& c : cases) {
        const float ms = time_kernel(c.kind, d_in, d_out, n, iters, block, timed_reps);

        HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
        int errors = 0;
        for (int i = 0; i < n; ++i) {
            const float want = ref_table[expected_path(i, prop.warpSize, c.kind)][i % NUM_SEEDS];
            if (std::fabs(h_out[i] - want) > 1e-4f) ++errors;
        }

        // Useful FLOPs only: one FMA = 2 FLOP. The divergent kernel ISSUES twice this
        // many - the gap between issued and useful work IS the divergence tax.
        const double gflops = 2.0 * n * iters / (ms / 1.0e3) / 1.0e9;
        printf("%-42s %8.3f %10.1f %8.2fx %s\n", c.name, ms, gflops, ms / base_ms,
               errors == 0 ? "PASSED" : "FAILED");
    }

    printf("\nExpect lane-divergent ~2x the branchless baseline (both paths execute per\n"
           "wavefront) and warp-uniform ~1x (each wavefront runs one path). If the two\n"
           "are equal, your compiler if-converted the branch for you - check the ISA\n"
           "with: hipcc -O3 --offload-arch=%s --save-temps -c hip/divergence.cpp\n",
           prop.gcnArchName);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
