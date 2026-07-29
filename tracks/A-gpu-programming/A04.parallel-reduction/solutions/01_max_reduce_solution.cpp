// Solution 1 - Max reduction, with the identity element handled correctly.
//
// Two changes from the starter:
//
//   1. IDENTITY becomes -INFINITY. The identity of a reduction operator is the
//      value that leaves the operator unchanged: 0 for +, 1 for *, -inf for max,
//      +inf for min, false for ||. Lanes with no data must contribute it, and
//      0.0f is emphatically NOT the identity for max - it beats every negative
//      input, so an all-negative array reduces to 0.
//
//   2. `+` becomes fmaxf() in both combine steps (the grid-stride fold and the
//      shared-memory tree). Nothing else about the reduction changes, which is
//      the real lesson: the skeleton is operator-agnostic.
//
// Why -INFINITY and not -FLT_MAX: -inf is the true identity (fmaxf(-inf, x) == x
// for every finite x, and also for x == -inf). -FLT_MAX works for any input that
// is not itself -inf, and is the right choice when a downstream kernel would do
// arithmetic on the result and cannot tolerate an infinity (e.g. an empty tile
// feeding a softmax). Pick deliberately; do not default to 0.
//
// Bonus property: unlike a sum, max is order-independent in floating point, so a
// correct GPU tree matches the CPU answer BIT FOR BIT. The check below uses ==,
// not a tolerance. (NaN is the exception - fmaxf propagates the non-NaN operand,
// so a NaN in the input is silently dropped rather than poisoning the result.
// If NaN must propagate, reduce with an explicit comparison instead.)
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_max_reduce_solution.cpp -o 01_max_reduce_solution.exe
// Run:   ./01_max_reduce_solution.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256u

// Cascade grid, capped so the second pass always fits in a single block.
constexpr unsigned kMaxBlocks = BLOCK;

#define IDENTITY (-INFINITY)

__global__ void reduce_max(const float* in, float* out, unsigned n) {
    __shared__ float sdata[BLOCK];
    const unsigned tid = threadIdx.x;

    // Grid-stride loop: every thread folds many elements into one register
    // before the block tree runs. Threads with no work keep the identity.
    float best = IDENTITY;
    for (unsigned i = blockIdx.x * BLOCK + tid; i < n; i += BLOCK * gridDim.x) {
        best = fmaxf(best, in[i]);
    }

    sdata[tid] = best;
    __syncthreads();

    for (unsigned s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// Two passes: n -> at most kMaxBlocks partials -> 1.
static float gpu_max(const std::vector<float>& h_in) {
    const unsigned n     = static_cast<unsigned>(h_in.size());
    const size_t   bytes = sizeof(float) * h_in.size();

    unsigned grid = (n + BLOCK - 1) / BLOCK;
    if (grid == 0) grid = 1;
    if (grid > kMaxBlocks) grid = kMaxBlocks;

    float *d_in = nullptr, *d_partial = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_partial, sizeof(float) * kMaxBlocks));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    reduce_max<<<grid, BLOCK>>>(d_in, d_partial, n);
    HIP_CHECK(hipGetLastError());

    // The second pass sees only `grid` valid partials, so its own tail lanes
    // take the identity too. This is the launch that punishes a wrong identity
    // even when the first pass looked fine.
    reduce_max<<<1, BLOCK>>>(d_partial, d_out, grid);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    float h_out = 0.0f;
    HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_partial));
    HIP_CHECK(hipFree(d_out));
    return h_out;
}

static bool check(const char* name, const std::vector<float>& h_in) {
    float ref = -std::numeric_limits<float>::infinity();
    for (float v : h_in) ref = (v > ref) ? v : ref;

    const float got = gpu_max(h_in);
    const bool  ok  = (got == ref);
    printf("%-46s expected %14.6f  got %14.6f  %s\n",
           name, ref, got, ok ? "PASSED" : "FAILED");
    return ok;
}

int main() {
    std::vector<float> mixed(1u << 20);
    for (size_t i = 0; i < mixed.size(); ++i) {
        mixed[i] = std::sin(static_cast<float>(i) * 0.001f) * 100.0f;
    }

    std::vector<float> negatives(1000003);
    for (size_t i = 0; i < negatives.size(); ++i) {
        negatives[i] = -1.0f - static_cast<float>(i % 997);
    }

    const std::vector<float> single = {-42.5f};

    int failures = 0;
    failures += !check("A: mixed signs, n = 1048576", mixed);
    failures += !check("B: all negative, n = 1000003 (identity!)", negatives);
    failures += !check("C: single element, n = 1", single);

    printf("\n%d of 3 cases failed.\n", failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
