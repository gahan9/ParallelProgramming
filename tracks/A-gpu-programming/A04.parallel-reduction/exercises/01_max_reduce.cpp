// Exercise 1 - Turn the sum reduction into a MAX reduction.
//
// A reduction is defined by two things: a binary operator and its IDENTITY
// element (the value that leaves the operator unchanged: 0 for +, 1 for *,
// -infinity for max). The tree shape, the barriers, and the grid-stride loop
// are all operator-agnostic - which is why one skeleton serves every reduction.
//
// The identity is where people get burned. A GPU reduction almost never gets an
// input whose length is an exact multiple of the tile size, so some lanes have
// no data and must contribute the identity. Pick the wrong one and the answer
// is silently wrong ONLY for certain inputs - here, arrays whose true maximum is
// negative. That is exactly the bug this drill plants.
//
// YOUR TASKS
//   TODO 1: fix IDENTITY so it is the identity element for max, not for +.
//   TODO 2: change the two combine steps from `+` to a max.
//
// All three test cases must print PASSED. Case B (all-negative values, length
// not a multiple of the tile) is the one that catches a wrong identity; case C
// (a single element) catches an off-by-one in the tail handling.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_max_reduce.cpp -o 01_max_reduce.exe
// Run:   ./01_max_reduce.exe
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

// TODO 1: 0.0f is the identity for addition. What is the identity for max?
//         (Hint: it must lose to every possible input value, including -1e30f.)
#define IDENTITY 0.0f

__global__ void reduce_max(const float* in, float* out, unsigned n) {
    __shared__ float sdata[BLOCK];
    const unsigned tid = threadIdx.x;

    // Grid-stride loop: every thread folds many elements into one register
    // before the block tree runs. Threads with no work keep the identity.
    float best = IDENTITY;
    for (unsigned i = blockIdx.x * BLOCK + tid; i < n; i += BLOCK * gridDim.x) {
        best = best + in[i];             // TODO 2: this is a SUM, make it a max
    }

    sdata[tid] = best;
    __syncthreads();

    for (unsigned s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];   // TODO 2: same here
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

// max is exactly reproducible: unlike a float sum, no rounding depends on the
// order, so the GPU answer must match the CPU answer bit for bit.
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

    // Length is deliberately not a multiple of BLOCK, and every value is
    // negative - so any lane that contributes 0.0f wins the max and lies.
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
