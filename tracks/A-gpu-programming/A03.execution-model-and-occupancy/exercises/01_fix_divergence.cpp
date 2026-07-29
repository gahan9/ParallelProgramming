// Exercise 1 - Fix a kernel whose divergence is destroying throughput.
//
// The workload: element `e` gets one of two ITERATIVE transforms depending on its
// parity. Even elements run a sqrt chain, odd elements run a Newton division chain.
// The two bodies are genuinely different instruction sequences, so no compiler can
// collapse them into a predicated select - the only way out is to change WHICH LANE
// HANDLES WHICH ELEMENT.
//
//   checkerboard_divergent : thread t handles element t. Lane parity alternates, so
//                            every wavefront contains both classes and executes BOTH
//                            chains with half its lanes masked off. ~2x the work.
//
// YOUR TASK (TODO 1): in `checkerboard_uniform`, remap thread -> element so that every
// wavefront handles a single parity. Hint: the first half of the threads should take
// the even elements (0,2,4,...) and the second half the odd ones (1,3,5,...). `n` is
// chosen so that n/2 is a multiple of 64, which puts the class boundary exactly on a
// wavefront boundary on both AMD (64) and NVIDIA (32).
//
// Both kernels must stay PASSED - the bug here is performance, not correctness.
// Target: the remapped kernel should be clearly faster (roughly the average of the two
// chains instead of their sum).
//
// Think about what the remap COSTS: adjacent lanes now touch every other element
// (stride 2), so global access efficiency drops. That is a deliberate trade - this
// kernel is arithmetic-bound (128 iterations per element), so the divergence tax is
// worth far more than the coalescing. State that trade in your own words when you are
// done; the solution spells it out.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_fix_divergence.cpp -o 01_fix_divergence.exe
// Run:   ./01_fix_divergence.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define NUM_SEEDS 10          // inputs are 1.0 .. 10.0

// Two contracting maps with distinct fixed points (4.0 and 3.0), so a wrong
// thread -> element mapping shows up immediately as FAILED rather than as noise.
__device__ __forceinline__ float even_chain(float x, int iters) {
    for (int k = 0; k < iters; ++k) x = sqrtf(4.0f * x);            // -> 4.0
    return x;
}

__device__ __forceinline__ float odd_chain(float x, int iters) {
    for (int k = 0; k < iters; ++k) x = 0.5f * (x + 9.0f / x);      // -> 3.0
    return x;
}

__global__ void checkerboard_divergent(const float* __restrict__ in,
                                       float* __restrict__ out, int n, int iters) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    const int e = t;                       // lane parity alternates -> divergent wavefronts
    out[e] = (e & 1) ? odd_chain(in[e], iters) : even_chain(in[e], iters);
}

__global__ void checkerboard_uniform(const float* __restrict__ in,
                                     float* __restrict__ out, int n, int iters) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    // TODO 1: replace this mapping so that all lanes of a wavefront get elements of the
    //         SAME parity. Right now it is identical to the divergent kernel, which is
    //         why the two timings match.
    const int e = t;

    out[e] = (e & 1) ? odd_chain(in[e], iters) : even_chain(in[e], iters);
}

static float time_kernel(bool remapped, const float* d_in, float* d_out, int n, int iters,
                         int block, int timed_reps) {
    const int grid = (n + block - 1) / block;

    auto launch = [&]() {
        if (remapped) checkerboard_uniform  <<<grid, block>>>(d_in, d_out, n, iters);
        else          checkerboard_divergent<<<grid, block>>>(d_in, d_out, n, iters);
    };

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    launch();                                             // warmup
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

static int verify(const std::vector<float>& ref, const float* d_out, int n) {
    std::vector<float> host(n);
    HIP_CHECK(hipMemcpy(host.data(), d_out, sizeof(float) * n, hipMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i)
        if (std::fabs(host[i] - ref[i]) > 1e-3f) ++errors;
    return errors;
}

int main() {
    const int    n          = 1 << 22;      // n/2 is a multiple of 64 -> clean class boundary
    const int    iters      = 128;
    const int    block      = 256;
    const int    timed_reps = 10;
    const size_t bytes      = static_cast<size_t>(n) * sizeof(float);

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::vector<float> h_in(n), h_ref(n);
    for (int i = 0; i < n; ++i) h_in[i] = 1.0f + static_cast<float>(i % NUM_SEEDS);

    // A chain's result depends only on (seed, parity), so the reference is a 2 x NUM_SEEDS
    // table - computed once, then every one of the n outputs is checked against it.
    float table[2][NUM_SEEDS];
    for (int s = 0; s < NUM_SEEDS; ++s) {
        float even = 1.0f + static_cast<float>(s);
        float odd  = even;
        for (int k = 0; k < iters; ++k) even = std::sqrt(4.0f * even);
        for (int k = 0; k < iters; ++k) odd  = 0.5f * (odd + 9.0f / odd);
        table[0][s] = even;
        table[1][s] = odd;
    }
    for (int i = 0; i < n; ++i) h_ref[i] = table[i & 1][i % NUM_SEEDS];

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    printf("device: %s, warpSize = %d, n = %d, %d iterations per element\n\n",
           prop.name, prop.warpSize, n, iters);

    HIP_CHECK(hipMemset(d_out, 0, bytes));
    const float div_ms  = time_kernel(false, d_in, d_out, n, iters, block, timed_reps);
    const int   div_err = verify(h_ref, d_out, n);

    HIP_CHECK(hipMemset(d_out, 0, bytes));
    const float uni_ms  = time_kernel(true, d_in, d_out, n, iters, block, timed_reps);
    const int   uni_err = verify(h_ref, d_out, n);

    printf("%-34s : %7.3f ms  %s\n", "divergent  (thread t -> element t)",
           div_ms, div_err == 0 ? "PASSED" : "FAILED");
    printf("%-34s : %7.3f ms  %s\n", "remapped   (your TODO)",
           uni_ms, uni_err == 0 ? "PASSED" : "FAILED");
    printf("%-34s : %7.2fx\n", "speedup", div_ms / uni_ms);

    if (div_ms / uni_ms < 1.15f)
        printf("\nNo real speedup yet - TODO 1 is still the identity mapping.\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return (div_err == 0 && uni_err == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
