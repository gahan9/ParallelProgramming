// Exercise 1 solution - Kill the divergence by remapping thread -> element.
//
// The two chains are different instruction sequences, so no predicated select can merge
// them. The only lever left is WHICH LANE HANDLES WHICH ELEMENT. Sending the first half
// of the threads to the even elements and the second half to the odd ones makes every
// wavefront class-uniform:
//
//   t <  n/2  ->  e = 2t        (all even  -> even_chain only)
//   t >= n/2  ->  e = 2(t-n/2)+1 (all odd  -> odd_chain only)
//
// The mapping is a bijection onto [0, n), so the output is identical to the divergent
// kernel and the same CPU reference verifies both. Because n/2 is a multiple of 64, the
// class boundary lands exactly on a wavefront boundary on AMD (64 lanes) and NVIDIA (32
// lanes) alike - not one wavefront straddles it.
//
// WHAT IT COSTS: adjacent lanes now touch every other element (stride 2), so global
// load/store efficiency drops toward 50% (A02's coalescing lesson, in reverse). That is
// a deliberate trade: at 128 iterations per element this kernel is arithmetic-bound, so
// halving the wavefront's issued work is worth far more than the wasted bytes. Flip the
// balance - drop `iters` to 1 - and the remap becomes a pessimization. Always name the
// resource you are trading.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_fix_divergence_solution.cpp -o 01_fix_divergence_solution.exe
// Run:   ./01_fix_divergence_solution.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define NUM_SEEDS 10          // inputs are 1.0 .. 10.0

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

    const int half = n >> 1;
    const int e    = (t < half) ? (t << 1) : (((t - half) << 1) | 1);

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
    // table - computed once, then every one of the n outputs is checked against it. The
    // reference is indexed by ELEMENT parity, which is what makes it valid for both the
    // divergent and the remapped kernel: the remap changes who does the work, not the answer.
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
    printf("%-34s : %7.3f ms  %s\n", "remapped   (wavefront-uniform)",
           uni_ms, uni_err == 0 ? "PASSED" : "FAILED");
    printf("%-34s : %7.2fx\n", "speedup", div_ms / uni_ms);
    printf("\nThe divergent wavefront runs sqrt-chain AND Newton-chain with half its lanes\n"
           "masked; the remapped one runs exactly one chain per wavefront. Same results,\n"
           "same total useful FLOPs.\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return (div_err == 0 && uni_err == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
