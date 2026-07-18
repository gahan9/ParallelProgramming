// Exercise 2 solution - Fix the histogram data race with atomics.
//
// WHY the original was wrong:
//   `bins[b]++` compiles to load -> add -> store. When two threads target the same
//   bin concurrently, both may load the same old value, both add 1, and both store
//   old+1 - so two increments collapse into one. Counts come out low and vary run to
//   run. This is a classic read-modify-write data race.
//
// THE FIX: atomicAdd(&bins[b], 1) performs the whole read-modify-write as one
//   indivisible hardware operation, so concurrent updates to the same bin serialize
//   correctly while updates to DIFFERENT bins still run in parallel.
//
// TRADEOFF / next step: if many threads hit the same few bins, atomics on global
//   memory contend. The standard optimization (Module A04-adjacent) is a per-block
//   histogram in shared memory (LDS), then one atomic per bin into global memory.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_race_condition_solution.cpp -o 02_race.exe
// Run:   ./02_race.exe   ->   "Histogram PASSED"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define NBINS 16

__global__ void histogram(const int* data, int n, int* bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int b = data[idx] % NBINS;
        atomicAdd(&bins[b], 1);        // FIX: indivisible increment
    }
}

int main() {
    const int n = 1 << 20;
    std::vector<int> h_data(n);
    std::vector<int> h_ref(NBINS, 0);
    for (int i = 0; i < n; ++i) {
        h_data[i] = i * 2654435761u % 1000;
        h_ref[h_data[i] % NBINS]++;
    }

    int *d_data = nullptr, *d_bins = nullptr;
    HIP_CHECK(hipMalloc(&d_data, n * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_bins, NBINS * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_data, h_data.data(), n * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_bins, 0, NBINS * sizeof(int)));

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    histogram<<<blocks, threads_per_block>>>(d_data, n, d_bins);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<int> h_bins(NBINS);
    HIP_CHECK(hipMemcpy(h_bins.data(), d_bins, NBINS * sizeof(int), hipMemcpyDeviceToHost));

    int errors = 0;
    for (int b = 0; b < NBINS; ++b) {
        if (h_bins[b] != h_ref[b]) ++errors;
    }
    printf("Histogram %s\n", errors == 0 ? "PASSED" : "FAILED (data race?)");

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_bins));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
