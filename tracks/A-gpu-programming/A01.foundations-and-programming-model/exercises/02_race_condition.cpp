// Exercise 2 - Fix the data race.
//
// This histogram kernel counts how many input values fall in each of NBINS bins.
// As written it is WRONG: many threads do `bins[b]++` on the same location at the
// same time. That read-modify-write is not atomic, so increments are lost and the
// totals come out too low and nondeterministic (run it twice - different answers).
//
// YOUR TASK: make the counting correct without serializing the whole kernel.
// Hint: the hardware has an instruction for exactly this. Search "atomicAdd".
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_race_condition.cpp -o 02_race.exe
// Run:   ./02_race.exe        (fix it until it prints PASSED)
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
        bins[b]++;                 // BUG: data race - concurrent read-modify-write.
                                   // TODO: replace with a correct atomic update.
    }
}

int main() {
    const int n = 1 << 20;
    std::vector<int> h_data(n);
    std::vector<int> h_ref(NBINS, 0);
    for (int i = 0; i < n; ++i) {
        h_data[i] = i * 2654435761u % 1000;      // arbitrary spread
        h_ref[h_data[i] % NBINS]++;              // CPU reference counts
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
        if (h_bins[b] != h_ref[b]) {
            printf("bin %2d: gpu=%d cpu=%d\n", b, h_bins[b], h_ref[b]);
            ++errors;
        }
    }
    printf("Histogram %s\n", errors == 0 ? "PASSED" : "FAILED (data race?)");

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_bins));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
