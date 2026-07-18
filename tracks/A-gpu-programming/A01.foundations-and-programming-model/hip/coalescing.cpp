// Lab 3 (HIP) - Coalescing: why HOW you read memory dominates.
//
// Identical arithmetic, different access pattern. Thread i touches element i*stride.
//   stride = 1  -> consecutive addresses within a wavefront -> coalesced -> fast
//   stride > 1  -> scattered addresses -> many wasted transactions -> slow
// We sweep strides and report the effective bandwidth of USEFUL bytes moved.
// This is the classic stride experiment (Harris, "How to Access Global Memory
// Efficiently"); it is the concrete on-ramp to the roofline model (Track B02).
//
// Build: hipcc -O3 --offload-arch=gfx942 coalescing.cpp -o coalescing.exe
// Run:   ./coalescing.exe
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "check.h"

// Each active thread reads and writes one element at position i*stride.
__global__ void strided_copy(const float* in, float* out, int n, int stride) {
    long long i   = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long idx = i * stride;
    if (idx < n) {
        out[idx] = in[idx] + 1.0f;
    }
}

static double time_stride(const float* d_in, float* d_out, int n, int stride,
                          int iters) {
    const int threads_per_block = 256;
    long long active_threads = (n + stride - 1) / stride;
    int blocks = static_cast<int>(
        (active_threads + threads_per_block - 1) / threads_per_block);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warmup (first launch pays one-time costs we don't want in the measurement).
    strided_copy<<<blocks, threads_per_block>>>(d_in, d_out, n, stride);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        strided_copy<<<blocks, threads_per_block>>>(d_in, d_out, n, stride);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    double seconds = (ms / 1.0e3) / iters;
    double useful_bytes = 2.0 * active_threads * sizeof(float);  // 1 read + 1 write
    return useful_bytes / seconds / 1.0e9;                        // GB/s
}

int main() {
    const int    n     = 1 << 26;              // ~67M elements (~256 MB per buffer)
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int    iters = 20;

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemset(d_in, 0, bytes));

    const int strides[] = {1, 2, 4, 8, 16, 32};
    printf("%-8s %-16s\n", "stride", "effective GB/s");
    printf("-------- ----------------\n");
    for (int s : strides) {
        double gbps = time_stride(d_in, d_out, n, s, iters);
        printf("%-8d %-16.1f\n", s, gbps);
    }
    printf("\nExpect: stride 1 (coalesced) near peak HBM; higher strides drop as\n"
           "each wavefront's lanes fall into separate cache lines. Same math, worse\n"
           "memory pattern -> much lower effective bandwidth.\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
