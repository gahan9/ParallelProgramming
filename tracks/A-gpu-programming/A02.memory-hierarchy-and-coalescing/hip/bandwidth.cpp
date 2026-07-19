// Lab 1 (HIP) - Establish the top of the memory hierarchy: peak achievable HBM bandwidth.
//
// Before you can judge whether a kernel is "good", you need the ceiling. This kernel
// does the most bandwidth-friendly thing possible: a fully coalesced grid-stride copy
// (out[i] = in[i]). Its effective GB/s is the practical HBM roofline you compare every
// other memory-bound kernel against (see README S5 and Track B02).
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip bandwidth.cpp -o bandwidth.exe
// Run:   ./bandwidth.exe
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "check.h"

// Grid-stride loop: every thread strides by the whole grid, so consecutive threads
// always touch consecutive addresses -> perfectly coalesced, and any N is covered.
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = in[i];
    }
}

int main() {
    const int    n     = 1 << 26;                       // ~67M floats (~256 MB per buffer)
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int    iters = 50;

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemset(d_in, 1, bytes));

    const int threads_per_block = 256;
    const int blocks = 1024;                            // enough to saturate the GPU

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    copy_kernel<<<blocks, threads_per_block>>>(d_in, d_out, n);   // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        copy_kernel<<<blocks, threads_per_block>>>(d_in, d_out, n);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double seconds = (ms / 1.0e3) / iters;
    double gbps = (2.0 * bytes) / seconds / 1.0e9;      // 1 read + 1 write per element

    printf("copy: n = %d, %.1f MB per buffer\n", n, bytes / 1.0e6);
    printf("time / iter      : %.3f ms\n", ms / iters);
    printf("effective BW     : %.1f GB/s  (your practical HBM ceiling)\n", gbps);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
