// Solution 1 - Coalesced grid-stride scale kernel.
//
// The fix: index with a grid-stride loop where the *base* index is the global thread
// id and the step is the total thread count. On any single iteration, lane i and lane
// i+1 differ by exactly 1 element -> consecutive addresses within a warp -> one
// coalesced transaction per warp. Successive iterations stride by the whole grid, so
// each "wave" is independently coalesced and any n is covered.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_coalesce_solution.cpp -o 01_coalesce.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

__global__ void scale_coalesced(const float* in, float* out, int n, float alpha) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = alpha * in[i];      // lane i, lane i+1 -> addresses i, i+1: coalesced
    }
}

int main() {
    const int    n     = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float  alpha = 3.0f;
    const int    iters = 30;

    std::vector<float> h_in(n);
    for (int i = 0; i < n; ++i) h_in[i] = static_cast<float>(i % 100);

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = 1024;

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    scale_coalesced<<<blocks, threads_per_block>>>(d_in, d_out, n, alpha);  // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < iters; ++it)
        scale_coalesced<<<blocks, threads_per_block>>>(d_in, d_out, n, alpha);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double gbps = (2.0 * bytes) / ((ms / 1.0e3) / iters) / 1.0e9;

    std::vector<float> h_out(n);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < n; ++i)
        if (std::fabs(h_out[i] - alpha * h_in[i]) > 1e-3f) ++errors;

    printf("effective BW : %.1f GB/s\n", gbps);
    printf("result       : %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
