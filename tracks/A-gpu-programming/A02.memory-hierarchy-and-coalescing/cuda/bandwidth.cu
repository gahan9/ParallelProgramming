// Lab 1 (CUDA) - Establish the top of the memory hierarchy: peak achievable HBM bandwidth.
//
// Line-for-line mirror of hip/bandwidth.cpp. A fully coalesced grid-stride copy gives
// the practical HBM roofline every other memory-bound kernel is compared against.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda bandwidth.cu -o bandwidth.exe
// Run:   ./bandwidth.exe
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "check.cuh"

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int n) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = in[i];
    }
}

int main() {
    const int    n     = 1 << 26;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int    iters = 50;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 1, bytes));

    const int threads_per_block = 256;
    const int blocks = 1024;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    copy_kernel<<<blocks, threads_per_block>>>(d_in, d_out, n);   // warmup
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        copy_kernel<<<blocks, threads_per_block>>>(d_in, d_out, n);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seconds = (ms / 1.0e3) / iters;
    double gbps = (2.0 * bytes) / seconds / 1.0e9;

    printf("copy: n = %d, %.1f MB per buffer\n", n, bytes / 1.0e6);
    printf("time / iter      : %.3f ms\n", ms / iters);
    printf("effective BW     : %.1f GB/s  (your practical HBM ceiling)\n", gbps);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
