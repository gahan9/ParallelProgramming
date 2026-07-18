// Lab 3 (CUDA) - Coalescing: why HOW you read memory dominates.
//
// Line-for-line comparable to hip/coalescing.cpp. Thread i touches element i*stride;
// we sweep strides and report effective bandwidth of useful bytes moved.
//
// Build: nvcc -O3 -arch=sm_90 coalescing.cu -o coalescing.exe
// Run:   ./coalescing.exe
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "check.cuh"

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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup (first launch pays one-time costs we don't want in the measurement).
    strided_copy<<<blocks, threads_per_block>>>(d_in, d_out, n, stride);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        strided_copy<<<blocks, threads_per_block>>>(d_in, d_out, n, stride);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double seconds = (ms / 1.0e3) / iters;
    double useful_bytes = 2.0 * active_threads * sizeof(float);  // 1 read + 1 write
    return useful_bytes / seconds / 1.0e9;                        // GB/s
}

int main() {
    const int    n     = 1 << 26;              // ~67M elements (~256 MB per buffer)
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int    iters = 20;

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_in, 0, bytes));

    const int strides[] = {1, 2, 4, 8, 16, 32};
    printf("%-8s %-16s\n", "stride", "effective GB/s");
    printf("-------- ----------------\n");
    for (int s : strides) {
        double gbps = time_stride(d_in, d_out, n, s, iters);
        printf("%-8d %-16.1f\n", s, gbps);
    }
    printf("\nExpect: stride 1 (coalesced) near peak HBM; higher strides drop as\n"
           "each warp's lanes fall into separate cache lines. Same math, worse\n"
           "memory pattern -> much lower effective bandwidth.\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
