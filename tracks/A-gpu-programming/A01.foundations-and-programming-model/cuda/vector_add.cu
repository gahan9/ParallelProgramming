// Lab 2 (CUDA) - Vector addition, production-grade.
//
// Line-for-line comparable to hip/vector_add.cpp. Demonstrates: bounds guard,
// full error checking, CPU verification, event-based timing, effective bandwidth.
//
// Build: nvcc -O3 -arch=sm_90 vector_add.cu -o vector_add.exe
// Run:   ./vector_add.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {                 // mandatory bounds guard: last block is partial
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int    n     = 1 << 24;              // ~16.7M elements
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // Host allocation + init.
    std::vector<float> h_A(n), h_B(n), h_C(n);
    for (int i = 0; i < n; ++i) {
        h_A[i] = static_cast<float>(i % 100);
        h_B[i] = static_cast<float>((n - i) % 100);
    }

    // Device allocation.
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Host -> device.
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    const int threads_per_block = 256;         // multiple of 32 (and 64) -> safe everywhere
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Time the kernel with GPU events (correct way to time an async launch).
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads_per_block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());            // launch-config error?
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Device -> host (this also synchronizes).
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify against CPU reference.
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        float expected = h_A[i] + h_B[i];
        if (std::fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 5) {
                std::fprintf(stderr, "Mismatch at %d: got %f, expected %f\n",
                             i, h_C[i], expected);
            }
            ++errors;
        }
    }

    // Effective bandwidth: 2 reads + 1 write per element.
    double gbps = (3.0 * bytes) / (ms / 1.0e3) / 1.0e9;

    printf("n = %d, block = %d, grid = %d\n", n, threads_per_block, blocks);
    printf("kernel time      : %.3f ms\n", ms);
    printf("effective BW     : %.1f GB/s\n", gbps);
    printf("result           : %s\n", errors == 0 ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
