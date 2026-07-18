// Lab 2 (HIP) - Vector addition, production-grade.
//
// Upgraded from the repo's legacy 02.vector_add. What changed and why:
//   * Every runtime call is wrapped in HIP_CHECK (legacy ignored hipError_t).
//   * The kernel launch error is checked explicitly (launches are async).
//   * Result is verified against a CPU reference -> prints PASSED/FAILED.
//   * Kernel time is measured with GPU events (not a CPU clock across an async call).
//   * Effective memory bandwidth is reported (vector add is memory-bound; see README S5).
//
// Build: hipcc -O3 --offload-arch=gfx942 vector_add.cpp -o vector_add.exe
// Run:   ./vector_add.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

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
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_B, bytes));
    HIP_CHECK(hipMalloc(&d_C, bytes));

    // Host -> device.
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

    const int threads_per_block = 256;         // multiple of 64 -> safe on AMD + NVIDIA
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Time the kernel with GPU events (correct way to time an async launch).
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    vector_add<<<blocks, threads_per_block>>>(d_A, d_B, d_C, n);
    HIP_CHECK(hipGetLastError());              // launch-config error?
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    // Device -> host (this also synchronizes).
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

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

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
