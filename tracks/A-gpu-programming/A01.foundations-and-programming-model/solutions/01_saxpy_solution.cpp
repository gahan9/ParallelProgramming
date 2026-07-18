// Exercise 1 solution - SAXPY: Y[i] = a * X[i] + Y[i].
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_saxpy_solution.cpp -o 01_saxpy.exe
// Run:   ./01_saxpy.exe   ->   "SAXPY PASSED (0 errors)"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

__global__ void saxpy(int n, float a, const float* X, float* Y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // who am I?
    if (idx < n) {                                       // guard the partial last block
        Y[idx] = a * X[idx] + Y[idx];
    }
}

int main() {
    const int    n     = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float  a     = 2.0f;

    std::vector<float> h_X(n), h_Y(n), h_Y_ref(n);
    for (int i = 0; i < n; ++i) {
        h_X[i]     = 1.0f;
        h_Y[i]     = static_cast<float>(i % 10);
        h_Y_ref[i] = a * h_X[i] + h_Y[i];
    }

    float *d_X = nullptr, *d_Y = nullptr;
    HIP_CHECK(hipMalloc(&d_X, bytes));
    HIP_CHECK(hipMalloc(&d_Y, bytes));
    HIP_CHECK(hipMemcpy(d_X, h_X.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Y, h_Y.data(), bytes, hipMemcpyHostToDevice));

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;  // ceil division

    saxpy<<<blocks, threads_per_block>>>(n, a, d_X, d_Y);
    HIP_CHECK(hipGetLastError());        // launch-config error?
    HIP_CHECK(hipDeviceSynchronize());   // execution error?

    HIP_CHECK(hipMemcpy(h_Y.data(), d_Y, bytes, hipMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_Y[i] - h_Y_ref[i]) > 1e-5f) ++errors;
    }
    printf("SAXPY %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    HIP_CHECK(hipFree(d_X));
    HIP_CHECK(hipFree(d_Y));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
