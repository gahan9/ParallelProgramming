// check.cuh - CUDA runtime error-checking macro (shared across the module's cuda/ code).
//
// Same contract as A01: every CUDA runtime call returns a cudaError_t that must be
// inspected. Ignoring it turns a clear failure into garbage output or a crash far
// from the cause. Kernel launches are asynchronous, so also check cudaGetLastError()
// right after a launch, and let the next cudaDeviceSynchronize() surface run errors.
//
// Usage:
//   CUDA_CHECK(cudaMalloc(&p, bytes));
//   kernel<<<g, b>>>(...);
//   CUDA_CHECK(cudaGetLastError());        // launch-config errors
//   CUDA_CHECK(cudaDeviceSynchronize());   // execution errors
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                       \
    do {                                                                      \
        cudaError_t err_ = (expr);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: '%s' -> %s\n",          \
                         __FILE__, __LINE__, #expr,                           \
                         cudaGetErrorString(err_));                          \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
