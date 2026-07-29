// check.cuh - CUDA runtime error-checking macro.
//
// Why this exists: CUDA runtime calls return a cudaError_t that is almost always
// ignored by beginners (see the repo's legacy 02.vector_add). Ignoring it turns a
// clear "allocation failed" into garbage output or a crash far from the real cause.
// Kernel launches are asynchronous, so their errors surface at the *next* sync call,
// not at the launch line - which is why we also check hipGetLastError()/cudaGetLastError()
// right after every launch (see the .cu files).
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
