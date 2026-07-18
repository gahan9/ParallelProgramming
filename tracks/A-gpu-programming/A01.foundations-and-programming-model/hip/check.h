// check.h - HIP runtime error-checking macro.
//
// Why this exists: HIP runtime calls return a hipError_t that is almost always
// ignored by beginners (see the repo's legacy 02.vector_add, which assigns
// hip_error = hipMalloc(...) and never inspects it). Ignoring it turns a clear
// "allocation failed" into garbage output or a crash far from the real cause.
// Kernel launches are asynchronous, so their errors surface at the *next* sync call,
// not at the launch line - which is why we also check hipGetLastError() right after
// every launch (see the .cpp files).
//
// Usage:
//   HIP_CHECK(hipMalloc(&p, bytes));
//   kernel<<<g, b>>>(...);
//   HIP_CHECK(hipGetLastError());        // launch-config errors
//   HIP_CHECK(hipDeviceSynchronize());   // execution errors
#pragma once

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expr)                                                        \
    do {                                                                      \
        hipError_t err_ = (expr);                                             \
        if (err_ != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error %s:%d: '%s' -> %s\n",           \
                         __FILE__, __LINE__, #expr,                           \
                         hipGetErrorString(err_));                           \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)
