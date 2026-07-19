// check.h - HIP runtime error-checking macro (shared across the module's hip/ code).
//
// Same contract as A01: every HIP runtime call returns a hipError_t that must be
// inspected. Ignoring it turns a clear failure into garbage output or a crash far
// from the cause. Kernel launches are asynchronous, so also check hipGetLastError()
// right after a launch, and let the next hipDeviceSynchronize() surface run errors.
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
