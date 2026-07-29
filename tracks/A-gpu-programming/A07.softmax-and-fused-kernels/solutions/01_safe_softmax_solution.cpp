// Solution 1 - Safe (max-subtracted) 3-pass row softmax.
//
// Two fixes relative to the exercise:
//
//   TODO 1  block_reduce_max is implemented as a mirror of block_reduce_sum, with
//           fmaxf() in place of `+=` and -INFINITY in place of 0.0f as the identity.
//           The identity is the subtle half: with 0.0f, every all-negative row would
//           report a max of 0, the shift would be wrong, and the kernel would be
//           quietly incorrect instead of loudly non-finite. That is why this drill's
//           input is all negative apart from the poison.
//
//   TODO 2  pass 3 subtracts the same m as pass 2. exp(x_i - m) / sum_j exp(x_j - m)
//           is only the softmax when BOTH sides use the same shift; mixing shifts
//           scales the whole row by exp(m).
//
// Why 3 passes and not 2: the safe formula needs the max before it can accumulate the
// normalizer, and the normalizer before it can normalize. The online formulation
// (../hip/softmax.cpp variant 3) removes one of those reads; see
// 03_online_softmax_math_solution.md for the derivation.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_safe_softmax_solution.cpp -o 01_safe_softmax_solution.exe
// Run:   ./01_safe_softmax_solution.exe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256
#define MAX_WAVES 8

// warpSize is a device built-in: 64 on a CDNA wavefront, 32 on an NVIDIA warp. The
// butterfly (__shfl_xor) leaves the reduced value in every lane, so no broadcast is
// needed at the wave level.
__device__ __forceinline__ float wave_max(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off));
    return v;
}

__device__ __forceinline__ float wave_sum(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

__device__ __forceinline__ float block_reduce_max(float v, float* red) {
    const int lane   = threadIdx.x % warpSize;
    const int wave   = threadIdx.x / warpSize;
    const int nwaves = (blockDim.x + warpSize - 1) / warpSize;

    v = wave_max(v);
    if (lane == 0) red[wave] = v;
    __syncthreads();
    float r = (threadIdx.x < nwaves) ? red[threadIdx.x] : -INFINITY;   // identity: -inf
    __syncthreads();                       // all reads done before slot 0 is rewritten
    if (wave == 0) {
        r = wave_max(r);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    r = red[0];
    __syncthreads();                       // red[] is reusable by the next reduction
    return r;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
    const int lane   = threadIdx.x % warpSize;
    const int wave   = threadIdx.x / warpSize;
    const int nwaves = (blockDim.x + warpSize - 1) / warpSize;

    v = wave_sum(v);
    if (lane == 0) red[wave] = v;
    __syncthreads();
    float r = (threadIdx.x < nwaves) ? red[threadIdx.x] : 0.0f;        // identity: 0
    __syncthreads();
    if (wave == 0) {
        r = wave_sum(r);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    r = red[0];
    __syncthreads();
    return r;
}

__global__ void softmax_safe(const float* __restrict__ x, float* __restrict__ y,
                             int rows, int cols) {
    __shared__ float red[MAX_WAVES];

    const int row = blockIdx.x;
    if (row >= rows) return;               // uniform across the block: barriers stay safe
    const float* xr = x + static_cast<size_t>(row) * cols;
    float*       yr = y + static_cast<size_t>(row) * cols;

    // Pass 1 - the row max.
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local_max = fmaxf(local_max, xr[c]);
    const float m = block_reduce_max(local_max, red);

    // Pass 2 - the normalizer. Every exponent is now <= 0, so exp() cannot overflow,
    // and at least one term is exactly 1, so the sum is in [1, cols].
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local_sum += expf(xr[c] - m);
    const float s = block_reduce_sum(local_sum, red);

    // Pass 3 - normalize with the SAME shift. The `s > 0` guard costs one compare and
    // covers the fully masked row (see 02_masked_softmax_solution.cpp).
    const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        yr[c] = (inv > 0.0f) ? expf(xr[c] - m) * inv : 0.0f;
}

static void reference(const std::vector<float>& x, std::vector<double>& ref, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + static_cast<size_t>(r) * cols;
        double* rr = ref.data() + static_cast<size_t>(r) * cols;

        double m = -INFINITY;
        for (int c = 0; c < cols; ++c) m = (xr[c] > m) ? xr[c] : m;
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += std::exp(static_cast<double>(xr[c]) - m);
        for (int c = 0; c < cols; ++c) rr[c] = std::exp(static_cast<double>(xr[c]) - m) / s;
    }
}

int main() {
    const int    rows  = 4096;
    const int    cols  = 512;
    const int    iters = 20;
    const size_t n     = static_cast<size_t>(rows) * cols;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n);
    uint32_t seed = 7u;
    for (size_t i = 0; i < n; ++i) {       // all negative: catches a 0.0f max identity
        seed = seed * 1664525u + 1013904223u;
        h_x[i] = 2.0f * (static_cast<float>(seed >> 8) / 16777216.0f) - 3.0f;
    }
    h_x[7] = 100.0f;                       // row 0, col 7: expf(100) overflows fp32

    std::vector<double> ref(n);
    reference(h_x, ref, rows, cols);

    float *d_x = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    softmax_safe<<<rows, BLOCK>>>(d_x, d_y, rows, cols);       // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) softmax_safe<<<rows, BLOCK>>>(d_x, d_y, rows, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    ms /= iters;

    std::vector<float> h_y(n);
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, bytes, hipMemcpyDeviceToHost));

    double max_err = 0.0;
    long   non_finite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(h_y[i])) { ++non_finite; continue; }
        const double e = std::fabs(static_cast<double>(h_y[i]) - ref[i]);
        if (e > max_err) max_err = e;
    }
    const double gbps = 4.0 * bytes / (ms / 1.0e3) / 1.0e9;    // 3 reads + 1 write
    const bool   ok   = (non_finite == 0) && (max_err < 5.0e-6);

    std::printf("rows = %d, cols = %d, block = %d\n", rows, cols, BLOCK);
    std::printf("time / iter    : %.3f ms\n", ms);
    std::printf("effective BW   : %.1f GB/s   (3 reads + 1 write)\n", gbps);
    std::printf("non-finite     : %ld\n", non_finite);
    std::printf("max |err|      : %.3e\n", max_err);
    std::printf("row 0 sum      : ");
    double row0 = 0.0;
    for (int c = 0; c < cols; ++c) row0 += h_y[c];
    std::printf("%.6f  (must be 1.0)\n", row0);
    std::printf("result         : %s\n", ok ? "PASSED" : "FAILED");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
