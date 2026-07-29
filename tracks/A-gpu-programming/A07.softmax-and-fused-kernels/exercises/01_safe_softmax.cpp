// Exercise 1 - Make softmax numerically safe.
//
// This is the 3-pass row softmax: find the row max, sum exp(x - max), normalize.
// Two things are missing, and the program tells you loudly:
//
//   TODO 1: block_reduce_max() is a stub that returns 0.0f. With max == 0 the second
//           pass computes exp(x) unshifted, so the poisoned score x = 100 overflows
//           fp32 (expf overflows above ~88.7) and row 0 fills with inf / nan.
//   TODO 2: the third pass forgot to subtract the max. Even with TODO 1 done, the
//           numerator and the denominator are then computed with different shifts,
//           so the row no longer sums to 1.
//
// Expected progression:
//   as shipped       -> FAILED: row 0 overflows in pass 2, non-finite outputs > 0.
//                       (Every other row happens to be right, because pass 2 and
//                        pass 3 agree on the wrong shift of 0.)
//   after TODO 1     -> FAILED: pass 2 shifts by the max and pass 3 does not, so each
//                       row sums to exp(max) instead of 1 - and the poisoned row now
//                       overflows in pass 3 instead of pass 2.
//   after TODO 1 + 2 -> PASSED, max |err| ~1e-7, row 0 sum = 1.000000
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_safe_softmax.cpp -o 01_safe_softmax.exe
// Run:   ./01_safe_softmax.exe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256
#define MAX_WAVES 8

// ---------------------------------------------------------------------------
// GIVEN: the sum reduction. Use it as the template for TODO 1.
//
// warpSize is a device built-in (64 on a CDNA wavefront, 32 on an NVIDIA warp), so
// nothing here hard-codes a lane count. __shfl_xor is a butterfly: after the loop
// EVERY lane holds the reduced value, so no separate broadcast is needed.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float wave_sum(float v) {
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
    const int lane   = threadIdx.x % warpSize;
    const int wave   = threadIdx.x / warpSize;
    const int nwaves = (blockDim.x + warpSize - 1) / warpSize;

    v = wave_sum(v);                       // 1. reduce inside the wave
    if (lane == 0) red[wave] = v;          // 2. one partial per wave
    __syncthreads();
    float r = (threadIdx.x < nwaves) ? red[threadIdx.x] : 0.0f;   // identity = 0
    __syncthreads();                       // all reads done before slot 0 is rewritten
    if (wave == 0) {                       // 3. wave 0 reduces the partials
        r = wave_sum(r);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    r = red[0];                            // 4. broadcast to the whole block
    __syncthreads();                       // red[] is reusable after this point
    return r;
}

// ---------------------------------------------------------------------------
// TODO 1: implement the block max reduction.
//
// Mirror block_reduce_sum exactly, with two substitutions:
//   * `+=`  becomes  fmaxf(...)
//   * the identity `0.0f` becomes `-INFINITY`   <- getting this wrong is the classic
//                                                  bug: 0.0f as the identity silently
//                                                  clamps every all-negative row.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_reduce_max(float v, float* red) {
    (void)v;
    (void)red;
    return 0.0f;                           // placeholder: NOT the row max
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

    // Pass 2 - the normalizer, computed on the shifted scores.
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local_sum += expf(xr[c] - m);
    const float s = block_reduce_sum(local_sum, red);

    // Pass 3 - normalize.
    const float inv = 1.0f / s;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        // TODO 2: this must use the SAME shift as pass 2. Subtract m here too.
        yr[c] = expf(xr[c]) * inv;
    }
}

// Double-precision CPU reference (with the max trick, so it is the correct answer).
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
    for (size_t i = 0; i < n; ++i) {       // deterministic LCG in [-3, -1): all negative,
        seed = seed * 1664525u + 1013904223u;   // so a 0.0f max identity is also caught
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
    // The 3-pass kernel touches the row three times and writes it once.
    const double gbps = 4.0 * bytes / (ms / 1.0e3) / 1.0e9;
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
