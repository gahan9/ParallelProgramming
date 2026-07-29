// Exercise 2 - Additive masking, and the fully-masked row that produces NaN.
//
// Real attention softmax never sees a clean row. Padding and causal masks arrive as an
// ADDITIVE mask of 0 (keep) or -inf (drop), because adding -inf before the exp is the
// only formulation that survives the max trick unchanged: exp(-inf - m) is exactly 0
// for any finite m, so a dropped position contributes nothing to the normalizer.
//
// The batch below includes row 3 with valid length 0 - an entry whose real tokens were
// all truncated away. Every score in that row is -inf, so the row max is -inf, and
// exp(-inf - -inf) = exp(nan) = nan. One such row silently NaNs an entire training step.
//
//   TODO 1: the mask is applied only in the final pass. Apply it in the max pass and
//           the normalizer pass too, or the distribution is normalized over positions
//           that were supposed to be dropped.
//   TODO 2: fixing TODO 1 makes row 3's max -inf and introduces the NaN. Guard it.
//           Convention for this drill: a fully masked row emits all zeros.
//
// Expected progression:
//   as shipped       -> FAILED, finite but wrong (masked positions still normalized in)
//   after TODO 1     -> FAILED, row 3 is nan
//   after TODO 1 + 2 -> PASSED, row 3 sums to exactly 0
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_masked_softmax.cpp -o 02_masked_softmax.exe
// Run:   ./02_masked_softmax.exe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256
#define MAX_WAVES 8

// GIVEN: the block reductions from exercise 1 (identity -inf for max, 0 for sum).
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
    float r = (threadIdx.x < nwaves) ? red[threadIdx.x] : -INFINITY;
    __syncthreads();
    if (wave == 0) {
        r = wave_max(r);
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
    const int lane   = threadIdx.x % warpSize;
    const int wave   = threadIdx.x / warpSize;
    const int nwaves = (blockDim.x + warpSize - 1) / warpSize;

    v = wave_sum(v);
    if (lane == 0) red[wave] = v;
    __syncthreads();
    float r = (threadIdx.x < nwaves) ? red[threadIdx.x] : 0.0f;
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

__global__ void masked_softmax(const float* __restrict__ x, const float* __restrict__ mask,
                               float* __restrict__ y, int rows, int cols) {
    __shared__ float red[MAX_WAVES];

    const int row = blockIdx.x;
    if (row >= rows) return;               // uniform across the block: barriers stay safe
    const float* xr = x    + static_cast<size_t>(row) * cols;
    const float* mr = mask + static_cast<size_t>(row) * cols;
    float*       yr = y    + static_cast<size_t>(row) * cols;

    // Pass 1 - the row max.
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, xr[c]);        // TODO 1: the mask belongs here
    const float m = block_reduce_max(local_max, red);

    // Pass 2 - the normalizer.
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_sum += expf(xr[c] - m);               // TODO 1: ...and here
    const float s = block_reduce_sum(local_sum, red);

    // Pass 3 - normalize. The mask IS applied here, which is why the output has zeros
    // in the right places while the values themselves are wrong.
    const float inv = 1.0f / s;                     // TODO 2: s can be 0 (empty row)
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        yr[c] = expf(xr[c] + mr[c] - m) * inv;      // TODO 2: m can be -inf -> exp(nan)
}

int main() {
    const int rows = 8;
    const int cols = 256;
    const size_t n = static_cast<size_t>(rows) * cols;
    const size_t bytes = n * sizeof(float);

    // Valid (unmasked) prefix length per row. Row 3 is the killer: nothing survives.
    const int valid[rows] = {256, 200, 128, 0, 64, 1, 250, 17};

    std::vector<float> h_x(n), h_mask(n);
    uint32_t seed = 99u;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            seed = seed * 1664525u + 1013904223u;
            const size_t i = static_cast<size_t>(r) * cols + c;
            h_x[i]    = 6.0f * (static_cast<float>(seed >> 8) / 16777216.0f) - 3.0f;
            h_mask[i] = (c < valid[r]) ? 0.0f : -INFINITY;
        }

    // Double-precision reference: softmax over the valid prefix, 0 elsewhere.
    // A row with no valid positions is defined to be all zeros.
    std::vector<double> ref(n, 0.0);
    for (int r = 0; r < rows; ++r) {
        if (valid[r] == 0) continue;
        const float* xr = h_x.data() + static_cast<size_t>(r) * cols;
        double m = -INFINITY;
        for (int c = 0; c < valid[r]; ++c) m = (xr[c] > m) ? xr[c] : m;
        double sum = 0.0;
        for (int c = 0; c < valid[r]; ++c) sum += std::exp(static_cast<double>(xr[c]) - m);
        for (int c = 0; c < valid[r]; ++c)
            ref[static_cast<size_t>(r) * cols + c] =
                std::exp(static_cast<double>(xr[c]) - m) / sum;
    }

    float *d_x = nullptr, *d_mask = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_mask, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_mask, h_mask.data(), bytes, hipMemcpyHostToDevice));

    masked_softmax<<<rows, BLOCK>>>(d_x, d_mask, d_y, rows, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_y(n);
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, bytes, hipMemcpyDeviceToHost));

    std::printf("%-5s %-7s %-13s %-13s %-9s %s\n",
                "row", "valid", "row sum", "max |err|", "non-fin", "result");
    std::printf("----- ------- ------------- ------------- --------- ------\n");

    int failed_rows = 0;
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0, err = 0.0;
        long   bad = 0;
        for (int c = 0; c < cols; ++c) {
            const size_t i = static_cast<size_t>(r) * cols + c;
            if (!std::isfinite(h_y[i])) { ++bad; continue; }
            sum += h_y[i];
            const double e = std::fabs(static_cast<double>(h_y[i]) - ref[i]);
            if (e > err) err = e;
        }
        const double want = (valid[r] == 0) ? 0.0 : 1.0;
        const bool ok = (bad == 0) && (err < 5.0e-6) && (std::fabs(sum - want) < 1.0e-4);
        if (!ok) ++failed_rows;
        std::printf("%-5d %-7d %-13.6f %-13.3e %-9ld %s\n",
                    r, valid[r], sum, err, bad, ok ? "PASSED" : "FAILED");
    }

    const bool all_ok = (failed_rows == 0);
    std::printf("\noverall: %s\n", all_ok ? "PASSED" : "FAILED");
    std::printf("Row 3 is the one that matters: valid = 0, so every score is -inf. It must\n"
                "produce a row of exact zeros, never a NaN - a single NaN here propagates\n"
                "through the whole backward pass and poisons every weight in the model.\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_mask));
    HIP_CHECK(hipFree(d_y));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
