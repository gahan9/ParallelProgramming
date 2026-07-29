// Solution 2 - Additive -inf masking with the fully-masked row handled.
//
//   TODO 1  The mask is added in ALL THREE passes, not just the output pass. A masked
//           position must not raise the row max and must not contribute to the
//           normalizer; applying the mask only at the end normalizes over positions
//           that were supposed to be dropped, which scales every probability down by
//           roughly (valid / cols).
//
//   TODO 2  Two guards, both one comparison:
//             (a) `if (m > -INFINITY)` around the normalizer pass. When every score in
//                 the row is -inf the max is -inf, and exp(-inf - -inf) = exp(nan).
//                 The branch is uniform across the block (m is a block-wide reduction
//                 result), so it contains no barrier and causes no divergence.
//             (b) `inv = (s > 0) ? 1/s : 0` plus the matching `inv > 0` test in the
//                 output pass, which turns an empty row into exact zeros instead of
//                 0/0 = nan.
//
// Convention chosen here: a fully masked row emits ALL ZEROS. That is what PyTorch's
// scaled_dot_product_attention effectively produces for a fully masked query, and it
// keeps the downstream P·V product finite. The other defensible convention is a uniform
// 1/cols distribution; whichever you pick, write it down, because the two disagree and
// the choice is invisible in the kernel signature.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_masked_softmax_solution.cpp -o 02_masked_softmax_solution.exe
// Run:   ./02_masked_softmax_solution.exe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256
#define MAX_WAVES 8

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
    if (row >= rows) return;
    const float* xr = x    + static_cast<size_t>(row) * cols;
    const float* mr = mask + static_cast<size_t>(row) * cols;
    float*       yr = y    + static_cast<size_t>(row) * cols;

    // Pass 1 - the max OF THE MASKED SCORES.
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        local_max = fmaxf(local_max, xr[c] + mr[c]);
    const float m = block_reduce_max(local_max, red);

    // Pass 2 - the normalizer. Skipped entirely when the row is empty: with m == -inf
    // the expression exp((x + mask) - m) is exp(-inf - -inf) = exp(nan).
    float local_sum = 0.0f;
    if (m > -INFINITY) {                   // uniform across the block: no barrier inside
        for (int c = threadIdx.x; c < cols; c += blockDim.x)
            local_sum += expf(xr[c] + mr[c] - m);
    }
    const float s = block_reduce_sum(local_sum, red);

    // Pass 3 - normalize, or emit zeros for an empty row.
    const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        yr[c] = (inv > 0.0f) ? expf(xr[c] + mr[c] - m) * inv : 0.0f;
}

int main() {
    const int rows = 8;
    const int cols = 256;
    const size_t n = static_cast<size_t>(rows) * cols;
    const size_t bytes = n * sizeof(float);

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

    std::vector<double> ref(n, 0.0);
    for (int r = 0; r < rows; ++r) {
        if (valid[r] == 0) continue;       // empty row stays all zeros
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
    std::printf("Row 3 (valid = 0) emits exact zeros. The two guards that make that true -\n"
                "skipping the normalizer pass when m == -inf, and treating s == 0 as \"emit 0\" -\n"
                "cost one comparison each and are the difference between a training run that\n"
                "converges and one that turns every weight into nan.\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_mask));
    HIP_CHECK(hipFree(d_y));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
