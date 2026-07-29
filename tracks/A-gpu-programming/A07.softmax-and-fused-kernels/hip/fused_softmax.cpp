// Lab 2 (HIP) - Kernel fusion: scale -> mask -> softmax in one kernel instead of three.
//
// The attention score pipeline is elementwise-then-rowwise:
//
//     y = softmax( (Q K^T) / sqrt(d) + causal_mask )
//
// Written the obvious way that is three kernel launches, and each one makes a full
// round trip to HBM: read the whole tensor, write the whole tensor. Nothing is reused
// across launches - a GPU register file does not survive a kernel boundary, and an
// L2 cache will not hold a multi-MiB tensor.
//
//   unfused  scale kernel   1 read + 1 write
//            mask  kernel   1 read + 1 write
//            softmax        2 reads + 1 write   (online 2-pass, see softmax.cpp)
//            ---------------------------------
//            total          7 traversals of the tensor + 3 launch overheads
//
//   fused    one kernel     1 read + 1 write = 2 traversals, 1 launch
//
// 3.5x less HBM traffic for identical arithmetic. On a ~1 FLOP/byte kernel that is
// the whole game: scale and mask are free once the data is already in a register.
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip fused_softmax.cpp -o fused_softmax.exe
// Run:   ./fused_softmax.exe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256                 // power of two, multiple of 32 and 64
#define ITEMS 8                   // registers per thread -> handles cols <= 2048
#define ELEMENTWISE_BLOCK 256

// ---------------------------------------------------------------------------
// The online-softmax combine operator (see hip/softmax.cpp for the derivation).
// The `mn == -inf` guard is what makes additive -inf masking safe.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void combine(float& m, float& s, float m2, float s2) {
    const float mn = fmaxf(m, m2);
    s = (mn == -INFINITY) ? 0.0f : s * expf(m - mn) + s2 * expf(m2 - mn);
    m = mn;
}

__device__ __forceinline__ void block_combine(float& m, float& s, float* sm, float* ss) {
    const int tid = threadIdx.x;
    sm[tid] = m;
    ss[tid] = s;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float mm = sm[tid], sk = ss[tid];
            combine(mm, sk, sm[tid + stride], ss[tid + stride]);
            sm[tid] = mm;
            ss[tid] = sk;
        }
        __syncthreads();
    }
    m = sm[0];
    s = ss[0];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// The unfused pipeline: three kernels, three HBM round trips.
// ---------------------------------------------------------------------------

__global__ void scale_kernel(const float* __restrict__ x, float* __restrict__ t,
                             size_t n, float scale) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride)
        t[i] = x[i] * scale;
}

// Additive causal mask: everything strictly above the diagonal becomes -inf, so its
// exp() is exactly 0. Adding -inf (rather than multiplying by 0) is what lets the same
// softmax kernel serve causal, padding, and sliding-window masks unchanged.
__global__ void causal_mask_kernel(float* __restrict__ t, int s) {
    const int row = blockIdx.y;
    if (row >= s) return;
    float* tr = t + static_cast<size_t>(row) * s;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < s; c += blockDim.x * gridDim.x)
        tr[c] = tr[c] + ((c > row) ? -INFINITY : 0.0f);
}

__global__ void softmax_rows_kernel(const float* __restrict__ t, float* __restrict__ y,
                                    int rows, int cols) {
    __shared__ float sm[BLOCK];
    __shared__ float ss[BLOCK];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* tr = t + static_cast<size_t>(row) * cols;
    float*       yr = y + static_cast<size_t>(row) * cols;

    float m = -INFINITY, s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) combine(m, s, tr[c], 1.0f);
    block_combine(m, s, sm, ss);

    const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        yr[c] = (inv > 0.0f) ? expf(tr[c] - m) * inv : 0.0f;
}

// ---------------------------------------------------------------------------
// The fused kernel: read x once into registers, apply scale and mask there, run the
// online statistics, write y once. The scratch tensor `t` disappears entirely.
//
// The whole row is loaded even though a causal row only needs its lower triangle, so
// that the traffic comparison against the unfused path stays apples-to-apples. A
// production causal kernel skips the fully masked blocks and saves ~2x again - that is
// FlashAttention's block-sparsity optimisation, covered in A09.
// ---------------------------------------------------------------------------
template <int PER_THREAD>
__global__ void fused_scale_mask_softmax(const float* __restrict__ x, float* __restrict__ y,
                                         int rows, int cols, float scale) {
    __shared__ float sm[BLOCK];
    __shared__ float ss[BLOCK];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float*       yr = y + static_cast<size_t>(row) * cols;

    float v[PER_THREAD];
    float m = -INFINITY, s = 0.0f;
#pragma unroll
    for (int k = 0; k < PER_THREAD; ++k) {
        const int   c    = threadIdx.x + k * BLOCK;          // coalesced segment per k
        const bool  in   = (c < cols);                       // bounds guard
        const float raw  = in ? xr[c] : 0.0f;                // the ONE read of the row
        const bool  keep = in && (c <= row);                 // additive causal mask
        v[k] = keep ? raw * scale : -INFINITY;               // scale happens in a register
        combine(m, s, v[k], keep ? 1.0f : 0.0f);
    }
    block_combine(m, s, sm, ss);

    const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
#pragma unroll
    for (int k = 0; k < PER_THREAD; ++k) {
        const int c = threadIdx.x + k * BLOCK;
        if (c < cols) yr[c] = (inv > 0.0f) ? expf(v[k] - m) * inv : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

static void run_unfused(const float* d_x, float* d_t, float* d_y, int s, float scale) {
    const size_t n = static_cast<size_t>(s) * s;

    const int grid_elem = static_cast<int>((n + ELEMENTWISE_BLOCK - 1) / ELEMENTWISE_BLOCK);
    scale_kernel<<<grid_elem, ELEMENTWISE_BLOCK>>>(d_x, d_t, n, scale);

    dim3 mask_grid((s + ELEMENTWISE_BLOCK - 1) / ELEMENTWISE_BLOCK, s);
    causal_mask_kernel<<<mask_grid, ELEMENTWISE_BLOCK>>>(d_t, s);

    softmax_rows_kernel<<<s, BLOCK>>>(d_t, d_y, s, s);
}

static void run_fused(const float* d_x, float* d_y, int s, float scale) {
    fused_scale_mask_softmax<ITEMS><<<s, BLOCK>>>(d_x, d_y, s, s, scale);
}

static double time_ms(bool fused, const float* d_x, float* d_t, float* d_y,
                      int s, float scale, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    if (fused) run_fused(d_x, d_y, s, scale); else run_unfused(d_x, d_t, d_y, s, scale);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        if (fused) run_fused(d_x, d_y, s, scale); else run_unfused(d_x, d_t, d_y, s, scale);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

// Double-precision reference for softmax(scale * x + causal_mask), row by row.
// Convention: a masked position gets probability exactly 0.
static void reference(const std::vector<float>& x, std::vector<double>& ref, int s, double scale) {
    for (int r = 0; r < s; ++r) {
        const float* xr = x.data() + static_cast<size_t>(r) * s;
        double* rr = ref.data() + static_cast<size_t>(r) * s;

        double m = -INFINITY;
        for (int c = 0; c <= r; ++c) {
            const double v = static_cast<double>(xr[c]) * scale;
            if (v > m) m = v;
        }
        double sum = 0.0;
        for (int c = 0; c <= r; ++c) sum += std::exp(static_cast<double>(xr[c]) * scale - m);
        for (int c = 0; c < s; ++c)
            rr[c] = (c <= r) ? std::exp(static_cast<double>(xr[c]) * scale - m) / sum : 0.0;
    }
}

struct Check {
    double max_err;
    long   non_finite;
};

static Check verify(const float* d_y, const std::vector<double>& ref, int s) {
    const size_t n = static_cast<size_t>(s) * s;
    std::vector<float> y(n);
    HIP_CHECK(hipMemcpy(y.data(), d_y, n * sizeof(float), hipMemcpyDeviceToHost));

    Check c{0.0, 0};
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(y[i])) {
            ++c.non_finite;
            c.max_err = INFINITY;
            continue;
        }
        const double e = std::fabs(static_cast<double>(y[i]) - ref[i]);
        if (e > c.max_err) c.max_err = e;
    }
    return c;
}

int main() {
    const int    s      = 2048;                 // 2048 x 2048 causal attention scores
    const int    d_head = 128;                  // scale = 1/sqrt(d_head), as in attention
    const int    iters  = 20;
    const size_t n      = static_cast<size_t>(s) * s;
    const size_t bytes  = n * sizeof(float);    // 16 MiB per tensor
    const float  scale  = 1.0f / std::sqrt(static_cast<float>(d_head));

    if (s > BLOCK * ITEMS) {
        std::fprintf(stderr, "s=%d exceeds BLOCK*ITEMS=%d; raise ITEMS.\n", s, BLOCK * ITEMS);
        return EXIT_FAILURE;
    }

    std::vector<float> h_x(n);
    uint32_t seed = 2024u;
    for (size_t i = 0; i < n; ++i) {            // deterministic LCG in [-4, 4)
        seed = seed * 1664525u + 1013904223u;
        h_x[i] = 8.0f * (static_cast<float>(seed >> 8) / 16777216.0f) - 4.0f;
    }

    std::vector<double> ref(n);
    reference(h_x, ref, s, static_cast<double>(scale));

    float *d_x = nullptr, *d_t = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_t, bytes));          // scratch that only the unfused path needs
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));

    std::printf("scores            : %d x %d fp32 (%.1f MiB per tensor), scale = 1/sqrt(%d)\n",
                s, s, bytes / (1024.0 * 1024.0), d_head);
    std::printf("mask              : causal (additive -inf strictly above the diagonal)\n\n");

    struct Case {
        bool        fused;
        const char* name;
        int         launches;
        double      traversals;                 // modelled reads + writes of the tensor
    } cases[] = {
        {false, "unfused: scale | mask | softmax", 3, 7.0},
        {true,  "fused:   one kernel",             1, 2.0},
    };

    std::printf("%-32s %-9s %-11s %-11s %-9s %-11s %-11s %s\n",
                "pipeline", "launches", "traversals", "HBM MiB", "ms", "GB/s",
                "max |err|", "result");
    std::printf("-------------------------------- --------- ----------- ----------- "
                "--------- ----------- ----------- ------\n");

    double ms_unfused = 0.0, ms_fused = 0.0;
    for (const Case& c : cases) {
        const double ms = time_ms(c.fused, d_x, d_t, d_y, s, scale, iters);

        HIP_CHECK(hipMemset(d_y, 0, bytes));
        if (c.fused) run_fused(d_x, d_y, s, scale); else run_unfused(d_x, d_t, d_y, s, scale);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        const Check  chk  = verify(d_y, ref, s);
        const double mib  = c.traversals * bytes / (1024.0 * 1024.0);
        const double gbps = c.traversals * bytes / (ms / 1.0e3) / 1.0e9;
        const bool   ok   = (chk.non_finite == 0) && (chk.max_err < 5.0e-6);

        std::printf("%-32s %-9d %-11.0f %-11.1f %-9.3f %-11.1f %-11.2e %s\n",
                    c.name, c.launches, c.traversals, mib, ms, gbps, chk.max_err,
                    ok ? "PASSED" : "FAILED");

        if (c.fused) ms_fused = ms; else ms_unfused = ms;
    }

    std::printf("\nHBM traffic  : %.1f MiB unfused vs %.1f MiB fused -> %.2fx less\n",
                7.0 * bytes / (1024.0 * 1024.0), 2.0 * bytes / (1024.0 * 1024.0), 7.0 / 2.0);
    std::printf("measured     : %.3f ms unfused vs %.3f ms fused -> %.2fx faster\n",
                ms_unfused, ms_fused, (ms_fused > 0.0) ? ms_unfused / ms_fused : 0.0);
    std::printf("\nThe speedup normally lands BELOW the 3.5x traffic ratio: the scale and mask\n"
                "kernels are pure streaming (they hit close to peak bandwidth) while the fused\n"
                "kernel carries reduction barriers, and part of the unfused traffic may be\n"
                "served by L2 rather than HBM. Compare the two numbers on your GPU and explain\n"
                "the gap - that reasoning is the point of the lab.\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_t));
    HIP_CHECK(hipFree(d_y));
    return 0;
}
