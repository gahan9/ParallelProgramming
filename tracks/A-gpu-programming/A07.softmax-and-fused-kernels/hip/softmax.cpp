// Lab 1 (HIP) - Row-wise softmax: overflow, the max trick, and the online recurrence.
//
// Softmax turns a row of scores into a probability distribution. The textbook formula
// exp(x_i) / sum_j exp(x_j) is numerically hostile: expf() overflows to +inf for
// x > ~88.7 in fp32 (ln(FLT_MAX) = 88.72), so a single large score poisons the whole
// row with inf/nan. Four kernels, one row-wise result:
//
//   1) naive_unsafe   - the textbook formula (2 reads + 1 write). FAILS on purpose:
//                       row 0 holds x = 100.
//   2) safe_3pass     - subtract the row max first. Mathematically identical, safe -
//                       but safety costs a pass: THREE reads (max, sum, normalize)
//                       + 1 write.
//   3) online_2pass   - Milakov & Gimelshein: one pass produces BOTH statistics by
//                       rescaling the running sum when a new max appears. TWO reads
//                       + 1 write - safety is now free relative to the naive version.
//   4) online_1pass   - keep the row in registers, so the second read never touches
//                       HBM. ONE read + 1 write == the algorithmic minimum. This is
//                       the seed of FlashAttention (keep the tile on-chip, combine
//                       partial (max, sum) pairs).
//
// Softmax is deeply memory-bound (~1 FLOP/byte), so the pass count IS the runtime.
// Reference: Milakov & Gimelshein, "Online normalizer calculation for softmax",
// https://arxiv.org/abs/1805.02867
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip softmax.cpp -o softmax.exe
// Run:   ./softmax.exe
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

// One block per row. 256 is a multiple of both 32 (warp) and 64 (wavefront), and it
// must be a power of two: the pair-combine tree below halves it each step.
#define BLOCK 256

// Upper bound on waves per block: BLOCK / 32 on NVIDIA, BLOCK / 64 on AMD.
#define MAX_WAVES 8

// Registers per thread for the cached 1-pass kernel -> handles cols <= BLOCK * ITEMS.
#define ITEMS 4

// ---------------------------------------------------------------------------
// Reductions (the A04 skill, reused). warpSize is a device built-in - 64 on CDNA
// wavefronts, 32 on NVIDIA warps - so nothing here hard-codes a lane count.
// A butterfly (__shfl_xor) is used instead of __shfl_down so that *every* lane ends
// up holding the reduced value; no follow-up broadcast is needed.
// ---------------------------------------------------------------------------

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
    __syncthreads();                      // every read is done before slot 0 is rewritten
    if (wave == 0) {
        r = wave_max(r);                  // whole wave 0 enters: the shuffle stays uniform
        if (lane == 0) red[0] = r;
    }
    __syncthreads();
    r = red[0];
    __syncthreads();                      // red[] is safe for the next reduction to reuse
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

// ---------------------------------------------------------------------------
// The online-softmax combine operator.
//
// A partial result is the pair (m, s) = (running max, sum of exp(x - m)) over the
// elements seen so far. Merging two partials:
//
//     m = max(m_a, m_b)
//     s = s_a * exp(m_a - m) + s_b * exp(m_b - m)
//
// Both rescale factors are <= 1, so the merge can never overflow. The operator is
// associative and commutative with identity (-inf, 0), which is exactly why it
// parallelises across lanes, blocks, and - in FlashAttention - across K/V tiles.
//
// The `m == -inf` guard is load-bearing: two empty/fully-masked partials would
// otherwise evaluate exp(-inf - -inf) = exp(nan) = nan and poison the whole row.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void combine(float& m, float& s, float m2, float s2) {
    const float mn = fmaxf(m, m2);
    s = (mn == -INFINITY) ? 0.0f : s * expf(m - mn) + s2 * expf(m2 - mn);
    m = mn;
}

// Tree-combine the per-thread (m, s) pairs. Requires a power-of-two blockDim.x.
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
    __syncthreads();                      // sm/ss are safe to reuse after this point
}

// ---------------------------------------------------------------------------
// 1) Naive, UNSAFE: the textbook formula, no max subtraction.
// ---------------------------------------------------------------------------
__global__ void softmax_naive_unsafe(const float* __restrict__ x, float* __restrict__ y,
                                     int rows, int cols) {
    __shared__ float red[MAX_WAVES];

    const int row = blockIdx.x;
    if (row >= rows) return;              // uniform across the block: no barrier divergence
    const float* xr = x + static_cast<size_t>(row) * cols;
    float*       yr = y + static_cast<size_t>(row) * cols;

    float partial = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        partial += expf(xr[c]);           // expf(100.0f) == +inf in fp32
    const float denom = block_reduce_sum(partial, red);

    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        yr[c] = expf(xr[c]) / denom;      // inf/inf -> nan, finite/inf -> 0
}

// ---------------------------------------------------------------------------
// 2) Safe 3-pass: subtract the row max. Three full reads of the row.
// ---------------------------------------------------------------------------
__global__ void softmax_safe_3pass(const float* __restrict__ x, float* __restrict__ y,
                                   int rows, int cols) {
    __shared__ float red[MAX_WAVES];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float*       yr = y + static_cast<size_t>(row) * cols;

    float m = -INFINITY;                                    // pass 1: the max
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, xr[c]);
    m = block_reduce_max(m, red);

    float s = 0.0f;                                         // pass 2: the normalizer
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += expf(xr[c] - m);
    s = block_reduce_sum(s, red);

    // s == 0 only if every entry was -inf (a fully masked row); see 02_masked_softmax.
    const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)     // pass 3: normalize
        yr[c] = (inv > 0.0f) ? expf(xr[c] - m) * inv : 0.0f;
}

// ---------------------------------------------------------------------------
// 3) Online 2-pass: one pass computes (max, sum) together, the second normalizes.
// ---------------------------------------------------------------------------
__global__ void softmax_online_2pass(const float* __restrict__ x, float* __restrict__ y,
                                     int rows, int cols) {
    __shared__ float sm[BLOCK];
    __shared__ float ss[BLOCK];

    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + static_cast<size_t>(row) * cols;
    float*       yr = y + static_cast<size_t>(row) * cols;

    float m = -INFINITY, s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        combine(m, s, xr[c], 1.0f);       // one element contributes the pair (x_c, 1)
    block_combine(m, s, sm, ss);

    const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        yr[c] = (inv > 0.0f) ? expf(xr[c] - m) * inv : 0.0f;
}

// ---------------------------------------------------------------------------
// 4) Online 1-pass: the row lives in registers, so HBM sees one read + one write.
//    Requires cols <= BLOCK * ITEMS and exactly BLOCK threads (the stride is static).
// ---------------------------------------------------------------------------
template <int PER_THREAD>
__global__ void softmax_online_1pass(const float* __restrict__ x, float* __restrict__ y,
                                     int rows, int cols) {
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
        const int c   = threadIdx.x + k * BLOCK;            // each k is a coalesced segment
        const bool in = (c < cols);
        v[k] = in ? xr[c] : -INFINITY;                       // pad lanes get the identity
        combine(m, s, v[k], in ? 1.0f : 0.0f);
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

enum Variant { NAIVE_UNSAFE, SAFE_3PASS, ONLINE_2PASS, ONLINE_1PASS };

static void launch(Variant v, const float* d_x, float* d_y, int rows, int cols) {
    switch (v) {
        case NAIVE_UNSAFE: softmax_naive_unsafe<<<rows, BLOCK>>>(d_x, d_y, rows, cols); break;
        case SAFE_3PASS:   softmax_safe_3pass<<<rows, BLOCK>>>(d_x, d_y, rows, cols);   break;
        case ONLINE_2PASS: softmax_online_2pass<<<rows, BLOCK>>>(d_x, d_y, rows, cols); break;
        case ONLINE_1PASS: softmax_online_1pass<ITEMS><<<rows, BLOCK>>>(d_x, d_y, rows, cols); break;
    }
}

// Median-free but warmed-up event timing: the first launch pays code-object load and
// cold caches, which would otherwise dominate a sub-millisecond kernel.
static double time_ms(Variant v, const float* d_x, float* d_y, int rows, int cols, int iters) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    launch(v, d_x, d_y, rows, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch(v, d_x, d_y, rows, cols);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

// Double-precision CPU reference. Uses the max trick itself - in fp64 the naive form
// would survive x = 100 (ln(DBL_MAX) = 709), but the reference must be the *correct*
// distribution, not a second demonstration of luck.
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

struct Check {
    double max_err;
    long   non_finite;
};

static Check verify(const float* d_y, const std::vector<double>& ref, int rows, int cols) {
    const size_t n = static_cast<size_t>(rows) * cols;
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
    const int rows = 8192;
    const int cols = 1024;                // 8192 x 1024 fp32 -> 32 MiB per tensor
    const int iters = 20;
    const size_t n = static_cast<size_t>(rows) * cols;
    const size_t bytes = n * sizeof(float);

    if (cols > BLOCK * ITEMS) {
        std::fprintf(stderr, "cols=%d exceeds BLOCK*ITEMS=%d; raise ITEMS.\n", cols, BLOCK * ITEMS);
        return EXIT_FAILURE;
    }

    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    std::vector<float> h_x(n);
    uint32_t seed = 12345u;
    for (size_t i = 0; i < n; ++i) {      // deterministic LCG in [-2, 2)
        seed = seed * 1664525u + 1013904223u;
        h_x[i] = 4.0f * (static_cast<float>(seed >> 8) / 16777216.0f) - 2.0f;
    }
    // The poison: one score large enough that expf() overflows fp32. Everything else
    // in the tensor is tame, so a failure can only come from the missing max trick.
    const int poison_row = 0, poison_col = 3;
    h_x[static_cast<size_t>(poison_row) * cols + poison_col] = 100.0f;

    std::vector<double> ref(n);
    reference(h_x, ref, rows, cols);

    float *d_x = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));

    std::printf("device            : %s (warpSize = %d)\n", props.name, props.warpSize);
    std::printf("softmax over rows : rows = %d, cols = %d (%.1f MiB per tensor), block = %d\n",
                rows, cols, bytes / (1024.0 * 1024.0), BLOCK);
    std::printf("poison            : x[%d][%d] = 100.0  (expf overflows fp32 above ~88.7)\n\n",
                poison_row, poison_col);

    struct Case {
        Variant     v;
        const char* name;
        double      row_traversals;       // modelled HBM passes: reads + the single write
    } cases[] = {
        {NAIVE_UNSAFE, "1 naive unsafe (no max sub)",     3.0},
        {SAFE_3PASS,   "2 safe 3-pass (max trick)",       4.0},
        {ONLINE_2PASS, "3 online 2-pass (running max)",   3.0},
        {ONLINE_1PASS, "4 online 1-pass (row in regs)",   2.0},
    };

    std::printf("%-32s %-7s %-9s %-11s %-11s %-6s %s\n",
                "kernel", "passes", "ms", "GB/s", "max |err|", "nonfin", "result");
    std::printf("-------------------------------- ------- --------- ----------- "
                "----------- ------ ------\n");

    // Keep the poisoned row from each kernel so the failure can be shown, not asserted.
    std::vector<std::vector<float>> row0(sizeof(cases) / sizeof(cases[0]),
                                         std::vector<float>(cols));
    int ci = 0;
    for (const Case& c : cases) {
        const double ms = time_ms(c.v, d_x, d_y, rows, cols, iters);

        HIP_CHECK(hipMemset(d_y, 0, bytes));
        launch(c.v, d_x, d_y, rows, cols);            // repopulate for verification
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        const Check chk = verify(d_y, ref, rows, cols);
        HIP_CHECK(hipMemcpy(row0[ci].data(), d_y + static_cast<size_t>(poison_row) * cols,
                            row0[ci].size() * sizeof(float), hipMemcpyDeviceToHost));

        const double gbps = c.row_traversals * bytes / (ms / 1.0e3) / 1.0e9;
        const bool   ok   = (chk.non_finite == 0) && (chk.max_err < 5.0e-6);
        std::printf("%-32s %-7.0f %-9.3f %-11.1f %-11.2e %-6ld %s\n",
                    c.name, c.row_traversals, ms, gbps, chk.max_err, chk.non_finite,
                    ok ? "PASSED" : "FAILED");
        ++ci;
    }

    std::printf("\nrow %d (the poisoned row): sum over the row, then the first 6 outputs.\n"
                "Expected: sum = 1.0, and ~1.0 at col %d with ~0 everywhere else.\n",
                poison_row, poison_col);
    ci = 0;
    for (const Case& c : cases) {
        double sum = 0.0;
        for (float f : row0[ci]) sum += f;
        std::printf("  %-32s sum=%10.6f |", c.name, sum);
        for (int k = 0; k < 6; ++k) std::printf(" %10.6f", row0[ci][k]);
        std::printf("\n");
        ++ci;
    }

    std::printf("\nKernel 1 is SUPPOSED to fail: exp(100) = inf, so the normalizer is inf,\n"
                "inf/inf = nan for the big score and finite/inf = 0 for every other one.\n"
                "Kernels 2-4 are bit-for-bit the same mathematics with the max subtracted;\n"
                "the only thing that changes between them is how many times the row is read\n"
                "from HBM (4 -> 3 -> 2 traversals), which for a ~1 FLOP/byte kernel is the\n"
                "entire runtime story.\n");

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    return 0;
}
