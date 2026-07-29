// Lab (HIP) - Parallel reduction: the classic GPU optimization ladder.
//
// One program, one input array, nine implementations of "sum N floats into 1".
// Every stage is verified against a double-precision CPU reference and timed with
// GPU events, so the printed table IS the lesson: each rung of the ladder removes
// one specific hardware inefficiency, and you can see exactly what it buys.
//
//   1 interleaved + modulo branch  - correct, but half the warp idles every step
//   2 interleaved + strided index  - divergence gone, LDS bank conflicts appear
//   3 sequential addressing        - conflict-free; active lanes stay contiguous
//   4 + first add during load      - halve the grid, two loads per thread
//   5 + unrolled last warp         - no barrier needed inside one warp/wavefront
//   6 + complete unroll            - block size as a template parameter
//   7 + grid-stride cascade        - many elements per thread; this is the one
//                                    that actually saturates HBM
//   8 warp shuffle + grid stride   - the modern idiom; registers, not LDS
//   9 shuffle + atomicAdd          - single launch, but summation order varies
//
// Reference: Mark Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA).
//   https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip reduction.cpp -o reduction.exe
// Run:   ./reduction.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

// One block size for the whole ladder so the stages are compared like for like.
// 256 is a multiple of both a 32-lane warp (NVIDIA, RDNA) and a 64-lane
// wavefront (CDNA), so no stage silently loses lanes on either vendor.
constexpr unsigned kBlock = 256;

// Shuffle sub-group width. 32 divides every shipping warp/wavefront size, and
// HIP indexes shuffle sub-groups by `laneIdx % width`, so a width-32 shuffle is
// correct on wave32 and wave64 alike. Hard-coding 64 would break on RDNA;
// hard-coding `warpSize` is not possible because it is not a compile-time
// constant in HIP (see README S3).
constexpr unsigned kShflWidth = 32u;

// ---------------------------------------------------------------------------
// Warp-level tails, shared by several stages
// ---------------------------------------------------------------------------

// The last 32 lanes of a block live inside one warp/wavefront, which AMD
// guarantees executes in lockstep - so no __syncthreads() is needed here.
// `volatile` is load-bearing: without it the compiler may keep s[tid] in a
// register across the steps and never publish the intermediate values to LDS.
// This trick is NOT valid on NVIDIA Volta+ (independent thread scheduling); the
// cuda/ twin therefore adds __syncwarp() between the steps. See README S3.
__device__ inline void warp_tail_volatile(volatile float* s, unsigned tid) {
    s[tid] += s[tid + 32];
    s[tid] += s[tid + 16];
    s[tid] += s[tid + 8];
    s[tid] += s[tid + 4];
    s[tid] += s[tid + 2];
    s[tid] += s[tid + 1];
}

// Register-to-register tree reduction across one 32-lane sub-group. No shared
// memory, no barrier, no address arithmetic - lane 0 ends up with the total.
__device__ inline float warp_reduce_shfl(float v) {
#pragma unroll
    for (unsigned off = kShflWidth / 2; off > 0; off >>= 1) {
        v += __shfl_down(v, off, static_cast<int>(kShflWidth));
    }
    return v;
}

// Whole-block reduction built from sub-group shuffles: reduce inside each
// sub-group, publish one value per sub-group to LDS, then reduce those.
// Only thread 0 holds a meaningful result on return.
template <unsigned BLOCK>
__device__ inline float block_reduce_shfl(float v) {
    static_assert(BLOCK % kShflWidth == 0, "block must be a whole number of sub-groups");
    constexpr unsigned kSubgroups = BLOCK / kShflWidth;
    __shared__ float subgroup_sums[kSubgroups];

    const unsigned lane = threadIdx.x % kShflWidth;
    const unsigned sub  = threadIdx.x / kShflWidth;

    v = warp_reduce_shfl(v);
    if (lane == 0) subgroup_sums[sub] = v;
    __syncthreads();

    // Every thread reloads so the final shuffle stays fully uniform; lanes past
    // the sub-group count contribute the additive identity instead of garbage.
    v = (threadIdx.x < kSubgroups) ? subgroup_sums[threadIdx.x] : 0.0f;
    return warp_reduce_shfl(v);
}

// ---------------------------------------------------------------------------
// Stage 1 - interleaved addressing with a modulo branch
// ---------------------------------------------------------------------------
// The tree is correct but the *branch* is the problem: `tid % (2*s) == 0` is
// true for a scattered subset of lanes, so on every step the warp executes both
// sides of the branch with most lanes masked off. Maximum divergence.
__global__ void reduce1_modulo(const float* in, float* out, unsigned n) {
    extern __shared__ float sdata[];
    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0.0f;   // pad the tail with the identity
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();                   // every thread must reach this
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 2 - interleaved addressing with a strided index
// ---------------------------------------------------------------------------
// Same tree, but the active threads are now the *first* `blockDim/2s` threads,
// so whole warps are either fully active or fully idle: divergence is gone.
// The cost: `index = 2*s*tid` makes the LDS addresses stride by 2, 4, 8, ...
// words, which collides in the 32 LDS banks (a 2-way conflict at s=1, 4-way at
// s=2, ...). Faster than stage 1, still leaving throughput on the table.
__global__ void reduce2_strided_index(const float* in, float* out, unsigned n) {
    extern __shared__ float sdata[];
    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = 1; s < blockDim.x; s *= 2) {
        const unsigned index = 2 * s * tid;
        if (index < blockDim.x) sdata[index] += sdata[index + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 3 - sequential addressing
// ---------------------------------------------------------------------------
// Walk the stride *down* instead of up. Active threads are always tid < s, a
// contiguous range, and the two operands sdata[tid] / sdata[tid + s] are
// consecutive words - one word per bank, no conflicts. This is the shape every
// production reduction uses.
__global__ void reduce3_sequential(const float* in, float* out, unsigned n) {
    extern __shared__ float sdata[];
    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 4 - first add during the global load
// ---------------------------------------------------------------------------
// In stage 3, half the threads do nothing after the very first step - the block
// is half idle for the entire tree. Fix the *launch* instead of the loop: give
// each block twice the data, add the two halves while loading, and halve the
// grid. Same total work, half the blocks, half the tree launches downstream.
__global__ void reduce4_first_add(const float* in, float* out, unsigned n) {
    extern __shared__ float sdata[];
    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];   // both loads coalesced
    sdata[tid] = v;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 5 - unroll the last warp
// ---------------------------------------------------------------------------
// Once s <= 32 only one warp is still active, so both the barrier and the
// `if (tid < s)` guard are pure overhead. Stop the loop at 32 and hand the tail
// to warp_tail_volatile(). Requires blockDim >= 64 so that sdata[tid + 32] is
// in range - enforced by the launcher, which always uses kBlock == 256.
__global__ void reduce5_unroll_warp(const float* in, float* out, unsigned n) {
    extern __shared__ float sdata[];
    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float v = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    sdata[tid] = v;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) warp_tail_volatile(sdata, tid);

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 6 - complete unroll with the block size as a template parameter
// ---------------------------------------------------------------------------
// With BLOCK known at compile time the whole tree collapses into straight-line
// code: the `if (BLOCK >= ...)` tests are folded away, the loop counter and its
// comparison disappear. Note the barriers sit OUTSIDE the `tid < ...` guards -
// a __syncthreads() that only some threads reach is undefined behavior.
template <unsigned BLOCK>
__global__ void reduce6_complete_unroll(const float* in, float* out, unsigned n) {
    extern __shared__ float sdata[];
    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * (BLOCK * 2) + threadIdx.x;

    float v = (i < n) ? in[i] : 0.0f;
    if (i + BLOCK < n) v += in[i + BLOCK];
    sdata[tid] = v;
    __syncthreads();

    if (BLOCK >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (BLOCK >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (BLOCK >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    if (tid < 32) warp_tail_volatile(sdata, tid);

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 7 - algorithm cascade: many elements per thread via a grid-stride loop
// ---------------------------------------------------------------------------
// Stages 1-6 all spend O(log BLOCK) barrier-separated steps to consume only
// one or two elements per thread - the tree overhead dominates. Launch a grid
// sized to the *machine* instead of to the data, and let each block stream many
// elements with a coalesced grid-stride loop before doing a single tree. The
// serial accumulation is free (it is just adds into a register) and the loads
// stay perfectly coalesced, so this is the stage that reaches bandwidth peak.
template <unsigned BLOCK>
__global__ void reduce7_cascade(const float* in, float* out, unsigned n) {
    extern __shared__ float sdata[];
    const unsigned tid    = threadIdx.x;
    const unsigned stride = BLOCK * 2 * gridDim.x;

    float sum = 0.0f;
    for (unsigned i = blockIdx.x * (BLOCK * 2) + tid; i < n; i += stride) {
        sum += in[i];
        if (i + BLOCK < n) sum += in[i + BLOCK];
    }
    sdata[tid] = sum;
    __syncthreads();

    if (BLOCK >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (BLOCK >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (BLOCK >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    if (tid < 32) warp_tail_volatile(sdata, tid);

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 8 - warp shuffle + grid stride (the idiom to actually write in 2020s code)
// ---------------------------------------------------------------------------
// Same cascade, but the intra-block tree is register-to-register shuffles plus
// exactly one barrier, instead of log2(BLOCK) LDS round trips and barriers.
// Shorter, uses BLOCK/32 floats of LDS instead of BLOCK, and no `volatile`
// sharp edges. Performance is typically within noise of stage 7 because both
// are memory-bound - the win is clarity and LDS pressure, not raw time.
template <unsigned BLOCK>
__global__ void reduce8_shfl_cascade(const float* in, float* out, unsigned n) {
    const unsigned stride = BLOCK * gridDim.x;

    float sum = 0.0f;
    for (unsigned i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += stride) {
        sum += in[i];
    }
    sum = block_reduce_shfl<BLOCK>(sum);

    if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// ---------------------------------------------------------------------------
// Stage 9 - shuffle cascade finished with atomicAdd (single launch)
// ---------------------------------------------------------------------------
// Identical to stage 8 except the cross-block combine happens on the device:
// one atomicAdd per block instead of a second kernel launch. Saves a launch
// (~5-10 us) and the partials buffer, but the order in which blocks land their
// atomics depends on scheduling, and float addition is NOT associative - so the
// last bits of the answer can change from run to run. See the determinism probe
// in main() and README S6.
template <unsigned BLOCK>
__global__ void reduce9_shfl_atomic(const float* in, float* out, unsigned n) {
    const unsigned stride = BLOCK * gridDim.x;

    float sum = 0.0f;
    for (unsigned i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += stride) {
        sum += in[i];
    }
    sum = block_reduce_shfl<BLOCK>(sum);

    if (threadIdx.x == 0) atomicAdd(out, sum);
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

enum Stage { S1, S2, S3, S4, S5, S6, S7, S8, S9 };

struct StageDesc {
    Stage       id;
    const char* name;
};

static const StageDesc kLadder[] = {
    {S1, "1 interleaved, modulo branch (divergent)"},
    {S2, "2 interleaved, strided index (bank conflicts)"},
    {S3, "3 sequential addressing (conflict-free)"},
    {S4, "4 + first add during global load"},
    {S5, "5 + unrolled last warp"},
    {S6, "6 + complete unroll (templated block size)"},
    {S7, "7 + grid-stride cascade (many elems/thread)"},
    {S8, "8 warp shuffle + grid stride (modern idiom)"},
    {S9, "9 shuffle + atomicAdd (1 launch, nondeterministic)"},
};

// Elements a block consumes per grid-stride iteration. Stages 4-7 load two.
static unsigned elems_per_block(Stage s) {
    switch (s) {
        case S4: case S5: case S6: case S7: return kBlock * 2;
        default:                            return kBlock;
    }
}

// Stages 7-9 size the grid to the machine, not the data (the cascade); the rest
// need one block per tile of input.
static bool is_cascade(Stage s) { return s == S7 || s == S8 || s == S9; }

static unsigned grid_for(Stage s, unsigned n, unsigned cascade_blocks) {
    unsigned need = (n + elems_per_block(s) - 1) / elems_per_block(s);
    if (need == 0) need = 1;
    if (is_cascade(s) && need > cascade_blocks) need = cascade_blocks;
    return need;
}

static void launch(Stage s, const float* in, float* out, unsigned n, unsigned grid) {
    const size_t shmem = kBlock * sizeof(float);
    switch (s) {
        case S1: reduce1_modulo<<<grid, kBlock, shmem>>>(in, out, n); break;
        case S2: reduce2_strided_index<<<grid, kBlock, shmem>>>(in, out, n); break;
        case S3: reduce3_sequential<<<grid, kBlock, shmem>>>(in, out, n); break;
        case S4: reduce4_first_add<<<grid, kBlock, shmem>>>(in, out, n); break;
        case S5: reduce5_unroll_warp<<<grid, kBlock, shmem>>>(in, out, n); break;
        case S6: reduce6_complete_unroll<kBlock><<<grid, kBlock, shmem>>>(in, out, n); break;
        case S7: reduce7_cascade<kBlock><<<grid, kBlock, shmem>>>(in, out, n); break;
        case S8: reduce8_shfl_cascade<kBlock><<<grid, kBlock>>>(in, out, n); break;
        case S9: reduce9_shfl_atomic<kBlock><<<grid, kBlock>>>(in, out, n); break;
    }
    HIP_CHECK(hipGetLastError());   // launch-config errors surface here, not later
}

// Reduce `n` inputs to a single float on the device and return where it landed.
// Stages 1-8 need one launch per level of the block tree (16.7M -> 65536 -> 256
// -> 1); stage 9 finishes in one launch because the blocks combine themselves.
static const float* reduce_full(Stage s, const float* d_in, unsigned n,
                                float* d_ping, float* d_pong, float* d_atomic,
                                unsigned cascade_blocks) {
    if (s == S9) {
        HIP_CHECK(hipMemsetAsync(d_atomic, 0, sizeof(float)));
        launch(S9, d_in, d_atomic, n, grid_for(S9, n, cascade_blocks));
        return d_atomic;
    }

    const float* src   = d_in;
    float*       dst   = d_ping;
    unsigned     count = n;
    for (;;) {
        const unsigned grid = grid_for(s, count, cascade_blocks);
        launch(s, src, dst, count, grid);
        if (grid == 1) return dst;
        src   = dst;
        dst   = (dst == d_ping) ? d_pong : d_ping;
        count = grid;
    }
}

static float reduce_to_host(Stage s, const float* d_in, unsigned n,
                            float* d_ping, float* d_pong, float* d_atomic,
                            unsigned cascade_blocks) {
    const float* d_result =
        reduce_full(s, d_in, n, d_ping, d_pong, d_atomic, cascade_blocks);
    float h = 0.0f;
    HIP_CHECK(hipMemcpy(&h, d_result, sizeof(float), hipMemcpyDeviceToHost));
    return h;
}

// Milliseconds for one full N -> 1 reduction, averaged over `iters`.
static double time_stage(Stage s, const float* d_in, unsigned n,
                         float* d_ping, float* d_pong, float* d_atomic,
                         unsigned cascade_blocks, int iters) {
    reduce_full(s, d_in, n, d_ping, d_pong, d_atomic, cascade_blocks);  // warmup
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        reduce_full(s, d_in, n, d_ping, d_pong, d_atomic, cascade_blocks);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

int main() {
    const unsigned n     = 1u << 24;              // 16,777,216 floats = 64 MiB
    const size_t   bytes = sizeof(float) * static_cast<size_t>(n);
    const int      iters = 50;

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, 0));

    // Cascade grid: several blocks per CU, so the machine stays saturated
    // without launching one block per 512 elements. The 1024 cap has a second
    // purpose - stage 9 accumulates one atomic per block into a float, and that
    // serial accumulation's worst-case relative error is about
    // cascade_blocks * eps/2, which must stay comfortably under the pass/fail
    // tolerance below (1024 * 6e-8 / 2 ~ 3e-5 vs a 1e-4 tolerance).
    unsigned cascade_blocks = static_cast<unsigned>(props.multiProcessorCount) * 8u;
    if (cascade_blocks == 0) cascade_blocks = 256u;
    if (cascade_blocks > 1024u) cascade_blocks = 1024u;

    // Values in [0, 1) with a repeating pattern: large enough that float
    // summation order is visible in the low bits, deterministic across runs.
    std::vector<float> h_in(n);
    double             ref = 0.0;                 // double reference: the tree
    for (unsigned i = 0; i < n; ++i) {            // result must match THIS, not
        h_in[i] = static_cast<float>(i % 100) * 0.01f;   // a float CPU loop
        ref += static_cast<double>(h_in[i]);
    }

    const unsigned max_partials = (n + kBlock - 1) / kBlock;
    float *d_in = nullptr, *d_ping = nullptr, *d_pong = nullptr, *d_atomic = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_ping, sizeof(float) * max_partials));
    HIP_CHECK(hipMalloc(&d_pong, sizeof(float) * max_partials));
    HIP_CHECK(hipMalloc(&d_atomic, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    printf("device          : %s (%d CUs/SMs)\n", props.name, props.multiProcessorCount);
    printf("n               : %u floats (%.1f MiB), block = %u, cascade grid = %u\n",
           n, bytes / (1024.0 * 1024.0), kBlock, cascade_blocks);
    printf("reference (fp64): %.6f\n\n", ref);

    printf("%-52s %8s %9s %8s %10s %s\n",
           "stage", "ms", "GB/s", "vs #1", "rel err", "result");
    printf("---------------------------------------------------- "
           "-------- --------- -------- ---------- ------\n");

    double baseline_ms = 0.0;
    for (const StageDesc& stage : kLadder) {
        const double ms = time_stage(stage.id, d_in, n, d_ping, d_pong, d_atomic,
                                     cascade_blocks, iters);
        const float  got = reduce_to_host(stage.id, d_in, n, d_ping, d_pong,
                                          d_atomic, cascade_blocks);

        const double rel  = std::fabs(static_cast<double>(got) - ref) / std::fabs(ref);
        const double gbps = static_cast<double>(bytes) / (ms / 1.0e3) / 1.0e9;
        if (stage.id == S1) baseline_ms = ms;

        // The tolerance is a bound, not a guess. A correct float32 tree drifts
        // by about log2(n) * eps ~ 1e-6 against the fp64 reference, and stage 9's
        // atomic accumulation adds at most cascade_blocks * eps/2 ~ 3e-5. 1e-4
        // sits above both. It is also well below the smallest interesting bug:
        // dropping a single block out of 1024 would show up as ~1e-3.
        const bool ok = rel < 1.0e-4;
        printf("%-52s %8.4f %9.1f %8.2fx %10.2e %s\n",
               stage.name, ms, gbps, baseline_ms / ms, rel, ok ? "PASSED" : "FAILED");
    }

    // Determinism probe: stage 9's cross-block combine is a race of atomicAdds,
    // so the summation order changes with block scheduling.
    printf("\ndeterminism probe (8 repeats of the same input):\n");
    const Stage probe[] = {S8, S9};
    for (Stage s : probe) {
        unsigned first_bits = 0;
        bool     identical  = true;
        float    lo = 0.0f, hi = 0.0f;
        for (int r = 0; r < 8; ++r) {
            const float v = reduce_to_host(s, d_in, n, d_ping, d_pong, d_atomic,
                                           cascade_blocks);
            unsigned bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            if (r == 0) { first_bits = bits; lo = hi = v; }
            if (bits != first_bits) identical = false;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        printf("  stage %d (%s): %s   [min %.6f, max %.6f]\n",
               s == S8 ? 8 : 9,
               s == S8 ? "second kernel launch" : "atomicAdd",
               identical ? "bitwise identical every run"
                         : "results DIFFER between runs",
               lo, hi);
    }

    printf("\nExpect roughly: #1 slowest; a large jump at #3 (divergence + bank\n"
           "conflicts gone) and again at #7 (enough work per thread to saturate\n"
           "HBM); #7/#8/#9 within noise of each other because all three are\n"
           "bandwidth-bound. Absolute numbers are hardware-dependent - the\n"
           "ORDERING is the lesson. Compare the top GB/s against your GPU's peak\n"
           "HBM bandwidth (A02 Lab 1 measures it).\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_ping));
    HIP_CHECK(hipFree(d_pong));
    HIP_CHECK(hipFree(d_atomic));
    return 0;
}
