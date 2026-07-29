// Lab (CUDA) - Parallel reduction: the classic GPU optimization ladder.
//
// Line-for-line twin of hip/reduction.cpp. One program, one input array, nine
// implementations of "sum N floats into 1", each verified against a
// double-precision CPU reference and timed with CUDA events.
//
//   1 interleaved + modulo branch  - correct, but half the warp idles every step
//   2 interleaved + strided index  - divergence gone, shared-mem bank conflicts
//   3 sequential addressing        - conflict-free; active lanes stay contiguous
//   4 + first add during load      - halve the grid, two loads per thread
//   5 + unrolled last warp         - warp-synchronous tail, guarded by __syncwarp
//   6 + complete unroll            - block size as a template parameter
//   7 + grid-stride cascade        - many elements per thread; saturates HBM
//   8 warp shuffle + grid stride   - the modern idiom; registers, not shared mem
//   9 shuffle + atomicAdd          - single launch, but summation order varies
//
// The ONLY substantive difference from the HIP version is the warp tail. AMD
// guarantees wavefront lanes run in lockstep, so `volatile` alone is enough
// there. NVIDIA Volta and later have independent thread scheduling, which voids
// that assumption - the classic volatile-only tail is undefined behavior on
// modern NVIDIA hardware and must be fenced with __syncwarp() (or replaced by
// shuffles, as stages 8-9 do).
//
// References:
//   Mark Harris, "Optimizing Parallel Reduction in CUDA"
//     https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//   Lin & Grover, "Using CUDA Warp-Level Primitives"
//     https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
//
// Build: nvcc -O3 -arch=sm_90 -Icuda reduction.cu -o reduction.exe
// Run:   ./reduction.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

// One block size for the whole ladder so the stages are compared like for like.
constexpr unsigned kBlock = 256;

// A CUDA warp is 32 lanes; keeping the width explicit mirrors the HIP twin,
// where 32 is a *sub-group* of a possibly 64-lane wavefront.
constexpr unsigned kShflWidth = 32u;
constexpr unsigned kFullMask  = 0xffffffffu;

// ---------------------------------------------------------------------------
// Warp-level tails, shared by several stages
// ---------------------------------------------------------------------------

// The warp-synchronous tail, written the way NVIDIA recommends for Volta+.
// `volatile` keeps the intermediate values in shared memory instead of a
// register; __syncwarp() re-converges the lanes and orders the shared accesses,
// which independent thread scheduling no longer does for free.
__device__ inline void warp_tail_syncwarp(volatile float* s, unsigned tid) {
    float v = s[tid];
    v += s[tid + 32]; __syncwarp();
    s[tid] = v;       __syncwarp();
    v += s[tid + 16]; __syncwarp();
    s[tid] = v;       __syncwarp();
    v += s[tid + 8];  __syncwarp();
    s[tid] = v;       __syncwarp();
    v += s[tid + 4];  __syncwarp();
    s[tid] = v;       __syncwarp();
    v += s[tid + 2];  __syncwarp();
    s[tid] = v;       __syncwarp();
    v += s[tid + 1];
    s[tid] = v;
}

// Register-to-register tree reduction across one warp. No shared memory, no
// barrier, no address arithmetic - lane 0 ends up with the total.
__device__ inline float warp_reduce_shfl(float v) {
#pragma unroll
    for (unsigned off = kShflWidth / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(kFullMask, v, off, static_cast<int>(kShflWidth));
    }
    return v;
}

// Whole-block reduction built from warp shuffles: reduce inside each warp,
// publish one value per warp to shared memory, then reduce those.
// Only thread 0 holds a meaningful result on return.
template <unsigned BLOCK>
__device__ inline float block_reduce_shfl(float v) {
    static_assert(BLOCK % kShflWidth == 0, "block must be a whole number of warps");
    constexpr unsigned kWarps = BLOCK / kShflWidth;
    __shared__ float warp_sums[kWarps];

    const unsigned lane = threadIdx.x % kShflWidth;
    const unsigned warp = threadIdx.x / kShflWidth;

    v = warp_reduce_shfl(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    // Every thread reloads so the final shuffle stays fully uniform (all 32
    // lanes of warp 0 present, as kFullMask promises); lanes past the warp count
    // contribute the additive identity instead of garbage.
    v = (threadIdx.x < kWarps) ? warp_sums[threadIdx.x] : 0.0f;
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
// The cost: `index = 2*s*tid` makes the shared addresses stride by 2, 4, 8, ...
// words, which collides in the 32 shared-memory banks.
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
// contiguous range, and the two operands are consecutive words - one word per
// bank, no conflicts. This is the shape every production reduction uses.
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
// to warp_tail_syncwarp(). Requires blockDim >= 64 so sdata[tid + 32] is in
// range - enforced by the launcher, which always uses kBlock == 256.
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
    if (tid < 32) warp_tail_syncwarp(sdata, tid);

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
    if (tid < 32) warp_tail_syncwarp(sdata, tid);

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 7 - algorithm cascade: many elements per thread via a grid-stride loop
// ---------------------------------------------------------------------------
// Stages 1-6 all spend O(log BLOCK) barrier-separated steps to consume only one
// or two elements per thread - the tree overhead dominates. Launch a grid sized
// to the *machine* instead of to the data, and let each block stream many
// elements with a coalesced grid-stride loop before doing a single tree.
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
    if (tid < 32) warp_tail_syncwarp(sdata, tid);

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

// ---------------------------------------------------------------------------
// Stage 8 - warp shuffle + grid stride (the idiom to actually write today)
// ---------------------------------------------------------------------------
// Same cascade, but the intra-block tree is register-to-register shuffles plus
// exactly one barrier, instead of log2(BLOCK) shared round trips and barriers.
// Shorter, uses BLOCK/32 floats of shared memory instead of BLOCK, and no
// `volatile` sharp edges. Performance is typically within noise of stage 7
// because both are memory-bound - the win is clarity, not raw time.
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
    CUDA_CHECK(cudaGetLastError());   // launch-config errors surface here, not later
}

// Reduce `n` inputs to a single float on the device and return where it landed.
// Stages 1-8 need one launch per level of the block tree (16.7M -> 65536 -> 256
// -> 1); stage 9 finishes in one launch because the blocks combine themselves.
static const float* reduce_full(Stage s, const float* d_in, unsigned n,
                                float* d_ping, float* d_pong, float* d_atomic,
                                unsigned cascade_blocks) {
    if (s == S9) {
        CUDA_CHECK(cudaMemsetAsync(d_atomic, 0, sizeof(float)));
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
    CUDA_CHECK(cudaMemcpy(&h, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// Milliseconds for one full N -> 1 reduction, averaged over `iters`.
static double time_stage(Stage s, const float* d_in, unsigned n,
                         float* d_ping, float* d_pong, float* d_atomic,
                         unsigned cascade_blocks, int iters) {
    reduce_full(s, d_in, n, d_ping, d_pong, d_atomic, cascade_blocks);  // warmup
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        reduce_full(s, d_in, n, d_ping, d_pong, d_atomic, cascade_blocks);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

int main() {
    const unsigned n     = 1u << 24;              // 16,777,216 floats = 64 MiB
    const size_t   bytes = sizeof(float) * static_cast<size_t>(n);
    const int      iters = 50;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

    // Cascade grid: several blocks per SM, so the machine stays saturated
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
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_ping, sizeof(float) * max_partials));
    CUDA_CHECK(cudaMalloc(&d_pong, sizeof(float) * max_partials));
    CUDA_CHECK(cudaMalloc(&d_atomic, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    printf("device          : %s (%d SMs)\n", props.name, props.multiProcessorCount);
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

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_ping));
    CUDA_CHECK(cudaFree(d_pong));
    CUDA_CHECK(cudaFree(d_atomic));
    return 0;
}
