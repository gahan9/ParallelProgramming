// Lab 1 (CUDA) - Parallel scan (prefix sum): the sequential-looking problem, parallelized.
//
// Line-for-line counterpart of hip/scan.cpp (only the runtime prefixes differ), so the two
// files can be diffed side by side. See that file's header and README S2/S3 for the
// derivation; the short version:
//
//   1) hillis_steele_pingpong - inclusive, O(n log n) work, O(log n) depth, 1 elem/thread,
//                               two shared buffers swapped each round.
//   2) hillis_steele_inplace  - same math on one buffer: stage the operand in a register,
//                               barrier, then write. Two barriers per step.
//   3) blelloch_scan_tile     - exclusive, O(n) work, O(log n) depth, 2 elems/thread,
//                               up-sweep + down-sweep with GPU Gems 3 bank padding.
//
// Arrays larger than one tile use the 3-phase decomposition in scan_exclusive().
//
// Reference: Blelloch, "Prefix Sums and Their Applications" (CMU-CS-90-190);
//            Harris, Sengupta, Owens, GPU Gems 3 Ch.39.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda scan.cu -o scan.exe
// Run:   ./scan.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

// One TILE is the chunk of the array a single block scans. Both algorithms use the
// SAME tile size so the work-efficiency comparison in Part 2 is apples-to-apples:
// Hillis-Steele spends TILE_ELEMS threads on it, Blelloch spends TILE_ELEMS/2.
#define TILE_ELEMS 256
#define HS_THREADS TILE_ELEMS
#define BL_THREADS (TILE_ELEMS / 2)

// Shared memory is banked in 32 four-byte words. Blelloch's tree indexes with
// power-of-two strides, so at the upper levels every active lane lands in the same
// bank; shifting one extra slot in every 32 spreads them again (GPU Gems 3 Ch.39).
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)

// --- 1) Hillis-Steele with ping-pong buffers: INCLUSIVE scan of one tile. ---
__global__ void hillis_steele_pingpong(const float* in, float* out, int n) {
    extern __shared__ float buf[];                  // 2 * blockDim.x floats
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    int pin = 0;
    buf[tid] = (idx < n) ? in[idx] : 0.0f;          // past-the-end lanes contribute the identity
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        const float* src = buf + pin * blockDim.x;
        float*       dst = buf + (1 - pin) * blockDim.x;
        dst[tid] = (tid >= offset) ? src[tid] + src[tid - offset] : src[tid];
        pin = 1 - pin;
        __syncthreads();
    }

    if (idx < n) out[idx] = buf[pin * blockDim.x + tid];
}

// --- 2) Hillis-Steele in place: same math, one buffer, register staging. ---
__global__ void hillis_steele_inplace(const float* in, float* out, int n) {
    extern __shared__ float s[];                    // blockDim.x floats
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    s[tid] = (idx < n) ? in[idx] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        const float addend = (tid >= offset) ? s[tid - offset] : 0.0f;
        __syncthreads();                            // all reads of the pre-step values are done
        s[tid] += addend;
        __syncthreads();                            // all writes visible before the next read
    }

    if (idx < n) out[idx] = s[tid];
}

// --- 3) Blelloch: work-efficient EXCLUSIVE scan of one tile (2 elements/thread). ---
__global__ void blelloch_scan_tile(const float* in, float* out, float* tile_sums, int n) {
    extern __shared__ float temp[];                 // 2*blockDim.x + padding floats
    const int tid   = threadIdx.x;
    const int elems = 2 * blockDim.x;
    const int base  = blockIdx.x * elems;

    const int ai = tid;
    const int bi = tid + blockDim.x;
    const int pad_a = CONFLICT_FREE_OFFSET(ai);
    const int pad_b = CONFLICT_FREE_OFFSET(bi);
    temp[ai + pad_a] = (base + ai < n) ? in[base + ai] : 0.0f;
    temp[bi + pad_b] = (base + bi < n) ? in[base + bi] : 0.0f;

    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {      // up-sweep: d active lanes per level
        __syncthreads();
        if (tid < d) {
            int li = offset * (2 * tid + 1) - 1;
            int ri = offset * (2 * tid + 2) - 1;
            li += CONFLICT_FREE_OFFSET(li);
            ri += CONFLICT_FREE_OFFSET(ri);
            temp[ri] += temp[li];
        }
        offset *= 2;
    }

    if (tid == 0) {
        const int last = (elems - 1) + CONFLICT_FREE_OFFSET(elems - 1);
        if (tile_sums != nullptr) tile_sums[blockIdx.x] = temp[last];
        temp[last] = 0.0f;                          // identity at the root makes it EXCLUSIVE
    }

    for (int d = 1; d < elems; d *= 2) {            // down-sweep: distribute the prefixes
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int li = offset * (2 * tid + 1) - 1;
            int ri = offset * (2 * tid + 2) - 1;
            li += CONFLICT_FREE_OFFSET(li);
            ri += CONFLICT_FREE_OFFSET(ri);
            const float left = temp[li];
            temp[li] = temp[ri];
            temp[ri] += left;
        }
    }
    __syncthreads();

    if (base + ai < n) out[base + ai] = temp[ai + pad_a];
    if (base + bi < n) out[base + bi] = temp[bi + pad_b];
}

// --- Phase 3: shift every element of tile t by that tile's exclusive offset. ---
__global__ void add_tile_offsets(float* data, const float* tile_offsets, int n,
                                 int elems_per_tile) {
    const float offset = tile_offsets[blockIdx.x];
    const int   base   = blockIdx.x * elems_per_tile;
    for (int i = threadIdx.x; i < elems_per_tile; i += blockDim.x) {
        const int idx = base + i;
        if (idx < n) data[idx] += offset;
    }
}

static size_t tile_shared_bytes(int elems) {
    return sizeof(float) * (elems + CONFLICT_FREE_OFFSET(elems - 1) + 1);
}

static int log2_int(int v) {
    int bits = 0;
    while ((1 << bits) < v) ++bits;
    return bits;
}

// Scratch for the tile-sum levels of the recursion. Allocated once: cudaMalloc costs
// far more than the kernels themselves, so it must never sit inside a timed region.
struct ScanWorkspace {
    std::vector<float*> sums;
    std::vector<float*> offsets;
};

static ScanWorkspace make_workspace(int n) {
    ScanWorkspace ws;
    const int elems = 2 * BL_THREADS;
    for (int count = n;;) {
        const int tiles = (count + elems - 1) / elems;
        if (tiles <= 1) break;                      // this level fits in one block: no scratch
        float *s = nullptr, *o = nullptr;
        CUDA_CHECK(cudaMalloc(&s, sizeof(float) * tiles));
        CUDA_CHECK(cudaMalloc(&o, sizeof(float) * tiles));
        ws.sums.push_back(s);
        ws.offsets.push_back(o);
        count = tiles;
    }
    return ws;
}

static void free_workspace(ScanWorkspace& ws) {
    for (float* p : ws.sums) CUDA_CHECK(cudaFree(p));
    for (float* p : ws.offsets) CUDA_CHECK(cudaFree(p));
    ws.sums.clear();
    ws.offsets.clear();
}

// Full exclusive scan of n elements. d_in and d_out must not overlap.
// Phase 1: every block scans its own tile and reports the tile total.
// Phase 2: scan the (much smaller) array of tile totals - same routine, one level down.
// Phase 3: add tile t's exclusive offset to all of tile t's elements.
static void scan_exclusive(const float* d_in, float* d_out, int n,
                           const ScanWorkspace& ws, int level = 0) {
    const int    elems = 2 * BL_THREADS;
    const int    tiles = (n + elems - 1) / elems;
    const size_t shmem = tile_shared_bytes(elems);

    if (tiles <= 1) {
        blelloch_scan_tile<<<1, BL_THREADS, shmem>>>(d_in, d_out, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    float* d_sums    = ws.sums[level];
    float* d_offsets = ws.offsets[level];

    blelloch_scan_tile<<<tiles, BL_THREADS, shmem>>>(d_in, d_out, d_sums, n);
    CUDA_CHECK(cudaGetLastError());

    scan_exclusive(d_sums, d_offsets, tiles, ws, level + 1);

    add_tile_offsets<<<tiles, BL_THREADS>>>(d_out, d_offsets, n, elems);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Launch>
static double time_ms(Launch&& launch, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();                                       // warmup: the first launch pays init costs
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / iters;
}

// Exact comparison is deliberate: the inputs are small integers, so every partial sum
// is an integer below 2^24 and therefore exactly representable in float. Any mismatch
// is an algorithm bug, not rounding. Re-associating REAL float data does change the
// last bits - see README S6.
static int mismatches(const float* d_out, const std::vector<float>& expect, int count) {
    std::vector<float> got(static_cast<size_t>(count));
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, sizeof(float) * count, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < count; ++i)
        if (got[static_cast<size_t>(i)] != expect[static_cast<size_t>(i)]) ++bad;
    return bad;
}

int main() {
    const int    n     = 1 << 22;                   // 4,194,304 elements = 16 MiB
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int    iters = 50;

    // Values 0..3 keep every prefix an exact integer < 2^24 (see mismatches()).
    std::vector<float> h_in(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h_in[static_cast<size_t>(i)] = static_cast<float>(i % 4);

    // CPU references: whole-array inclusive/exclusive, and the per-tile versions that
    // phase 1 alone produces.
    std::vector<float> h_incl(static_cast<size_t>(n)), h_excl(static_cast<size_t>(n));
    float running = 0.0f;
    for (int i = 0; i < n; ++i) {
        h_excl[static_cast<size_t>(i)] = running;
        running += h_in[static_cast<size_t>(i)];
        h_incl[static_cast<size_t>(i)] = running;
    }

    std::vector<float> h_tile_incl(static_cast<size_t>(n)), h_tile_excl(static_cast<size_t>(n));
    for (int t = 0; t < n; t += TILE_ELEMS) {
        float run = 0.0f;
        for (int i = t; i < t + TILE_ELEMS && i < n; ++i) {
            h_tile_excl[static_cast<size_t>(i)] = run;
            run += h_in[static_cast<size_t>(i)];
            h_tile_incl[static_cast<size_t>(i)] = run;
        }
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    const int    tiles     = n / TILE_ELEMS;
    const size_t hs_shmem2 = sizeof(float) * 2 * HS_THREADS;
    const size_t hs_shmem1 = sizeof(float) * HS_THREADS;
    const size_t bl_shmem  = tile_shared_bytes(TILE_ELEMS);

    printf("parallel scan: n = %d elements (%.1f MiB), tile = %d elements\n",
           n, static_cast<double>(bytes) / (1024.0 * 1024.0), TILE_ELEMS);
    const int levels = log2_int(TILE_ELEMS);
    printf("  hillis-steele: %d threads/tile (1 elem/thread), %d steps\n", HS_THREADS, levels);
    printf("  blelloch     : %d threads/tile (2 elems/thread), %d steps (up+down)\n\n",
           BL_THREADS, 2 * levels);

    // ---- Part 1: single-tile correctness ----
    printf("Part 1 - single-tile correctness (first %d elements)\n", TILE_ELEMS);
    printf("%-40s %-11s %-8s\n", "kernel", "scan type", "result");
    printf("---------------------------------------- ----------- --------\n");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    hillis_steele_pingpong<<<1, HS_THREADS, hs_shmem2>>>(d_in, d_out, TILE_ELEMS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int bad = mismatches(d_out, h_incl, TILE_ELEMS);
    printf("%-40s %-11s %-8s\n", "hillis-steele (ping-pong buffers)", "inclusive",
           bad == 0 ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    hillis_steele_inplace<<<1, HS_THREADS, hs_shmem1>>>(d_in, d_out, TILE_ELEMS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    bad = mismatches(d_out, h_incl, TILE_ELEMS);
    printf("%-40s %-11s %-8s\n", "hillis-steele (in-place + register)", "inclusive",
           bad == 0 ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));
    blelloch_scan_tile<<<1, BL_THREADS, bl_shmem>>>(d_in, d_out, nullptr, TILE_ELEMS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    bad = mismatches(d_out, h_excl, TILE_ELEMS);
    printf("%-40s %-11s %-8s\n\n", "blelloch (up-sweep / down-sweep)", "exclusive",
           bad == 0 ? "PASSED" : "FAILED");

    // ---- Part 2: per-tile scans of the whole array (work efficiency) ----
    // Identical tile count and identical bytes moved; the only difference is how much
    // arithmetic and how many barriers each algorithm spends per tile.
    printf("Part 2 - per-tile scans of the whole array (%d tiles, same for both)\n", tiles);
    printf("%-40s %-11s %-9s %-8s\n", "kernel", "time (ms)", "GB/s", "result");
    printf("---------------------------------------- ----------- --------- --------\n");

    double ms = time_ms([&] {
        hillis_steele_pingpong<<<tiles, HS_THREADS, hs_shmem2>>>(d_in, d_out, n);
    }, iters);
    bad = mismatches(d_out, h_tile_incl, n);
    printf("%-40s %-11.3f %-9.1f %-8s\n", "hillis-steele  O(n log n) work", ms,
           (2.0 * static_cast<double>(bytes)) / (ms / 1.0e3) / 1.0e9,
           bad == 0 ? "PASSED" : "FAILED");

    ms = time_ms([&] {
        blelloch_scan_tile<<<tiles, BL_THREADS, bl_shmem>>>(d_in, d_out, nullptr, n);
    }, iters);
    bad = mismatches(d_out, h_tile_excl, n);
    printf("%-40s %-11.3f %-9.1f %-8s\n\n", "blelloch       O(n) work", ms,
           (2.0 * static_cast<double>(bytes)) / (ms / 1.0e3) / 1.0e9,
           bad == 0 ? "PASSED" : "FAILED");

    // ---- Part 3: full multi-block exclusive scan ----
    ScanWorkspace ws = make_workspace(n);
    ms = time_ms([&] { scan_exclusive(d_in, d_out, n, ws); }, iters);
    bad = mismatches(d_out, h_excl, n);

    printf("Part 3 - full exclusive scan of all %d elements (3-phase decomposition)\n", n);
    printf("recursion levels : %zu (tile sums, then sums of those, ...)\n", ws.sums.size() + 1);
    printf("time             : %.3f ms\n", ms);
    printf("effective BW     : %.1f GB/s  (useful bytes only: read n + write n)\n",
           (2.0 * static_cast<double>(bytes)) / (ms / 1.0e3) / 1.0e9);
    printf("result           : %s\n\n", bad == 0 ? "PASSED" : "FAILED");

    printf("The 3-phase scan re-reads and re-writes the output in phase 3, so it moves\n"
           "roughly twice the useful bytes above. A single-pass chained scan (decoupled\n"
           "look-back, as in rocPRIM/CUB) is what closes that gap - see README S3/S5.\n");

    free_workspace(ws);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return bad == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
