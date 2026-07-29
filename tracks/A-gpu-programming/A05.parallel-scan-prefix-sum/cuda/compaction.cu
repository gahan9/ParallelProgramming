// Lab 2 (CUDA) - Stream compaction: scan as infrastructure, not as a toy.
//
// Line-for-line counterpart of hip/compaction.cpp. "Keep the elements that pass a
// predicate, preserving order" is std::copy_if; on a GPU the hard part is deciding WHERE
// each survivor goes. An exclusive scan of the 0/1 flags answers it deterministically:
//
//   flags   = 1 1 0 1 0 0 1 1        (1 = keep)
//   scan    = 0 1 2 2 3 3 3 4        <- exclusive scan of flags
//                                       for a kept element, scan[i] IS its output index
//   total   = scan[n-1] + flags[n-1] (the sum that "falls off" an exclusive scan)
//
// mark -> scan -> scatter. No atomics, no ordering ambiguity, and the result matches
// std::copy_if element for element (a STABLE compaction).
//
// Build: nvcc -O3 -arch=sm_90 -Icuda compaction.cu -o compaction.exe
// Run:   ./compaction.exe
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

#define TILE_ELEMS 256
#define BL_THREADS (TILE_ELEMS / 2)
#define SCATTER_THREADS 256

#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)

// --- Phase A: the predicate. One flag per element, 1 = survives. ---
__global__ void mark_nonzero(const float* in, unsigned int* flags, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) flags[idx] = (in[idx] != 0.0f) ? 1u : 0u;
}

// --- Phase B: Blelloch exclusive scan of one tile (see scan.cu for the derivation). ---
__global__ void blelloch_scan_tile_u32(const unsigned int* in, unsigned int* out,
                                       unsigned int* tile_sums, int n) {
    extern __shared__ unsigned int temp[];
    const int tid   = threadIdx.x;
    const int elems = 2 * blockDim.x;
    const int base  = blockIdx.x * elems;

    const int ai = tid;
    const int bi = tid + blockDim.x;
    const int pad_a = CONFLICT_FREE_OFFSET(ai);
    const int pad_b = CONFLICT_FREE_OFFSET(bi);
    temp[ai + pad_a] = (base + ai < n) ? in[base + ai] : 0u;
    temp[bi + pad_b] = (base + bi < n) ? in[base + bi] : 0u;

    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {      // up-sweep (reduce)
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
        temp[last] = 0u;                            // identity at the root -> exclusive result
    }

    for (int d = 1; d < elems; d *= 2) {            // down-sweep
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int li = offset * (2 * tid + 1) - 1;
            int ri = offset * (2 * tid + 2) - 1;
            li += CONFLICT_FREE_OFFSET(li);
            ri += CONFLICT_FREE_OFFSET(ri);
            const unsigned int left = temp[li];
            temp[li] = temp[ri];
            temp[ri] += left;
        }
    }
    __syncthreads();

    if (base + ai < n) out[base + ai] = temp[ai + pad_a];
    if (base + bi < n) out[base + bi] = temp[bi + pad_b];
}

__global__ void add_tile_offsets_u32(unsigned int* data, const unsigned int* tile_offsets,
                                    int n, int elems_per_tile) {
    const unsigned int offset = tile_offsets[blockIdx.x];
    const int          base   = blockIdx.x * elems_per_tile;
    for (int i = threadIdx.x; i < elems_per_tile; i += blockDim.x) {
        const int idx = base + i;
        if (idx < n) data[idx] += offset;
    }
}

// --- Phase C: every survivor writes itself to the slot the scan assigned it. ---
// Distinct flags guarantee distinct offsets, so no two threads can collide.
__global__ void scatter_kept(const float* in, const unsigned int* flags,
                             const unsigned int* offsets, float* out, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && flags[idx] != 0u) out[offsets[idx]] = in[idx];
}

static size_t tile_shared_bytes(int elems) {
    return sizeof(unsigned int) * (elems + CONFLICT_FREE_OFFSET(elems - 1) + 1);
}

// Scratch for the tile-sum levels, allocated once outside any timed region.
struct ScanWorkspace {
    std::vector<unsigned int*> sums;
    std::vector<unsigned int*> offsets;
};

static ScanWorkspace make_workspace(int n) {
    ScanWorkspace ws;
    const int elems = 2 * BL_THREADS;
    for (int count = n;;) {
        const int tiles = (count + elems - 1) / elems;
        if (tiles <= 1) break;
        unsigned int *s = nullptr, *o = nullptr;
        CUDA_CHECK(cudaMalloc(&s, sizeof(unsigned int) * tiles));
        CUDA_CHECK(cudaMalloc(&o, sizeof(unsigned int) * tiles));
        ws.sums.push_back(s);
        ws.offsets.push_back(o);
        count = tiles;
    }
    return ws;
}

static void free_workspace(ScanWorkspace& ws) {
    for (unsigned int* p : ws.sums) CUDA_CHECK(cudaFree(p));
    for (unsigned int* p : ws.offsets) CUDA_CHECK(cudaFree(p));
    ws.sums.clear();
    ws.offsets.clear();
}

static void scan_exclusive_u32(const unsigned int* d_in, unsigned int* d_out, int n,
                               const ScanWorkspace& ws, int level = 0) {
    const int    elems = 2 * BL_THREADS;
    const int    tiles = (n + elems - 1) / elems;
    const size_t shmem = tile_shared_bytes(elems);

    if (tiles <= 1) {
        blelloch_scan_tile_u32<<<1, BL_THREADS, shmem>>>(d_in, d_out, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    blelloch_scan_tile_u32<<<tiles, BL_THREADS, shmem>>>(d_in, d_out, ws.sums[level], n);
    CUDA_CHECK(cudaGetLastError());

    scan_exclusive_u32(ws.sums[level], ws.offsets[level], tiles, ws, level + 1);

    add_tile_offsets_u32<<<tiles, BL_THREADS>>>(d_out, ws.offsets[level], n, elems);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Launch>
static double time_ms(Launch&& launch, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();                                       // warmup
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

int main() {
    const int    n     = 1 << 22;                   // 4,194,304 elements = 16 MiB
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const int    iters = 30;

    // Roughly one third of the elements are zeros, i.e. fail the predicate. The kept
    // values are copied unchanged, so the comparison against std::copy_if is exact.
    std::vector<float> h_in(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        h_in[static_cast<size_t>(i)] =
            (i % 3 == 0) ? 0.0f : static_cast<float>(i % 97 + 1);

    std::vector<float> h_expect;
    h_expect.reserve(h_in.size());
    std::copy_if(h_in.begin(), h_in.end(), std::back_inserter(h_expect),
                 [](float v) { return v != 0.0f; });

    float *d_in = nullptr, *d_out = nullptr;
    unsigned int *d_flags = nullptr, *d_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));          // worst case: nothing is discarded
    CUDA_CHECK(cudaMalloc(&d_flags, sizeof(unsigned int) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(unsigned int) * static_cast<size_t>(n)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    ScanWorkspace ws = make_workspace(n);

    const int mark_blocks    = (n + SCATTER_THREADS - 1) / SCATTER_THREADS;
    const int scatter_blocks = mark_blocks;

    auto compact = [&] {
        mark_nonzero<<<mark_blocks, SCATTER_THREADS>>>(d_in, d_flags, n);
        CUDA_CHECK(cudaGetLastError());
        scan_exclusive_u32(d_flags, d_offsets, n, ws);
        scatter_kept<<<scatter_blocks, SCATTER_THREADS>>>(d_in, d_flags, d_offsets, d_out, n);
        CUDA_CHECK(cudaGetLastError());
    };

    const double ms = time_ms(compact, iters);

    // The kept count is the total the exclusive scan dropped off its end: the last
    // offset plus the last flag. Both live on the device, so read the two tail words.
    unsigned int last_offset = 0u, last_flag = 0u;
    CUDA_CHECK(cudaMemcpy(&last_offset, d_offsets + (n - 1), sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_flag, d_flags + (n - 1), sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));
    const int kept = static_cast<int>(last_offset + last_flag);

    std::vector<float> h_got(static_cast<size_t>(kept));
    if (kept > 0)
        CUDA_CHECK(cudaMemcpy(h_got.data(), d_out, sizeof(float) * static_cast<size_t>(kept),
                              cudaMemcpyDeviceToHost));

    const bool count_ok = (kept == static_cast<int>(h_expect.size()));
    int        bad      = count_ok ? 0 : -1;
    if (count_ok) {
        for (int i = 0; i < kept; ++i)
            if (h_got[static_cast<size_t>(i)] != h_expect[static_cast<size_t>(i)]) ++bad;
    }

    const double useful_bytes =
        static_cast<double>(n) * sizeof(float) + static_cast<double>(kept) * sizeof(float);

    printf("stream compaction (keep non-zeros): n = %d\n", n);
    printf("kept (GPU)       : %d\n", kept);
    printf("kept (std::copy_if): %zu\n", h_expect.size());
    printf("time             : %.3f ms  (mark + 3-phase scan + scatter)\n", ms);
    printf("effective BW     : %.1f GB/s  (useful bytes: read n + write kept)\n",
           useful_bytes / (ms / 1.0e3) / 1.0e9);
    if (!count_ok)
        printf("result           : FAILED (count mismatch)\n");
    else
        printf("result           : %s (%d value mismatches)\n",
               bad == 0 ? "PASSED" : "FAILED", bad);

    printf("\nThe flags array and the scan add traffic the 'useful bytes' number ignores,\n"
           "so the real cost is well above the copy ceiling per useful byte. Fusing the\n"
           "predicate into the scan (single-pass, decoupled look-back) is how rocPRIM/CUB\n"
           "get compaction close to a plain copy - see README S5.\n");

    free_workspace(ws);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_offsets));
    return (count_ok && bad == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
