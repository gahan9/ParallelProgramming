// Lab 2 (HIP) - Matrix transpose: the canonical memory-hierarchy lesson.
//
// Transpose is pure data movement (zero arithmetic), so it is a clean microscope on
// the memory system. Three versions, same result, very different bandwidth:
//
//   1) naive        - reads are coalesced, WRITES are strided by the row width ->
//                      each wavefront scatters across N cache lines. Slow.
//   2) tiled         - stage a tile in shared memory (LDS), so BOTH global reads and
//                      global writes are coalesced. The transpose happens in fast
//                      on-chip memory. Fast -- but shared-memory BANK CONFLICTS remain.
//   3) tiled_padded  - pad the shared tile by one column (TILE+1) so column accesses
//                      hit distinct banks. Removes the conflict. Fastest.
//
// This is the shared-memory / coalescing / bank-conflict trifecta in one file.
// Reference: Ruetsch & Micikevicius, "Optimizing Matrix Transpose in CUDA" (NVIDIA).
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip transpose.cpp -o transpose.exe
// Run:   ./transpose.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define TILE 32

// --- 1) Naive: out[x*N + y] = in[y*N + x]. Reads coalesced, writes strided. ---
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;   // column
    int y = blockIdx.y * TILE + threadIdx.y;   // row
    if (x < n && y < n) {
        out[x * n + y] = in[y * n + x];        // write index strided by n -> scattered
    }
}

// --- 2) Tiled: stage TILE x TILE block in shared memory, then write coalesced. ---
__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];         // conflict-prone: column stride == 32 banks

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];   // coalesced global read
    }
    __syncthreads();

    // Transposed output tile origin: swap block indices.
    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < n && ty < n) {
        out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];  // coalesced global write
    }
}

// --- 3) Tiled + padded: +1 column breaks the shared-memory bank conflict. ---
__global__ void transpose_tiled_padded(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];     // padding shifts each row by one bank

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    }
    __syncthreads();

    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < n && ty < n) {
        out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

enum Kind { NAIVE, TILED, PADDED };

static double run(Kind kind, const float* d_in, float* d_out, int n, int iters) {
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    auto launch = [&]() {
        switch (kind) {
            case NAIVE:  transpose_naive<<<grid, block>>>(d_in, d_out, n); break;
            case TILED:  transpose_tiled<<<grid, block>>>(d_in, d_out, n); break;
            case PADDED: transpose_tiled_padded<<<grid, block>>>(d_in, d_out, n); break;
        }
    };

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    launch();                                   // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < iters; ++it) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    double seconds = (ms / 1.0e3) / iters;
    double bytes = 2.0 * static_cast<double>(n) * n * sizeof(float);  // read + write
    return bytes / seconds / 1.0e9;             // GB/s
}

static bool verify(const std::vector<float>& in, const float* d_out, int n) {
    std::vector<float> out(static_cast<size_t>(n) * n);
    HIP_CHECK(hipMemcpy(out.data(), d_out,
                        sizeof(float) * static_cast<size_t>(n) * n,
                        hipMemcpyDeviceToHost));
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            if (out[static_cast<size_t>(y) * n + x] != in[static_cast<size_t>(x) * n + y])
                return false;
    return true;
}

int main() {
    const int    n     = 4096;                  // 4096 x 4096 -> 64 MB per matrix
    const size_t bytes = sizeof(float) * static_cast<size_t>(n) * n;
    const int    iters = 20;

    std::vector<float> h_in(static_cast<size_t>(n) * n);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(i % 1000);

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    struct { Kind kind; const char* name; } cases[] = {
        {NAIVE,  "naive (strided writes)"},
        {TILED,  "tiled  (shared mem, bank conflicts)"},
        {PADDED, "tiled+padded (conflict-free)"},
    };

    printf("%-38s %-14s %-8s\n", "kernel", "effective GB/s", "result");
    printf("-------------------------------------- -------------- --------\n");
    for (auto c : cases) {
        double gbps = run(c.kind, d_in, d_out, n, iters);
        HIP_CHECK(hipMemset(d_out, 0, bytes));
        run(c.kind, d_in, d_out, n, 1);         // repopulate for verify
        bool ok = verify(h_in, d_out, n);
        printf("%-38s %-14.1f %-8s\n", c.name, gbps, ok ? "PASSED" : "FAILED");
    }
    printf("\nExpect: naive << tiled < tiled+padded. Same result, the whole delta is\n"
           "memory access pattern (coalescing + shared memory + bank conflicts).\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
