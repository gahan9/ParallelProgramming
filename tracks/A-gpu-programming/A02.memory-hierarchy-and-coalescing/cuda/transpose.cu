// Lab 2 (CUDA) - Matrix transpose: the canonical memory-hierarchy lesson.
//
// Line-for-line mirror of hip/transpose.cpp. Three versions, same result:
//   1) naive        - coalesced reads, strided (scattered) writes. Slow.
//   2) tiled         - shared-memory staging -> both global accesses coalesced.
//                      Fast, but shared-memory bank conflicts remain.
//   3) tiled_padded  - +1 column padding removes the bank conflict. Fastest.
//
// Reference: Ruetsch & Micikevicius, "Optimizing Matrix Transpose in CUDA" (NVIDIA).
//
// Build: nvcc -O3 -arch=sm_90 -Icuda transpose.cu -o transpose.exe
// Run:   ./transpose.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

#define TILE 32

__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) {
        out[x * n + y] = in[y * n + x];
    }
}

__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];

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

__global__ void transpose_tiled_padded(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];

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

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();                                   // warmup
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

    double seconds = (ms / 1.0e3) / iters;
    double bytes = 2.0 * static_cast<double>(n) * n * sizeof(float);
    return bytes / seconds / 1.0e9;
}

static bool verify(const std::vector<float>& in, const float* d_out, int n) {
    std::vector<float> out(static_cast<size_t>(n) * n);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out,
                          sizeof(float) * static_cast<size_t>(n) * n,
                          cudaMemcpyDeviceToHost));
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x)
            if (out[static_cast<size_t>(y) * n + x] != in[static_cast<size_t>(x) * n + y])
                return false;
    return true;
}

int main() {
    const int    n     = 4096;
    const size_t bytes = sizeof(float) * static_cast<size_t>(n) * n;
    const int    iters = 20;

    std::vector<float> h_in(static_cast<size_t>(n) * n);
    for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = static_cast<float>(i % 1000);

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    struct { Kind kind; const char* name; } cases[] = {
        {NAIVE,  "naive (strided writes)"},
        {TILED,  "tiled  (shared mem, bank conflicts)"},
        {PADDED, "tiled+padded (conflict-free)"},
    };

    printf("%-38s %-14s %-8s\n", "kernel", "effective GB/s", "result");
    printf("-------------------------------------- -------------- --------\n");
    for (auto c : cases) {
        double gbps = run(c.kind, d_in, d_out, n, iters);
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        run(c.kind, d_in, d_out, n, 1);
        bool ok = verify(h_in, d_out, n);
        printf("%-38s %-14.1f %-8s\n", c.name, gbps, ok ? "PASSED" : "FAILED");
    }
    printf("\nExpect: naive << tiled < tiled+padded. Same result, the whole delta is\n"
           "memory access pattern (coalescing + shared memory + bank conflicts).\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
