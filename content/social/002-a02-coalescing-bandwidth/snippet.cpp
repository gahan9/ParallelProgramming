// Teaching core — A02 transpose ladder (HIP twin of CUDA lab).
// Full file: tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/hip/transpose.cpp
// Expect: naive << tiled < tiled+padded. Same result; the delta is access pattern.

#define TILE 32

// 1) Naive: reads coalesced, WRITES strided by row width → scattered global stores.
__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) {
        out[x * n + y] = in[y * n + x];  // write index jumps by n each lane
    }
}

// 2) Tiled: load coalesced → shared → store coalesced (column reads hit bank conflicts).
__global__ void transpose_tiled(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE];  // column stride == 32 banks → conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < n && ty < n)
        out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];
}

// 3) +1 pad: shifts each row by one bank → conflict-free column reads.
__global__ void transpose_tiled_padded(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];  // the one-character win
    // same body as tiled — only the shared declaration changes
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();
    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < n && ty < n)
        out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];
}

// Stretch: profile tiled vs padded with ncu / rocprof-compute bank-conflict counters;
// confirm the GB/s gap matches conflict → conflict-free, not a timing artifact.
