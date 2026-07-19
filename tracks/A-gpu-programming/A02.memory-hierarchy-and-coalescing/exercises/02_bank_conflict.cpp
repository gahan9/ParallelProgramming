// Exercise 2 - Fix the shared-memory bank conflict.
//
// This kernel loads a 32x32 tile into shared memory and then sums each COLUMN
// (writing 32 column-sums). Reading the tile by column with a [32][32] layout means
// every lane of a warp hits the SAME bank -> a 32-way bank conflict -> the shared
// reads serialize. Correct result, but far slower than it should be.
//
// YOUR TASK (TODO 1): change the shared tile declaration so the column reads become
// conflict-free (hint: pad the minor dimension). The result must stay PASSED and the
// kernel should speed up (confirm with `ncu --set full` / rocprofv3 bank-conflict metric).
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_bank_conflict.cpp -o 02_bank.exe
// Run:   ./02_bank.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define TILE 32

__global__ void col_sum(const float* in, float* out, int n) {
    // TODO 1: this declaration causes a 32-way bank conflict on the column read below.
    //         Pad the minor dimension to make column accesses hit distinct banks.
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;   // global column
    int y = blockIdx.y * TILE + threadIdx.y;   // global row

    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();

    // Thread (threadIdx.x == c, threadIdx.y == 0) sums column c of the tile.
    if (threadIdx.y == 0 && x < n) {
        float s = 0.0f;
        for (int r = 0; r < TILE; ++r)
            s += tile[r][threadIdx.x];         // column read -> conflict with [32][32]
        atomicAdd(&out[x], s);
    }
}

int main() {
    const int    n     = 4096;
    const size_t bytes = sizeof(float) * static_cast<size_t>(n) * n;

    std::vector<float> h_in(static_cast<size_t>(n) * n);
    std::vector<float> h_ref(n, 0.0f);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) {
            float v = static_cast<float>((x + y) % 7);
            h_in[static_cast<size_t>(y) * n + x] = v;
            h_ref[x] += v;
        }

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, n * sizeof(float)));

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    col_sum<<<grid, block>>>(d_in, d_out, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_out(n);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, n * sizeof(float), hipMemcpyDeviceToHost));

    int errors = 0;
    for (int c = 0; c < n; ++c)
        if (std::abs(h_out[c] - h_ref[c]) > 1e-1f) ++errors;
    printf("column-sum %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
