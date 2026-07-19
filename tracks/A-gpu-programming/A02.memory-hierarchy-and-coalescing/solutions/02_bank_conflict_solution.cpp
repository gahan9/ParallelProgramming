// Solution 2 - Conflict-free column reduction via +1 padding.
//
// The only change from the exercise is the shared tile declaration:
//     __shared__ float tile[TILE][TILE];      // 32-way bank conflict on column read
//  -> __shared__ float tile[TILE][TILE + 1];  // conflict-free
//
// Why it works: shared memory has 32 banks of 4-byte words. With a row width of 33
// words, element tile[r][c] maps to bank (r*33 + c) % 32 = (r + c) % 32. Holding c
// fixed and varying r across the 32 lanes of the column read yields 32 distinct banks
// -> one cycle instead of 32. Cost: 32 extra floats (128 B) of shared memory per block.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_bank_conflict_solution.cpp -o 02_bank.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define TILE 32

__global__ void col_sum(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];     // <-- the fix: pad the minor dimension

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];
    __syncthreads();

    if (threadIdx.y == 0 && x < n) {
        float s = 0.0f;
        for (int r = 0; r < TILE; ++r)
            s += tile[r][threadIdx.x];         // now hits 32 distinct banks
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
