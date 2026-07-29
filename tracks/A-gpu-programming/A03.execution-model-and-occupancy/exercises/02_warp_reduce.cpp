// Exercise 2 - A wavefront-level reduction that is portable across warpSize 32 and 64.
//
// Summing an array is the "hello world" of GPU cooperation. The fast idiom does the
// first log2(warpSize) steps entirely in registers with __shfl_down - no shared memory,
// no __syncthreads(), because lanes of one wavefront are already in lockstep. Then one
// wavefront combines the per-wavefront partials.
//
// The starter is written the way most CUDA tutorials write it: with 32 baked in. On an
// AMD CDNA GPU (warpSize == 64) that silently drops half of every wavefront's data, and
// the second bug drops all but the first wavefront of each block. The program prints the
// ratio got/expected, so the factor tells you which bug you are looking at.
//
// YOUR TASKS
//   TODO 1: make `warp_reduce_sum` portable - derive the starting offset from `warpSize`
//           instead of hardcoding 16, so lane 0 ends up with the sum of ALL lanes on both
//           a 32-lane warp and a 64-lane wavefront.
//   TODO 2: combine the per-wavefront partials. Wavefront 0 should reduce the
//           `warp_partial[]` slots that were actually written and add the block total to
//           out[0] exactly once.
//
// Values are small integers, so every partial sum is exactly representable in float -
// a FAILED result means a real bug, never accumulated rounding.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_warp_reduce.cpp -o 02_warp_reduce.exe
// Run:   ./02_warp_reduce.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

// warpSize is a runtime device value in HIP (64 on CDNA, 32 on RDNA/NVIDIA), so it
// cannot size a __shared__ array. 1024 threads / 32 lanes = 32 is the worst case.
#define MAX_WARPS_PER_BLOCK 32

__device__ __forceinline__ float warp_reduce_sum(float v) {
    // TODO 1: this only reduces lanes within each group of 32. On a 64-lane wavefront
    //         lane 0 therefore holds half of the wavefront's sum. Start from warpSize/2.
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset, warpSize);
    return v;
}

__global__ void block_sum(const float* __restrict__ in, float* __restrict__ out, int n) {
    __shared__ float warp_partial[MAX_WARPS_PER_BLOCK];

    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    const int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    // Out-of-range lanes contribute the identity instead of returning early: every lane
    // of the wavefront must reach the shuffles below, or the reduction reads garbage.
    float v = (tid < n) ? in[tid] : 0.0f;

    v = warp_reduce_sum(v);
    if (lane == 0) warp_partial[warp] = v;
    __syncthreads();

    // TODO 2: this only publishes wavefront 0's partial, so the block total is short by
    //         every other wavefront. Reduce warp_partial[0 .. warps_per_block-1] with
    //         warp_reduce_sum and add the result to out[0] once.
    if (threadIdx.x == 0) atomicAdd(out, warp_partial[0]);
    (void)warps_per_block;   // computed for you; drop this line once TODO 2 uses it
}

int main() {
    const int    n     = 1 << 20;
    const int    block = 256;                  // multiple of both 32 and 64
    const int    grid  = (n + block - 1) / block;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    std::vector<float> h_in(n);
    double expected = 0.0;
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>((i % 8) + 1);   // 1..8, all sums exact in float
        expected += h_in[i];
    }

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(float)));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    block_sum<<<grid, block>>>(d_in, d_out, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    float got = 0.0f;
    HIP_CHECK(hipMemcpy(&got, d_out, sizeof(float), hipMemcpyDeviceToHost));

    const bool ok = std::fabs(got - static_cast<float>(expected)) < 0.5f;
    printf("warpSize     : %d lanes\n", prop.warpSize);
    printf("sum          : %.1f  (expected %.1f, ratio %.3fx)\n",
           got, expected, got / expected);
    printf("kernel time  : %.3f ms   effective BW %.1f GB/s\n",
           ms, bytes / (ms / 1.0e3) / 1.0e9);
    printf("result       : %s\n", ok ? "PASSED" : "FAILED");
    if (!ok)
        printf("\nRatio 0.5x -> TODO 1 (half the wavefront). Ratio 1/warps_per_block ->\n"
               "TODO 2 (only one wavefront published). Both -> the product of the two.\n");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
