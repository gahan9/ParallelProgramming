// Exercise 2 solution - Portable wavefront reduction (warpSize 32 and 64).
//
// Two fixes, both about never hardcoding the lockstep width:
//
//   1) Start the shuffle ladder at `warpSize / 2`, not 16. On a 64-lane wavefront the
//      ladder is 32,16,8,4,2,1 (6 steps); on a 32-lane warp it is 16,...,1 (5 steps).
//      Passing `warpSize` as the shuffle `width` keeps the exchange inside one wavefront.
//      After the ladder, lane 0 holds the sum of the whole wavefront. The other lanes hold
//      partial garbage - by design, __shfl_down leaves the top lanes with stale values, so
//      only lane 0's result may be used.
//
//   2) Reduce the per-wavefront partials with the SAME routine. `warps_per_block` is
//      blockDim.x / warpSize, which is 4 on CDNA and 8 on NVIDIA for a 256-thread block -
//      so it must be computed, never assumed. Lanes past that count feed in 0.0f (the
//      identity) rather than skipping the shuffle, because every lane of wavefront 0 has
//      to execute the same shuffle instructions.
//
// Why no __syncthreads() inside warp_reduce_sum: lanes of one wavefront share a program
// counter (CDNA) or are converged by the shuffle's mask (NVIDIA Volta+), so register
// exchange needs no barrier. The single __syncthreads() is only there because the
// per-wavefront partials cross wavefront boundaries via shared memory.
//
// Portability note: on NVIDIA the modern spelling is __shfl_down_sync(mask, v, offset)
// with an explicit mask, because Volta+ has independent thread scheduling and the
// implicitly-synchronized __shfl_down is deprecated. HIP's __shfl_down maps to the AMD
// DPP/permute path. If you need one source for both, wrap the shuffle in a macro or use
// rocPRIM / CUB, which already do.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_warp_reduce_solution.cpp -o 02_warp_reduce_solution.exe
// Run:   ./02_warp_reduce_solution.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

// warpSize is a runtime device value in HIP, so it cannot size a __shared__ array.
#define MAX_WARPS_PER_BLOCK 32

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset, warpSize);
    return v;                                  // valid in lane 0 only
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

    if (warp == 0) {
        float p = (lane < warps_per_block) ? warp_partial[lane] : 0.0f;
        p = warp_reduce_sum(p);
        if (lane == 0) atomicAdd(out, p);
    }
}

int main() {
    const int    n     = 1 << 20;
    const int    block = 256;                  // multiple of both 32 and 64
    const int    grid  = (n + block - 1) / block;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));

    // A 256-thread block is 4 wavefronts on CDNA and 8 warps on NVIDIA; both fit the
    // MAX_WARPS_PER_BLOCK slots, which is what makes the static array safe.
    if (block / prop.warpSize > MAX_WARPS_PER_BLOCK) {
        std::fprintf(stderr, "block too large for warp_partial[]\n");
        return EXIT_FAILURE;
    }

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
    printf("warpSize     : %d lanes  (%d per 256-thread block)\n",
           prop.warpSize, block / prop.warpSize);
    printf("sum          : %.1f  (expected %.1f, ratio %.3fx)\n",
           got, expected, got / expected);
    printf("kernel time  : %.3f ms   effective BW %.1f GB/s\n",
           ms, bytes / (ms / 1.0e3) / 1.0e9);
    printf("result       : %s\n", ok ? "PASSED" : "FAILED");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
