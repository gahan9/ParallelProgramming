// Solution 2 - The two missing barriers, and why each one is load-bearing.
//
// BARRIER 1, after the shared-memory load.
//   Thread `tid` writes sdata[tid], then the first tree step has thread `tid`
//   read sdata[tid + 128] - a slot owned by a thread in a DIFFERENT wavefront.
//   Nothing orders that write before that read. With 256 threads and 64-lane
//   wavefronts the block is 4 independent wavefronts that the scheduler
//   interleaves however it likes, so wavefront 0 can reach the tree before
//   wavefront 2 has finished its store. The barrier is what makes "all stores by
//   all threads of this block are visible" true.
//
// BARRIER 2, between the tree steps.
//   Step s writes sdata[0 .. s-1]; step s/2 reads sdata[0 .. s-1]. Same
//   cross-wavefront hazard, once per level. One barrier at the bottom of the
//   loop body covers every level.
//
// WHERE THE BARRIER MUST NOT GO.
//   Inside `if (tid < s)`. __syncthreads() is a *block-wide* rendezvous: every
//   thread of the block that has not exited must execute it, and all of them
//   must execute the SAME barrier. Wavefronts whose lanes all have tid >= s skip
//   the branch entirely, never arrive, and the wavefronts that did arrive wait
//   forever - a hang, not a wrong answer. Keep barriers at uniform control flow.
//   The same rule is why `if (tid < s) { ... }` with the barrier *outside* is
//   fine: the branch diverges, the barrier does not.
//
// WHAT YOU CANNOT REMOVE, AND WHAT YOU CAN.
//   Once s <= 32 the surviving threads are all inside one warp/wavefront, which
//   executes in lockstep on AMD, so the barrier is redundant there - that is
//   exactly ladder stage 5 in the main lab. On NVIDIA Volta and later,
//   independent thread scheduling means "same warp" no longer implies lockstep,
//   and the tail needs __syncwarp() or warp shuffles instead. Do not delete a
//   barrier because a run looked green; delete it because the memory model says
//   you may.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_fix_sync_solution.cpp -o 02_fix_sync_solution.exe
// Run:   ./02_fix_sync_solution.exe
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256u

__global__ void reduce_sum(const float* in, float* out, unsigned n) {
    __shared__ float sdata[BLOCK];
    const unsigned tid = threadIdx.x;
    const unsigned i   = blockIdx.x * BLOCK + tid;

    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();                       // BARRIER 1: tile is fully populated

    for (unsigned s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();                   // BARRIER 2: level s is complete
    }

    if (tid == 0) out[blockIdx.x] = sdata[0];
}

int main() {
    const unsigned n      = 1u << 22;            // 4,194,304 floats
    const unsigned grid   = (n + BLOCK - 1) / BLOCK;
    const size_t   bytes  = sizeof(float) * n;
    const int      trials = 20;

    std::vector<float> h_in(n);
    double             ref = 0.0;                // fp64 reference
    for (unsigned i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i % 100) * 0.01f;
        ref += static_cast<double>(h_in[i]);
    }

    float *d_in = nullptr, *d_partial = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_partial, sizeof(float) * grid));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    std::vector<float> partials(grid);
    unsigned           first_bits = 0;
    bool               stable     = true;
    int                mismatches = 0;
    double             worst_rel  = 0.0;

    for (int t = 0; t < trials; ++t) {
        HIP_CHECK(hipMemset(d_partial, 0, sizeof(float) * grid));
        reduce_sum<<<grid, BLOCK>>>(d_in, d_partial, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(partials.data(), d_partial, sizeof(float) * grid,
                            hipMemcpyDeviceToHost));

        double total = 0.0;
        for (float p : partials) total += static_cast<double>(p);

        const float  as_float = static_cast<float>(total);
        unsigned     bits     = 0;
        std::memcpy(&bits, &as_float, sizeof(bits));
        if (t == 0) first_bits = bits;
        if (bits != first_bits) stable = false;

        const double rel = std::fabs(total - ref) / std::fabs(ref);
        if (rel > worst_rel) worst_rel = rel;
        if (rel >= 1.0e-4) ++mismatches;
    }

    printf("n = %u, grid = %u, %d trials\n", n, grid, trials);
    printf("reference (fp64) : %.6f\n", ref);
    printf("run-to-run       : %s\n",
           stable ? "identical every trial" : "DIFFERENT between trials (race!)");
    printf("worst rel error  : %.3e\n", worst_rel);
    printf("result           : %s (%d/%d trials outside tolerance)\n",
           (mismatches == 0 && stable) ? "PASSED" : "FAILED", mismatches, trials);

    // With both barriers in place the kernel is deterministic: the same input
    // and the same launch geometry visit the additions in the same order every
    // time. That is a property of the two-launch tree, not of GPUs in general -
    // the atomicAdd variant in the main lab gives it up. See README S6.
    printf("\nWith both barriers the answer is reproducible run to run, and the\n"
           "residual ~1e-7 error is float summation order versus fp64 - not a bug.\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_partial));
    return (mismatches == 0 && stable) ? EXIT_SUCCESS : EXIT_FAILURE;
}
