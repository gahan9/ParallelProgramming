// Exercise 2 - Find and fix the missing barriers.
//
// This block reduction has the right tree, the right identity, and the right
// bounds guard. It is still wrong, because two `__syncthreads()` calls are
// missing. Shared memory is only coherent across a block *at a barrier*: without
// one, a thread in wavefront 0 can read a slot that a thread in wavefront 3 has
// not written yet. The result is a data race - which is worse than a crash,
// because it often produces the right answer anyway.
//
// YOUR TASKS
//   TODO 1: a barrier is missing between the shared-memory load and the tree.
//   TODO 2: a barrier is missing between the tree steps.
//
// HOW TO READ THE OUTPUT
// The program runs the same kernel on the same data 20 times. A correct kernel
// gives one answer, every time, matching the CPU reference. The broken kernel
// typically gives several different answers, or one consistently wrong answer -
// but it MAY print PASSED on some hardware or some runs. That is the whole
// point: a missing barrier is undefined behavior, and "it worked on my GPU" is
// not evidence of correctness. Confirm with a race detector:
//     compute-sanitizer --tool racecheck ./02_fix_sync.exe      (NVIDIA)
//
// A THIRD TRAP, deliberately not shipped as runnable code:
//     for (s = BLOCK/2; s > 0; s >>= 1) {
//         if (tid < s) { sdata[tid] += sdata[tid + s]; __syncthreads(); }
//     }
// Here the barrier sits INSIDE a divergent branch, so the wavefronts with
// tid >= s never arrive. `__syncthreads()` requires every non-exited thread of
// the block to reach it; anything else is undefined and on real hardware
// usually *hangs* the kernel until the watchdog kills it. Barriers must always
// be at uniform (block-wide) control flow - keep them outside the `if`.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_fix_sync.cpp -o 02_fix_sync.exe
// Run:   ./02_fix_sync.exe
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

    // TODO 1: the tree below reads slots written by OTHER threads. What has to
    //         happen here before a single one of those reads is legal?

    for (unsigned s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // TODO 2: step s+1 reads slots that step s just wrote, from threads in
        //         other wavefronts. What is missing between the steps?
        //         (Put it here, at uniform control flow - NOT inside the `if`.)
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

        // Sum the per-block partials on the host in double, so any error the
        // table reports comes from the kernel and not from this last step.
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
    if (mismatches == 0 && !stable) {
        printf("note             : close enough to pass the tolerance, but NOT\n"
               "                   reproducible - still a race. Fix the TODOs.\n");
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_partial));
    return (mismatches == 0 && stable) ? EXIT_SUCCESS : EXIT_FAILURE;
}
