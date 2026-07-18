// Lab 1 (HIP) - Hello from thousands of threads.
//
// Each thread prints its (block, thread, global) index. Run it a few times and
// notice the print order is NOT sequential: the GPU schedules blocks/wavefronts
// in whatever order it likes. Lesson: never assume an ordering across threads.
//
// Build: hipcc -O3 --offload-arch=gfx942 hello.cpp -o hello.exe
// Run:   ./hello.exe
#include <cstdio>
#include <hip/hip_runtime.h>
#include "check.h"

__global__ void hello_kernel() {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from block %d, thread %d -> global index %d\n",
           blockIdx.x, threadIdx.x, global_id);
}

int main() {
    const int blocks = 2;
    const int threads_per_block = 4;

    printf("Launching %d blocks x %d threads = %d GPU threads\n",
           blocks, threads_per_block, blocks * threads_per_block);

    hello_kernel<<<blocks, threads_per_block>>>();
    HIP_CHECK(hipGetLastError());        // catch launch-configuration errors
    HIP_CHECK(hipDeviceSynchronize());   // wait for device printf to flush

    printf("Hello from the CPU (host) - after the GPU finished.\n");
    return 0;
}
