// Lab 1 (CUDA) - Hello from thousands of threads.
//
// Line-for-line comparable to hip/hello.cpp. Each thread prints its indices;
// the print order is nondeterministic across threads by design.
//
// Build: nvcc -O3 -arch=sm_90 hello.cu -o hello.exe
// Run:   ./hello.exe
#include <cstdio>
#include <cuda_runtime.h>
#include "check.cuh"

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
    CUDA_CHECK(cudaGetLastError());        // catch launch-configuration errors
    CUDA_CHECK(cudaDeviceSynchronize());   // wait for device printf to flush

    printf("Hello from the CPU (host) - after the GPU finished.\n");
    return 0;
}
