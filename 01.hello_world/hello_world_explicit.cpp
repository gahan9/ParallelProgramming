#include <stdio.h>
#include <hip/hip_runtime.h>
#include <string.h>

// Kernel that runs on the GPU
__global__ void hello_gpu() {
    printf("Hello World from GPU! (Thread %d, Block %d)\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    hipStream_t hip_stream_id = hipStreamDefault; // Default stream ID would be 0x00
    hipError_t hip_error;

    // Launch kernel with 2 block and 4 threads
    int total_blocks = 2; // Number of blocks in the grid
    int threads_per_block = 4; // Number of threads per block

    // 1. Standard Triple Chevron Syntax
    hip_error = hipStreamCreate(&hip_stream_id);
    printf("Kernel launch...!\n");
    printf("hip_stream_id : %p\n", (void*)hip_stream_id);
    hello_gpu<<<dim3(total_blocks), // 3D grid specifying number of blocks to launch: (2, 1, 1)
                dim3(threads_per_block), // 3D grid specifying number of threads to launch: (4, 1, 1)
                0, // number of bytes of additional shared memory to allocate
                hip_stream_id // stream where the kernel should execute: default stream
                >>>();

    // Wait for GPU to finish before accessing results
    hip_error = hipDeviceSynchronize();

    // 2. hipLaunchKernelGGL macro
    hip_error = hipStreamCreate(&hip_stream_id);
    printf("Kernel Launch..........!\n");
    printf("hip_stream_id : %p\n", (void*)hip_stream_id);
    hipLaunchKernelGGL(hello_gpu, // kernel function to launch
                       dim3(total_blocks), // 3D grid specifying number of blocks to launch: (2, 1, 1)
                       dim3(threads_per_block), // 3D grid specifying number of threads to launch: (4, 1, 1)
                       0, // number of bytes of additional shared memory to allocate
                       hip_stream_id // stream where the kernel should execute: default stream
                      );

    // Wait for GPU to finish before accessing results
    hip_error = hipDeviceSynchronize();

    // Check for errors
    hip_error = hipGetLastError();
    if (hip_error != hipSuccess) {
        fprintf(stderr, "HIP error: %s\n", hipGetErrorString(hip_error));
        return -1;
    }

    printf("GPU execution completed!\n");

    printf("Hello World from CPU!\n");
    return 0;
}