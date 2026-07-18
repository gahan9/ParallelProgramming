#include <stdio.h>
#include <hip/hip_runtime.h>
#include <string.h>

// Kernel that runs on the GPU
__global__ void hello_gpu() {
    printf("Hello World from GPU! (Thread %d, Block %d)\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    hipError_t hip_error;
    
    printf("Halting Execution before kernel launch...!\n");
    getchar();

    // Launch kernel with 1 block and 4 threads
    hello_gpu<<<2, 4>>>();
    
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