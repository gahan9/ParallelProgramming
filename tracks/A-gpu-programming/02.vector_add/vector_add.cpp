/*
Vector Addition Program


REF: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html#heterogeneous-programming

*/

#include <stdio.h>
#include <hip/hip_runtime.h>

#define N 1024

// HIP kernel for vector addition
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;
    
    hipError_t hip_error;

    // Allocate host memory
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = (N - i) * 1.0f;
    }

    // Allocate device memory
    hip_error = hipMalloc(&d_A, N * sizeof(float));
    hip_error = hipMalloc(&d_B, N * sizeof(float));
    hip_error = hipMalloc(&d_C, N * sizeof(float));

    // Copy data from host to device
    hip_error = hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice);
    hip_error = hipMemcpy(d_B, B, N * sizeof(float), hipMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vector_add, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);

    // Copy result back to host
    hip_error = hipMemcpy(C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);

    // Print a few results
    for (int i = 0; i < 5; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    // Free device and host memory
    hip_error = hipFree(d_A);
    hip_error = hipFree(d_B);
    hip_error = hipFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
