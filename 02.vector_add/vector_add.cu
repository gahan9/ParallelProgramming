#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 300000000

void vector_add(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    float* a, * b, * out;

    // Allocate memory
    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    printf("Vector addition on CPU...\n");
    time_t start = time(NULL);
    vector_add(out, a, b, N);
    time_t end = time(NULL);
    printf("Time taken: %ld seconds\n", end - start);


    // Allocate memory on GPU
    float* d_a, * d_b, * d_out;
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Copy data from host to device
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Launch kernel
    printf("Vector addition on GPU...\n");
    time_t start_gpu = time(NULL);
    vector_add_gpu << <1, 1 >> > (d_out, d_a, d_b, N);
    // cudaDeviceSynchronize();
    time_t end_gpu = time(NULL);
    printf("Time taken: %ld seconds\n", end_gpu - start_gpu);
    // Free memory
    free(a);
    free(b);
    free(out);
    return 0;
}
