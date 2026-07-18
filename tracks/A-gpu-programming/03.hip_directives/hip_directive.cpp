// Define the method to be timed
__global__ void my_cuda_method() {
    // Some CUDA operations
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Example: perform some calculations
    float result = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        result += sin(tid + i) * cos(tid - i);
    }
}

int main() {
    cudaEvent_t start, stop;
    float elapsedTime;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Execute the CUDA method
    int blockSize = 256;
    int numBlocks = 32;
    my_cuda_method<<<numBlocks, blockSize>>>();

    // Record the stop event
    cudaEventRecord(stop, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the elapsed time
    std::cout << "Execution time of my_cuda_method: " << elapsedTime << " ms" << std::endl;

    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}