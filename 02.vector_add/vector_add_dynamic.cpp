// filepath: c:\Projects\ParallelProgramming\02.vector_add\vector_add_dynamic.cpp
/*
Vector Addition Program with Dynamic GPU Resource Querying

This program:
1. Finds the number of GPU devices
2. Queries GPU memory and compute resources
3. Chooses an array size to utilize a large fraction of free memory
   while ensuring there's enough work to keep the GPU busy.

REF: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html#heterogeneous-programming
*/

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <hip/hip_runtime.h>

// Memory unit conversion macros
#define BYTES_TO_KB(bytes) ((bytes) / 1024.0)
#define BYTES_TO_MB(bytes) ((bytes) / (1024.0 * 1024.0))
#define BYTES_TO_GB(bytes) ((bytes) / (1024.0 * 1024.0 * 1024.0))

// Simple macro to check HIP errors and exit early
#define HIP_CHECK(call)                                                       \
    do {                                                                      \
        hipError_t _err = (call);                                             \
        if (_err != hipSuccess) {                                             \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n",                  \
                         __FILE__, __LINE__, hipGetErrorString(_err));        \
            return -1;                                                        \
        }                                                                     \
    } while (0)

// HIP kernel for vector addition
__global__ void vector_add(const float* A, const float* B, float* C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Helper to compute launch configuration (1D)
static void compute_launch_config(size_t N, int& threadsPerBlock, int& blocksPerGrid) {
    // Reasonable default; many GPUs like 128/256/512
    const int kDefaultBlockSize = 256;
    threadsPerBlock = kDefaultBlockSize;
    blocksPerGrid = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);
}

int main() {
    // -------------------------------------------------------------------------
    // 1. Device discovery and selection
    // -------------------------------------------------------------------------
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::fprintf(stderr, "No GPU devices found!\n");
        return -1;
    }

    std::printf("Number of GPU devices: %d\n", device_count);

    // For simplicity, use device 0
    const int device_id = 0;
    HIP_CHECK(hipSetDevice(device_id));

    hipDeviceProp_t devProp{};
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    size_t free_memory = 0, total_memory = 0;
    HIP_CHECK(hipMemGetInfo(&free_memory, &total_memory));

    // Display GPU information
    std::printf("\n=== GPU Device Information ===\n");
    std::printf("Device Name              : %s\n", devProp.name);
    std::printf("Total Global Memory      : %.2f GB\n", BYTES_TO_GB(devProp.totalGlobalMem));
    std::printf("Free Global Memory       : %.2f GB\n", BYTES_TO_GB(free_memory));
    std::printf("Compute Units            : %d\n", devProp.multiProcessorCount);
    std::printf("Max Threads Per Block    : %d\n", devProp.maxThreadsPerBlock);
    std::printf("Max Grid Size            : [%d, %d, %d]\n",
                devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);

    // -------------------------------------------------------------------------
    // 2. Compute problem size (N) from memory and core constraints
    // -------------------------------------------------------------------------
    const double kMemoryUsageFraction = 0.80;    // Use 80% of free memory
    const size_t kMinN = 1024;                   // Minimum size for a meaningful test
    const size_t kAlignment = 256;               // Align N to multiple of 256 for nicer blocks

    // 3 arrays: A, B, C
    size_t usable_memory = static_cast<size_t>(free_memory * kMemoryUsageFraction);
    size_t memory_per_array = usable_memory / 3;

    size_t max_elements_by_memory = memory_per_array / sizeof(float);

    // Ensure enough work to keep cores busy; 4x oversubscription is common
    size_t min_elements_by_cores =
        static_cast<size_t>(devProp.multiProcessorCount) *
        static_cast<size_t>(devProp.maxThreadsPerBlock) * 4;

    // Memory is usually the tight constraint, but we ensure we meet core minimum
    size_t N = std::max(max_elements_by_memory, min_elements_by_cores);

    // Enforce a lower bound
    if (N < kMinN) {
        N = kMinN;
    }

    // Align down to nearest multiple of kAlignment
    N = (N / kAlignment) * kAlignment;

    // Recompute the total bytes we plan to allocate
    size_t bytes_per_array = N * sizeof(float);
    size_t total_bytes = 3 * bytes_per_array;

    // Clamp to 80% of free memory in case we overshot due to alignment
    if (total_bytes > usable_memory) {
        N = (usable_memory / (3 * sizeof(float)));
        N = (N / kAlignment) * kAlignment;
        bytes_per_array = N * sizeof(float);
        total_bytes = 3 * bytes_per_array;
    }

    // Final safety check
    if (N == 0 || total_bytes == 0 || total_bytes > free_memory) {
        std::fprintf(stderr,
                     "Unable to choose a valid N: N=%zu, total_bytes=%zu, free_memory=%zu\n",
                     N, total_bytes, free_memory);
        return -1;
    }

    std::printf("\n=== Calculated Array Size ===\n");
    std::printf("Max elements by memory   : %zu\n", max_elements_by_memory);
    std::printf("Min elements by cores    : %zu\n", min_elements_by_cores);
    std::printf("Selected array size (N)  : %zu\n", N);
    std::printf("Memory per array         : %.2f GB (%.0f bytes)\n",
                BYTES_TO_GB(bytes_per_array), static_cast<double>(bytes_per_array));
    std::printf("Total memory to allocate : %.2f GB (%.0f bytes)\n",
                BYTES_TO_GB(total_bytes), static_cast<double>(total_bytes));

    // -------------------------------------------------------------------------
    // 3. Host memory allocation and initialization
    // -------------------------------------------------------------------------
    float* A = static_cast<float*>(std::malloc(bytes_per_array));
    float* B = static_cast<float*>(std::malloc(bytes_per_array));
    float* C = static_cast<float*>(std::malloc(bytes_per_array));

    if (!A || !B || !C) {
        std::fprintf(stderr, "Error allocating host memory!\n");
        std::free(A);
        std::free(B);
        std::free(C);
        return -1;
    }

    for (size_t i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(N - i);
    }

    // -------------------------------------------------------------------------
    // 4. Device memory allocation
    // -------------------------------------------------------------------------
    std::printf("\n=== Allocating Device Memory ===\n");
    std::printf("Allocating %.2f GB per array\n", BYTES_TO_GB(bytes_per_array));

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    HIP_CHECK(hipMalloc(&d_A, bytes_per_array));
    HIP_CHECK(hipMalloc(&d_B, bytes_per_array));
    HIP_CHECK(hipMalloc(&d_C, bytes_per_array));

    // Quick memory status check (optional but informative)
    size_t free_after_alloc = 0, total_after_alloc = 0;
    HIP_CHECK(hipMemGetInfo(&free_after_alloc, &total_after_alloc));
    std::printf("Free GPU memory after alloc: %.2f GB\n", BYTES_TO_GB(free_after_alloc));

    // -------------------------------------------------------------------------
    // 5. Copy input data to device
    // -------------------------------------------------------------------------
    HIP_CHECK(hipMemcpy(d_A, A, bytes_per_array, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, B, bytes_per_array, hipMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // 6. Kernel launch
    // -------------------------------------------------------------------------
    int threadsPerBlock = 0, blocksPerGrid = 0;
    compute_launch_config(N, threadsPerBlock, blocksPerGrid);

    std::printf("\n=== Kernel Launch Configuration ===\n");
    std::printf("Threads per block : %d\n", threadsPerBlock);
    std::printf("Blocks per grid   : %d\n", blocksPerGrid);
    std::printf("Total threads     : %lld\n",
                static_cast<long long>(blocksPerGrid) * threadsPerBlock);

    hipLaunchKernelGGL(vector_add,
                       dim3(blocksPerGrid),
                       dim3(threadsPerBlock),
                       0, 0,
                       d_A, d_B, d_C, N);

    HIP_CHECK(hipDeviceSynchronize());

    // -------------------------------------------------------------------------
    // 7. Copy result back and verify
    // -------------------------------------------------------------------------
    HIP_CHECK(hipMemcpy(C, d_C, bytes_per_array, hipMemcpyDeviceToHost));

    std::printf("\n=== Verification ===\n");
    bool correct = true;
    const size_t kCheckCount = 10;
    for (size_t i = 0; i < std::min(kCheckCount, N); ++i) {
        float expected = A[i] + B[i];
        if (std::fabs(C[i] - expected) > 1e-5f) {
            std::printf("Error at index %zu: expected %f, got %f\n", i, expected, C[i]);
            correct = false;
        }
    }

    if (correct) {
        std::printf("First %zu elements verified correctly.\n", std::min(kCheckCount, N));
        std::printf("Sample results:\n");
        for (size_t i = 0; i < std::min<size_t>(5, N); ++i) {
            std::printf("C[%zu] = %f (A[%zu]=%f + B[%zu]=%f)\n",
                        i, C[i], i, A[i], i, B[i]);
        }
    }

    // -------------------------------------------------------------------------
    // 8. Cleanup
    // -------------------------------------------------------------------------
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    std::free(A);
    std::free(B);
    std::free(C);

    std::printf("\nProgram completed successfully!\n");
    return 0;
}
