// Lab 2 (CUDA) - Occupancy: theory (the runtime's occupancy API) vs measurement (the clock).
//
// Line-for-line twin of hip/occupancy.cpp. The kernel is a DEPENDENT pointer chase over
// a cache-resident buffer: each hop's address is unknown until the previous load returns,
// so a single warp spends almost all its time stalled. The only way to go fast is to have
// MANY warps resident so the scheduler always has a ready one - i.e. latency hiding,
// which is exactly what occupancy buys. (`next[p] = (p + STRIDE) % n` keeps lanes
// contiguous, so this measures dependent-load latency, not coalescing - that was A02.)
//
// Two controlled sweeps:
//   1) block size {64,128,256,512,1024}, zero dynamic shared memory. Theoretical
//      occupancy is usually FLAT here for a simple kernel; the measured time still moves
//      (scheduling granularity, tail effect). Seeing "occupancy flat, time not" is the point.
//   2) fixed 256-thread block, dynamic shared memory per block {2,8,16,32} KB. Shared
//      memory becomes the binding limiter, occupancy falls, and the time rises with it.
//      (Above 48 KB per block CUDA requires opting in via
//      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes).)
//
// The two tables use different kernels, so compare rows WITHIN a table, never across.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda occupancy.cu -o occupancy.exe
// Run:   ./occupancy.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

#define STRIDE 65537          // prime; coprime with a power-of-two n, so the walk never
                              // short-cycles and every lane keeps moving

__global__ void chase(const int* __restrict__ next, long long* __restrict__ out,
                      int n, int hops) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        int       p   = tid;
        long long acc = 0;
        for (int h = 0; h < hops; ++h) {
            p = next[p];      // the next address is unknown until THIS load returns
            acc += p;
        }
        out[tid] = acc;
    }
}

// Same chase, but the accumulator lives in dynamic shared memory. Bytes beyond the first
// blockDim.x slots are reserved and untouched on purpose: sweep 2 varies the shared-memory
// FOOTPRINT to move the occupancy limiter while holding the work fixed.
__global__ void chase_smem(const int* __restrict__ next, long long* __restrict__ out,
                           int n, int hops) {
    extern __shared__ long long scratch[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        scratch[threadIdx.x] = 0;
        int p = tid;
        for (int h = 0; h < hops; ++h) {
            p = next[p];
            scratch[threadIdx.x] += p;
        }
        out[tid] = scratch[threadIdx.x];
    }
}

struct Measurement {
    int    blocks_per_sm;     // from cudaOccupancyMaxActiveBlocksPerMultiprocessor
    double occupancy;         // resident warps / max warps, as a fraction
    float  ms;                // per-launch, event-timed
    double ghops;             // giga dependent hops per second - the metric occupancy moves
    bool   ok;                // verified against the CPU reference
};

static Measurement run_config(bool use_smem, int block, size_t dyn_smem,
                              const int* d_next, long long* d_out,
                              const std::vector<long long>& ref,
                              int n, int hops, int timed_reps,
                              const cudaDeviceProp& prop) {
    const int grid = (n + block - 1) / block;

    int blocks_per_sm = 0;
    if (use_smem) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, chase_smem, block, dyn_smem));
    } else {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, chase, block, dyn_smem));
    }

    auto launch = [&]() {
        if (use_smem) chase_smem<<<grid, block, dyn_smem>>>(d_next, d_out, n, hops);
        else          chase     <<<grid, block>>>          (d_next, d_out, n, hops);
    };

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();                                  // warmup also pulls the walk into cache
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < timed_reps; ++r) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::vector<long long> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), d_out, sizeof(long long) * n, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) ok = (host[i] == ref[i]);

    const float ms = total_ms / timed_reps;
    Measurement m{};
    m.blocks_per_sm = blocks_per_sm;
    m.occupancy     = static_cast<double>(blocks_per_sm) * block / prop.maxThreadsPerMultiProcessor;
    m.ms            = ms;
    m.ghops         = static_cast<double>(n) * hops / (ms / 1.0e3) / 1.0e9;
    m.ok            = ok;
    return m;
}

int main() {
    const int n          = 1 << 20;            // 4 MB walk table -> cache-resident
    const int hops       = 64;                 // dependent loads per thread
    const int timed_reps = 20;

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    const int max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    // Host-side walk table and reference: p advances by STRIDE with wraparound, so the
    // reference is exact integer arithmetic - no floating-point tolerance to argue about.
    std::vector<int> h_next(n);
    for (int i = 0; i < n; ++i) {
        int p = i + STRIDE;
        h_next[i] = (p >= n) ? p - n : p;
    }
    std::vector<long long> h_ref(n);
    for (int tid = 0; tid < n; ++tid) {
        int       p   = tid;
        long long acc = 0;
        for (int h = 0; h < hops; ++h) {
            p = h_next[p];
            acc += p;
        }
        h_ref[tid] = acc;
    }

    int*       d_next = nullptr;
    long long* d_out  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_next, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(long long) * n));
    CUDA_CHECK(cudaMemcpy(d_next, h_next.data(), sizeof(int) * n, cudaMemcpyHostToDevice));

    printf("device              : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs                 : %d\n", prop.multiProcessorCount);
    printf("warpSize            : %d lanes per warp\n", prop.warpSize);
    printf("max threads / SM    : %d  (= %d warps)\n",
           prop.maxThreadsPerMultiProcessor, max_warps);
    printf("shared mem / SM     : %zu B\n", prop.sharedMemPerMultiprocessor);
    printf("regs / block        : %d\n", prop.regsPerBlock);
    printf("workload            : %d threads, %d dependent hops, %d timed reps\n\n",
           n, hops, timed_reps);

    printf("sweep 1 - block size, no dynamic shared memory\n");
    printf("%7s %10s %9s %9s %9s %10s %s\n",
           "block", "blocks/SM", "warps/SM", "theo.occ", "ms", "Ghops/s", "result");
    printf("------- ---------- --------- --------- --------- ---------- ------\n");
    const int blocks[] = {64, 128, 256, 512, 1024};
    for (int b : blocks) {
        Measurement m = run_config(false, b, 0, d_next, d_out, h_ref, n, hops, timed_reps, prop);
        printf("%7d %10d %9d %8.1f%% %9.3f %10.2f %s\n",
               b, m.blocks_per_sm, m.blocks_per_sm * b / prop.warpSize,
               100.0 * m.occupancy, m.ms, m.ghops, m.ok ? "PASSED" : "FAILED");
    }

    printf("\nsweep 2 - dynamic shared memory per block (block = 256, shared mem is the limiter)\n");
    printf("%7s %10s %9s %9s %9s %10s %s\n",
           "smem KB", "blocks/SM", "warps/SM", "theo.occ", "ms", "Ghops/s", "result");
    printf("------- ---------- --------- --------- --------- ---------- ------\n");
    const int smem_block = 256;
    const int smem_kb[]  = {2, 8, 16, 32};     // 2 KB is the minimum this block size needs
    for (int kb : smem_kb) {
        const size_t bytes = static_cast<size_t>(kb) * 1024;
        if (bytes < sizeof(long long) * static_cast<size_t>(smem_block)) continue;
        Measurement m = run_config(true, smem_block, bytes, d_next, d_out, h_ref,
                                   n, hops, timed_reps, prop);
        printf("%7d %10d %9d %8.1f%% %9.3f %10.2f %s\n",
               kb, m.blocks_per_sm, m.blocks_per_sm * smem_block / prop.warpSize,
               100.0 * m.occupancy, m.ms, m.ghops, m.ok ? "PASSED" : "FAILED");
    }

    printf("\nRead sweep 1 for 'theoretical occupancy barely moves, time still does' and\n"
           "sweep 2 for the real occupancy cliff: every doubling of the shared-memory\n"
           "request halves the resident blocks. Occupancy is a means (latency hiding), not\n"
           "the goal - if Ghops/s stops improving while occupancy still rises, you have\n"
           "found the point of diminishing returns on this GPU.\n");

    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
