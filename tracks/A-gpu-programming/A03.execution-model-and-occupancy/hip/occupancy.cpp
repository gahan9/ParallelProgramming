// Lab 2 (HIP) - Occupancy: theory (the runtime's occupancy API) vs measurement (the clock).
//
// The kernel is a DEPENDENT pointer chase over a cache-resident buffer: each hop's
// address is unknown until the previous load returns, so a single wavefront spends
// almost all its time stalled. The only way to go fast is to have MANY wavefronts
// resident so the scheduler always has a ready one - i.e. latency hiding, which is
// exactly what occupancy buys. (`next[p] = (p + STRIDE) % n` keeps lanes contiguous, so
// this measures dependent-load latency, not coalescing - that was A02.)
//
// Two controlled sweeps:
//   1) block size {64,128,256,512,1024}, zero dynamic LDS. Theoretical occupancy is
//      usually FLAT here for a simple kernel; the measured time still moves (scheduling
//      granularity, tail effect). Seeing "occupancy flat, time not" is the point.
//   2) fixed 256-thread block, dynamic LDS per block {2,8,16,32} KB. LDS becomes the
//      binding limiter, occupancy falls, and the time rises with it.
//
// The two tables use different kernels, so compare rows WITHIN a table, never across.
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip occupancy.cpp -o occupancy.exe
// Run:   ./occupancy.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

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

// Same chase, but the accumulator lives in dynamic LDS. Bytes beyond the first
// blockDim.x slots are reserved and untouched on purpose: sweep 2 varies the LDS
// FOOTPRINT to move the occupancy limiter while holding the work fixed.
__global__ void chase_lds(const int* __restrict__ next, long long* __restrict__ out,
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
    int    blocks_per_cu;     // from hipOccupancyMaxActiveBlocksPerMultiprocessor
    double occupancy;         // resident wavefronts / max wavefronts, as a fraction
    float  ms;                // per-launch, event-timed
    double ghops;             // giga dependent hops per second - the metric occupancy moves
    bool   ok;                // verified against the CPU reference
};

static Measurement run_config(bool use_lds, int block, size_t dyn_lds,
                              const int* d_next, long long* d_out,
                              const std::vector<long long>& ref,
                              int n, int hops, int timed_reps,
                              const hipDeviceProp_t& prop) {
    const int grid = (n + block - 1) / block;

    int blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_cu,
        use_lds ? reinterpret_cast<const void*>(chase_lds)
                : reinterpret_cast<const void*>(chase),
        block, dyn_lds));

    auto launch = [&]() {
        if (use_lds) chase_lds<<<grid, block, dyn_lds>>>(d_next, d_out, n, hops);
        else         chase    <<<grid, block>>>          (d_next, d_out, n, hops);
    };

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    launch();                                  // warmup also pulls the walk into cache
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int r = 0; r < timed_reps; ++r) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    std::vector<long long> host(n);
    HIP_CHECK(hipMemcpy(host.data(), d_out, sizeof(long long) * n, hipMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n && ok; ++i) ok = (host[i] == ref[i]);

    const float  ms = total_ms / timed_reps;
    Measurement  m{};
    m.blocks_per_cu = blocks_per_cu;
    m.occupancy     = static_cast<double>(blocks_per_cu) * block / prop.maxThreadsPerMultiProcessor;
    m.ms            = ms;
    m.ghops         = static_cast<double>(n) * hops / (ms / 1.0e3) / 1.0e9;
    m.ok            = ok;
    return m;
}

int main() {
    const int n          = 1 << 20;            // 4 MB walk table -> cache-resident
    const int hops        = 64;                // dependent loads per thread
    const int timed_reps  = 20;

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    const int max_waves = prop.maxThreadsPerMultiProcessor / prop.warpSize;

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
    HIP_CHECK(hipMalloc(&d_next, sizeof(int) * n));
    HIP_CHECK(hipMalloc(&d_out, sizeof(long long) * n));
    HIP_CHECK(hipMemcpy(d_next, h_next.data(), sizeof(int) * n, hipMemcpyHostToDevice));

    printf("device            : %s (%s)\n", prop.name, prop.gcnArchName);
    printf("CUs               : %d\n", prop.multiProcessorCount);
    printf("warpSize          : %d lanes per wavefront\n", prop.warpSize);
    printf("max threads / CU  : %d  (= %d wavefronts)\n",
           prop.maxThreadsPerMultiProcessor, max_waves);
    printf("LDS / CU          : %zu B\n", static_cast<size_t>(prop.maxSharedMemoryPerMultiProcessor));
    printf("regs / block      : %d\n", prop.regsPerBlock);
    printf("workload          : %d threads, %d dependent hops, %d timed reps\n\n",
           n, hops, timed_reps);

    printf("sweep 1 - block size, no dynamic LDS\n");
    printf("%7s %10s %9s %9s %9s %10s %s\n",
           "block", "blocks/CU", "waves/CU", "theo.occ", "ms", "Ghops/s", "result");
    printf("------- ---------- --------- --------- --------- ---------- ------\n");
    const int blocks[] = {64, 128, 256, 512, 1024};
    for (int b : blocks) {
        Measurement m = run_config(false, b, 0, d_next, d_out, h_ref, n, hops, timed_reps, prop);
        printf("%7d %10d %9d %8.1f%% %9.3f %10.2f %s\n",
               b, m.blocks_per_cu, m.blocks_per_cu * b / prop.warpSize,
               100.0 * m.occupancy, m.ms, m.ghops, m.ok ? "PASSED" : "FAILED");
    }

    printf("\nsweep 2 - dynamic LDS per block (block = 256, LDS is the limiter)\n");
    printf("%7s %10s %9s %9s %9s %10s %s\n",
           "LDS KB", "blocks/CU", "waves/CU", "theo.occ", "ms", "Ghops/s", "result");
    printf("------- ---------- --------- --------- --------- ---------- ------\n");
    const int lds_block = 256;
    const int lds_kb[]  = {2, 8, 16, 32};      // 2 KB is the minimum this block size needs
    for (int kb : lds_kb) {
        const size_t bytes = static_cast<size_t>(kb) * 1024;
        if (bytes < sizeof(long long) * static_cast<size_t>(lds_block)) continue;
        Measurement m = run_config(true, lds_block, bytes, d_next, d_out, h_ref,
                                   n, hops, timed_reps, prop);
        printf("%7d %10d %9d %8.1f%% %9.3f %10.2f %s\n",
               kb, m.blocks_per_cu, m.blocks_per_cu * lds_block / prop.warpSize,
               100.0 * m.occupancy, m.ms, m.ghops, m.ok ? "PASSED" : "FAILED");
    }

    printf("\nRead sweep 1 for 'theoretical occupancy barely moves, time still does' and\n"
           "sweep 2 for the real occupancy cliff: every doubling of the LDS request halves\n"
           "the resident blocks. Occupancy is a means (latency hiding), not the goal -\n"
           "if Ghops/s stops improving while occupancy still rises, you have found the\n"
           "point of diminishing returns on this GPU.\n");

    HIP_CHECK(hipFree(d_next));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
