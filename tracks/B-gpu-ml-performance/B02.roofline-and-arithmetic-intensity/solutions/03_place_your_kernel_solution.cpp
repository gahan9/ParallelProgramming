// Solution 3 - Put your own kernel on the roofline.
//
// Reference answer for exercises/03_place_your_kernel.cpp. The kernels and harness are
// unchanged; the six accounting functions are filled in, each with the reasoning that
// produced it.
//
// Headline numbers a correct solution produces (rates are hardware-specific, the
// intensities are not):
//   sum_reduce  I ~= 0.25 FLOP/byte  -> memory-bound on every current GPU
//   fma_heavy   I  = 16.0 FLOP/byte  -> straddles the fp32 ridge point (~15-30), so
//                                       its verdict genuinely depends on the machine
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 03_place_your_kernel_solution.cpp -o 03_place_your_kernel_solution.exe
// Run:   ./03_place_your_kernel_solution.exe 78500
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

constexpr int kThreadsPerBlock = 256;
constexpr int kFmaChains       = 4;
constexpr int kFmaCount        = 64;
constexpr int kWarmupIters     = 3;
constexpr int kTimedIters      = 20;

constexpr double kFallbackPeakGflops = 60000.0;

// ===========================================================================
// Kernels - identical to the exercise.
// ===========================================================================

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out,
                            long long n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = in[i];
    }
}

__global__ void sum_reduce_kernel(const float* __restrict__ in,
                                  float* __restrict__ partials, long long n) {
    __shared__ float scratch[kThreadsPerBlock];
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

    float local = 0.0f;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        local += in[i];
    }
    scratch[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            scratch[threadIdx.x] += scratch[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = scratch[0];
    }
}

__global__ void fma_heavy_kernel(const float* __restrict__ in, float* __restrict__ out,
                                 long long n, float a, float b) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long tid    = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long group  = kFmaChains * stride;
    const long long full   = (n / group) * group;

    for (long long i = tid; i < full; i += group) {
        float v0 = in[i];
        float v1 = in[i + stride];
        float v2 = in[i + 2 * stride];
        float v3 = in[i + 3 * stride];
#pragma unroll
        for (int j = 0; j < kFmaCount; ++j) {
            v0 = fmaf(v0, a, b);
            v1 = fmaf(v1, a, b);
            v2 = fmaf(v2, a, b);
            v3 = fmaf(v3, a, b);
        }
        out[i]              = v0;
        out[i + stride]     = v1;
        out[i + 2 * stride] = v2;
        out[i + 3 * stride] = v3;
    }
    for (long long i = full + tid; i < n; i += stride) {
        float v = in[i];
#pragma unroll
        for (int j = 0; j < kFmaCount; ++j) {
            v = fmaf(v, a, b);
        }
        out[i] = v;
    }
}

// ===========================================================================
// SOLUTION 1 - sum_reduce accounting.
//
// FLOPs. Two phases add. The grid-stride phase starts every thread's accumulator at
// zero and adds each of the n elements exactly once: n adds. The shared-memory tree
// then performs blockDim.x - 1 adds per block. The tree term is real but tiny (a few
// hundred adds per block against millions of elements), and it is included rather
// than hand-waved because "roughly n" is how people end up with a 20% error they
// cannot explain later.
//
// Bytes. Each element is read from HBM exactly once (4 bytes), and each block writes
// one float back (4 bytes). Nothing else crosses HBM - the tree lives in LDS. The
// write term is ~0.004% of the traffic; again, count it and let the numbers say so.
//
// This function returns the grid-stride term only, because the tree term needs the
// launch shape; it is added in main() where `blocks` is in scope.
// ===========================================================================

double sum_reduce_flops(long long n) {
    return static_cast<double>(n);
}

double sum_reduce_bytes(long long n, int blocks) {
    const double read  = static_cast<double>(n) * sizeof(float);
    const double write = static_cast<double>(blocks) * sizeof(float);
    return read + write;
}

// ===========================================================================
// SOLUTION 2 - fma_heavy accounting.
//
// FLOPs. Every element gets kFmaCount fused multiply-adds, and an FMA is 2 FLOPs
// (one multiply + one add). The chain structure (4 chains) changes the schedule, not
// the op count. So 2 * kFmaCount * n.
//
// Bytes. One 4-byte read and one 4-byte write per element: 8n. The intermediate
// values never leave registers, which is exactly why intensity is tunable here.
// ===========================================================================

double fma_heavy_flops(long long n) {
    return 2.0 * kFmaCount * static_cast<double>(n);
}

double fma_heavy_bytes(long long n) {
    return 2.0 * static_cast<double>(n) * sizeof(float);
}

// ===========================================================================
// SOLUTION 3 - the model.
//
// The whole roofline is one min(). Units: [FLOP/byte] x [GB/s] = [GFLOP/s], because
// the 1e9 in "giga" cancels on both sides.
//
// The +/-2x dead band around the ridge is deliberate. Both ceilings are measured with
// a few percent of noise, real kernels rarely overlap load and math perfectly, and a
// kernel sitting at 1.1x the ridge is limited by both. Declaring a winner there would
// be false precision.
// ===========================================================================

double roofline_gflops(double ai, double peak_gflops, double peak_gbps) {
    const double memory_roof = ai * peak_gbps;
    return (memory_roof < peak_gflops) ? memory_roof : peak_gflops;
}

const char* verdict(double ai, double ridge) {
    if (ai < ridge / 2.0) return "memory-bound";
    if (ai > ridge * 2.0) return "compute-bound";
    return "near the ridge";
}

// ===========================================================================
// Harness - identical to the exercise.
// ===========================================================================

template <typename Launch>
double time_kernel_ms(Launch launch) {
    for (int i = 0; i < kWarmupIters; ++i) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < kTimedIters; ++i) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return static_cast<double>(total_ms) / kTimedIters;
}

int main(int argc, char** argv) {
    double peak_gflops = kFallbackPeakGflops;
    if (argc > 1) {
        peak_gflops = std::atof(argv[1]);
    }
    if (peak_gflops <= 0.0) {
        std::fprintf(stderr, "usage: %s [peak_gflops > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const int blocks = (prop.multiProcessorCount * 8 < 4096)
                           ? prop.multiProcessorCount * 8 : 4096;

    const long long n     = 1LL << 26;
    const size_t    bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_in(static_cast<size_t>(n));
    double host_sum = 0.0;
    for (long long i = 0; i < n; ++i) {
        h_in[static_cast<size_t>(i)] = static_cast<float>(i % 8) + 1.0f;
        host_sum += h_in[static_cast<size_t>(i)];
    }

    float *d_in = nullptr, *d_out = nullptr, *d_partials = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMalloc(&d_partials, static_cast<size_t>(blocks) * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    printf("device      : %s (%s), %d CUs\n", prop.name, prop.gcnArchName,
           prop.multiProcessorCount);
    printf("n           : %lld floats, %d blocks x %d threads\n",
           n, blocks, kThreadsPerBlock);
    if (argc <= 1) {
        printf("\nNOTE: no compute ceiling given, using the placeholder %.0f GFLOP/s.\n",
               kFallbackPeakGflops);
        printf("      Run ../hip/measure_ceilings.exe and pass its P as argv[1].\n");
    }

    const double copy_ms = time_kernel_ms([&] {
        copy_kernel<<<blocks, kThreadsPerBlock>>>(d_in, d_out, n);
    });
    const double peak_gbps = (2.0 * static_cast<double>(bytes)) / (copy_ms / 1.0e3) / 1.0e9;
    const double ridge     = peak_gflops / peak_gbps;

    printf("\nceilings\n");
    printf("  memory roof  B : %.1f GB/s     (measured here, coalesced copy)\n", peak_gbps);
    printf("  compute roof P : %.1f GFLOP/s  (%s)\n", peak_gflops,
           argc > 1 ? "from measure_ceilings" : "PLACEHOLDER");
    printf("  ridge point    : %.2f FLOP/byte\n", ridge);

    const double reduce_ms = time_kernel_ms([&] {
        sum_reduce_kernel<<<blocks, kThreadsPerBlock>>>(d_in, d_partials, n);
    });
    std::vector<float> h_partials(static_cast<size_t>(blocks));
    HIP_CHECK(hipMemcpy(h_partials.data(), d_partials,
                        static_cast<size_t>(blocks) * sizeof(float), hipMemcpyDeviceToHost));
    double gpu_sum = 0.0;
    for (int b = 0; b < blocks; ++b) gpu_sum += h_partials[b];
    const bool reduce_ok = std::fabs(gpu_sum - host_sum) / host_sum < 1.0e-4;

    const double fma_ms = time_kernel_ms([&] {
        fma_heavy_kernel<<<blocks, kThreadsPerBlock>>>(d_in, d_out, n, 1.0f, 0.0f);
    });
    std::vector<float> h_out(static_cast<size_t>(n));
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    bool fma_ok = true;
    for (long long i = 0; i < n; ++i) {
        if (h_out[static_cast<size_t>(i)] != h_in[static_cast<size_t>(i)]) {
            fma_ok = false;
            break;
        }
    }

    struct Row {
        const char* name;
        double      ms;
        double      flops;
        double      bytes;
        bool        verified;
    };
    // The tree-reduction adds depend on the launch shape, so they are added here where
    // `blocks` is in scope rather than smuggled into sum_reduce_flops(n).
    const double reduce_flops = sum_reduce_flops(n)
                                + static_cast<double>(blocks) * (kThreadsPerBlock - 1);
    const Row rows[] = {
        {"sum_reduce", reduce_ms, reduce_flops, sum_reduce_bytes(n, blocks), reduce_ok},
        {"fma_heavy",  fma_ms,    fma_heavy_flops(n), fma_heavy_bytes(n),    fma_ok},
    };

    printf("\n%-12s %9s %11s %11s %9s %11s %11s %8s %-15s %s\n",
           "kernel", "ms", "FLOPs", "bytes", "I (F/B)", "GFLOP/s", "roof GF/s",
           "% roof", "verdict", "verify");
    printf("------------ --------- ----------- ----------- --------- ----------- "
           "----------- -------- --------------- ------\n");

    for (const Row& r : rows) {
        const double seconds  = r.ms / 1.0e3;
        const double ai       = r.flops / r.bytes;
        const double achieved = r.flops / seconds / 1.0e9;
        const double roof     = roofline_gflops(ai, peak_gflops, peak_gbps);
        printf("%-12s %9.3f %11.3e %11.3e %9.3f %11.1f %11.1f %7.1f%% %-15s %6s\n",
               r.name, r.ms, r.flops, r.bytes, ai, achieved, roof,
               100.0 * achieved / roof, verdict(ai, ridge), r.verified ? "PASS" : "FAIL");
    }

    // For a memory-bound kernel the honest headline is bandwidth, not FLOP/s: a
    // reduction reaching 95% of the copy ceiling is finished, whatever its GFLOP/s says.
    const double reduce_gbps = sum_reduce_bytes(n, blocks) / (reduce_ms / 1.0e3) / 1.0e9;
    printf("\nsum_reduce achieved %.1f GB/s = %.1f%% of the copy ceiling.\n",
           reduce_gbps, 100.0 * reduce_gbps / peak_gbps);
    printf("That percentage, not its GFLOP/s, is the number to optimize for a\n");
    printf("memory-bound kernel. Read Section 5 of the README for why.\n");

    const bool all_verified = reduce_ok && fma_ok;
    printf("\nkernels verified : %s\n", all_verified ? "PASSED" : "FAILED");
    printf("instrumentation  : PASSED\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_partials));
    return all_verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
