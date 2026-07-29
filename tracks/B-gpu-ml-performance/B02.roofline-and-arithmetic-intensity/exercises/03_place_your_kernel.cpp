// Exercise 3 - Put your own kernel on the roofline.
//
// The kernels here are already written, verified, and timed. What is missing is the
// part that matters in a real performance investigation: the INSTRUMENTATION. A
// kernel that cannot report its own FLOPs and bytes cannot be placed on a roofline,
// and a kernel you cannot place is a kernel you are guessing about.
//
// Two kernels are provided, chosen so that a correct solution produces two DIFFERENT
// verdicts:
//   sum_reduce  - reads n floats, does ~n adds, writes one partial per block.
//   fma_heavy   - reads n floats, does kFmaCount FMAs on each, writes n floats.
//
// The memory ceiling B is measured for you by a coalesced copy. The compute ceiling P
// comes from hip/measure_ceilings.exe - pass it as argv[1].
//
// YOUR TASKS
//   TODO 1: sum_reduce_flops / sum_reduce_bytes
//   TODO 2: fma_heavy_flops  / fma_heavy_bytes
//   TODO 3: roofline_gflops  / verdict
//
// Until all six are filled in, the program reports FAILED and exits non-zero.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 03_place_your_kernel.cpp -o 03_place_your_kernel.exe
// Run:   ./03_place_your_kernel.exe 78500        # 78500 = your measured P in GFLOP/s
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

constexpr int kThreadsPerBlock = 256;
constexpr int kFmaChains       = 4;    // independent chains -> ILP, not latency-bound
constexpr int kFmaCount        = 64;   // FMAs applied to every loaded element
constexpr int kWarmupIters     = 3;
constexpr int kTimedIters      = 20;

// Fallback compute ceiling if argv[1] is missing: Device R's fp32 peak from the paper
// drills. Provisional by construction - measure your own.
constexpr double kFallbackPeakGflops = 60000.0;

// ===========================================================================
// Provided kernels - do not change these. The exercise is the accounting.
// ===========================================================================

__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out,
                            long long n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = in[i];
    }
}

// Grid-stride accumulate into registers, then a shared-memory tree reduction per
// block. One float per block leaves the kernel; the host finishes the sum.
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

// kFmaCount FMAs per loaded element across kFmaChains independent chains. Called with
// a = 1.0f, b = 0.0f so the chain is the identity and verification is exact.
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
// TODO 1 - Account for sum_reduce.
//
// Look at sum_reduce_kernel above and count what actually crosses HBM. Every
// element is read exactly once. Each block writes exactly one float. Count one
// floating-point add as 1 FLOP.
// ===========================================================================

double sum_reduce_flops(long long n) {
    (void)n;
    return 0.0;   // TODO 1a: how many adds does the whole reduction perform?
}

double sum_reduce_bytes(long long n, int blocks) {
    (void)n;
    (void)blocks;
    return 0.0;   // TODO 1b: bytes read from HBM + bytes written back to HBM.
}

// ===========================================================================
// TODO 2 - Account for fma_heavy.
//
// One read and one write per element, and kFmaCount fused multiply-adds applied to
// each element. An FMA is 2 FLOPs.
// ===========================================================================

double fma_heavy_flops(long long n) {
    (void)n;
    return 0.0;   // TODO 2a
}

double fma_heavy_bytes(long long n) {
    (void)n;
    return 0.0;   // TODO 2b
}

// ===========================================================================
// TODO 3 - The model itself.
//
// roofline_gflops: the attainable rate for a kernel of intensity `ai` on a machine
//   with compute ceiling `peak_gflops` (GFLOP/s) and memory ceiling `peak_gbps` (GB/s).
//   One line. Mind the units: GB/s x FLOP/byte = GFLOP/s.
//
// verdict: "memory-bound", "compute-bound", or "near the ridge" (use "near the ridge"
//   when `ai` is within a factor of 2 of the ridge point either way - the model is not
//   precise enough to call a winner in that band).
// ===========================================================================

double roofline_gflops(double ai, double peak_gflops, double peak_gbps) {
    (void)ai;
    (void)peak_gbps;
    (void)peak_gflops;
    return 0.0;   // TODO 3a
}

const char* verdict(double ai, double ridge) {
    (void)ai;
    (void)ridge;
    return "TODO";   // TODO 3b
}

// ===========================================================================
// Harness - provided.
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

    const long long n     = 1LL << 26;                      // ~67M floats, 256 MB
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

    // ---- memory ceiling B: coalesced copy ----
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

    // ---- kernel 1: sum_reduce ----
    const double reduce_ms = time_kernel_ms([&] {
        sum_reduce_kernel<<<blocks, kThreadsPerBlock>>>(d_in, d_partials, n);
    });
    std::vector<float> h_partials(static_cast<size_t>(blocks));
    HIP_CHECK(hipMemcpy(h_partials.data(), d_partials,
                        static_cast<size_t>(blocks) * sizeof(float), hipMemcpyDeviceToHost));
    double gpu_sum = 0.0;
    for (int b = 0; b < blocks; ++b) gpu_sum += h_partials[b];
    const bool reduce_ok = std::fabs(gpu_sum - host_sum) / host_sum < 1.0e-4;

    // ---- kernel 2: fma_heavy ----
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

    // ---- place both on the roofline using YOUR accounting ----
    struct Row {
        const char* name;
        double      ms;
        double      flops;
        double      bytes;
        bool        verified;
    };
    const Row rows[] = {
        {"sum_reduce", reduce_ms, sum_reduce_flops(n), sum_reduce_bytes(n, blocks), reduce_ok},
        {"fma_heavy",  fma_ms,    fma_heavy_flops(n),  fma_heavy_bytes(n),          fma_ok},
    };

    printf("\n%-12s %9s %11s %11s %9s %11s %11s %8s %-15s %s\n",
           "kernel", "ms", "FLOPs", "bytes", "I (F/B)", "GFLOP/s", "roof GF/s",
           "% roof", "verdict", "verify");
    printf("------------ --------- ----------- ----------- --------- ----------- "
           "----------- -------- --------------- ------\n");

    bool todos_done = true;
    for (const Row& r : rows) {
        if (r.flops <= 0.0 || r.bytes <= 0.0) todos_done = false;
        const double seconds  = r.ms / 1.0e3;
        const double ai       = (r.bytes > 0.0) ? r.flops / r.bytes : 0.0;
        const double achieved = r.flops / seconds / 1.0e9;
        const double roof     = roofline_gflops(ai, peak_gflops, peak_gbps);
        const double pct      = (roof > 0.0) ? 100.0 * achieved / roof : 0.0;
        printf("%-12s %9.3f %11.3e %11.3e %9.3f %11.1f %11.1f %7.1f%% %-15s %6s\n",
               r.name, r.ms, r.flops, r.bytes, ai, achieved, roof, pct,
               verdict(ai, ridge), r.verified ? "PASS" : "FAIL");
    }

    if (roofline_gflops(1.0, peak_gflops, peak_gbps) <= 0.0) todos_done = false;

    const bool all_verified = reduce_ok && fma_ok;
    printf("\nkernels verified : %s\n", all_verified ? "PASSED" : "FAILED");
    printf("instrumentation  : %s\n", todos_done ? "PASSED (TODOs filled in)"
                                                 : "FAILED (TODOs still return 0)");
    if (!todos_done) {
        printf("\nFill in TODO 1-3 above. When correct you should see roughly\n");
        printf("I = 0.25 FLOP/byte for sum_reduce (memory-bound) and I = 16 FLOP/byte\n");
        printf("for fma_heavy, whose verdict depends on where your ridge point landed.\n");
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_partials));
    return (all_verified && todos_done) ? EXIT_SUCCESS : EXIT_FAILURE;
}
