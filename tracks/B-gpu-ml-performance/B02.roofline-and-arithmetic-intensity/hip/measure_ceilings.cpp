// Lab 2 (HIP) - Measure the two axes of your roofline on the actual machine.
//
// A roofline drawn from a datasheet is a marketing plot. A roofline drawn from
// microbenchmarks is an engineering tool, because it is the ceiling your kernel can
// actually reach on this GPU, this driver, this clock state. This program measures:
//
//   B, the achievable HBM bandwidth  - a fully coalesced grid-stride copy, the most
//                                      bandwidth-friendly kernel that exists.
//   P, the achievable fp32 FLOP/s    - a register-resident FMA loop that touches
//                                      memory only once at the very end, so nothing
//                                      but the FMA units limits it.
//
// The ridge point follows: I_ridge = P / B FLOP/byte. Below it, kernels are
// memory-bound; above it, compute-bound.
//
// Both kernels verify. The FMA loop uses a = 1.0f, b = 0.0f passed at runtime, so
// each chain is the identity and the expected output is the sum of its seeds -
// checkable exactly on the host. The compiler cannot fold the chain away because it
// cannot see the operand values, and IEEE semantics forbid reassociating FMAs.
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip measure_ceilings.cpp -o measure_ceilings.exe
// Run:   ./measure_ceilings.exe [outer_iters]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

constexpr int kThreadsPerBlock = 256;
constexpr int kBlocksPerCU     = 8;
constexpr int kFmaChains       = 8;    // independent chains -> enough ILP to saturate
constexpr int kFmaUnroll       = 32;   // FMAs issued per chain per outer iteration
constexpr int kWarmupIters     = 3;
constexpr int kCopyIters       = 30;

// Coalesced grid-stride copy: adjacent lanes touch adjacent addresses, so every
// byte in every fetched cache line is used. This is the memory roof.
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out,
                            long long n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = in[i];
    }
}

// Register-resident FMA loop: one store per thread for the whole kernel, so HBM
// traffic is negligible and the FMA pipelines are the only limiter. This is the
// compute roof. kFmaChains breaks the dependency chain so the measurement reflects
// FMA *throughput* rather than FMA *latency*.
__global__ void fma_peak_kernel(float* __restrict__ out, int outer_iters, float a, float b) {
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    float v[kFmaChains];
#pragma unroll
    for (int c = 0; c < kFmaChains; ++c) {
        v[c] = static_cast<float>((tid + c) % 13) + 1.0f;
    }

    // outer_iters is a runtime argument on purpose: it keeps the compiler from
    // unrolling the whole thing into a constant-folded expression.
    for (int it = 0; it < outer_iters; ++it) {
#pragma unroll
        for (int u = 0; u < kFmaUnroll; ++u) {
#pragma unroll
            for (int c = 0; c < kFmaChains; ++c) {
                v[c] = fmaf(v[c], a, b);
            }
        }
    }

    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < kFmaChains; ++c) {
        acc += v[c];
    }
    out[tid] = acc;
}

double measure_copy_gbps(int blocks, long long n, const std::vector<float>& h_in,
                         std::vector<float>& h_out, bool* passed) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

    for (int i = 0; i < kWarmupIters; ++i) {
        copy_kernel<<<blocks, kThreadsPerBlock>>>(d_in, d_out, n);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < kCopyIters; ++i) {
        copy_kernel<<<blocks, kThreadsPerBlock>>>(d_in, d_out, n);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    *passed = true;
    for (long long i = 0; i < n; ++i) {
        if (h_out[static_cast<size_t>(i)] != h_in[static_cast<size_t>(i)]) {
            *passed = false;
            break;
        }
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    const double seconds = (total_ms / 1.0e3) / kCopyIters;
    return (2.0 * static_cast<double>(bytes)) / seconds / 1.0e9;   // 1 read + 1 write
}

double measure_fma_gflops(int blocks, int outer_iters, bool* passed) {
    const long long total_threads = static_cast<long long>(blocks) * kThreadsPerBlock;
    const size_t    bytes         = static_cast<size_t>(total_threads) * sizeof(float);
    const float     a             = 1.0f;   // identity FMA -> exact expected output
    const float     b             = 0.0f;

    float* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, bytes));

    for (int i = 0; i < kWarmupIters; ++i) {
        fma_peak_kernel<<<blocks, kThreadsPerBlock>>>(d_out, outer_iters, a, b);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    fma_peak_kernel<<<blocks, kThreadsPerBlock>>>(d_out, outer_iters, a, b);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    std::vector<float> h_out(static_cast<size_t>(total_threads));
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    *passed = true;
    for (long long t = 0; t < total_threads; ++t) {
        float expected = 0.0f;
        for (int c = 0; c < kFmaChains; ++c) {
            expected += static_cast<float>((t + c) % 13) + 1.0f;
        }
        if (h_out[static_cast<size_t>(t)] != expected) {
            *passed = false;
            break;
        }
    }
    HIP_CHECK(hipFree(d_out));

    // Count only the FMAs (2 FLOPs each). The kFmaChains-1 reduction adds at the end
    // are ~1e-4 of the total and are deliberately excluded rather than inflated in.
    const double fmas = static_cast<double>(total_threads) * outer_iters
                        * kFmaUnroll * kFmaChains;
    const double seconds = ms / 1.0e3;
    return (2.0 * fmas) / seconds / 1.0e9;
}

int main(int argc, char** argv) {
    const int outer_iters = (argc > 1) ? std::atoi(argv[1]) : 128;
    if (outer_iters <= 0) {
        std::fprintf(stderr, "usage: %s [outer_iters > 0]\n", argv[0]);
        return EXIT_FAILURE;
    }

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    const int blocks = prop.multiProcessorCount * kBlocksPerCU;

    const long long n = 1LL << 26;                                  // ~67M floats, 256 MB
    std::vector<float> h_in(static_cast<size_t>(n));
    std::vector<float> h_scratch(static_cast<size_t>(n));
    for (long long i = 0; i < n; ++i) {
        h_in[static_cast<size_t>(i)] = static_cast<float>(i % 101) + 1.0f;
    }

    printf("device            : %s (%s)\n", prop.name, prop.gcnArchName);
    printf("compute units     : %d\n", prop.multiProcessorCount);
    if (prop.memoryClockRate > 0 && prop.memoryBusWidth > 0) {
        // Datasheet-style theoretical peak, for context only. HBM stacks report this
        // inconsistently across ROCm versions, so it is printed, never relied upon.
        const double theoretical = 2.0 * prop.memoryClockRate * 1.0e3
                                   * (prop.memoryBusWidth / 8.0) / 1.0e9;
        printf("theoretical HBM BW: %.1f GB/s (from device properties, informational)\n",
               theoretical);
    }
    printf("launch            : %d blocks x %d threads\n\n", blocks, kThreadsPerBlock);

    bool copy_ok = false;
    bool fma_ok  = false;
    const double gbps   = measure_copy_gbps(blocks, n, h_in, h_scratch, &copy_ok);
    const double gflops = measure_fma_gflops(blocks, outer_iters, &fma_ok);
    const double ridge  = gflops / gbps;

    printf("ceiling                             value        verify   how\n");
    printf("----------------------------------- ------------ -------- ---------------------------\n");
    printf("memory roof   B  (GB/s)             %12.1f %8s coalesced grid-stride copy\n",
           gbps, copy_ok ? "PASSED" : "FAILED");
    printf("compute roof  P  (GFLOP/s, fp32)    %12.1f %8s register-resident FMA loop\n",
           gflops, fma_ok ? "PASSED" : "FAILED");
    printf("\nridge point   I_ridge = P / B      = %.2f FLOP/byte\n", ridge);
    printf("  below %.2f FLOP/byte -> memory-bound (bandwidth decides)\n", ridge);
    printf("  above %.2f FLOP/byte -> compute-bound (FLOP/s decides)\n", ridge);
    printf("\nWhat this means in practice: a kernel needs to do more than %.0f FLOPs for\n",
           ridge);
    printf("every byte it pulls from HBM before arithmetic becomes its limit. An\n");
    printf("elementwise add does about 0.08. That gap is why fusion pays.\n");
    printf("\nDraw the chart:\n");
    printf("  python ../tools/roofline_plot.py --peak-gflops %.0f --peak-gbps %.0f --demo-kernels\n",
           gflops, gbps);
    printf("\nCaveat: P here is the fp32 SIMT FMA roof. Matrix cores (AMD MFMA / NVIDIA\n");
    printf("Tensor Cores) have their own, far higher roof for fp16/bf16/fp8, and that\n");
    printf("is the ceiling ML kernels are actually measured against. See README S6.\n");

    return (copy_ok && fma_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
