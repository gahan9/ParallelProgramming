// Lab 1 (CUDA) - Discover the roofline ridge point by measurement, not by assertion.
//
// Line-for-line counterpart of hip/roofline_sweep.cpp. One kernel, one tunable knob:
// every thread loads a float, performs K fused multiply-adds on it in registers, and
// stores it. Bytes moved per element is fixed at 8 (one 4-byte read + one 4-byte
// write); FLOPs per element is 2K. So arithmetic intensity is exactly
//
//     I(K) = 2K FLOP / 8 byte = K / 4  FLOP/byte
//
// and sweeping K walks a single kernel across the entire roofline. At small K the
// achieved GB/s pins near the copy ceiling while GFLOP/s is tiny; at large K the
// GB/s collapses while GFLOP/s pins near the fp32 ceiling. The knee between those
// two regimes IS the ridge point, and this program locates it from the numbers it
// just measured on this machine.
//
// Correctness: a and b are passed at runtime as 1.0f and 0.0f, so the K-fold chain
// v = fma(v, 1, 0) is the identity for every K. That makes verification exact and
// K-independent (out must equal in bit-for-bit) while the hardware still issues K
// real FFMA instructions per element - the compiler cannot fold them away because it
// cannot see the operand values, and IEEE semantics forbid reassociation.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda roofline_sweep.cu -o roofline_sweep.exe
// Run:   ./roofline_sweep.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

constexpr int kThreadsPerBlock = 256;
constexpr int kChains          = 4;   // independent FMA chains per thread (see below)
constexpr int kWarmupIters     = 3;
constexpr int kTimedIters      = 20;

// K is a template parameter, not a kernel argument, so the FMA sequence fully
// unrolls into straight-line code. A runtime K would add loop overhead that grows
// as the arithmetic shrinks - exactly the regime the sweep is trying to measure.
//
// kChains matters as much as K: a single dependent FMA chain per thread would make
// the high-K end latency-bound on the FMA pipeline depth rather than throughput-bound,
// putting a false ceiling under the compute roof. Four independent chains give the
// scheduler enough ILP to saturate the FMA units.
template <int K>
__global__ void fma_sweep(const float* __restrict__ in, float* __restrict__ out,
                          long long n, float a, float b) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    const long long tid    = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long group  = kChains * stride;
    const long long full   = (n / group) * group;   // largest whole-group prefix

    // Main body: inside [0, full) all four accesses are provably in range because
    // tid < stride, so no per-access bounds check is needed in the hot loop.
    for (long long i = tid; i < full; i += group) {
        float v0 = in[i];
        float v1 = in[i + stride];
        float v2 = in[i + 2 * stride];
        float v3 = in[i + 3 * stride];
#pragma unroll
        for (int j = 0; j < K; ++j) {
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

    // Ragged tail: fewer than one group of elements remain, so at most kChains
    // iterations run here. Kept separate so the main loop stays branch-free.
    for (long long i = full + tid; i < n; i += stride) {
        float v = in[i];
#pragma unroll
        for (int j = 0; j < K; ++j) {
            v = fmaf(v, a, b);
        }
        out[i] = v;
    }
}

struct SweepResult {
    int    fmas;        // K
    double intensity;   // FLOP/byte, analytic
    double ms;          // per-iteration kernel time
    double gbps;        // achieved HBM traffic
    double gflops;      // achieved arithmetic throughput
    bool   passed;
};

// One sweep point: warm up, time kTimedIters launches with GPU events, then verify.
template <int K>
SweepResult run_case(const float* d_in, float* d_out, long long n, int blocks,
                     const std::vector<float>& h_in, std::vector<float>& h_out) {
    const float a = 1.0f;   // identity FMA: keeps verification exact for every K
    const float b = 0.0f;

    for (int i = 0; i < kWarmupIters; ++i) {
        fma_sweep<K><<<blocks, kThreadsPerBlock>>>(d_in, d_out, n, a, b);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kTimedIters; ++i) {
        fma_sweep<K><<<blocks, kThreadsPerBlock>>>(d_in, d_out, n, a, b);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double seconds = (total_ms / 1.0e3) / kTimedIters;
    const double bytes   = 2.0 * static_cast<double>(n) * sizeof(float);
    const double flops   = 2.0 * K * static_cast<double>(n);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out,
                          static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost));
    bool passed = true;
    for (long long i = 0; i < n; ++i) {
        if (h_out[static_cast<size_t>(i)] != h_in[static_cast<size_t>(i)]) {
            passed = false;
            break;
        }
    }

    SweepResult r;
    r.fmas      = K;
    r.intensity = flops / bytes;
    r.ms        = total_ms / kTimedIters;
    r.gbps      = bytes / seconds / 1.0e9;
    r.gflops    = flops / seconds / 1.0e9;
    r.passed    = passed;
    return r;
}

using CaseFn = SweepResult (*)(const float*, float*, long long, int,
                               const std::vector<float>&, std::vector<float>&);

// K values chosen so I = K/4 spans ~0 to 128 FLOP/byte, which brackets the fp32
// ridge point of every current data-center GPU (roughly 15-30 FLOP/byte).
const CaseFn kCases[] = {
    run_case<0>,   run_case<1>,   run_case<2>,   run_case<4>,
    run_case<8>,   run_case<16>,  run_case<32>,  run_case<64>,
    run_case<128>, run_case<256>, run_case<512>,
};
constexpr int kNumCases = static_cast<int>(sizeof(kCases) / sizeof(kCases[0]));

void render_bar(char* buf, int width, double fraction) {
    const int filled = static_cast<int>(fraction * width + 0.5);
    for (int i = 0; i < width; ++i) {
        buf[i] = (i < filled) ? '#' : '.';
    }
    buf[width] = '\0';
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Enough blocks to keep every SM busy with several waves; more than this only
    // lengthens the grid-stride loop and does not change the steady-state rates.
    const int blocks = prop.multiProcessorCount * 32;

    const long long n     = 1LL << 25;                                  // ~33.5M floats
    const size_t    bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_in(static_cast<size_t>(n));
    std::vector<float> h_out(static_cast<size_t>(n));
    for (long long i = 0; i < n; ++i) {
        h_in[static_cast<size_t>(i)] = static_cast<float>(i % 97) + 1.0f;
    }

    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    printf("device      : %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("n           : %lld floats (%.1f MB per buffer)\n", n, bytes / 1.0e6);
    printf("launch      : %d blocks x %d threads, %d chains/thread\n",
           blocks, kThreadsPerBlock, kChains);
    printf("per element : 8 bytes moved (1 read + 1 write), 2K FLOPs -> I = K/4\n\n");

    SweepResult results[kNumCases];
    printf("  K  I (FLOP/byte)   ms/iter      GB/s     GFLOP/s  verify\n");
    printf("---- -------------- --------- --------- ----------- ------\n");
    for (int c = 0; c < kNumCases; ++c) {
        results[c] = kCases[c](d_in, d_out, n, blocks, h_in, h_out);
        printf("%4d %14.3f %9.3f %9.1f %11.1f %6s\n",
               results[c].fmas, results[c].intensity, results[c].ms,
               results[c].gbps, results[c].gflops,
               results[c].passed ? "PASS" : "FAIL");
    }

    // The two ceilings, taken from this sweep alone: the highest traffic rate any
    // point reached is the memory roof; the highest arithmetic rate is the compute
    // roof. Neither is asserted - both are the best number the hardware produced.
    double bw_roof   = 0.0;
    double flop_roof = 0.0;
    bool   all_pass  = true;
    for (int c = 0; c < kNumCases; ++c) {
        if (results[c].gbps > bw_roof)     bw_roof   = results[c].gbps;
        if (results[c].gflops > flop_roof) flop_roof = results[c].gflops;
        all_pass = all_pass && results[c].passed;
    }
    const double ridge = flop_roof / bw_roof;

    printf("\nmeasured ceilings (this sweep, this machine)\n");
    printf("  memory roof  B : %.1f GB/s      (best traffic rate, reached at low K)\n", bw_roof);
    printf("  compute roof P : %.1f GFLOP/s   (best arithmetic rate, reached at high K)\n", flop_roof);
    printf("  ridge point    : I_ridge = P / B = %.2f FLOP/byte  -> K ~= %.0f\n",
           ridge, ridge * 4.0);

    printf("\nplacement vs the roofline min(P, I x B)\n");
    printf("  K  I (FLOP/byte)   roof GF/s   achieved   %% of roof  regime          GFLOP/s bar\n");
    printf("---- -------------- ----------- ----------- --------- --------------- ----------------------\n");
    char bar[33];
    for (int c = 0; c < kNumCases; ++c) {
        const SweepResult& r = results[c];
        if (r.fmas == 0) {
            render_bar(bar, 20, 0.0);
            printf("%4d %14.3f %11s %11s %9s %-15s %s\n", r.fmas, r.intensity,
                   "-", "-", "-", "bandwidth cal.", bar);
            continue;
        }
        const double roof = (r.intensity * bw_roof < flop_roof)
                                ? r.intensity * bw_roof
                                : flop_roof;
        const char* regime = (r.intensity < ridge / 2.0)   ? "memory-bound"
                             : (r.intensity > ridge * 2.0) ? "compute-bound"
                                                           : "near the ridge";
        render_bar(bar, 20, r.gflops / flop_roof);
        printf("%4d %14.3f %11.1f %11.1f %8.1f%% %-15s %s\n", r.fmas, r.intensity, roof,
               r.gflops, 100.0 * r.gflops / roof, regime, bar);
    }

    printf("\nExpected shape: GFLOP/s rises roughly linearly with I while the kernel is\n");
    printf("on the memory roof, then flattens once it hits the compute roof. GB/s does\n");
    printf("the mirror image. The K where the two curves cross is the ridge point above.\n");
    printf("Absolute values are hardware-specific; the shape is not.\n");
    printf("\nFeed these ceilings to the plotter:\n");
    printf("  python ../tools/roofline_plot.py --peak-gflops %.0f --peak-gbps %.0f --demo-kernels\n",
           flop_roof, bw_roof);
    printf("\nverification: %s\n", all_pass ? "PASSED (all K reproduce the input exactly)"
                                            : "FAILED");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
