// Exercise 3 (HIP) - Measure achieved bandwidth as a percentage of YOUR device's
// theoretical peak.
//
// The point of this drill is the gap. Peak bandwidth is an arithmetic product of a
// clock and a bus width; achieved bandwidth is what silicon, refresh, ECC, and your
// access pattern leave you. Knowing the ratio for your own machine is what stops
// you from chasing a number you can never reach in B02's roofline.
//
// Nothing here may hardcode a device property. Every figure comes from
// hipGetDeviceProperties() plus a timed copy.
//
// Build: make 03_measure_vs_spec.exe GPU_ARCH=gfx942
// Run:   ./03_measure_vs_spec.exe
//
// As shipped this FAILS the correctness check and reports an unknown peak. Three
// TODOs fix it. Reference answer: ../solutions/03_measure_vs_spec_solution.cpp
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

namespace {

constexpr int kIters = 50;

// TODO 1 -----------------------------------------------------------------------
// Derive the theoretical peak HBM bandwidth in GB/s from the two properties
// passed in. The identity you need:
//
//     bytes/s = transfers/s x bus_width_bytes
//             = (memory_clock_hz x DDR_FACTOR) x (bus_width_bits / 8)
//
// memory_clock_khz is in kHz. Return 0.0 if either input is non-positive - some
// virtualized drivers report zero, and main() prints an honest "unavailable"
// rather than a fake number when you do.
double theoretical_peak_gbps(int memory_clock_khz, int bus_width_bits) {
    (void)memory_clock_khz;
    (void)bus_width_bits;
    return 0.0;   // <-- replace
}

// TODO 2 -----------------------------------------------------------------------
// This is the most bandwidth-friendly kernel there is: a fully coalesced copy.
// As written it only copies the first (gridDim.x * blockDim.x) elements, so most
// of the buffer is never touched and the verification below fails.
//
// Turn it into a grid-stride loop so that (a) every element is copied for any n,
// and (b) consecutive threads still touch consecutive addresses.
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, long long n) {
    long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i];
    }
}

}  // namespace

int main() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::printf("No HIP-capable devices visible - nothing to measure.\n");
        std::printf("Check with: rocm-smi   (and unset HIP_VISIBLE_DEVICES if it is set)\n");
        return EXIT_SUCCESS;
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, 0));

    const long long n     = 1LL << 26;                                  // ~67M floats
    const size_t    bytes = static_cast<size_t>(n) * sizeof(float);

    std::printf("device        : %s (%s)\n", p.name, p.gcnArchName);
    std::printf("CUs           : %d, wavefront %d\n", p.multiProcessorCount, p.warpSize);
    std::printf("buffers       : 2 x %.1f MiB\n", bytes / (1024.0 * 1024.0));

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_out, bytes));

    // A recognizable pattern, so the verification below catches "never written"
    // as well as "written wrong".
    std::vector<float> h_in(static_cast<size_t>(n));
    for (long long i = 0; i < n; ++i) {
        h_in[static_cast<size_t>(i)] = static_cast<float>(i % 1024) + 0.5f;
    }
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    const int threads = 256;
    const int blocks  = 1024;   // enough resident work to saturate any current GPU

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    copy_kernel<<<blocks, threads>>>(d_in, d_out, n);   // warmup: skip first-call costs
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int it = 0; it < kIters; ++it) {
        copy_kernel<<<blocks, threads>>>(d_in, d_out, n);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    const double seconds = (ms / 1.0e3) / kIters;

    // TODO 3 -------------------------------------------------------------------
    // A copy moves each element twice: one read and one write. Compute the
    // achieved bandwidth in GB/s from `bytes` and `seconds`, then express it as a
    // percentage of the theoretical peak from TODO 1 (guard against a zero peak).
    const double achieved_gbps = 0.0;   // <-- replace
    const double peak_gbps     = theoretical_peak_gbps(p.memoryClockRate, p.memoryBusWidth);
    const double efficiency    = 0.0;   // <-- replace (percent of peak)

    std::vector<float> h_out(static_cast<size_t>(n));
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    long long errors = 0;
    for (long long i = 0; i < n; ++i) {
        if (h_out[static_cast<size_t>(i)] != h_in[static_cast<size_t>(i)]) {
            ++errors;
        }
    }

    std::printf("time / iter   : %.3f ms\n", ms / kIters);
    std::printf("achieved BW   : %.1f GB/s\n", achieved_gbps);
    if (peak_gbps > 0.0) {
        std::printf("theoretical   : %.1f GB/s  (2 x %.0f MHz x %d bits / 8)\n",
                    peak_gbps, p.memoryClockRate / 1.0e3, p.memoryBusWidth);
        std::printf("efficiency    : %.1f%% of peak\n", efficiency);
    } else {
        std::printf("theoretical   : unavailable (driver reported clock=%d kHz, bus=%d bits)\n",
                    p.memoryClockRate, p.memoryBusWidth);
    }
    std::printf("result        : %s (%lld mismatches)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
