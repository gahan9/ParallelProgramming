// Solution 3 (HIP) - Achieved copy bandwidth as a percentage of the device's own
// theoretical peak.
//
// Every figure is derived at runtime. The three changes against the starter:
//   TODO 1  theoretical_peak_gbps implements  2 x clock x bus_bytes
//   TODO 2  copy_kernel became a grid-stride loop, so any n is fully covered
//   TODO 3  achieved bandwidth counts 2 x bytes (one read + one write), and the
//           efficiency ratio guards against a zero peak
//
// Build: make 03_measure_vs_spec_solution.exe GPU_ARCH=gfx942
// Run:   ./03_measure_vs_spec_solution.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

namespace {

constexpr int    kIters     = 50;
constexpr double kDdrFactor = 2.0;   // HBM transfers on both clock edges

// bytes/s = transfers/s x bus_width_bytes = (clock_hz x 2) x (bus_bits / 8).
// Returns 0.0 when the driver leaves either property at zero (seen under some
// virtualization and container configurations) so the caller can say "unavailable"
// instead of printing a fabricated number.
double theoretical_peak_gbps(int memory_clock_khz, int bus_width_bits) {
    if (memory_clock_khz <= 0 || bus_width_bits <= 0) {
        return 0.0;
    }
    const double clock_hz  = static_cast<double>(memory_clock_khz) * 1.0e3;
    const double bus_bytes = static_cast<double>(bus_width_bits) / 8.0;
    return (kDdrFactor * clock_hz * bus_bytes) / 1.0e9;
}

// Grid-stride loop: every thread advances by the whole grid, so consecutive lanes
// always touch consecutive addresses (fully coalesced) and any n is covered no
// matter how the launch is sized.
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, long long n) {
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
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

    std::vector<float> h_in(static_cast<size_t>(n));
    for (long long i = 0; i < n; ++i) {
        h_in[static_cast<size_t>(i)] = static_cast<float>(i % 1024) + 0.5f;
    }
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, bytes));

    const int threads = 256;
    const int blocks  = 1024;

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

    // A copy touches every element twice: one read from d_in, one write to d_out.
    const double achieved_gbps = (2.0 * static_cast<double>(bytes)) / seconds / 1.0e9;
    const double peak_gbps     = theoretical_peak_gbps(p.memoryClockRate, p.memoryBusWidth);
    const double efficiency    = peak_gbps > 0.0 ? (100.0 * achieved_gbps / peak_gbps) : 0.0;

    std::vector<float> h_out(static_cast<size_t>(n));
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    long long errors = 0;
    for (long long i = 0; i < n; ++i) {
        if (h_out[static_cast<size_t>(i)] != h_in[static_cast<size_t>(i)]) {
            ++errors;
        }
    }

    std::printf("time / iter   : %.3f ms\n", ms / kIters);
    std::printf("achieved BW   : %.1f GB/s  (2 x %.1f MiB / iter)\n",
                achieved_gbps, bytes / (1024.0 * 1024.0));
    if (peak_gbps > 0.0) {
        std::printf("theoretical   : %.1f GB/s  (2 x %.0f MHz x %d bits / 8)\n",
                    peak_gbps, p.memoryClockRate / 1.0e3, p.memoryBusWidth);
        std::printf("efficiency    : %.1f%% of peak\n", efficiency);
        std::printf("\nUse the ACHIEVED number as the memory ceiling of your roofline in B02.\n");
        std::printf("The peak is the number on the datasheet; the gap is DRAM refresh, ECC,\n");
        std::printf("read/write turnaround and command overhead, and it never goes to zero.\n");
    } else {
        std::printf("theoretical   : unavailable (driver reported clock=%d kHz, bus=%d bits)\n",
                    p.memoryClockRate, p.memoryBusWidth);
        std::printf("\nWith no usable clock/bus width, the measured number IS your ceiling.\n");
        std::printf("That is the safer number to build a roofline on anyway.\n");
    }
    std::printf("result        : %s (%lld mismatches)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
