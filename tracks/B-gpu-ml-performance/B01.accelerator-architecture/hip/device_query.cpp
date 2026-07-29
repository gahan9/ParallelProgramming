// Lab 1 (HIP) - Print the REAL architecture of the machine this runs on.
//
// This module's whole thesis is that a vendor spec sheet is a marketing artifact
// and your GPU is the only ground truth. So every number below comes from
// hipGetDeviceProperties() at runtime - there is no hardcoded table anywhere in
// this file, deliberately. Run it on an MI300X and an MI355X and a laptop RDNA
// part and the output changes accordingly.
//
// The derived numbers at the end are the ones you carry into B02 (roofline):
// theoretical peak HBM bandwidth is the memory ceiling, and the vector-FMA
// estimate is a *lower bound* on the compute ceiling (see the caveat printed
// alongside it - matrix cores are not visible in device properties at all).
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip device_query.cpp -o device_query.exe
// Run:   ./device_query.exe
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "check.h"

namespace {

// Bytes -> the unit a human actually wants. GiB for capacities (that is how
// vendors quote HBM), MiB/KiB for caches and scratchpads.
void print_bytes(const char* label, unsigned long long bytes) {
    const double kib = static_cast<double>(bytes) / 1024.0;
    if (kib < 1024.0) {
        std::printf("  %-38s %llu B (%.1f KiB)\n", label, bytes, kib);
    } else if (kib < 1024.0 * 1024.0) {
        std::printf("  %-38s %llu B (%.1f MiB)\n", label, bytes, kib / 1024.0);
    } else {
        std::printf("  %-38s %llu B (%.2f GiB)\n", label, bytes, kib / (1024.0 * 1024.0));
    }
}

// Peak theoretical HBM bandwidth from first principles:
//
//   bytes/s = data_rate_transfers_per_s  x  bus_width_bytes
//           = (memory_clock_hz x 2)      x  (bus_width_bits / 8)
//
// The x2 is the DDR (double data rate) factor: HBM transfers on both clock
// edges, so the effective transfer rate is twice the clock. Returns 0.0 when
// the driver does not populate the clock or bus width, which happens on some
// virtualized and container setups - the caller must handle that.
double theoretical_peak_gbps(int memory_clock_khz, int bus_width_bits) {
    if (memory_clock_khz <= 0 || bus_width_bits <= 0) {
        return 0.0;
    }
    const double clock_hz   = static_cast<double>(memory_clock_khz) * 1.0e3;
    const double bus_bytes  = static_cast<double>(bus_width_bits) / 8.0;
    return (2.0 * clock_hz * bus_bytes) / 1.0e9;
}

void report_device(int dev) {
    hipDeviceProp_t p{};
    HIP_CHECK(hipGetDeviceProperties(&p, dev));

    std::printf("Device %d: %s\n", dev, p.name);
    std::printf("  %-38s %s\n", "LLVM target / arch name", p.gcnArchName);
    std::printf("  %-38s %d.%d\n", "gfxip (major.minor)", p.major, p.minor);
    std::printf("  %-38s %04x:%02x:%02x\n", "PCI domain:bus:device",
                static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID),
                static_cast<unsigned>(p.pciDeviceID));

    std::printf("\n  -- Execution resources --\n");
    std::printf("  %-38s %d\n", "Compute units (CUs)", p.multiProcessorCount);
    std::printf("  %-38s %d\n", "Wavefront / warp size", p.warpSize);
    std::printf("  %-38s %d\n", "Max threads per block", p.maxThreadsPerBlock);
    std::printf("  %-38s %d\n", "Max threads per CU", p.maxThreadsPerMultiProcessor);
    std::printf("  %-38s %d\n", "32-bit registers per block", p.regsPerBlock);
    std::printf("  %-38s %d x %d x %d\n", "Max block dims (x,y,z)",
                p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    std::printf("  %-38s %d x %d x %d\n", "Max grid dims (x,y,z)",
                p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    std::printf("  %-38s %d\n", "Cooperative launch supported", p.cooperativeLaunch);

    std::printf("\n  -- Clocks --\n");
    std::printf("  %-38s %.0f MHz\n", "Peak engine clock", p.clockRate / 1.0e3);
    std::printf("  %-38s %.0f MHz\n", "Memory clock", p.memoryClockRate / 1.0e3);

    std::printf("\n  -- Memory system --\n");
    print_bytes("Global memory (HBM/VRAM)", static_cast<unsigned long long>(p.totalGlobalMem));
    std::printf("  %-38s %d bits\n", "Memory bus width", p.memoryBusWidth);
    print_bytes("L2 cache", static_cast<unsigned long long>(p.l2CacheSize));
    print_bytes("LDS / shared memory per block", static_cast<unsigned long long>(p.sharedMemPerBlock));
    print_bytes("LDS / shared memory per CU",
                static_cast<unsigned long long>(p.maxSharedMemoryPerMultiProcessor));
    print_bytes("Constant memory", static_cast<unsigned long long>(p.totalConstMem));

    std::printf("\n  -- Host/device memory model (matters for MI300A-class APUs) --\n");
    std::printf("  %-38s %d\n", "Integrated (shares host memory)", p.integrated);
    std::printf("  %-38s %d\n", "Can map host memory", p.canMapHostMemory);
    std::printf("  %-38s %d\n", "Managed memory supported", p.managedMemory);
    std::printf("  %-38s %d\n", "Concurrent managed access", p.concurrentManagedAccess);

    std::printf("\n  -- Derived ceilings (the input to B02's roofline) --\n");
    const double peak_bw = theoretical_peak_gbps(p.memoryClockRate, p.memoryBusWidth);
    if (peak_bw > 0.0) {
        std::printf("  %-38s %.1f GB/s\n", "Theoretical peak HBM bandwidth", peak_bw);
        std::printf("      = 2 (DDR) x %.0f MHz x %d bits / 8\n",
                    p.memoryClockRate / 1.0e3, p.memoryBusWidth);
    } else {
        std::printf("  %-38s unavailable\n", "Theoretical peak HBM bandwidth");
        std::printf("      driver reported clock=%d kHz, bus=%d bits - one is zero, so the\n",
                    p.memoryClockRate, p.memoryBusWidth);
        std::printf("      derivation is impossible here. Measure it instead (see A02 bandwidth lab).\n");
    }

    // A CU issues one wavefront-wide FP32 FMA per clock per SIMD group; two FLOP
    // per lane per clock is the classic vector-FMA model. This is intentionally a
    // LOWER bound: packed-FP32 dual issue and matrix cores both multiply it, and
    // neither is exposed in device properties.
    const double vector_tflops =
        (2.0 * static_cast<double>(p.multiProcessorCount) * static_cast<double>(p.warpSize) *
         (p.clockRate * 1.0e3)) / 1.0e12;
    std::printf("  %-38s %.1f TFLOP/s\n", "FP32 vector-FMA estimate (lower bnd)", vector_tflops);
    std::printf("      = 2 FLOP x %d CUs x %d lanes x %.2f GHz\n",
                p.multiProcessorCount, p.warpSize, p.clockRate / 1.0e6);
    std::printf("      NOTE: matrix-core (MFMA) throughput is NOT in device properties.\n");
    std::printf("            On CDNA3/CDNA4 the fp16/bf16/fp8 matrix peak is roughly an\n");
    std::printf("            order of magnitude above this vector number - consult the\n");
    std::printf("            vendor datasheet for it, and see README S3.\n");

    if (peak_bw > 0.0 && vector_tflops > 0.0) {
        // Ridge point: the arithmetic intensity (FLOP per byte) where the compute
        // ceiling and the bandwidth ceiling cross. Left of it you are memory-bound.
        const double ridge = (vector_tflops * 1.0e12) / (peak_bw * 1.0e9);
        std::printf("  %-38s %.1f FLOP/byte\n", "Roofline ridge point (vector fp32)", ridge);
        std::printf("      kernels below this intensity are memory-bound - B02 formalizes it.\n");
    }
}

}  // namespace

int main() {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    int runtime_version = 0, driver_version = 0;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    HIP_CHECK(hipDriverGetVersion(&driver_version));
    std::printf("HIP runtime %d, driver %d\n", runtime_version, driver_version);

    if (device_count == 0) {
        std::printf("\nNo HIP-capable devices visible.\n");
        std::printf("Nothing to introspect. Common causes:\n");
        std::printf("  - no AMD GPU on this host\n");
        std::printf("  - HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES masks every device\n");
        std::printf("  - the container was started without --device=/dev/kfd --device=/dev/dri\n");
        std::printf("Check with: rocm-smi   (or: %s)\n", "/opt/rocm/bin/rocminfo");
        return EXIT_SUCCESS;   // nothing measured, but not a program failure
    }

    std::printf("Visible devices: %d\n", device_count);
    for (int dev = 0; dev < device_count; ++dev) {
        std::printf("\n================================================================\n");
        report_device(dev);
    }
    std::printf("\n");
    std::printf("Record the CU count, wavefront size, LDS/CU, L2 size and peak bandwidth\n");
    std::printf("above: exercises/03_measure_vs_spec.cpp measures against them, and B02\n");
    std::printf("builds your roofline from them.\n");
    return EXIT_SUCCESS;
}
