// Lab 1 (CUDA) - Print the REAL architecture of the machine this runs on.
//
// Line-for-line counterpart of hip/device_query.cpp. Same thesis: a vendor spec
// sheet is a marketing artifact and your GPU is the only ground truth, so every
// number below comes from cudaGetDeviceProperties() at runtime. There is no
// hardcoded table in this file, deliberately.
//
// The derived numbers at the end are the ones you carry into B02 (roofline):
// theoretical peak HBM bandwidth is the memory ceiling, and the vector-FMA
// estimate is a *lower bound* on the compute ceiling - tensor cores are not
// visible in device properties at all.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda device_query.cu -o device_query.exe
// Run:   ./device_query.exe
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "check.cuh"

namespace {

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
// The x2 is the DDR (double data rate) factor. Returns 0.0 when the driver does
// not populate the clock or bus width (some MIG and vGPU configurations), which
// the caller must handle rather than printing a nonsense zero.
double theoretical_peak_gbps(int memory_clock_khz, int bus_width_bits) {
    if (memory_clock_khz <= 0 || bus_width_bits <= 0) {
        return 0.0;
    }
    const double clock_hz  = static_cast<double>(memory_clock_khz) * 1.0e3;
    const double bus_bytes = static_cast<double>(bus_width_bits) / 8.0;
    return (2.0 * clock_hz * bus_bytes) / 1.0e9;
}

void report_device(int dev) {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    std::printf("Device %d: %s\n", dev, p.name);
    std::printf("  %-38s %d.%d\n", "Compute capability", p.major, p.minor);
    std::printf("  %-38s %04x:%02x:%02x\n", "PCI domain:bus:device",
                static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID),
                static_cast<unsigned>(p.pciDeviceID));

    std::printf("\n  -- Execution resources --\n");
    std::printf("  %-38s %d\n", "Streaming multiprocessors (SMs)", p.multiProcessorCount);
    std::printf("  %-38s %d\n", "Warp size", p.warpSize);
    std::printf("  %-38s %d\n", "Max threads per block", p.maxThreadsPerBlock);
    std::printf("  %-38s %d\n", "Max threads per SM", p.maxThreadsPerMultiProcessor);
    std::printf("  %-38s %d\n", "Max blocks per SM", p.maxBlocksPerMultiProcessor);
    std::printf("  %-38s %d\n", "32-bit registers per block", p.regsPerBlock);
    std::printf("  %-38s %d\n", "32-bit registers per SM", p.regsPerMultiprocessor);
    std::printf("  %-38s %d x %d x %d\n", "Max block dims (x,y,z)",
                p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
    std::printf("  %-38s %d x %d x %d\n", "Max grid dims (x,y,z)",
                p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    std::printf("  %-38s %d\n", "Cooperative launch supported", p.cooperativeLaunch);

    std::printf("\n  -- Clocks --\n");
    std::printf("  %-38s %.0f MHz\n", "Peak SM clock", p.clockRate / 1.0e3);
    std::printf("  %-38s %.0f MHz\n", "Memory clock", p.memoryClockRate / 1.0e3);

    std::printf("\n  -- Memory system --\n");
    print_bytes("Global memory (HBM/VRAM)", static_cast<unsigned long long>(p.totalGlobalMem));
    std::printf("  %-38s %d bits\n", "Memory bus width", p.memoryBusWidth);
    print_bytes("L2 cache", static_cast<unsigned long long>(p.l2CacheSize));
    print_bytes("L2 persisting-access window (max)",
                static_cast<unsigned long long>(p.persistingL2CacheMaxSize));
    print_bytes("Shared memory per block", static_cast<unsigned long long>(p.sharedMemPerBlock));
    print_bytes("Shared memory per SM",
                static_cast<unsigned long long>(p.sharedMemPerMultiprocessor));
    print_bytes("Constant memory", static_cast<unsigned long long>(p.totalConstMem));

    std::printf("\n  -- Host/device memory model --\n");
    std::printf("  %-38s %d\n", "Integrated (shares host memory)", p.integrated);
    std::printf("  %-38s %d\n", "Can map host memory", p.canMapHostMemory);
    std::printf("  %-38s %d\n", "Unified addressing", p.unifiedAddressing);
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

    // One FP32 FMA per core per clock is two FLOP; using warpSize as the lane
    // count per scheduler issue makes this a deliberate LOWER bound, because an
    // SM has several schedulers and more FP32 cores than one warp wide. Tensor
    // cores are invisible here. Consult the datasheet for the real ceiling.
    const double vector_tflops =
        (2.0 * static_cast<double>(p.multiProcessorCount) * static_cast<double>(p.warpSize) *
         (p.clockRate * 1.0e3)) / 1.0e12;
    std::printf("  %-38s %.1f TFLOP/s\n", "FP32 vector-FMA estimate (lower bnd)", vector_tflops);
    std::printf("      = 2 FLOP x %d SMs x %d lanes x %.2f GHz\n",
                p.multiProcessorCount, p.warpSize, p.clockRate / 1.0e6);
    std::printf("      NOTE: tensor-core throughput is NOT in device properties, and an SM\n");
    std::printf("            issues from several schedulers, so the true FP32 vector peak is\n");
    std::printf("            a multiple of this. The fp16/bf16/fp8 tensor peak is roughly an\n");
    std::printf("            order of magnitude above it - see README S3.\n");

    if (peak_bw > 0.0 && vector_tflops > 0.0) {
        const double ridge = (vector_tflops * 1.0e12) / (peak_bw * 1.0e9);
        std::printf("  %-38s %.1f FLOP/byte\n", "Roofline ridge point (vector fp32)", ridge);
        std::printf("      kernels below this intensity are memory-bound - B02 formalizes it.\n");
    }
}

}  // namespace

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    std::printf("CUDA runtime %d, driver %d\n", runtime_version, driver_version);

    if (device_count == 0) {
        std::printf("\nNo CUDA-capable devices visible.\n");
        std::printf("Nothing to introspect. Common causes:\n");
        std::printf("  - no NVIDIA GPU on this host\n");
        std::printf("  - CUDA_VISIBLE_DEVICES masks every device\n");
        std::printf("  - the container was started without --gpus all\n");
        std::printf("Check with: nvidia-smi\n");
        return EXIT_SUCCESS;   // nothing measured, but not a program failure
    }

    std::printf("Visible devices: %d\n", device_count);
    for (int dev = 0; dev < device_count; ++dev) {
        std::printf("\n================================================================\n");
        report_device(dev);
    }
    std::printf("\n");
    std::printf("Record the SM count, warp size, shared memory/SM, L2 size and peak bandwidth\n");
    std::printf("above: exercises/03_measure_vs_spec.cpp measures against them, and B02\n");
    std::printf("builds your roofline from them.\n");
    return EXIT_SUCCESS;
}
