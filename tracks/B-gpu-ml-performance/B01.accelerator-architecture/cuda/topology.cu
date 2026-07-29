// Lab 2 (CUDA) - The interconnect story, measured instead of quoted.
//
// Line-for-line counterpart of hip/topology.cpp. A datasheet says "18 NVLink 4
// links, 900 GB/s per GPU". This program asks the machine in front of you: which
// GPU pairs can address each other's memory directly, and how fast is a copy
// between them in practice? That matters because tensor parallelism (B07) is
// bounded by exactly this number, not by the number on the slide.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda topology.cu -o topology.exe
// Run:   ./topology.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

namespace {

constexpr size_t kCopyBytes = 256u * 1024u * 1024u;   // large enough to hide launch overhead
constexpr int    kIters     = 20;

// Peer access is enabled per (src -> dst) ordered pair and is idempotent from the
// caller's point of view: a second enable returns cudaErrorPeerAccessAlreadyEnabled.
// That is not a failure, so this one call is checked by hand rather than with
// CUDA_CHECK - every other runtime call in this file goes through the macro.
bool try_enable_peer(int src, int dst) {
    CUDA_CHECK(cudaSetDevice(src));
    const cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
    if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
        return true;
    }
    std::fprintf(stderr, "note: could not enable peer access %d->%d (%s)\n",
                 src, dst, cudaGetErrorString(err));
    return false;
}

double measure_peer_gbps(int src, int dst) {
    void* d_src = nullptr;
    void* d_dst = nullptr;

    CUDA_CHECK(cudaSetDevice(src));
    CUDA_CHECK(cudaMalloc(&d_src, kCopyBytes));
    CUDA_CHECK(cudaMemset(d_src, 1, kCopyBytes));

    CUDA_CHECK(cudaSetDevice(dst));
    CUDA_CHECK(cudaMalloc(&d_dst, kCopyBytes));

    CUDA_CHECK(cudaSetDevice(src));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup: the first peer copy pays one-time mapping cost that would otherwise
    // be attributed to the link.
    CUDA_CHECK(cudaMemcpyPeer(d_dst, dst, d_src, src, kCopyBytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < kIters; ++it) {
        CUDA_CHECK(cudaMemcpyPeer(d_dst, dst, d_src, src, kCopyBytes));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double seconds = (ms / 1.0e3) / kIters;
    const double gbps    = static_cast<double>(kCopyBytes) / seconds / 1.0e9;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaSetDevice(dst));
    CUDA_CHECK(cudaFree(d_dst));

    return gbps;
}

}  // namespace

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::printf("No CUDA-capable devices visible - there is no topology to report.\n");
        std::printf("Check with: nvidia-smi   (and unset CUDA_VISIBLE_DEVICES if it is set)\n");
        return EXIT_SUCCESS;
    }

    std::printf("Visible devices: %d\n\n", device_count);
    std::printf("%-4s %-28s %-8s %-16s\n", "dev", "name", "cc", "PCI dom:bus:dev");
    std::printf("---- ---------------------------- -------- ----------------\n");
    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
        std::printf("%-4d %-28.28s %d.%-6d %04x:%02x:%02x\n",
                    dev, p.name, p.major, p.minor,
                    static_cast<unsigned>(p.pciDomainID), static_cast<unsigned>(p.pciBusID),
                    static_cast<unsigned>(p.pciDeviceID));
    }

    if (device_count == 1) {
        std::printf("\nOnly one device is visible, so there are no peer links to probe.\n");
        std::printf("The single-GPU story is complete without them; come back to this lab on\n");
        std::printf("an 8-GPU node (an HGX baseboard or an MI300X/MI355X OAM board) to see the\n");
        std::printf("fabric. Everything below needs at least two devices.\n");
        return EXIT_SUCCESS;
    }

    // ---- Reachability matrix ----
    std::printf("\nPeer access matrix (row = source, col = destination)\n");
    std::printf("  Y = direct peer addressing possible, . = must route through the host\n\n");
    std::printf("     ");
    for (int dst = 0; dst < device_count; ++dst) {
        std::printf("%4d", dst);
    }
    std::printf("\n");

    std::vector<int> can_access(static_cast<size_t>(device_count) * device_count, 0);
    for (int src = 0; src < device_count; ++src) {
        std::printf("%4d ", src);
        for (int dst = 0; dst < device_count; ++dst) {
            if (src == dst) {
                std::printf("%4s", "-");
                continue;
            }
            int ok = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&ok, src, dst));
            can_access[static_cast<size_t>(src) * device_count + dst] = ok;
            std::printf("%4s", ok ? "Y" : ".");
        }
        std::printf("\n");
    }

    // ---- Runtime's own link-quality ranking ----
    // A performance rank of 0 is the best link this device has; larger numbers mean
    // a worse path. On a fully NVSwitch-connected baseboard every pair ranks
    // equally; across a PCIe host bridge the ranks separate.
    std::printf("\nPeer performance rank (lower is better, '.' = no direct path)\n\n");
    std::printf("     ");
    for (int dst = 0; dst < device_count; ++dst) {
        std::printf("%4d", dst);
    }
    std::printf("\n");
    for (int src = 0; src < device_count; ++src) {
        std::printf("%4d ", src);
        for (int dst = 0; dst < device_count; ++dst) {
            if (src == dst) {
                std::printf("%4s", "-");
                continue;
            }
            if (!can_access[static_cast<size_t>(src) * device_count + dst]) {
                std::printf("%4s", ".");
                continue;
            }
            int rank = 0;
            CUDA_CHECK(cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, src, dst));
            std::printf("%4d", rank);
        }
        std::printf("\n");
    }

    // ---- Native atomics across the link ----
    // Device-scope atomics that work across a peer link are what make a
    // multi-GPU reduction cheap; without them the runtime falls back to slower
    // paths. This is a real capability difference between fabrics.
    std::printf("\nNative device-scope atomics over the peer link (Y/./-)\n\n");
    std::printf("     ");
    for (int dst = 0; dst < device_count; ++dst) {
        std::printf("%4d", dst);
    }
    std::printf("\n");
    for (int src = 0; src < device_count; ++src) {
        std::printf("%4d ", src);
        for (int dst = 0; dst < device_count; ++dst) {
            if (src == dst) {
                std::printf("%4s", "-");
                continue;
            }
            if (!can_access[static_cast<size_t>(src) * device_count + dst]) {
                std::printf("%4s", ".");
                continue;
            }
            int atomics = 0;
            CUDA_CHECK(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported,
                                                 src, dst));
            std::printf("%4s", atomics ? "Y" : ".");
        }
        std::printf("\n");
    }

    // ---- Measured bandwidth, device 0 outward ----
    // Probing every ordered pair on an 8-GPU node is 56 timed copies; one row is
    // enough to see the shape (uniform on a full mesh, stepped otherwise) without
    // making the lab slow.
    std::printf("\nMeasured unidirectional copy bandwidth from device 0 (%.0f MiB x %d iters)\n\n",
                static_cast<double>(kCopyBytes) / (1024.0 * 1024.0), kIters);
    std::printf("%-10s %-12s %-14s %s\n", "0 -> dst", "peer access", "GB/s", "path");
    std::printf("---------- ------------ -------------- ----------------------------\n");
    for (int dst = 1; dst < device_count; ++dst) {
        const bool direct = can_access[static_cast<size_t>(0) * device_count + dst] != 0;
        if (direct && !try_enable_peer(0, dst)) {
            std::printf("0 -> %-5d %-12s %-14s %s\n", dst, "Y", "n/a", "peer enable failed");
            continue;
        }
        const double gbps = measure_peer_gbps(0, dst);
        std::printf("0 -> %-5d %-12s %-14.1f %s\n", dst, direct ? "Y" : ".", gbps,
                    direct ? "direct fabric link" : "staged via host (PCIe both ways)");
    }

    CUDA_CHECK(cudaSetDevice(0));
    std::printf("\nWhat to take away: the 'direct fabric link' rows are your scale-up ceiling.\n");
    std::printf("Tensor parallelism moves activations across exactly these links on every\n");
    std::printf("layer, so this GB/s - not the HBM number - bounds how far a model can shard.\n");
    std::printf("B07 turns this into a sharding decision.\n");
    return EXIT_SUCCESS;
}
