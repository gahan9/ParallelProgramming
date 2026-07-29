// B03 lab (CUDA) - four kernels, four different bottlenecks. You name them.
//
// Line-for-line mirror of hip/mystery_kernels.cpp so the HIP and CUDA diagnoses
// are directly comparable. Each kernel below is slow (or not) for a DIFFERENT,
// diagnosable reason. The program deliberately reports only measurements - name,
// milliseconds, effective GB/s, PASS/FAIL - and never the diagnosis. Naming the
// bottleneck from profiler evidence is exercises/01_diagnose_mystery.md; the
// answer key lives in solutions/01_diagnose_mystery_solution.md.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda mystery_kernels.cu -o mystery_kernels.exe
// Run:   ./mystery_kernels.exe            # all four
//        ./mystery_kernels.exe gamma      # one kernel, for cheap ncu runs
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "check.cuh"

// n = ROWS * COLS exactly, so kernel_beta's index permutation is a bijection and
// every output element is written exactly once (same result as kernel_alpha).
constexpr int    ROWS  = 4096;
constexpr int    COLS  = 4096;
constexpr size_t N     = static_cast<size_t>(ROWS) * COLS;   // 16,777,216 floats

constexpr float  SCALE = 2.5f;
constexpr float  BIAS  = 1.25f;

// Horner trip counts for kernel_gamma. One lane in every 32 takes the long path.
constexpr int    HEAVY = 512;
constexpr int    LIGHT = 8;
constexpr float  HORNER_C = 0.5f;

// Timing policy, shared by all four kernels so the comparison is fair.
constexpr int    WARMUP = 3;
constexpr int    ITERS  = 20;

// ---------------------------------------------------------------------------
// kernel_alpha
// ---------------------------------------------------------------------------
__global__ void kernel_alpha(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = SCALE * in[i] + BIAS;
    }
}

// ---------------------------------------------------------------------------
// kernel_beta
// ---------------------------------------------------------------------------
__global__ void kernel_beta(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         t < n; t += stride) {
        // Walk the same n elements in column-major order: consecutive threads are
        // COLS floats apart in memory.
        size_t idx = (t % ROWS) * COLS + (t / ROWS);
        if (idx < n) {
            out[idx] = SCALE * in[idx] + BIAS;
        }
    }
}

// ---------------------------------------------------------------------------
// kernel_gamma
// ---------------------------------------------------------------------------
__global__ void kernel_gamma(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        const float x    = in[i];
        const int   reps = ((i & 31u) == 0u) ? HEAVY : LIGHT;

        // Dependent FMA chain: the trip count is a runtime value, so the compiler
        // can neither unroll it away nor hoist it out.
        float acc = 1.0f;
        for (int k = 0; k < reps; ++k) {
            acc = acc * x + HORNER_C;
        }
        out[i] = acc;
    }
}

// ---------------------------------------------------------------------------
// kernel_delta
// ---------------------------------------------------------------------------
__global__ void kernel_delta(const float* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        out[i] = SCALE * in[i] + BIAS;
    }
}

// ---------------------------------------------------------------------------

enum Which { ALPHA, BETA, GAMMA, DELTA };

struct Case {
    Which       which;
    const char* name;
    int         grid;
    int         block;
    // Bytes touched per element: 4 read + 4 written for all four kernels.
    double      bytes_per_elem;
};

static const Case CASES[] = {
    {ALPHA, "kernel_alpha", 4096, 256, 8.0},
    {BETA,  "kernel_beta",  4096, 256, 8.0},
    {GAMMA, "kernel_gamma", 4096, 256, 8.0},
    {DELTA, "kernel_delta",   64,  64, 8.0},
};

static void launch(const Case& c, const float* d_in, float* d_out) {
    switch (c.which) {
        case ALPHA: kernel_alpha<<<c.grid, c.block>>>(d_in, d_out, N); break;
        case BETA:  kernel_beta <<<c.grid, c.block>>>(d_in, d_out, N); break;
        case GAMMA: kernel_gamma<<<c.grid, c.block>>>(d_in, d_out, N); break;
        case DELTA: kernel_delta<<<c.grid, c.block>>>(d_in, d_out, N); break;
    }
    CUDA_CHECK(cudaGetLastError());   // launch-configuration errors surface here
}

// Returns milliseconds per iteration, measured with GPU events.
static double time_kernel(const Case& c, const float* d_in, float* d_out) {
    // Warmup exists because the FIRST launch of a kernel pays one-time costs that
    // have nothing to do with the kernel's steady-state speed: lazy module load
    // and JIT/PTX finalization, allocator first-touch, and clocks still ramping
    // from idle. Timing it would report the runtime's startup, not your kernel.
    // This is profiling lesson #2 in the README - and the reason the profiling
    // recipes below use --launch-skip / --kernel-iteration-range.
    for (int i = 0; i < WARMUP; ++i) launch(c, d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) launch(c, d_in, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));   // also surfaces execution errors

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(ms) / ITERS;
}

static float horner_ref(float x, int reps) {
    float acc = 1.0f;
    for (int k = 0; k < reps; ++k) acc = acc * x + HORNER_C;
    return acc;
}

// Relative tolerance: the GPU may contract `acc * x + C` into a single FMA while
// the host does a separate multiply and add, so the two differ in the last bits.
static bool verify(Which which, const std::vector<float>& h_in, const float* d_out) {
    std::vector<float> out(N);
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    size_t bad = 0;
    for (size_t i = 0; i < N; ++i) {
        const float expect = (which == GAMMA)
            ? horner_ref(h_in[i], ((i & 31u) == 0u) ? HEAVY : LIGHT)
            : SCALE * h_in[i] + BIAS;
        const float denom = std::fabs(expect) > 1.0f ? std::fabs(expect) : 1.0f;
        if (std::fabs(out[i] - expect) / denom > 1e-4f) {
            if (++bad > 8) break;
        }
    }
    return bad == 0;
}

static bool selected(const char* filter, const char* name) {
    // name is "kernel_alpha"; accept either the full name or just "alpha".
    return filter == nullptr
        || std::strcmp(filter, name) == 0
        || std::strcmp(filter, name + std::strlen("kernel_")) == 0;
}

int main(int argc, char** argv) {
    const char* filter = (argc > 1 && std::strcmp(argv[1], "all") != 0) ? argv[1] : nullptr;

    std::vector<float> h_in(N);
    // Keep |x| < 1 so kernel_gamma's Horner chain stays bounded for any trip count.
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = 0.25f + 0.5f * static_cast<float>(i % 1000) / 1000.0f;
    }

    const size_t bytes = N * sizeof(float);
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    std::printf("n = %zu elements (%.1f MB per buffer), %d warmup + %d timed iterations\n\n",
                N, bytes / 1.0e6, WARMUP, ITERS);
    std::printf("%-14s %12s %16s %8s\n", "kernel", "time (ms)", "effective GB/s", "result");
    std::printf("-------------- ------------ ---------------- --------\n");

    int matched = 0;
    for (const Case& c : CASES) {
        if (!selected(filter, c.name)) continue;
        ++matched;

        const double ms = time_kernel(c, d_in, d_out);
        CUDA_CHECK(cudaMemset(d_out, 0, bytes));
        launch(c, d_in, d_out);                   // repopulate for verification
        CUDA_CHECK(cudaDeviceSynchronize());
        const bool ok = verify(c.which, h_in, d_out);

        const double gbps = (c.bytes_per_elem * static_cast<double>(N)) / (ms * 1.0e-3) / 1.0e9;
        std::printf("%-14s %12.3f %16.1f %8s\n", c.name, ms, gbps, ok ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    if (matched == 0) {
        std::fprintf(stderr, "\nno kernel matched '%s'. Use: alpha | beta | gamma | delta | all\n",
                     argv[1]);
        return EXIT_FAILURE;
    }

    std::printf("\nAll four are correct. The differences are entirely performance.\n"
                "Profile them and name each bottleneck: see exercises/01_diagnose_mystery.md\n");
    return 0;
}
