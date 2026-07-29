// B04 Lab 2 (HIP) - The accuracy/speed tradeoff of low precision, measured.
//
// Two experiments, because low precision pays off through two different
// mechanisms and they behave very differently:
//
//   [A] Tiled GEMM (compute-bound-ish): the SAME kernel with operands stored in
//       fp32, fp16 and bf16, always accumulating in fp32 - plus one deliberately
//       broken variant that accumulates in fp16 so you can see that the
//       ACCUMULATOR, not the storage, is what destroys a long dot product.
//       Reported per variant: ms, GFLOP/s, analytic global traffic, effective
//       GB/s, and max/mean relative error against a DOUBLE reference.
//
//   [B] Elementwise y = a*x + y (hopelessly memory-bound): the same three storage
//       types over 16.7M elements. Here halving the bytes really does move the
//       ceiling, and the measured GB/s says by how much.
//
// The pair is the point. [B] is where "half the bytes" turns into "half the time";
// [A] is where it does not, because this teaching kernel is limited by LDS traffic
// and vector-ALU issue rather than HBM, and it does NOT use the matrix cores (MFMA
// on CDNA, Tensor Cores on Hopper/Ada) that are the real reason reduced precision
// is fast for GEMM. Section 5 of the README expands on this; a vendor library
// (hipBLASLt / cuBLASLt) is what actually converts fp16/bf16 into 2-8x throughput.
//
// bf16 is modelled as a 16-bit storage type with exact round-to-nearest-even
// conversion rather than __hip_bfloat16 / __nv_bfloat16, so one source file builds
// unchanged on both vendors (the vendor type names and headers differ). The
// conversion is bit-exact, so the accuracy column is real bf16 accuracy; only the
// arithmetic is done after widening to fp32, which is exactly what "bf16 storage
// with fp32 accumulate" means anyway.
//
// Build: hipcc -O3 --offload-arch=gfx942 -Ihip -DMDIM=1024 mixed_precision_gemm.cpp \
//              -o mixed_precision_gemm.exe
// Run:   ./mixed_precision_gemm.exe
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "check.h"

#ifndef MDIM
#define MDIM 1024
#endif

// One thread per output element, 32x32 tile: 1024 threads, the hardware maximum.
// A06 explains why this shape cannot reach the compute roof; here the shape is
// held fixed on purpose so that PRECISION is the only variable.
constexpr int TILE = 32;
constexpr int TILE_THREADS = TILE * TILE;

// ---------------------------------------------------------------------------
// Storage policies: how a value is kept in global memory, and how it converts
// ---------------------------------------------------------------------------
union FloatBits {
    float        f;
    unsigned int u;
};

// bf16 is literally the top half of an fp32. The 0x7FFF + lsb bias before the
// shift is round-to-nearest-even; without it you get truncation, which is biased
// toward zero and measurably worse.
__host__ __device__ inline unsigned short float_to_bf16(float x) {
    FloatBits fb;
    fb.f = x;
    if ((fb.u & 0x7F800000u) == 0x7F800000u && (fb.u & 0x007FFFFFu) != 0u) return 0x7FC0u;
    const unsigned int lsb = (fb.u >> 16) & 1u;
    return static_cast<unsigned short>((fb.u + 0x7FFFu + lsb) >> 16);
}
__host__ __device__ inline float bf16_to_float(unsigned short b) {
    FloatBits fb;
    fb.u = static_cast<unsigned int>(b) << 16;
    return fb.f;
}

struct StoreF32 {
    using type = float;
    static const char* name() { return "fp32"; }
    static int bytes() { return 4; }
    __host__ __device__ static float load(type v) { return v; }
    __host__ __device__ static type store(float v) { return v; }
};

struct StoreF16 {
    using type = __half;
    static const char* name() { return "fp16"; }
    static int bytes() { return 2; }
    __host__ __device__ static float load(type v) { return __half2float(v); }
    __host__ __device__ static type store(float v) { return __float2half(v); }
};

struct StoreBF16 {
    using type = unsigned short;
    static const char* name() { return "bf16"; }
    static int bytes() { return 2; }
    __host__ __device__ static float load(type v) { return bf16_to_float(v); }
    __host__ __device__ static type store(float v) { return float_to_bf16(v); }
};

// ---------------------------------------------------------------------------
// Accumulator policies: the width the running sum is kept in
// ---------------------------------------------------------------------------
struct AccF32 {
    using type = float;
    static const char* name() { return "fp32"; }
    __device__ static void init(type& a) { a = 0.0f; }
    __device__ static void mac(type& a, float x, float y) { a += x * y; }
    __device__ static float get(type a) { return a; }
};

struct AccF16 {
    using type = __half;
    static const char* name() { return "fp16"; }
    __device__ static void init(type& a) { a = __float2half(0.0f); }
    // Every partial sum is rounded back onto the 10-bit fp16 grid. This is the
    // variant that fails, and it fails for the reason lab 1 section 5 printed.
    __device__ static void mac(type& a, float x, float y) {
        a = __float2half(__half2float(a) + x * y);
    }
    __device__ static float get(type a) { return __half2float(a); }
};

// ---------------------------------------------------------------------------
// [A] Tiled GEMM, parameterised by storage and accumulator precision
// ---------------------------------------------------------------------------
// The shared tiles hold fp32: the operands are widened once on the way in from
// global memory, which is the cheap place to do it. (A production kernel keeps the
// tiles narrow too, halving LDS pressure as well - one more knob we are not
// touching so that precision stays the only variable.)
template <typename Store, typename Acc>
__global__ __launch_bounds__(TILE_THREADS) void gemm_tiled(
    const typename Store::type* __restrict__ A, const typename Store::type* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    typename Acc::type acc;
    Acc::init(acc);

    const int tiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < tiles; ++t) {
        const int aCol = t * TILE + tx;
        const int bRow = t * TILE + ty;

        // Out-of-range lanes must write the additive identity, not skip: every
        // thread reads the whole tile, so a stale slot corrupts the dot product.
        As[ty][tx] = (row < M && aCol < K)
                         ? Store::load(A[static_cast<size_t>(row) * K + aCol]) : 0.0f;
        Bs[ty][tx] = (bRow < K && col < N)
                         ? Store::load(B[static_cast<size_t>(bRow) * N + col]) : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; ++k) Acc::mac(acc, As[ty][k], Bs[k][tx]);
        __syncthreads();
    }

    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = Acc::get(acc);
    }
}

// ---------------------------------------------------------------------------
// [B] Elementwise y = a*x + y, parameterised by storage precision
// ---------------------------------------------------------------------------
template <typename Store>
__global__ void axpy_storage(const typename Store::type* __restrict__ x,
                             typename Store::type* __restrict__ y, float a, long long n) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        // Read, widen, one FMA, narrow, write: 1 FLOP per 3 stored elements, so the
        // element WIDTH is the entire performance story here.
        y[i] = Store::store(a * Store::load(x[i]) + Store::load(y[i]));
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct Timer {
    hipEvent_t start{}, stop{};
    Timer() {
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
    }
    ~Timer() {
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }
    void begin() { HIP_CHECK(hipEventRecord(start)); }
    float end() {
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// Reference in double, from the ORIGINAL fp32 inputs. That is the honest question
// to ask of a low-precision pipeline: how far is it from exact arithmetic on the
// data you actually have - not "how close is it to its own rounded inputs".
static void cpu_reference(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<double>& C, int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            const float* brow = &B[static_cast<size_t>(k) * N];
            double* crow = &C[static_cast<size_t>(i) * N];
            for (int j = 0; j < N; ++j) crow[j] += a * brow[j];
        }
    }
}

struct ErrorStats {
    double max_rel;
    double mean_rel;
};

static ErrorStats compare(const std::vector<double>& ref, const std::vector<float>& got) {
    ErrorStats s{0.0, 0.0};
    for (size_t i = 0; i < got.size(); ++i) {
        const double r = ref[i];
        // Inputs are drawn from [0,1) so every partial sum is positive and the
        // reference is O(K/4); the floor only guards a pathological zero.
        const double rel = std::fabs(static_cast<double>(got[i]) - r) / std::max(std::fabs(r), 1e-6);
        s.max_rel = std::max(s.max_rel, rel);
        s.mean_rel += rel;
    }
    s.mean_rel /= static_cast<double>(got.size());
    return s;
}

// Upload an fp32 host buffer converted to the policy's storage type.
template <typename Store>
static typename Store::type* upload(const std::vector<float>& host) {
    using T = typename Store::type;
    std::vector<T> narrow(host.size());
    for (size_t i = 0; i < host.size(); ++i) narrow[i] = Store::store(host[i]);
    T* dev = nullptr;
    HIP_CHECK(hipMalloc(&dev, sizeof(T) * narrow.size()));
    HIP_CHECK(hipMemcpy(dev, narrow.data(), sizeof(T) * narrow.size(), hipMemcpyHostToDevice));
    return dev;
}

// ---------------------------------------------------------------------------
// [A] driver
// ---------------------------------------------------------------------------
template <typename Store, typename Acc>
static void run_gemm_case(const std::vector<float>& hA, const std::vector<float>& hB,
                          const std::vector<double>& ref, int M, int N, int K, int iters,
                          double tol, bool expect_pass, double* baseline_ms) {
    typename Store::type* dA = upload<Store>(hA);
    typename Store::type* dB = upload<Store>(hB);
    float* dC = nullptr;
    HIP_CHECK(hipMalloc(&dC, sizeof(float) * static_cast<size_t>(M) * N));
    HIP_CHECK(hipMemset(dC, 0, sizeof(float) * static_cast<size_t>(M) * N));

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_tiled<Store, Acc><<<grid, block>>>(dA, dB, dC, M, N, K);   // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    Timer timer;
    timer.begin();
    for (int i = 0; i < iters; ++i) gemm_tiled<Store, Acc><<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    const double ms = timer.end() / iters;

    std::vector<float> hC(static_cast<size_t>(M) * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, sizeof(float) * hC.size(), hipMemcpyDeviceToHost));
    const ErrorStats err = compare(ref, hC);

    // Analytic global traffic for the tiled algorithm: each of the (N/T)^2 blocks
    // loads 2*T*K operand elements, and C is written once.
    const double operand_elems = 2.0 * static_cast<double>(M) * N * K / TILE;
    const double bytes = operand_elems * Store::bytes() + 4.0 * M * N;
    const double flops = 2.0 * M * N * K;

    if (*baseline_ms == 0.0) *baseline_ms = ms;
    const bool pass = err.max_rel <= tol;
    const char* verdict = pass ? (expect_pass ? "PASSED" : "PASSED (unexpected!)")
                               : (expect_pass ? "FAILED" : "FAILED (expected)");

    printf("%-6s %-6s %9.3f %9.1f %8.2fx %10.1f %9.1f %11.2e %11.2e %8.0e  %s\n",
           Store::name(), Acc::name(), ms, flops / (ms / 1e3) / 1e9, *baseline_ms / ms,
           bytes / 1e6, bytes / (ms / 1e3) / 1e9, err.max_rel, err.mean_rel, tol, verdict);

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
}

// ---------------------------------------------------------------------------
// [B] driver
// ---------------------------------------------------------------------------
template <typename Store>
static void run_axpy_case(const std::vector<float>& hx, const std::vector<float>& hy, float a,
                          int iters, double* baseline_ms) {
    using T = typename Store::type;
    const long long n = static_cast<long long>(hx.size());

    T* dx = upload<Store>(hx);
    T* dy = upload<Store>(hy);

    const int block = 256;
    const int grid = 1024;                    // fixed, machine-sized grid + grid-stride loop

    axpy_storage<Store><<<grid, block>>>(dx, dy, a, n);            // warmup
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Correctness before speed: one extra pass on a freshly uploaded y, checked
    // against the same arithmetic done in double on the host. The tolerance is the
    // storage format's own resolution - a 16-bit result cannot be closer than that.
    T* dy_check = upload<Store>(hy);
    axpy_storage<Store><<<grid, block>>>(dx, dy_check, a, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<T> narrow(hx.size());
    HIP_CHECK(hipMemcpy(narrow.data(), dy_check, sizeof(T) * narrow.size(),
                        hipMemcpyDeviceToHost));
    double max_rel = 0.0;
    for (size_t i = 0; i < hx.size(); ++i) {
        // Reference from the values as stored: this check is asking "did the kernel
        // do the arithmetic right", not "how lossy is the format" (that is [A]).
        const double r = static_cast<double>(a) * Store::load(Store::store(hx[i])) +
                         Store::load(Store::store(hy[i]));
        const double got = Store::load(narrow[i]);
        max_rel = std::max(max_rel, std::fabs(got - r) / std::max(std::fabs(r), 1e-6));
    }
    const double tol = (Store::bytes() == 4) ? 1e-6 : 1e-2;
    HIP_CHECK(hipFree(dy_check));

    Timer timer;
    timer.begin();
    for (int i = 0; i < iters; ++i) axpy_storage<Store><<<grid, block>>>(dx, dy, a, n);
    HIP_CHECK(hipGetLastError());
    const double ms = timer.end() / iters;

    const double bytes = 3.0 * static_cast<double>(n) * Store::bytes();   // read x, read+write y
    if (*baseline_ms == 0.0) *baseline_ms = ms;

    printf("%-6s %12.1f %10.4f %10.1f %9.2fx %11.2e  %s\n", Store::name(), bytes / 1e6, ms,
           bytes / (ms / 1e3) / 1e9, *baseline_ms / ms, max_rel,
           max_rel <= tol ? "PASSED" : "FAILED");

    HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy));
}

int main() {
    const int M = MDIM, N = MDIM, K = MDIM;
    const int iters = 10;

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("=== B04 Lab 2 - mixed precision: what you gain, what you pay ===\n");
    printf("device : %s\n", prop.name);
    printf("[A] GEMM  C[%d x %d] = A[%d x %d] * B[%d x %d], %.2f GFLOP per call, %d iters\n", M,
           N, M, K, K, N, 2.0 * M * N * K / 1e9, iters);

    std::vector<float> hA(static_cast<size_t>(M) * K), hB(static_cast<size_t>(K) * N);
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    printf("computing the double-precision reference on the host (this takes a moment)...\n");
    fflush(stdout);
    std::vector<double> ref(static_cast<size_t>(M) * N);
    cpu_reference(hA, hB, ref, M, N, K);

    printf("\n%-6s %-6s %9s %9s %9s %10s %9s %11s %11s %8s  %s\n", "store", "accum", "ms/iter",
           "GFLOP/s", "speedup", "MB global", "GB/s", "max rel", "mean rel", "tol", "result");
    printf("------ ------ --------- --------- --------- ---------- --------- ----------- "
           "----------- --------  ------\n");

    // Tolerances, justified rather than tuned until green:
    //   fp32 storage : rounding of a K-long fp32 sum, ~sqrt(K)*2^-24 ~ 2e-6. 1e-4 is
    //                  50x headroom and still 4 orders below what a real bug produces.
    //   fp16 storage : the INPUTS are already perturbed by up to half an fp16 ulp,
    //                  2^-11 = 4.9e-4 relative, and two operands compound: ~1e-3.
    //   bf16 storage : same argument with 7 mantissa bits, 2^-8 = 3.9e-3: ~1e-2.
    //   fp16 accum   : held to the fp16-storage tolerance on purpose. It is EXPECTED
    //                  to fail - that failure is the lesson, not a defect.
    double gemm_baseline = 0.0;
    run_gemm_case<StoreF32, AccF32>(hA, hB, ref, M, N, K, iters, 1e-4, true, &gemm_baseline);
    run_gemm_case<StoreF16, AccF32>(hA, hB, ref, M, N, K, iters, 2e-3, true, &gemm_baseline);
    run_gemm_case<StoreBF16, AccF32>(hA, hB, ref, M, N, K, iters, 2e-2, true, &gemm_baseline);
    run_gemm_case<StoreF16, AccF16>(hA, hB, ref, M, N, K, iters, 2e-3, false, &gemm_baseline);

    printf("\n  Read the accuracy columns first. fp16 and bf16 storage both stay within the\n"
           "  error their own mantissa allows, and bf16's error is about 8x fp16's - exactly\n"
           "  the ratio of their ulp(1) (2^-7 vs 2^-10). Now read the last row: the same\n"
           "  fp16 storage with an fp16 ACCUMULATOR is far worse, because %d roundings of a\n"
           "  growing sum compound. Storage precision costs you a bounded relative error;\n"
           "  accumulator precision costs you the whole result. That single asymmetry is why\n"
           "  every matrix core multiplies narrow and accumulates in fp32.\n", K);
    printf("\n  Now read the throughput columns, and be suspicious. Global traffic halves\n"
           "  exactly as the byte accounting predicts, but the GFLOP/s should move far less\n"
           "  than that (how much less depends on your GPU): at a 32x32 tile this kernel is\n"
           "  limited by LDS traffic and VALU issue rather than HBM, and it never touches the\n"
           "  matrix cores. Narrower operands only buy time where BYTES are the limit - which\n"
           "  is what [B] shows - or where a matrix-core path exists to consume them, which is\n"
           "  what hipBLASLt / cuBLASLt do and this kernel does not. README section 5.\n");

    // ---- [B] memory-bound elementwise --------------------------------------
    const long long n_elems = 1LL << 24;      // 16.7M, same size as the A01/A02 labs
    printf("\n[B] Elementwise y = a*x + y, n = %lld (1 FLOP per 3 stored elements)\n\n",
           n_elems);
    std::vector<float> hx(static_cast<size_t>(n_elems)), hy(static_cast<size_t>(n_elems));
    for (long long i = 0; i < n_elems; ++i) {
        hx[static_cast<size_t>(i)] = dist(rng);
        hy[static_cast<size_t>(i)] = dist(rng);
    }

    printf("%-6s %12s %10s %10s %9s %11s  %s\n", "store", "MB moved", "ms/iter", "GB/s",
           "speedup", "max rel", "result");
    printf("------ ------------ ---------- ---------- --------- -----------  ------\n");
    double axpy_baseline = 0.0;
    run_axpy_case<StoreF32>(hx, hy, 2.0f, iters, &axpy_baseline);
    run_axpy_case<StoreF16>(hx, hy, 2.0f, iters, &axpy_baseline);
    run_axpy_case<StoreBF16>(hx, hy, 2.0f, iters, &axpy_baseline);

    printf("\n  This is the roofline result from B02, in one table: the kernel moves 3 elements\n"
           "  per FLOP, so it sits far to the left of the ridge point and its ceiling IS the\n"
           "  bandwidth divided by the bytes per element. Halve the element width and the\n"
           "  ceiling doubles, so the time should approach half - expect a speedup near 2x but\n"
           "  short of it, because the GB/s column shows both variants are already close to\n"
           "  the same HBM limit and the fp16/bf16 runs have less work to hide launch and\n"
           "  scheduling overhead behind. Compare the GB/s across rows: if they match, you are\n"
           "  bandwidth-bound in both and the speedup is purely the byte count.\n");
    printf("\n  Neither table is a benchmark of your GPU. Run it on yours; the SHAPE (bytes\n"
           "  halve, error grows with the reciprocal of mantissa bits, fp16 accumulate loses)\n"
           "  is what transfers.\n");
    return 0;
}
