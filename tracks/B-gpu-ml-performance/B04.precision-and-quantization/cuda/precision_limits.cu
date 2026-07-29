// B04 Lab 1 (CUDA) - What each numeric format can and cannot represent.
//
// Line-for-line counterpart of hip/precision_limits.cpp. Seven sections, every
// number computed at run time:
//
//   [1] Format reference table  - max finite, min normal, min subnormal, ulp(1),
//                                 decimal digits and dynamic range for fp32, tf32,
//                                 bf16, fp16, fp8 E4M3 / E5M2 / E4M3FNUZ and int8,
//                                 DERIVED from the bit layout rather than hard-coded.
//   [2] Round-trip              - one value stored in every format, with the
//                                 relative error each one introduces.
//   [3] fp16 exp() overflow     - the softmax hazard: exp(x) leaves fp16 range at
//                                 x ~ 11.09 (= ln 65504). This is why A07 subtracts
//                                 the row max first.
//   [4] 1.0 + tiny              - the vanishing weight update, the reason mixed
//                                 precision keeps an fp32 master copy of weights.
//   [5] Long dot product        - fp16 vs fp32 vs fp64 accumulation, SAME summation
//                                 order (one thread), so the only variable is the
//                                 accumulator width.
//   [6] Summation order         - fp32 accumulation with a varying number of blocks:
//                                 the answer changes. Ties back to A04.
//   [7] Catastrophic cancellation - one-pass variance in fp32 on data with a large
//                                 mean, versus the two-pass formula.
//
// Formats narrower than fp32 are modelled in software (exact round-to-nearest-even
// on the fp32 bit pattern) instead of vendor types, for two reasons: the vendor
// spellings diverge (__nv_bfloat16 vs __hip_bfloat16; Hopper and CDNA4 expose the
// OCP fp8 variants while CDNA3 exposes the FNUZ ones), and a software converter
// makes the *bit layout* the visible object of study. fp16 is the exception: __half
// and __float2half/__half2float are spelled identically in CUDA and HIP, so the
// fp16 paths use the real hardware type. Where a vendor fp8 header is present we
// cross-check the software converter against it (section 2); otherwise we say so
// and skip, rather than failing to build.
//
// Build: nvcc -O3 -arch=sm_90 -Icuda precision_limits.cu -o precision_limits.exe
// Run:   ./precision_limits.exe
// Escape hatch: -DB04_NO_NATIVE_FP8 disables the native fp8 cross-check.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "check.cuh"

// Optional cross-check against the vendor fp8 type. NVIDIA (Hopper onward) and CDNA4
// implement the OCP variants (bias 7, no infinities, max 448); CDNA3 (gfx94x)
// implements the FNUZ variants (bias 8, max 240). We compare against whichever one
// the target actually has, which is itself the lesson.
#if !defined(B04_NO_NATIVE_FP8) && defined(__has_include)
#  if __has_include(<cuda_fp8.h>)
#    include <cuda_fp8.h>
#    define B04_NATIVE_FP8 1
using native_fp8_t = __nv_fp8_e4m3;
static const char* kNativeFp8Name = "__nv_fp8_e4m3";
static const char* kNativeFp8Model = "fp8 E4M3";
#  endif
#endif

// ---------------------------------------------------------------------------
// Software format model
// ---------------------------------------------------------------------------
// A binary floating-point format is three integers: how many exponent bits (which
// buy DYNAMIC RANGE), how many mantissa bits (which buy PRECISION), and the
// exponent bias. Everything printed in section 1 follows from those three plus the
// convention used for the all-ones exponent field.
enum SpecialsKind {
    SPEC_IEEE,       // all-ones exponent reserved for inf/NaN (fp32, tf32, bf16, fp16, E5M2)
    SPEC_OCP_E4M3,   // no inf; all-ones exponent usable, top mantissa pattern is NaN -> max 448
    SPEC_FNUZ        // no inf; single NaN encoding (0x80), no negative zero -> max 240
};

struct FormatSpec {
    const char*  name;
    int          exp_bits;
    int          man_bits;
    int          bias;
    SpecialsKind kind;
    int          storage_bits;
    const char*  overflow;   // what a too-large value becomes on conversion
};

static const FormatSpec kFormats[] = {
    {"fp32",         8, 23, 127, SPEC_IEEE,     32, "inf"},
    {"tf32",         8, 10, 127, SPEC_IEEE,     32, "inf"},
    {"bf16",         8,  7, 127, SPEC_IEEE,     16, "inf"},
    {"fp16",         5, 10,  15, SPEC_IEEE,     16, "inf"},
    {"fp8 E4M3",     4,  3,   7, SPEC_OCP_E4M3,  8, "saturate/NaN"},
    {"fp8 E5M2",     5,  2,  15, SPEC_IEEE,      8, "inf"},
    {"fp8 E4M3FNUZ", 4,  3,   8, SPEC_FNUZ,      8, "saturate/NaN"},
};
static const int kNumFormats = sizeof(kFormats) / sizeof(kFormats[0]);

// Look the format up by name so the call sites read as themselves; indexing
// kFormats[3] and hoping it is still fp16 is how tables like this rot.
static const FormatSpec& format_by_name(const char* name) {
    for (int i = 0; i < kNumFormats; ++i) {
        if (std::strcmp(kFormats[i].name, name) == 0) return kFormats[i];
    }
    std::fprintf(stderr, "unknown format '%s'\n", name);
    std::exit(EXIT_FAILURE);
}

static int format_emin(const FormatSpec& f) { return 1 - f.bias; }

static int format_emax(const FormatSpec& f) {
    // IEEE formats spend the all-ones exponent on inf/NaN and therefore lose one
    // binade at the top; the fp8 encodings that drop infinities get it back.
    const int fields = 1 << f.exp_bits;
    return (f.kind == SPEC_IEEE ? fields - 2 : fields - 1) - f.bias;
}

static double format_max_significand(const FormatSpec& f) {
    // OCP E4M3 keeps one mantissa pattern at the top exponent for NaN, so its
    // largest significand is 2 - 2^(1-m) (= 1.75 for m = 3) rather than 2 - 2^-m.
    return (f.kind == SPEC_OCP_E4M3) ? 2.0 - std::ldexp(1.0, 1 - f.man_bits)
                                     : 2.0 - std::ldexp(1.0, -f.man_bits);
}

static double format_max(const FormatSpec& f) {
    return format_max_significand(f) * std::ldexp(1.0, format_emax(f));
}
static double format_min_normal(const FormatSpec& f) {
    return std::ldexp(1.0, format_emin(f));
}
static double format_min_subnormal(const FormatSpec& f) {
    return std::ldexp(1.0, format_emin(f) - f.man_bits);
}
// ulp(1.0): the gap between 1.0 and the next representable value. This is the
// C library's FLT_EPSILON convention; the worst-case relative rounding error is
// half of it.
static double format_eps(const FormatSpec& f) { return std::ldexp(1.0, -f.man_bits); }

// Round an fp32 value onto the grid of `f` with round-to-nearest-even, the default
// rounding mode of every GPU conversion instruction. Host-only: it uses libm and is
// meant for the reference table, not for hot loops (the bf16 path below is the fast
// version used inside kernels).
static float quantize(float x, const FormatSpec& f) {
    if (std::isnan(x)) return x;

    const double maxv = format_max(f);
    const double sign = std::signbit(x) ? -1.0 : 1.0;
    const float  inf  = std::numeric_limits<float>::infinity();

    if (std::isinf(x)) {
        return (f.kind == SPEC_IEEE) ? x : static_cast<float>(sign * maxv);
    }
    double a = std::fabs(static_cast<double>(x));
    if (a == 0.0) return x;

    // Binade of the value, then the spacing of the representable grid inside it.
    // std::log2 can land one off at a binade boundary, so correct explicitly.
    int e = static_cast<int>(std::floor(std::log2(a)));
    while (std::ldexp(1.0, e) > a) --e;
    while (std::ldexp(1.0, e + 1) <= a) ++e;

    const int emin  = format_emin(f);
    const double scale = (e < emin) ? std::ldexp(1.0, emin - f.man_bits)    // subnormal grid
                                    : std::ldexp(1.0, e - f.man_bits);

    const double q    = a / scale;
    double       fl   = std::floor(q);
    const double frac = q - fl;
    // Round half to EVEN: this is what makes repeated rounding unbiased, and why a
    // long fp16 accumulation drifts like sqrt(n) rather than n.
    if (frac > 0.5 || (frac == 0.5 && std::fmod(fl, 2.0) != 0.0)) fl += 1.0;

    const double r = fl * scale;
    if (r > maxv) {
        // IEEE formats have an infinity to overflow into. The fp8 encodings do not,
        // so a conversion either saturates to the max finite value or produces NaN
        // depending on the mode (FP8 paper section 2). Saturating is the default of
        // both vendors' casts, so that is what we model.
        return (f.kind == SPEC_IEEE) ? static_cast<float>(sign) * inf
                                     : static_cast<float>(sign * maxv);
    }
    return static_cast<float>(sign * r);
}

// ---------------------------------------------------------------------------
// bf16: the fast, kernel-usable path
// ---------------------------------------------------------------------------
// bf16 is the top 16 bits of an fp32 - same sign, same 8 exponent bits, mantissa
// truncated from 23 to 7 - so the conversion is integer arithmetic on the bit
// pattern. Adding 0x7FFF + lsb before the shift performs round-to-nearest-even,
// and a carry out of the mantissa correctly bumps the exponent (and, at the very
// top of the range, produces inf).
union FloatBits {
    float        f;
    unsigned int u;
};

__host__ __device__ inline unsigned short float_to_bf16(float x) {
    FloatBits fb;
    fb.f = x;
    if ((fb.u & 0x7F800000u) == 0x7F800000u && (fb.u & 0x007FFFFFu) != 0u) {
        return 0x7FC0u;                       // NaN: never let the rounding bias make it inf
    }
    const unsigned int lsb  = (fb.u >> 16) & 1u;
    const unsigned int bias = 0x7FFFu + lsb;
    return static_cast<unsigned short>((fb.u + bias) >> 16);
}

__host__ __device__ inline float bf16_to_float(unsigned short b) {
    FloatBits fb;
    fb.u = static_cast<unsigned int>(b) << 16;
    return fb.f;
}

__host__ __device__ inline float round_trip_bf16(float x) {
    return bf16_to_float(float_to_bf16(x));
}
__host__ __device__ inline float round_trip_fp16(float x) {
    return __half2float(__float2half(x));
}

// ---------------------------------------------------------------------------
// [3] fp16 exp() overflow - the softmax hazard
// ---------------------------------------------------------------------------
__global__ void probe_exp(const float* __restrict__ x, float* __restrict__ out_fp32,
                          float* __restrict__ out_fp16, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                       // partial last block
    const float e = expf(x[i]);
    out_fp32[i] = e;
    // Storing an activation in fp16 is a conversion, and a conversion can overflow
    // even when the fp32 math was perfectly healthy.
    out_fp16[i] = round_trip_fp16(e);
}

// ---------------------------------------------------------------------------
// [4] 1.0 + tiny - the vanishing update
// ---------------------------------------------------------------------------
__global__ void probe_unit_plus_tiny(const float* __restrict__ tiny, float* __restrict__ o32,
                                     float* __restrict__ o16, float* __restrict__ obf, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float t = tiny[i];
    o32[i] = 1.0f + t;
    // Low precision does not lose the small number, it loses the SUM: the result is
    // rounded back onto a grid whose spacing near 1.0 is ulp(1) = 2^-mantissa_bits.
    o16[i] = round_trip_fp16(round_trip_fp16(1.0f) + round_trip_fp16(t));
    obf[i] = round_trip_bf16(round_trip_bf16(1.0f) + round_trip_bf16(t));
}

// ---------------------------------------------------------------------------
// [5] Dot product, three accumulator widths, ONE summation order
// ---------------------------------------------------------------------------
// Deliberately single-threaded. A parallel reduction would change the summation
// order at the same time as the precision, and we want exactly one variable here;
// section 6 changes the order with the precision held fixed.
__global__ void serial_dot(const float* __restrict__ a, const float* __restrict__ b, int n,
                           float* __restrict__ out16, float* __restrict__ out32,
                           double* __restrict__ out64) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    __half acc16 = __float2half(0.0f);
    float  acc32 = 0.0f;
    double acc64 = 0.0;

    for (int i = 0; i < n; ++i) {
        const float p = a[i] * b[i];
        // A true fp16 accumulator rounds the product to fp16 and rounds the sum
        // back to fp16 on every step; that double rounding is the error source.
        acc16 = __float2half(__half2float(acc16) + round_trip_fp16(p));
        acc32 += p;
        acc64 += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    *out16 = __half2float(acc16);
    *out32 = acc32;
    *out64 = acc64;
}

// ---------------------------------------------------------------------------
// [6] Same precision, different summation order (A04's reduction, revisited)
// ---------------------------------------------------------------------------
constexpr int kRedBlock = 256;

__global__ void blocked_dot_fp32(const float* __restrict__ a, const float* __restrict__ b,
                                 int n, float* __restrict__ partials) {
    __shared__ float smem[kRedBlock];

    float acc = 0.0f;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        acc += a[i] * b[i];                   // grid-stride: no out-of-range access
    }
    smem[threadIdx.x] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// [7] Catastrophic cancellation - one-pass variance
// ---------------------------------------------------------------------------
__global__ void serial_variance(const float* __restrict__ x, int n, float* __restrict__ naive32,
                                float* __restrict__ twopass32, double* __restrict__ naive64) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float  s32 = 0.0f, sq32 = 0.0f;
    double s64 = 0.0, sq64 = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = x[i];
        s32 += v;
        sq32 += v * v;
        s64 += static_cast<double>(v);
        sq64 += static_cast<double>(v) * static_cast<double>(v);
    }
    const float  m32 = s32 / n;
    const double m64 = s64 / n;
    // E[x^2] - E[x]^2: two huge, nearly equal numbers. Their leading digits cancel
    // and what survives is rounding noise, not signal.
    *naive32 = sq32 / n - m32 * m32;
    *naive64 = sq64 / n - m64 * m64;

    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float d = x[i] - m32;           // subtract first: the operands are now small
        acc += d * d;
    }
    *twopass32 = acc / n;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct Timer {
    cudaEvent_t start{}, stop{};
    Timer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~Timer() {
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

static void print_format_table() {
    printf("[1] Format reference table (derived from the bit layout, not tabulated)\n\n");
    printf("%-13s %5s %3s %3s %6s %12s %12s %13s %11s %7s %8s %-13s\n",
           "format", "bits", "E", "M", "bias", "max finite", "min normal", "min subnormal",
           "ulp(1)", "digits", "binades", "overflow");
    printf("------------- ----- --- --- ------ ------------ ------------ ------------- "
           "----------- ------- -------- -------------\n");
    for (int i = 0; i < kNumFormats; ++i) {
        const FormatSpec& f = kFormats[i];
        printf("%-13s %5d %3d %3d %6d %12.4e %12.4e %13.4e %11.3e %7.2f %8d %-13s\n",
               f.name, f.storage_bits, f.exp_bits, f.man_bits, f.bias, format_max(f),
               format_min_normal(f), format_min_subnormal(f), format_eps(f),
               (f.man_bits + 1) * std::log10(2.0), format_emax(f) - format_emin(f) + 1,
               f.overflow);
    }
    printf("%-13s %5d %3s %3s %6s %12.4e %12s %13s %11s %7.2f %8s %-13s\n",
           "int8", 8, "-", "-", "-", 127.0, "1 (the step)", "-", "1 (the step)",
           8 * std::log10(2.0), "-", "clamp");
    printf("\n  Read the table left to right: E buys RANGE (look at the max-finite and\n"
           "  binades columns), M buys PRECISION (ulp(1) and digits). bf16 and fp32 share\n"
           "  E = 8 and therefore share the range; fp16 has MORE mantissa than bf16 and so\n"
           "  more precision, but only 30 binades of range against 254.\n\n");
}

static void print_round_trip(const std::vector<float>& values) {
    printf("[2] The same value stored in each format (software round-to-nearest-even)\n\n");
    printf("%14s", "value");
    for (int i = 0; i < kNumFormats; ++i) printf(" %14s", kFormats[i].name);
    printf("\n");
    for (size_t v = 0; v < values.size(); ++v) {
        printf("%14.6g", values[v]);
        for (int i = 0; i < kNumFormats; ++i) {
            printf(" %14.6g", quantize(values[v], kFormats[i]));
        }
        printf("\n");
    }
    printf("\n  Note 65505 -> inf in E5M2 but 65504 in fp16 (same 5 exponent bits, more\n"
           "  mantissa), 3e38 -> inf in fp16 but survives in bf16 (same 8 exponent bits as\n"
           "  fp32), and 1e-8 -> 0 in every 16-bit-or-smaller format except bf16.\n\n");

#if defined(B04_NATIVE_FP8)
    // Cross-check the software converter against the vendor type, on in-range values
    // only: overflow behaviour is mode-dependent and not what we are validating.
    const FormatSpec& native_model = format_by_name(kNativeFp8Model);
    printf("    native fp8 cross-check vs %s (software model: %s)\n", kNativeFp8Name,
           native_model.name);
    const float probes[] = {0.1f, 1.0f / 3.0f, 1.0f, 7.5f, 240.0f, 0.001f, -2.75f};
    int mismatches = 0;
    for (float p : probes) {
        const native_fp8_t nv(p);
        const float native = static_cast<float>(nv);
        const float soft   = quantize(p, native_model);
        if (native != soft) ++mismatches;
        printf("      %-10.6g native %-12.6g software %-12.6g %s\n", p, native, soft,
               native == soft ? "match" : "MISMATCH");
    }
    printf("    -> %s\n\n", mismatches == 0 ? "software model agrees with the hardware type"
                                            : "MISMATCH: inspect the converter");
#else
    printf("    native fp8 cross-check SKIPPED: no vendor fp8 header found (needs a recent\n"
           "    ROCm/CUDA), or -DB04_NO_NATIVE_FP8 was passed. The software table above is\n"
           "    unaffected.\n\n");
#endif
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== B04 Lab 1 - precision limits, measured ===\ndevice: %s\n\n", prop.name);

    print_format_table();
    print_round_trip({0.1f, 1.0f / 3.0f, 1.0f + std::ldexp(1.0f, -11), 65505.0f, 1e-8f,
                      3.0e38f, 448.0f, 240.0f, 300.0f});

    Timer timer;
    double total_ms = 0.0;

    // ---- [3] fp16 exp() overflow -------------------------------------------
    {
        // Ascending, and deliberately dense either side of ln(65504) = 11.0899.
        const std::vector<float> hx = {9.0f,   10.0f,  10.5f,  11.0f,  11.05f, 11.08f,
                                       11.09f, 11.10f, 11.15f, 11.50f, 12.00f};
        const int n = static_cast<int>(hx.size());
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        float *dx = nullptr, *d32 = nullptr, *d16 = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, bytes));
        CUDA_CHECK(cudaMalloc(&d32, bytes));
        CUDA_CHECK(cudaMalloc(&d16, bytes));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));

        const int block = 64, grid = (n + block - 1) / block;
        probe_exp<<<grid, block>>>(dx, d32, d16, n);            // warmup
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.begin();
        probe_exp<<<grid, block>>>(dx, d32, d16, n);
        CUDA_CHECK(cudaGetLastError());
        const float ms = timer.end();
        total_ms += ms;

        std::vector<float> h32(n), h16(n);
        CUDA_CHECK(cudaMemcpy(h32.data(), d32, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h16.data(), d16, bytes, cudaMemcpyDeviceToHost));

        printf("[3] fp16 exp() overflow - the softmax hazard (kernel %.3f ms)\n\n", ms);
        printf("%8s %16s %16s %-10s\n", "x", "expf(x) in fp32", "stored as fp16", "status");
        printf("-------- ---------------- ---------------- ----------\n");
        float first_overflow = 0.0f;
        for (int i = 0; i < n; ++i) {
            const bool bad = std::isinf(h16[i]);
            if (bad && first_overflow == 0.0f) first_overflow = hx[i];
            printf("%8.3f %16.6e %16.6e %-10s\n", hx[i], h32[i], h16[i], bad ? "OVERFLOW" : "ok");
        }
        const double fp16_max = format_max(format_by_name("fp16"));
        printf("\n  fp16 max finite = %.0f, and ln(that) = %.4f, so exp() escapes fp16 at\n",
               fp16_max, std::log(fp16_max));
        printf("  x = %.3f - a completely ordinary logit. fp32 has headroom to x = %.1f.\n",
               first_overflow, std::log(format_max(format_by_name("fp32"))));
        printf("  A07's fix is not a micro-optimisation: subtracting the row max makes every\n"
               "  exponent argument <= 0, so exp() <= 1 and the format cannot overflow at all.\n\n");

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(d32));
        CUDA_CHECK(cudaFree(d16));
    }

    // ---- [4] 1.0 + tiny ----------------------------------------------------
    {
        const std::vector<float> tiny = {1e-1f, 1e-2f, 1e-3f, 1e-4f, 1e-5f, 1e-6f, 1e-8f};
        const int n = static_cast<int>(tiny.size());
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        float *dt = nullptr, *d32 = nullptr, *d16 = nullptr, *dbf = nullptr;
        CUDA_CHECK(cudaMalloc(&dt, bytes));
        CUDA_CHECK(cudaMalloc(&d32, bytes));
        CUDA_CHECK(cudaMalloc(&d16, bytes));
        CUDA_CHECK(cudaMalloc(&dbf, bytes));
        CUDA_CHECK(cudaMemcpy(dt, tiny.data(), bytes, cudaMemcpyHostToDevice));

        probe_unit_plus_tiny<<<1, 64>>>(dt, d32, d16, dbf, n);   // warmup
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.begin();
        probe_unit_plus_tiny<<<1, 64>>>(dt, d32, d16, dbf, n);
        CUDA_CHECK(cudaGetLastError());
        const float ms = timer.end();
        total_ms += ms;

        std::vector<float> h32(n), h16(n), hbf(n);
        CUDA_CHECK(cudaMemcpy(h32.data(), d32, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h16.data(), d16, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hbf.data(), dbf, bytes, cudaMemcpyDeviceToHost));

        printf("[4] 1.0 + tiny: where a weight update vanishes (kernel %.3f ms)\n\n", ms);
        printf("%10s %14s %14s %14s   %s\n", "tiny", "fp32 1+t", "fp16 1+t", "bf16 1+t",
               "update survives in");
        printf("---------- -------------- -------------- --------------   ------------------\n");
        for (int i = 0; i < n; ++i) {
            char kept[32];
            snprintf(kept, sizeof(kept), "%s%s%s", h32[i] != 1.0f ? "fp32 " : "",
                     h16[i] != 1.0f ? "fp16 " : "", hbf[i] != 1.0f ? "bf16" : "");
            printf("%10.0e %14.9f %14.9f %14.9f   %s\n", tiny[i], h32[i], h16[i], hbf[i],
                   kept[0] ? kept : "NOTHING (silently dropped)");
        }
        printf("\n  ulp(1) is %.3e in fp16 and %.3e in bf16, so any update smaller than half\n",
               format_eps(format_by_name("fp16")), format_eps(format_by_name("bf16")));
        printf("  of that rounds straight back to 1.0. A learning rate of 1e-3 against weights\n"
               "  of order 1 is exactly this regime - which is why mixed-precision training\n"
               "  applies the update to an fp32 MASTER copy and only casts down for the next\n"
               "  forward pass. Note the ordering: fp16 keeps more of these than bf16, because\n"
               "  precision near 1.0 is a mantissa question, not a range question.\n\n");

        CUDA_CHECK(cudaFree(dt));
        CUDA_CHECK(cudaFree(d32));
        CUDA_CHECK(cudaFree(d16));
        CUDA_CHECK(cudaFree(dbf));
    }

    // ---- [5] + [6] dot product: accumulator width, then summation order ----
    {
        const int nmax = 1 << 16;
        std::vector<float> ha(nmax), hb(nmax);
        std::mt19937 rng(0);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < nmax; ++i) {
            ha[i] = dist(rng);
            hb[i] = dist(rng);
        }

        float *da = nullptr, *db = nullptr, *d16 = nullptr, *d32 = nullptr, *dpart = nullptr;
        double* d64 = nullptr;
        const size_t bytes = static_cast<size_t>(nmax) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&da, bytes));
        CUDA_CHECK(cudaMalloc(&db, bytes));
        CUDA_CHECK(cudaMalloc(&d16, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d32, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d64, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dpart, sizeof(float) * 4096));
        CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

        printf("[5] Dot product: one summation order, three accumulator widths\n\n");
        printf("%8s %18s %12s %18s %12s   %s\n", "n", "fp16 accumulator", "rel err",
               "fp32 accumulator", "rel err", "verdict");
        printf("-------- ------------------ ------------ ------------------ ------------   "
               "-------\n");

        bool all_ok = true;
        for (int n : {256, 1024, 4096, 16384, 65536}) {
            serial_dot<<<1, 1>>>(da, db, n, d16, d32, d64);   // warmup
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.begin();
            serial_dot<<<1, 1>>>(da, db, n, d16, d32, d64);
            CUDA_CHECK(cudaGetLastError());
            total_ms += timer.end();

            float h16 = 0.0f, h32 = 0.0f;
            double h64 = 0.0;
            CUDA_CHECK(cudaMemcpy(&h16, d16, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h32, d32, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h64, d64, sizeof(double), cudaMemcpyDeviceToHost));

            // Host reference in long double, same order as the kernel. The fp64 GPU
            // result must match it to ~1e-12 relative; anything looser would mean the
            // kernel, not the arithmetic, is wrong.
            long double ref = 0.0L;
            for (int i = 0; i < n; ++i) {
                ref += static_cast<long double>(ha[i]) * static_cast<long double>(hb[i]);
            }
            const double refd = static_cast<double>(ref);
            const double err64 = std::fabs(h64 - refd) / refd;
            if (err64 > 1e-12) all_ok = false;

            // fp32 tolerance: random rounding accumulates like sqrt(n)*eps, i.e. about
            // 3e-5 at n = 65536. 1e-3 leaves headroom while still catching a real bug,
            // which would show up as an O(1) relative error.
            const double err32 = std::fabs(static_cast<double>(h32) - refd) / refd;
            if (err32 > 1e-3) all_ok = false;

            const double err16 = std::fabs(static_cast<double>(h16) - refd) / refd;
            printf("%8d %18.6f %12.3e %18.6f %12.3e   %s\n", n, h16, err16, h32, err32,
                   err32 <= 1e-3 ? "fp32 ok" : "fp32 SUSPECT");
        }
        printf("\n  fp64 GPU result vs a long-double host sum in the same order: %s\n",
               all_ok ? "PASSED (<=1e-12 rel)" : "FAILED - investigate");
        printf("  The fp16 column is the lesson. Its error grows with n because every one of\n"
               "  the n additions is rounded onto a grid with only 10 mantissa bits, and once\n"
               "  the running sum is large enough, each new term is smaller than half an ulp\n"
               "  of it and contributes NOTHING. This is why matrix cores multiply in 16 bits\n"
               "  but accumulate in fp32: the product is cheap to narrow, the SUM is not.\n\n");

        printf("[6] Same fp32 precision, different summation order (A04's reduction)\n\n");
        const int n = nmax;
        long double refl = 0.0L;
        for (int i = 0; i < n; ++i) {
            refl += static_cast<long double>(ha[i]) * static_cast<long double>(hb[i]);
        }
        const double refd = static_cast<double>(refl);
        printf("%8s %18s %14s\n", "blocks", "fp32 result", "rel err");
        printf("-------- ------------------ --------------\n");
        std::vector<float> hpart(4096);
        double spread_lo = 1e308, spread_hi = -1e308;
        for (int blocks : {1, 4, 16, 64, 256, 1024}) {
            blocked_dot_fp32<<<blocks, kRedBlock>>>(da, db, n, dpart);   // warmup
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.begin();
            blocked_dot_fp32<<<blocks, kRedBlock>>>(da, db, n, dpart);
            CUDA_CHECK(cudaGetLastError());
            total_ms += timer.end();

            CUDA_CHECK(cudaMemcpy(hpart.data(), dpart, sizeof(float) * blocks,
                                  cudaMemcpyDeviceToHost));
            float total = 0.0f;
            for (int i = 0; i < blocks; ++i) total += hpart[i];   // fp32 on purpose
            const double rel = std::fabs(static_cast<double>(total) - refd) / refd;
            spread_lo = std::min(spread_lo, static_cast<double>(total));
            spread_hi = std::max(spread_hi, static_cast<double>(total));
            printf("%8d %18.6f %14.3e\n", blocks, total, rel);
        }
        printf("\n  Spread across launch configurations: %.3e absolute. Nothing is broken and\n",
               spread_hi - spread_lo);
        printf("  no result is 'the' right one - fp32 addition is not associative, so the\n"
               "  partition of the sum is part of the answer. Consequences you will actually\n"
               "  hit: a reduction is not bit-reproducible across grid sizes, so a regression\n"
               "  test must use a tolerance, and 'the loss changed after I retuned the launch\n"
               "  config' is expected, not a bug.\n\n");

        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
        CUDA_CHECK(cudaFree(d16));
        CUDA_CHECK(cudaFree(d32));
        CUDA_CHECK(cudaFree(d64));
        CUDA_CHECK(cudaFree(dpart));
    }

    // ---- [7] catastrophic cancellation -------------------------------------
    {
        const int n = 4096;
        std::vector<float> hx(n);
        std::mt19937 rng(1);
        std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
        const float offset = 1.0e6f;          // large mean, small variance: the trap
        for (int i = 0; i < n; ++i) hx[i] = offset + noise(rng);

        float *dx = nullptr, *dnaive = nullptr, *dtwo = nullptr;
        double* dref = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, sizeof(float) * n));
        CUDA_CHECK(cudaMalloc(&dnaive, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dtwo, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dref, sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dx, hx.data(), sizeof(float) * n, cudaMemcpyHostToDevice));

        serial_variance<<<1, 1>>>(dx, n, dnaive, dtwo, dref);   // warmup
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.begin();
        serial_variance<<<1, 1>>>(dx, n, dnaive, dtwo, dref);
        CUDA_CHECK(cudaGetLastError());
        const float ms = timer.end();
        total_ms += ms;

        float naive = 0.0f, two = 0.0f;
        double ref = 0.0;
        CUDA_CHECK(cudaMemcpy(&naive, dnaive, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&two, dtwo, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&ref, dref, sizeof(double), cudaMemcpyDeviceToHost));

        // Uniform(-1, 1) has variance 1/3; the two-pass fp32 estimate should land
        // within sampling noise of it, which is O(1/sqrt(n)) ~ 1.5% at n = 4096.
        const double expected = 1.0 / 3.0;
        printf("[7] Catastrophic cancellation: one-pass variance (kernel %.3f ms)\n\n", ms);
        printf("  data          : %d values, x = %.0e + Uniform(-1, 1), so var(x) = 1/3\n", n,
               static_cast<double>(offset));
        printf("  E[x^2]-E[x]^2 in fp32 : %18.9f   <-- %s\n", naive,
               std::fabs(naive - expected) > 0.1 ? "garbage (sometimes negative!)" : "lucky");
        printf("  E[x^2]-E[x]^2 in fp64 : %18.9f   <-- the same formula, wider accumulator\n",
               ref);
        printf("  two-pass in fp32      : %18.9f   <-- %s\n", two,
               std::fabs(two - expected) < 0.05 ? "PASSED (within sampling noise of 1/3)"
                                                : "FAILED");
        printf("\n  Both fp32 sums are individually accurate to ~7 digits. The subtraction\n"
               "  destroys them: E[x^2] and E[x]^2 are both about 1e12 and agree to 12\n"
               "  digits, so their difference keeps almost no correct digits at all. The fix\n"
               "  is algebraic, not numeric - subtract the mean FIRST, while the operands are\n"
               "  still small. No amount of extra precision makes an unstable formula stable;\n"
               "  fp64 only moves the cliff.\n\n");

        CUDA_CHECK(cudaFree(dx));
        CUDA_CHECK(cudaFree(dnaive));
        CUDA_CHECK(cudaFree(dtwo));
        CUDA_CHECK(cudaFree(dref));
    }

    printf("total measured kernel time: %.3f ms (this lab is about numerics, not speed -\n"
           "see mixed_precision_gemm for the throughput side)\n", total_ms);
    return 0;
}
