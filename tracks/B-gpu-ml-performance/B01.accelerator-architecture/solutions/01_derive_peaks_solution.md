# Solution 1 — Derive the peaks yourself

Every published figure quoted here is linked in
[`../exercises/01_derive_peaks.md`](../exercises/01_derive_peaks.md) and in the
module README's Section 8, all verified as of **29 Jul 2026**.

---

## Task 1 — MI300X peak HBM bandwidth

The MI300X datasheet quotes a memory bus that "operates at **5.2 Gbps**". `Gbps` on
a per-pin basis is already a *transfer* rate — the DDR doubling is baked in. So:

```
bus width      = 8192 bits = 1024 bytes
transfer rate  = 5.2 x 10^9 transfers/s per pin
peak bandwidth = 1024 bytes x 5.2e9 /s
               = 5.3248 x 10^12 bytes/s
               = 5324.8 GB/s = 5.32 TB/s
```

**AMD publishes 5.3 TB/s.** Match.

If you had (wrongly) applied another ×2 you would have produced 10.6 TB/s — exactly
the error the drill warns about. The tell is the unit: `Gbps`/`GT/s` is post-DDR,
`MHz DDR` is pre-DDR.

Expressed the other way, the MI300X memory *clock* is 2600 MHz and
`2 × 2600 MHz × 1024 B = 5324.8 GB/s` — the same answer, which is why the
`device_query` lab prints the derivation alongside the result.

---

## Task 2 — H100 SXM5 peak HBM bandwidth

The Hopper whitepaper's table lists memory data rate as **2619 MHz DDR** — a clock,
so the DDR factor applies:

```
bus width      = 5120 bits = 640 bytes
transfer rate  = 2 x 2.619e9 = 5.238 x 10^9 transfers/s
peak bandwidth = 640 bytes x 5.238e9 /s
               = 3.352 x 10^12 bytes/s
               = 3352 GB/s = 3.35 TB/s
```

**NVIDIA publishes 3.35 TB/s.** Match.

Note the whitepaper prose says "over 3 TB/sec" while the product page says
"3.35 TB/s". Both are the same silicon; the prose is rounded conservatively. When two
official numbers differ, prefer the one with more significant figures *and* record
which document you took it from.

---

## Task 3 — MI300X peak FP64 vector

```
FLOP/s = clock x CUs x lanes/CU x FLOP/lane/clock
       = 2.1e9 x 304 x 64 x 2
       = 2.1e9 x 304 x 128
       = 81.69 x 10^12 = 81.7 TFLOP/s
```

**AMD publishes 81.7 TFLOP/s FP64 vector.** Match. The `× 2` is the fused
multiply-add: one FMA retires a multiply *and* an add, and the industry counts both.

---

## Task 4 — The factor-of-two puzzle

MI300X FP32 vector peak is **163.4 TFLOP/s**, exactly 2× the FP64 figure, even though
a lane is a lane and FP32 is *narrower* than FP64 (so you would naively expect
FP32 ≥ FP64, not exactly 2×).

The mechanism is **packed FP32**: CDNA exposes packed instructions (the
`v_pk_*_f32` family) that operate on a *pair* of adjacent FP32 values in one lane per
clock, giving 4 FLOP/lane/clock instead of 2.

```
2.1e9 x 304 x 64 x 4 = 163.4 x 10^12 = 163.4 TFLOP/s
```

**What it demands of the programmer:** the doubling is not free and not automatic in
the general case. Your data has to be laid out so that two FP32 values the kernel
wants to multiply-add sit in adjacent, aligned register halves, and the compiler has
to be able to *see* that (vectorized loads, `float2`-shaped access, no
pointer-aliasing ambiguity). A scalar `a[i] * b[i] + c[i]` loop that the compiler
cannot pack runs at the 81.7 TFLOP/s rate, not 163.4. This is the first instance of
the module's central theme: **a peak number is a claim about the hardware, and a
claim about your code.**

---

## Task 5 — H100 SXM5 peak FP32 and FP64 vector

FP32, using the non-tensor boost clock of 1980 MHz and 128 FP32 cores/SM:

```
2 x 132 SMs x 128 cores x 1.98e9 = 66.9 x 10^12 = 66.9 TFLOP/s
```

**NVIDIA publishes 66.9 TFLOP/s.** Match.

FP64 is published as **33.5 TFLOP/s**, exactly half. Solving for the core count:

```
33.5e12 = 2 x 132 x C x 1.98e9   ->   C = 64
```

So **64 FP64 cores per SM**, half the FP32 count — the classic 1:2 FP64:FP32 ratio
on NVIDIA data-center parts. (H100 *also* has an FP64 Tensor Core path at
66.9 TFLOP/s, i.e. the tensor unit does FP64 at the FP32 vector rate. That is a
different engine, which is exactly why the checklist in drill 2 asks "which
engine?".)

---

## Task 6 — MI300X fp16 matrix throughput

The CDNA 3 whitepaper's Table 1 gives 2048 fp16 matrix FLOP per CU per clock:

```
2.1e9 x 304 CUs x 2048 FLOP/CU/clock
       = 1307.4 x 10^12 = 1307.4 TFLOP/s = 1.31 PFLOP/s
```

which is the **1.3 PFLOP/s** AMD publishes for fp16/bf16 matrix (and 2.6 PF with
structured sparsity — a different claim, see drill 2).

As a multiple of the FP32 vector answer:

```
1307.4 / 163.4 = 8.0x
```

**Eight times.** And against the FP64 vector rate it is **16×**. On the *same silicon,
at the same clock*, the answer to "how many TFLOPs does this GPU have?" ranges from
81.7 to 2614.9 (fp8) — a **32× spread** — purely depending on precision and engine.
That spread is why the phrase "it does N TFLOPs" carries no information.

The per-clock numbers make the mechanism obvious: 128 FLOP/CU/clock for an FP64
vector FMA versus 2048 for an fp16 matrix instruction. A Matrix Core is not a faster
ALU; it is a **fixed-shape** unit that consumes small tiles of A and B and emits a
tile of C, amortizing one instruction issue over hundreds of multiply-accumulates. It
can only do that one shape.

---

## Task 7 — Ridge points

```
ridge = peak_matrix_FLOP_per_s / peak_bytes_per_s
```

Keep both in base units or the exponents will eat you.

**MI300X (fp16 matrix):**

```
1307.4e12 FLOP/s / 5.3248e12 B/s = 245.5 FLOP/byte
```

**H100 SXM5 (fp16 tensor, dense):**

```
989.4e12 FLOP/s / 3.352e12 B/s = 295.2 FLOP/byte
```

**Interpretation.** Both chips need roughly **250–300 FLOP per byte** moved before
the matrix units, rather than HBM, are the limit. For a GEMM in fp16 the arithmetic
intensity of a `K`-deep reduction is on the order of `K/2` FLOP/byte once you account
for 2 bytes per element and the 2 FLOP per MAC, so `K` has to be in the **many
hundreds** before either chip is genuinely compute-bound with a naive
(non-tiled) data path. Tiling through LDS/shared memory is what raises the effective
intensity far above that — which is precisely the A06 tiled-matmul lesson, and why
attention with short sequence lengths is memory-bound while a big training GEMM is
not.

The same arithmetic in fp8 pushes the ridge further right still (MI300X:
2614.9e12 / 5.3248e12 ≈ **491 FLOP/byte**), because halving the data type doubles
the compute peak *without* doubling bandwidth. **Every generation of low-precision
math makes the memory wall taller, not shorter.**

---

## Task 8 — The honest sentence

Any of these is defensible:

> "MI300X peak theoretical fp16 **matrix** throughput is 1.3 PFLOP/s **dense**
> (2.6 PFLOP/s only with 2:4 structured sparsity), against 163.4 TFLOP/s of fp32
> **vector** peak and 5.3 TB/s of peak HBM bandwidth — all per GPU, all peak
> theoretical, none of them measured."

> "Our GEMM budget assumes MI300X bf16 Matrix Core throughput, dense, per GPU:
> 1.3 PFLOP/s peak theoretical. We plan against 60% of that as achievable, and
> against measured HBM bandwidth (not the 5.3 TB/s peak) for the memory-bound layers."

What makes them honest is that each one names **precision, engine, sparsity
assumption, scope, and peak-vs-achievable**. Drop any one of the five and the
sentence becomes a number someone else will plan around and miss by 8×.

---

## The habit to keep

Every peak figure in this solution came out of a four-term multiplication over
numbers the vendor already published. That means two things:

1. You can **check any vendor claim in thirty seconds** — and you should, because
   secondary sources round, transcribe wrongly, and confuse sparse with dense
   constantly.
2. You can **derive the peak for hardware you have not seen yet** from a press
   release, before it ships, and know roughly where its ridge point will sit.

That is the whole reason this module comes before the roofline module.
