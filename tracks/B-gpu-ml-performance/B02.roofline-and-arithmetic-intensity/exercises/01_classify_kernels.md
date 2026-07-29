# Exercise 1 — Classify these kernels

No code. Do the arithmetic on paper (a calculator is fine), then check
[`../solutions/01_classify_kernels_solution.md`](../solutions/01_classify_kernels_solution.md).

This is the drill that turns the roofline from a picture into a tool. Every one of these
classifications takes under a minute once the habit is built, and it tells you which
optimizations are worth attempting *before* you write any code.

---

## The machine — "Device R"

Deliberately round numbers so the algebra stays visible.

| Property | Value |
|---|---|
| fp32 vector (SIMT FMA) peak, `P_v` | **60 TFLOP/s** |
| bf16 matrix-core peak, `P_m` | **1200 TFLOP/s** |
| HBM bandwidth, `B` | **4.0 TB/s** |
| Kernel launch overhead | **~4 µs** |

**Definitions.** Arithmetic intensity `I = FLOPs ÷ bytes moved between the GPU and HBM`.
The attainable rate is `min(P, I × B)`. The ridge point is `I_ridge = P / B`; below it a
kernel is memory-bound, above it compute-bound. Count one FMA as 2 FLOPs, one `exp` as
1 FLOP, and one `max` comparison as 1 FLOP.

Unless a row says otherwise, assume `N = 2^26` elements (~67.1M) and no cache reuse —
every global read comes from HBM. Question C3 asks you to attack that assumption.

---

## Part A — Compute the two ridge points

**A1.** Compute `I_ridge` for the fp32 vector ceiling and for the bf16 matrix-core ceiling.
State both in FLOP/byte.

**A2.** In one sentence, explain why the matrix-core ridge point being ~20× higher than
the fp32 one means *more* ML kernels are memory-bound on newer hardware, not fewer.

---

## Part B — Classify each kernel

For each row: write the FLOP count, the byte count, `I`, the attainable rate
`min(P, I × B)` in GFLOP/s, the verdict (memory-bound / compute-bound / near the ridge),
and **the one optimization you would try first**. Use the ceiling that matches the dtype.

| # | Kernel |
|---|--------|
| 1 | `saxpy` fp32: `y[i] = a*x[i] + y[i]`, N elements |
| 2 | `relu` fp32, out-of-place: `y[i] = max(x[i], 0)`, N elements |
| 3 | Sum reduction fp32: one scalar out of N elements |
| 4 | A chain of **5** separate elementwise fp32 kernels (1 FLOP each), each reading and writing the full N-element tensor |
| 5 | The same 5 operations **fused into one kernel** |
| 6 | Row softmax fp32, classic **3-pass** (pass 1 reads for the max, pass 2 reads + writes `exp`, pass 3 reads + writes the normalize), N elements total |
| 7 | fp32 GEMM, `M = N = K = 8192`, perfectly tiled so each of A, B, C crosses HBM exactly once |
| 8 | bf16 GEMM, `M = N = 8192`, `K = 128` (a thin output projection), same perfect tiling |
| 9 | bf16 GEMM, `M = N = K = 8192`, same perfect tiling |

**B10.** Two of the rows above have the *same* verdict but a **5× difference** in attainable
rate. Which two, and what does that tell you about what fusion actually buys you?

**B11.** Row 8 flips its verdict depending on which ceiling you compare it against. Show both
answers and say which one an engineer optimizing a bf16 projection layer should use.

---

## Part C — Where the model stops helping

**C1.** `vector_add` fp32 with **N = 4096** (not 2^26). Its arithmetic intensity is identical
to the large case. Compute the total bytes moved and the time that traffic would take at
4.0 TB/s, then compare that to the 4 µs launch overhead. Is the roofline verdict useful here?
What is the kernel actually bound by, and what would you change?

**C2.** A kernel has `I = 40` FLOP/byte (fp32) and measures **8,000 GFLOP/s**.
- What does the roofline say it should attain?
- What fraction of its roof is it reaching?
- Is it memory-bound, compute-bound, or neither? Name the three most likely causes and the
  profiler counter you would check for each.

**C3.** Rows 1–6 assumed no cache reuse. For the 3-pass softmax on a row of 4096 fp32 values
(16 KB), the row very likely stays resident in L2 between passes. Redo row 6's intensity with
respect to *L2* traffic instead of HBM traffic, assuming an L2 bandwidth of 12 TB/s. Does the
verdict change? What is the name for a roofline that carries one ceiling per memory level?

**C4.** You are told a kernel "hits 95% of roofline." Name two ways that statement can be true
and still describe a badly performing kernel.
