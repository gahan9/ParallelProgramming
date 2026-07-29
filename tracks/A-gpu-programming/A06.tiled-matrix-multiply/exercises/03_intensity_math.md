# Exercise 3 — Arithmetic-intensity drill

No code. Do the algebra on paper, then check
[`../solutions/03_intensity_math_solution.md`](../solutions/03_intensity_math_solution.md).

This is the single most transferable skill in the module: before you write a kernel, you can predict
whether it will be memory-bound or compute-bound, and how much tiling you need to change the answer.

---

## Setup

**Problem.** Square fp32 GEMM, `C = A · B`, with `M = N = K = 4096`. Row-major, no transposes.

**Device G** (a plausible data-center part; deliberately round numbers):

| Property | Value |
|---|---|
| fp32 vector (SIMT FMA) peak | **60 TFLOP/s** |
| HBM bandwidth | **3.0 TB/s** |
| Shared memory / LDS per CU or SM | **64 KB** |
| Max threads per block | **1024** |

**Definitions.** Arithmetic intensity `I = FLOPs / bytes moved between the GPU and HBM`.
The roofline **ridge point** is `I_ridge = peak FLOP/s ÷ peak bytes/s`. Below the ridge you are
memory-bound; above it you are compute-bound.

Assume the pessimistic model throughout: **no cache hits**, every global read goes to HBM. Question 8
asks you to criticise that assumption.

---

## Questions

1. **Total work.** How many floating-point operations does this GEMM perform? (Count one
   multiply-add as 2 FLOPs.) Express it symbolically in `N` first, then numerically.

2. **Ridge point.** Compute `I_ridge` for Device G in FLOP/byte. What does a kernel have to achieve
   to be compute-bound here?

3. **Naive kernel.** One thread computes one element of `C` and reads both operands from global
   memory.
   - How many bytes does *one thread* read, and how many FLOPs does it do?
   - What is the arithmetic intensity? Is it a function of `N`?
   - How many total bytes does the whole kernel move? Compare that to the 3 × 4096² × 4 bytes the
     three matrices actually occupy — what is the traffic amplification factor?

4. **Tiled kernel, symbolically.** A block computes a `T × T` tile of `C` by looping over `K/T`
   tile-pairs, staging a `T × T` slice of `A` and of `B` in shared memory each iteration.
   - How many global elements does one block load in total?
   - How many blocks are there?
   - Derive total global bytes for the whole kernel, then show that the intensity simplifies to
     **`I = T / 4` FLOP/byte** for fp32.

5. **Tiled kernel, numerically.** Fill in this table for Device G:

   | `T` | Intensity (FLOP/byte) | Total HBM traffic (GB) | Memory- or compute-bound? |
   |---|---|---|---|
   | naive | | | |
   | 16 | | | |
   | 32 | | | |
   | 64 | | | |

6. **Crossing the ridge.** Solve for the smallest `T` that makes the kernel compute-bound on
   Device G. Then check whether that `T` is *implementable* with one thread per output element:
   - How many threads would the block need?
   - How much shared memory would the two tiles need, and how many blocks stay resident in 64 KB?
   - State the conclusion this forces about one-thread-per-output designs.

7. **Register blocking.** Now let each thread compute a `TM × TN` micro-tile, so a block covers a
   `BM × BN` tile of `C` while looping over `BK`-deep slices.
   - Show that the global-memory intensity is `I = BM·BN / (2·(BM + BN))` FLOP/byte and confirm it
     does **not** depend on `BK`.
   - Evaluate it for `BM = BN = 64` and `BM = BN = 128`.
   - For `BM = BN = 128`, `TM = TN = 8`, `BK = 8`: how many threads per block, and how much shared
     memory? Does this fit the limits that defeated Q6?
8. **What breaks the model.** Once the kernel is above the ridge, HBM bandwidth is no longer the
   limiter. Name the next three resources that become the bottleneck, in the order you would expect
   to hit them, and say which profiler counter you would look at for each.

9. **Critique the assumption.** The "no cache hits" model overstates the naive kernel's traffic.
   Explain which operand the L2 cache actually captures reuse for and why the intensity is still
   nowhere near the ridge.

---

## Bonus — where matrix cores move the goalposts

Device G's fp32 SIMT peak is 60 TFLOP/s. Suppose its matrix cores (AMD **MFMA** / NVIDIA **Tensor
Cores**) deliver **8×** that throughput for `bf16` inputs with fp32 accumulation, while HBM
bandwidth is unchanged.

1. What is the new ridge point?
2. `bf16` operands are 2 bytes, not 4. Redo the tiled intensity derivation — what does `I = T / 4`
   become?
3. Combine the two: what tile size `T` do you now need to stay compute-bound? What does that imply
   about why production GEMM libraries use large tiles, deep software pipelining, and multi-level
   (block → warp → thread) blocking rather than the single-level tiling in this module?
