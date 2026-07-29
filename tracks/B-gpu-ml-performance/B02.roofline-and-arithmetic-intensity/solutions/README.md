# B02 Solutions

Reference answers with the reasoning, not just the results. Try the
[`../exercises/`](../exercises/) yourself first — for this module especially, the value is in
having done the derivation once under your own steam.

This directory has its own `Makefile`:

```bash
make all GPU_ARCH=gfx950                    # build every solution for your GPU
make run                                    # build + run all; every one must PASS
./03_place_your_kernel_solution.exe 78500   # pass your measured P in GFLOP/s
make clean
```

| # | File | Notes |
|---|------|-------|
| 1 | `01_classify_kernels_solution.md` | Nine operators with full FLOP/byte accounting; the thin-K GEMM that flips verdict depending on which ceiling you use; the three "roofline lies" cases. |
| 2 | `02_ai_of_attention_solution.md` | Full symbolic derivation of `I(s, d)` fused and unfused, the `s → ∞` limits, the crossover algebra, plus causal, decode, and GQA variants. |
| 3 | `03_place_your_kernel_solution.cpp` | The six accounting functions with the counting argument for each, and why a memory-bound kernel should be scored in GB/s rather than GFLOP/s. |

---

## The three things worth carrying out of this module

1. **`min(P, I × B)` is the whole model.** Two ceilings, one intensity, one comparison. If you
   can compute `I` you can predict which optimizations are even *possible*.
2. **Fusion usually does not change your classification — it deletes bytes.** Rows 4 and 5 of
   Exercise 1 are both memory-bound, 5× apart in attainable rate. The mistake is expecting
   fusion to make a kernel compute-bound; the win is that it finishes sooner.
3. **A point far below its own roof is not bound by that roof.** That is the answer to the most
   common ML-performance interview question, and it is the one place the simple roofline is
   silent: it has no axis for latency, occupancy, or launch overhead.

## Where the answers came from

Nothing in these solutions is a measurement. Every number is either arithmetic on the stated
"Device R" ceilings or an analytic FLOP/byte count you can rederive in a minute. The measured
numbers in this module live in `hip/measure_ceilings` and `hip/roofline_sweep`, where they are
computed on your hardware at run time — see [`../README.md`](../README.md) Section 5.
