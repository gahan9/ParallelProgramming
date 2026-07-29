# B04 Solutions

Reference answers for [`../exercises/`](../exercises/). Read them **after** you have
attempted the drills — the value here is in the derivations, not the conclusions, and a
conclusion you did not derive will not survive the next review you have to defend it in.

```bash
make            # list targets
make all        # syntax-check the Python solution
make run        # run it (must complete cleanly)
make clean
```

| # | File | What it settles |
|---|------|-----------------|
| 1 | `01_format_selection_solution.md` | The range-then-precision-then-throughput method, worked through six scenarios, ending in a filled decision table. Includes the KV-cache "128 KiB per token" anchor and the fp8 FNUZ-vs-OCP factor-of-two trap. |
| 2 | `02_loss_scaling_solution.md` | The full underflow census (≈64% subnormal, ≈3% zeroed), `S = 2^16`, the unscale-before-clip ordering and both of its silent failure modes, and the fact that mixed precision costs the *same* 16 bytes/param as fp32 training. |
| 3 | `03_quantize_error_solution.py` | Working per-channel and group-wise int8 quantization, plus a run-time experiment showing that group-wise only contains outlier damage as well as its *alignment* allows. |

---

## The three results worth carrying out of this module

1. **bf16 beat fp16 for training on a range argument, not a precision one.** Two thirds of
   a realistic gradient tensor is subnormal in fp16 and ~3% of it rounds to exactly zero.
   Loss scaling is the machinery that exists to work around that; bf16 deletes the
   problem, and pays with 8× coarser precision near 1.0 that the fp32 master copy absorbs.

2. **Storage precision and accumulator precision are independent decisions, and they fail
   differently.** Narrow storage costs a bounded relative error set by the mantissa;
   a narrow accumulator costs you the result, because the error compounds over the
   reduction length. Every matrix core is built on that asymmetry: multiply narrow,
   accumulate in fp32.

3. **Quantization is a quality decision that happens to buy speed.** The scale is only as
   good as the largest value it must cover, which is why per-channel, group-wise, and
   outlier-aware schemes exist — and why the acceptance criterion is a task-level eval,
   not a loss curve, a weight-space norm, or a perplexity number.

## If your numbers disagree

- **Exercise 3** is deterministic (`np.random.default_rng(0)`), so your table should match
  the reference digit for digit. If it does not, check that your symmetric quantizer uses
  `[-127, 127]` (not `[-128, 127]`) and rounds with `np.rint` (banker's rounding) rather
  than `np.floor(x + 0.5)`.
- **Exercises 1 and 2** are arithmetic, so a disagreement is a real disagreement. The most
  common one: forgetting that the round-to-zero threshold is **half** the min subnormal,
  and forgetting that fp16 loses one binade at the top to `inf`/`NaN`.
- The GPU labs are **not** deterministic across machines and are not supposed to be. Lab 1
  section 6 exists to prove that the same fp32 reduction gives different answers at
  different block counts.
