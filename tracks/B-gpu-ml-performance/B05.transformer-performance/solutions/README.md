# B05 Solutions

Reference answers for [`../exercises/`](../exercises/), with the full working shown rather than just
the results — the reasoning is the transferable part.

| # | Solution | Headline results |
|---|----------|------------------|
| 1 | [`01_size_a_model_solution.md`](01_size_a_model_solution.md) | 19.5 G parameters, 39.0 GB in fp16, **136 tokens/s** single-stream on MI300X (**272** in int8), and the `12d²` shortcut overshooting by 16% |
| 2 | [`02_kv_cache_budget_solution.md`](02_kv_cache_budget_solution.md) | 4 concurrent 32k sequences with GQA, **zero** with MHA, and int8 weights as the one lever that improves concurrency *and* latency |
| 3 | [`03_regime_diagnosis_solution.md`](03_regime_diagnosis_solution.md) | Three of five cases have a tempting wrong answer; one is limited by neither compute nor bandwidth |

---

## Read these three things even if you got the numbers right

**1. `12d²` works for the wrong reason, so know when it stops working.**
Solution 1 shows GQA pulling a layer *down* by `1.67 d²` while SwiGLU (with `d_ffn = 2.67d`) adds
nothing, so the shortcut overshoots by 16%. In Llama-3-8B the same two features nearly cancel and it
lands within 8%. The rule is robust because two modern design choices happen to oppose each other —
check the `d_ffn/d` ratio and the KV-head count before you trust it on an unfamiliar model.

**2. The number a tool prints and the number you provision are different numbers.**
Solution 2's calculator says 5 concurrent sequences; the answer is 4, because a real runtime needs a
reserve the formula does not model. Both are correct for what they are. Being able to say *which one
you are quoting* is the difference between a capacity plan that works and one that OOMs in week two.

**3. "Both utilizations are low" is a diagnosis, not a measurement error.**
Solution 3's case E is the most common real finding in a first profile of a decode loop, and neither
of the two optimizations everyone reaches for (more batch, fewer bytes) helps. Every regime in this
module saturates *something*; when nothing is saturated, the GPU is idle and you have a launch
overhead problem.

---

## Reproducing the analytical parts

The drills are paper exercises, but every numeric answer is reproducible:

```bash
cd ..

# Solution 1
python tools/model_calculator.py -L 48 -d 6144 --heads 48 --kv-heads 8 \
    --ffn 16384 --glu --vocab 64000 --seq 8192 --gpu mi300x
#   ... then --dtype int8, --gpu h100, --kv-heads 48 for the variants

# Solution 2
python tools/model_calculator.py --preset llama2-70b --seq 32768 --batch 4 --gpu mi300x
python tools/model_calculator.py --preset llama2-70b --seq 32768 --batch 8 --kv-dtype fp8 --gpu mi300x
python tools/model_calculator.py --preset llama2-70b --seq 32768 --kv-heads 64 --gpu mi300x
```

Solution 3 needs no tooling — it is read from the two utilization percentages and the token count per
pass, which is the point.

> **A reminder on what these figures are.** Everything the calculator prints is a **derived estimate**:
> an analytical ceiling assuming perfect overlap, zero launch overhead, and 100% of vendor-published
> bandwidth. Real hardware lands below it, and case E is a worked example of *how far* below and why.
> For measurements, run [`../bench/`](../bench/) on a real GPU.
