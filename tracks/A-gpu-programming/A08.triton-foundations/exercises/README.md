# A08 Exercises

Do these yourself before opening [`../solutions/`](../solutions/). Write real, running,
edge-case-correct code.

There is nothing to compile — Triton JIT-compiles on the first call — so run the files
directly, or drive them from the module root:

```bash
python 01_fused_relu_add.py          # from this directory
python 02_masked_mean.py

cd .. && make run-exercises          # both, from the module root
```

Both scripts refuse to run without a GPU and point you at
[`../../../../docs/SETUP.md`](../../../../docs/SETUP.md). Both ship with a placeholder
kernel that runs and reports **FAILED** — that gap is the exercise.

| # | File | Skill |
|---|------|-------|
| 1 | `01_fused_relu_add.py` | Fuse two elementwise ops into one kernel; quantify the HBM traffic you deleted. |
| 2 | `02_masked_mean.py` | Separate the bounds mask from the data mask; pick the right `other=` identity; survive the fully masked row without producing nan. |
| 3 | `03_triton_vs_cuda.md` | Written: what a Triton rewrite costs you in control, and what evidence you would demand in review. |

Each maps to an interview competency: **kernel-fusion reasoning** (1), **masking and
numerical edge cases** (2), **abstraction-tradeoff judgement** (3). Drill 3 is the one
that gets asked at senior level, and it is the one with no code.

---

## What the output should look like

No numbers are quoted here on purpose. Triton throughput depends on your GPU, your
ROCm/CUDA version, and your Triton version, and a table of invented GB/s would be worse
than none. What is stable is the **PASS/FAIL state and the direction of the comparison**
— that is what you check.

### Exercise 1 — before you solve it

The placeholder writes zeros, so the check fails while the timing looks great (writing
zeros is cheap):

```text
$ python 01_fused_relu_add.py
n                : 16777216   (67.1 MB per array)
torch (2 kernels): <ms>   <GB/s>   modelled traffic 5 units
triton (1 kernel): <ms>   <GB/s>   modelled traffic 3 units
HBM bytes saved  : 134.2 MB per call  (2 of 5 units, 40%)
speedup          : <x>
vs torch.relu(x+y): FAILED
```

> [!TIP]
> A fast kernel that reports FAILED is not a fast kernel. Check correctness first,
> every time — this is the single most common way a GPU benchmark lies to you.

### Exercise 1 — after you solve it

`FAILED` becomes `PASSED`, and the fused kernel should be **faster** than eager torch
while moving less memory. Expect the measured speedup to be **less than** the 40%
traffic saving predicts — launch overhead and cache hits on the intermediate both eat
into it. Report what you measured.

### Exercise 2 — before you solve it

The placeholder returns 0.0 for every row. That is coincidentally correct for the
fully masked row, which makes the failure instructive: two of the three checks pass:

```text
$ python 02_masked_mean.py
rows x cols      : 4096 x 1024
time / iter      : <ms>
effective BW     : <GB/s>
all finite       : True
empty row -> 0.0 : True
single kept elem : False
vs torch ref     : FAILED
```

### Exercise 2 — after you solve it

All four booleans read `True` / `PASSED`. The two traps to check you actually handled:

- Use `other=-float("inf")` instead of `0.0` on the `x` load and the sum is `-inf` for
  any row with padding — a wrong identity for a *sum*. (It is the *right* identity for
  the max in softmax. The identity belongs to the reduction, not to the habit.)
- Skip the denominator clamp and row 0 becomes `nan`, `all finite` flips to `False`,
  and the nan then propagates through everything downstream of this kernel.

### Exercise 3

Written drill, no program output. Fill in Parts A–D, then compare your reasoning with
[`../solutions/03_triton_vs_cuda_solution.md`](../solutions/03_triton_vs_cuda_solution.md).
Your line counts will differ if you chose a different kernel pair; the categories that
collapsed should not.
