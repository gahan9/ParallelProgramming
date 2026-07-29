# A09 Solutions

Reference answers for [`../exercises/`](../exercises/). Read them **after** your own
attempt — the reasoning is the payload, not the diff.

```bash
python 01_causal_mask_solution.py       # from this directory
python 02_online_rescale_solution.py

cd .. && make run-solutions             # from the module root: runs all of them
```

| # | File | What it demonstrates |
|---|------|----------------------|
| 1 | `01_causal_mask_solution.py` | Both halves of causal attention: the `min((pid_m+1)*BLOCK_M, n_ctx)` loop bound that skips future K/V blocks, and the `offs_m >= offs_n` diagonal mask. Prints the fraction of blocks visited and explains why the measured speedup lands below the ideal. |
| 2 | `02_online_rescale_solution.py` | The `(m, l, acc)` update with the algebra written out: why rebasing by `alpha = exp(m_old - m_new)` is exact, why every exponent is non-positive, and why the buggy version stays finite. Includes a two-block PyTorch walkthrough you can read the numbers off. |
| 3 | `03_flash_math_solution.md` | The full byte table (naive vs fused at N = 1k/4k/16k), the `1 + N/d` closed form, the cache and HBM breaking points, and the arithmetic-intensity limits `d/2` vs `N/2`. |

---

## The three findings worth carrying into a code review

1. **A correct causal kernel can still be a slow one.** Masking a score you have
   already loaded and multiplied saves nothing. The win is in the loop bound. If a PR
   adds causal support with no measured speedup, the block skipping is missing.

2. **A missing rescale does not crash.** It produces finite, plausible numbers biased
   toward whichever block held the largest scores. Any streaming reduction that
   carries a running max needs a test with *deliberately unbalanced* magnitudes —
   uniform random inputs will happily hide the bug, because the per-block maxima are
   all about the same.

3. **The traffic ratio is `1 + N/d`, and the dtype cancels.** Going from fp16 to fp8
   halves the bytes on both sides and does not change the argument for fusing at all.
   The lever is sequence length relative to head dimension.
