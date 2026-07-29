# A07 Solutions

Reference answers with reasoning. Try the [`../exercises/`](../exercises/) yourself first — the
learning is in the struggle, not the answer key.

This directory has its own `Makefile`:

```bash
make all GPU_ARCH=gfx950      # build every solution for your GPU
make run                      # build + run all; every one must PASS
make clean
```

| # | File | Notes |
|---|------|-------|
| 1 | `01_safe_softmax_solution.cpp` | Block max reduction with the `-INFINITY` identity + the same shift in every pass. |
| 2 | `02_masked_softmax_solution.cpp` | Additive mask in all three passes; `m == -inf` and `s == 0` guards for the empty row. |
| 3 | `03_online_softmax_math_solution.md` | Full derivation, monoid proof, fp16/bf16 thresholds, byte counts, FlashAttention bridge. |

## The three lines worth memorising

Everything in this module reduces to these:

```cpp
// 1. The max trick - mathematically free, numerically decisive.
y_i = expf(x_i - m) / sum_j expf(x_j - m);          // m = max_j x_j

// 2. The online recurrence - one pass produces both statistics.
m_new = fmaxf(m, x);
s     = s * expf(m - m_new) + expf(x - m_new);
m     = m_new;

// 3. The guard that makes -inf masking safe.
s = (m_new == -INFINITY) ? 0.0f : /* the update above */;
```

Line 2 with `x` replaced by a *tile* of scores, and a matching rescale applied to a running `O`
accumulator, **is** FlashAttention. That is the whole punchline of A07.
