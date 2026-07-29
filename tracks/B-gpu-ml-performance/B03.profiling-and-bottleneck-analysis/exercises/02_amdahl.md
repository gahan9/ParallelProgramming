# Exercise 2 — Amdahl's law: is this optimization worth doing?

**Competency:** prioritizing optimization work from a phase breakdown, and
saying *no* to a real speedup that does not matter.

No GPU needed. This is arithmetic and judgement — the two things that decide
whether a week of kernel work was well spent.

The law, in the only form you need:

```
S_end_to_end = 1 / ( (1 - p) + p / k )
```

where `p` is the fraction of end-to-end time spent in the part you are speeding
up, and `k` is the local speedup you achieve on it. Note the ceiling: as
`k → ∞`, `S → 1 / (1 - p)`. **You can never save time you were not spending.**

---

## Part 1 — A synthetic inference step

> The table below is a **hypothetical scenario constructed for this drill**, not
> a measurement from any machine. The shape is representative of a transformer
> inference step; the numbers are chosen to make the arithmetic clean.

One forward step takes **100.0 ms** end to end, broken down as:

| Phase | Time (ms) | Share |
|---|---:|---:|
| Host→device input copy | 18.0 | 18% |
| `gemm_qkv` | 34.0 | 34% |
| `attention_softmax` | 21.0 | 21% |
| `gemm_proj` | 12.0 | 12% |
| `layernorm` | 6.0 | 6% |
| `elementwise_gelu` | 4.0 | 4% |
| Host launch gaps / framework overhead | 5.0 | 5% |
| **Total** | **100.0** | **100%** |

Compute, showing your work:

1. A colleague has a FlashAttention-style rewrite that makes
   `attention_softmax` **10× faster**. What is the new end-to-end step time, and
   what is `S_end_to_end`? Express the win as a percentage of step time saved.
2. What is the **hard ceiling** on that work — the end-to-end speedup if
   `attention_softmax` became infinitely fast (`k → ∞`)? How much of the
   colleague's theoretical headroom does the 10× version already capture?
3. Someone else proposes a 10× speedup of `layernorm`. Compute `S_end_to_end`.
   The `layernorm` work is estimated at **two engineer-weeks**; the attention
   work at **three**. Rank the two proposals by *saving per engineer-week* and
   state which you would fund.
4. Which **single** kernel is the highest-value 10× target in this table, and
   what does that buy end to end?
5. You are asked for a **2× end-to-end** speedup. Assuming you can only apply the
   10× trick to whole kernels, what is the smallest set of kernels that gets you
   there? Show the running total. (If no set of kernel wins can reach 2×, say so
   and explain what that implies.)
6. **23% of the step is not kernel time at all** (copy + host gaps). Name one
   technique for each of those two rows that could reclaim most of it, and say
   why neither shows up as a "kernel optimization" in a counter profile.

## Part 2 — Real numbers, real disappointment

These figures are **real**, taken from the `rocprofv3` run documented in
[A01 § 4, "Reading the `rocprofv3` output"](../../../A-gpu-programming/A01.foundations-and-programming-model/README.md#reading-the-rocprofv3-output).
They are that run's *device-side, non-overlapping* activity for a `vector_add`
over 16.7M floats:

| Device activity | Duration (ns) |
|---|---:|
| `MEMORY_COPY_HOST_TO_DEVICE` | 2,357,053 |
| `MEMORY_COPY_DEVICE_TO_HOST` | 1,205,206 |
| `vector_add` kernel dispatch | 44,242 |

7. What fraction `p` of device activity is the kernel itself?
8. You spend a week and make `vector_add` **10× faster** — a genuinely excellent
   kernel result. What is `S` over device activity, and how many microseconds
   did you save?
9. In the same run's full `ROCPROFV3 SUMMARY`, the kernel is **0.014%** of
   captured time (`hipMemcpy` is 62.6% inclusive, `hipMalloc` 22.0%). Using that
   figure, what is the *absolute best case* end-to-end from any kernel work at
   all? What should the week have been spent on instead?
10. Write the one-sentence version of this result that you would put in a design
    review — the sentence that stops someone from starting that week of work.

## Part 3 — When the arithmetic is not the whole answer

Amdahl tells you the ceiling for **one run of one configuration**. Give a
concrete situation for each of the following where a "worthless" optimization by
the numbers above is still the right call, and one where it is not:

11. The kernel is 2% of *this* model's step time but 40% of a different,
    strategically important model's step time.
12. The 5% win is on a service that runs on 20,000 GPUs continuously.
13. The optimization is a 3% win that adds a hand-tuned assembly path with no
    portable fallback.
14. The 2% kernel is on the **critical path of a synchronization barrier** in a
    multi-GPU run, so every rank waits for the slowest one.

---

Worked answers, with the arithmetic shown line by line, are in
[`../solutions/02_amdahl_solution.md`](../solutions/02_amdahl_solution.md).
