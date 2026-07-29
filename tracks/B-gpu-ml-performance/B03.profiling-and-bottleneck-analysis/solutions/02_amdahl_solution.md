# Solution 2 — Amdahl's law: is this optimization worth doing?

```
S = 1 / ( (1 - p) + p / k )        ceiling as k -> infinity:  S_max = 1 / (1 - p)
```

---

## Part 1 — The synthetic inference step

*(Reminder: the 100 ms breakdown in the exercise is a constructed scenario, not
a measurement.)*

### 1. `attention_softmax` at 10×

```
p = 21.0 / 100.0 = 0.21,  k = 10
new step time = 100.0 - 21.0 + 2.1        = 81.1 ms
S             = 100.0 / 81.1              = 1.233x
saving        = 18.9 ms                   = 18.9% of step time
```

Check against the formula: `1 / (0.79 + 0.021) = 1 / 0.811 = 1.233`. ✔

### 2. The hard ceiling on that work

```
k -> infinity:  new step time = 79.0 ms,  S_max = 100 / 79 = 1.266x
```

The 10× version captures `18.9 / 21.0 = 90%` of all headroom that exists in that
phase. **This is the number that ends the conversation.** A colleague proposing a
further 2× on top of the 10× is fighting over the last 2.1 ms — 2.1% of the step —
no matter how brilliant the rewrite. Compute the ceiling *before* you fund the
follow-up, not after.

### 3. `layernorm` at 10×, and saving per engineer-week

```
p = 0.06, k = 10
new step time = 100.0 - 6.0 + 0.6 = 94.6 ms
S             = 100 / 94.6        = 1.057x        saving = 5.4 ms (5.4%)
```

| Proposal | Saving | Cost | Saving / engineer-week |
|---|---:|---:|---:|
| `attention_softmax` 10× | 18.9 ms | 3 weeks | **6.3 ms/week** |
| `layernorm` 10× | 5.4 ms | 2 weeks | 2.7 ms/week |

Fund **attention**: 2.3× the return per unit of engineering. Note that the
cheaper project is the worse project here — "it's only two weeks" is not an
argument, and rate-of-return is the comparison that survives review.

### 4. The best single 10× target

`gemm_qkv`, the largest phase:

```
p = 0.34, k = 10
new step time = 100.0 - 34.0 + 3.4 = 69.4 ms
S = 100 / 69.4 = 1.441x            saving = 30.6 ms (30.6%)
```

**Always rank targets by `p` first.** A mediocre speedup on the dominant phase
beats a heroic speedup on a small one — this is the single most valuable habit
Amdahl's law gives you.

### 5. Reaching 2× end to end

Target: step time ≤ 50.0 ms. Apply 10× greedily, largest `p` first:

| Step | Kernel | Saving (ms) | Running total (ms) | S |
|---|---|---:|---:|---:|
| 1 | `gemm_qkv` | 30.6 | 69.4 | 1.441× |
| 2 | `attention_softmax` | 18.9 | **50.5** | 1.980× |
| 3 | `elementwise_gelu` | 3.6 | 46.9 | **2.132×** |

**Two kernels are not enough** — the best possible pair lands at 50.5 ms, 1.98×,
agonizingly short. You need **three**, and the third can be the *smallest*
kernel in the table (`elementwise_gelu`, 4 ms): once the big two are done, any
third 10× win pushes you over. Choosing `layernorm` instead gives 45.1 ms
(2.217×) if it is cheaper to build.

Two things worth internalizing from that table:

- **Optimizing all five kernels 10×** gives `7.7 + 23.0 = 30.7 ms`, S = 3.26×.
- **Making every kernel take zero time** gives 23.0 ms, S = **4.35×**. That is
  the absolute ceiling on all kernel work combined, forever. Past 4.35× the only
  remaining work is the copy and the host gaps.

### 6. The 23% that is not kernel time

| Row | Technique | Why a counter profile never shows it |
|---|---|---|
| H2D copy, 18 ms | Overlap copy with compute (multiple streams + pinned host memory, double-buffered), or keep the tensor resident on-device across steps so the copy disappears entirely | Counters describe what happens *inside* a kernel while it executes. A copy is not a kernel. |
| Host gaps, 5 ms | Capture the step into a **HIP/CUDA graph** and replay it, so one host call launches the whole sequence instead of dozens; or batch/fuse small ops to cut launch count | These are the *gaps between* kernels. Every kernel can be at 100% Speed-of-Light while the GPU is idle half the wall clock. |

This is exactly why the module's workflow puts the **timeline before counters**.
A counter profile of this step would report five healthy kernels and completely
miss 23% of the step time.

---

## Part 2 — Real numbers, real disappointment

Figures from the `rocprofv3` run documented in
[A01 § 4](../../../A-gpu-programming/A01.foundations-and-programming-model/README.md#reading-the-rocprofv3-output).

```
device activity total = 2,357,053 (H2D) + 1,205,206 (D2H) + 44,242 (kernel)
                      = 3,606,501 ns
```

### 7. The kernel's share of device activity

```
p = 44,242 / 3,606,501 = 0.012268  =  1.23%
```

### 8. A 10× kernel speedup

```
new device time = 3,606,501 - 44,242 + 4,424 = 3,566,683 ns
S               = 3,606,501 / 3,566,683      = 1.011x
saved           = 39,818 ns                  = 39.8 microseconds  (1.1%)
```

A week of excellent kernel engineering bought **forty microseconds**.

### 9. The absolute best case, end to end

In the full summary the kernel is **0.014%** of captured time
(`hipMemcpy` 62.6% inclusive, `hipMalloc` 22.0%). Therefore:

```
S_max = 1 / (1 - 0.00014) = 1.00014x
```

**Even an infinitely fast kernel changes end-to-end time by 0.014%** — less than
run-to-run noise. The week belongs to the 84.6% that is allocation and host↔device
transfer: pool/reuse the device buffers instead of `hipMalloc`-ing per call, use
pinned host memory, issue the copies asynchronously so they overlap with compute,
and best of all keep the data resident on the device so most copies never happen.

> A caveat worth stating out loud in a review: the summary's `PERCENT (INC)` is
> **inclusive**, so `hipMemcpy`'s 62.6% *contains* the HSA calls and the device
> copy it triggers. Those percentages overlap and do not sum to 100 — read them
> as "which operations dominate", not as a pie chart. The device-activity
> arithmetic in questions 7–8 uses the non-overlapping traces precisely to avoid
> that ambiguity. Getting this distinction wrong is the most common way to
> misread a `rocprofv3` summary.

### 10. The one-sentence version for a design review

> "The kernel is 1.2% of device activity and 0.014% of the profiled run, so even
> making it infinitely fast is a rounding error; the 85% that is allocation and
> host↔device copy is where the week should go."

Short, quantified, and it names the alternative. A review sentence that only says
"don't do that" loses; one that redirects the effort wins.

---

## Part 3 — When the arithmetic is not the whole answer

Amdahl's law scopes to **one run of one configuration**. Four situations where
the scope is the thing being argued about:

### 11. 2% here, 40% on a strategic model — **do it**

You measured the wrong workload. The right response is not to override the
arithmetic but to **re-run it on the configuration that matters**: `p = 0.40`,
`k = 10` gives `S = 1.55×`, a serious win. Then say so explicitly — "this is
justified by model B, not model A" — so the next person does not re-derive your
2% and cancel the work.

*Where this flips:* if the strategic model is a roadmap item that has slipped
twice and has no committed launch date, you are funding a speedup for a workload
that may never ship.

### 12. 5% on 20,000 GPUs — **usually do it**

Multiply before you decide: 5% of 20,000 GPUs is the equivalent of **1,000 GPUs
of capacity**, recurring, against a one-off engineering cost. Fleet-scale
recurring wins are the classic case where a percentage that looks trivial per-run
is enormous in aggregate.

*Where this flips:* if the change raises the tail-latency variance of a
latency-SLO service, a 5% mean throughput win that costs you the p99 is a net
loss. Measure the distribution, not just the mean.

### 13. 3% via unportable hand-tuned assembly — **usually don't**

Amdahl says 3%. The cost side is not three weeks — it is a permanent maintenance
liability: one architecture supported, a silent correctness risk on every new
GPU generation, and a file most of the team cannot review. **Include the ongoing
cost in the comparison, not just the build cost.**

*Where this flips:* if the kernel is *the* product (a vendor library kernel, a
published FlashAttention-class primitive) and the asm path sits behind a tested
portable fallback, 3% is worth owning — that is precisely what vendor library
teams do.

### 14. 2% on the critical path of a multi-GPU barrier — **do it, and re-measure `p`**

This one is a **measurement error, not a judgement call**. With a barrier, the
step time is set by the *slowest* rank, so a straggler's 2% is not 2% of one
GPU's work — it is 2% × (all ranks waiting). Your single-GPU profile computed
`p` against the wrong denominator.

The fix is methodological: profile the **collective**, not the rank. Trace all
ranks (`rocprofv3 --rccl-trace`, or `nsys` with NCCL tracing) and measure how
long ranks spend *inside* the barrier. If they are waiting, the straggler's `p`
in the distributed sense is far larger than 2%, and Amdahl now agrees with your
instinct.

**The pattern across all four:** when Amdahl's law disagrees with a strong
instinct, the usual reason is that `p` was measured over the wrong scope — wrong
model, wrong time horizon, wrong number of GPUs. Fix the scope, then re-run the
arithmetic. Do not discard the arithmetic.
