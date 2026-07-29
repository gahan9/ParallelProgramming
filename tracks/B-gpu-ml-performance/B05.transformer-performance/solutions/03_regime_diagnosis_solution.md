# Solution 3 — Regime diagnosis

Reference working for [`../exercises/03_regime_diagnosis.md`](../exercises/03_regime_diagnosis.md).

Machine reference: MI300X, 5.3 TB/s HBM, 192 GB, 1307.4 TFLOP/s dense bf16, **ridge point
247 FLOP/byte**.

The tempting wrong answers are in **cases A, B, and D**. If you got all three, you have the module.

---

## Case A — The chatbot

**Regime: decode / bandwidth-bound — and already near optimal for that regime.**

**Evidence.** The decisive observation is the pair `4% of peak FLOP rate` against `78% of peak HBM
bandwidth`. That is the textbook bandwidth-bound signature: the memory system is nearly saturated
while the matrix cores idle. The `M = 1` GEMM shapes confirm the mechanism — every projection is a
matrix-*vector* product, arithmetic intensity ≈ 1 FLOP/byte against a ridge point of 247.

**But first, the comparison the drill asked for:**

```
batch-1 ceiling = 5300 GB/s / 26 GB = 204 tokens/s
measured        =                     160 tokens/s   = 78% of the ceiling
```

The measured 78% of ceiling and the profiler's 78% of peak bandwidth are the *same number* — they
must be, because at batch 1 the step is nothing but the weight read. **This system is not broken.**
It is running at 78% of the hard physical limit for its regime, which is a good result for a naive
serving loop.

**Fix, in priority order.** Since you cannot beat 204 tokens/s at batch 1, the only options change
the regime or change the bytes:

1. **Batch requests.** The traffic is described as bursty and low-volume, so use **continuous
   batching** — as requests arrive, fold them into the in-flight decode batch rather than queueing
   them. With a 13 B model the weight read (26 GB) dwarfs a single sequence's KV cache by roughly two
   orders of magnitude at chat-length contexts, so the `batch × KV ≪ weights` condition from
   [solution 2 §5(c)](02_kv_cache_budget_solution.md) holds comfortably into the tens of sequences —
   aggregate throughput scales nearly linearly there, at almost no cost in per-user latency. This is
   the single biggest available win.
2. **Quantize weights to int8.** 26 GB → 13 GB doubles the ceiling to 408 tokens/s, and it works at
   *any* batch size, including batch 1. This is the only lever that improves single-user latency.

**Reject: fusing the elementwise ops (norms, RoPE, residuals, activation).** At batch 1 these move
~0.04% of the bytes the weights do ([README §3.3](../README.md#33-the-elementwise-tax-quantified)) —
a perfect fusion would be invisible. Also reject **buying a GPU with more FLOPs**: you are using 4%
of the FLOPs you already own.

> The senior-engineer version of this answer starts with *"before we optimize anything, this is at
> 78% of its theoretical ceiling — the problem is the regime, not the implementation."*

---

## Case B — The training run

**Regime: compute-bound overall — but the first engineer is right, and the second engineer's
reasoning is wrong.**

**Evidence.** `61% of peak FLOP rate` is a genuinely compute-bound training step. But look at the
kernel breakdown: `layer_norm (8%) + add/mul/silu (9%) + to_copy (5%) = 22%` of kernel time in
operations with **essentially zero FLOPs**. Those 22% are contributing nothing to the 61%; they are
pure overhead dragging the average down.

**Do the crossover arithmetic.** From [README §3.3](../README.md#33-the-elementwise-tax-quantified),
elementwise HBM traffic overtakes the weight read at roughly **2,300 tokens per pass**. This step
processes `8 × 4096 = 32,768` tokens — **14× past the crossover.** The elementwise ops here are not a
rounding error; they are moving more bytes than the entire weight read.

**Fix.** Fuse them, in this order:

1. **Turn on `torch.compile`** (or your framework's equivalent) before hand-writing anything. It
   fuses exactly this class of op automatically and will capture much of the 22% for a day of work
   rather than two weeks.
2. **Investigate `to_copy` (5%) specifically.** Dtype casts appearing as top-level kernels usually
   means a mixed-precision boundary in the wrong place — often an fp32 upcast inside RMSNorm that
   never gets fused back down. This is frequently the cheapest 3–5% in the whole profile.
3. **Hand-fused norm/activation kernels** only if 1 and 2 leave a meaningful gap. Note the ceiling
   on the whole effort: eliminating all 22% raises the FLOP rate from 61% to about
   `61 / 0.78 = 78%` of peak — a real ~28% step-time improvement, but not 2×. Size the two weeks
   against that number, not against a hope.

**Reject: the second engineer's argument.** "We're compute-bound so memory-bound ops don't matter" is
a category error. Being compute-bound describes where the *bottleneck* is for the FLOP-carrying work;
it says nothing about time spent in kernels that carry no FLOPs at all. Those kernels are precisely
why you are at 61% and not 85%. Also reject **fusing attention** — it is already flash-attention at
7% of the time, so the maximum possible win is under 7%.

---

## Case C — The document summarizer

**Regime: prefill / compute-bound, running badly — and above the attention crossover.**

**Evidence.** The complaint is TTFT, which is by definition the prefill metric; ITL at 4 ms is
healthy. So this is entirely a prefill problem, and the two numbers to explain are the 40 GB
transient and the 20% FLOP rate.

**Where the 40 GB comes from.** The materialized attention score matrix, per layer:

```
h × N² × bytes = 32 × 25,000² × 2 B
               = 32 × 625,000,000 × 2
               = 40,000,000,000 B  =  40 GB
```

That is the `O(N²)` term from [README §2.3](../README.md#23-the-quadratic-term-nobody-can-optimize-away),
and it matches the profiler exactly. It is allocated and freed *per layer* — which is the only reason
this runs at all, since 32 layers' worth would be 1.28 TB.

**Why a compute-bound workload only reaches 20% of peak.** Because a large share of its time is spent
in memory-bound work, not arithmetic. Writing 40 GB of scores, reading them back for softmax, writing
them again, and reading them once more for the `@ V` product is ~160 GB of HBM traffic **per layer**,
or ~5 TB per prefill — about 1 second of pure memory traffic at 5.3 TB/s, doing almost no FLOPs. That
is what "attention kernels = 55% of prefill time" is measuring, and it is why the aggregate FLOP rate
collapses.

Cross-check the FLOP share against theory. For this shape the crossover is
`N = p_layer/(2d) = 26,624` tokens, so at `N = 25,000` attention should be
`25,000 / (25,000 + 26,624) = 48%` of the FLOPs — and it is taking 55% of the time. Slightly *worse*
than its FLOP share, exactly as you would expect for the least efficient kernel in the layer.

**Fix, in priority order.**

1. **FlashAttention.** This is the whole answer. It tiles the computation so the score matrix never
   leaves on-chip memory (Dao et al. 2022), which simultaneously removes the 40 GB allocation and the
   ~5 TB of HBM traffic. Expect the attention share to drop sharply and the overall FLOP rate to move
   toward the 60%+ range.
2. **Chunked prefill.** Split the 25k prompt into chunks so a long prefill cannot monopolize the GPU
   and stall other users' decode steps. This does not reduce total work; it protects everyone else's
   ITL, and it also bounds peak memory. B06's subject.

**Reject: batching.** The complaint is TTFT for a single long prompt, and prefill is already
compute-bound at one sequence — batching adds proportional work for proportional time and makes TTFT
*worse*, not better. Also reject **weight quantization**: at 25,000 tokens per pass the 16 GB weight
read is amortized to 0.6 MB per token and is nowhere near the bottleneck.

---

## Case D — The high-throughput API

**Regime: compute-bound. The vendor's pitch is wrong for this workload.**

**Evidence.** `71% of peak FLOP rate` against `30% of peak HBM bandwidth` — the inverse of case A's
signature. Confirm it independently from the batch size:

```
arithmetic intensity at batch 256, fp16  ≈  256 FLOP/byte
machine ridge point                      =  247 FLOP/byte
```

At batch 256 this workload has crossed the ridge point. Continuous batching did its job so well that
decode is no longer bandwidth-bound. The profiler and the back-of-envelope agree, which is exactly
the confirmation you want before telling a director "no."

**The answer on int4.** Quantization reduces *bytes*. This workload is not limited by bytes — it is at
30% of bandwidth and 71% of compute. Halving or quartering the weight bytes attacks a resource that
is 70% idle. Amdahl's law bounds the win at the fraction of time that is actually memory-stalled,
which is small here. Worse, int4 weights typically require **dequantization to a compute dtype before
the GEMM**, which adds work on the axis that *is* saturated — so an int4 kernel can be measurably
**slower** than fp16 at high batch unless the kernel is very carefully written.

**What to tell the director:**

> "Not for this endpoint. Int4 gives roughly 4× on *bandwidth-bound* decode — batch 1 to ~32. We run
> at batch 256, where the profiler shows us at 71% of compute and 30% of bandwidth, so we are past the
> crossover and quantization attacks the resource we are not short of. It may even cost us throughput
> through dequantization overhead.
>
> Int4 *would* pay off in two places: it frees ~51 GB of HBM, which lets us raise the batch further or
> serve longer contexts, and it would substantially help a low-latency single-user endpoint if we
> offer one. I'd like to benchmark it for capacity, not for throughput — and I'd want an accuracy
> evaluation either way."

**Fix, if the goal is more throughput here.** Attack compute: fp8 matrix-core GEMMs (which is a
*different* argument from int4 weight-only quantization — fp8 raises the compute ceiling rather than
lowering bytes), fused attention, better GEMM tiling, and reducing FLOPs outright via speculative
decoding or a smaller model.

**Reject: raising the batch much further.** You are past the ridge point, so additional batch now
buys throughput roughly proportionally to added time — and it makes per-user latency worse. The easy
gains from batching are behind you.

---

## Case E — The one that doesn't add up

**Regime: nominally bandwidth-bound, but currently limited by neither bandwidth nor compute — it is
kernel-launch-overhead-bound.**

**Evidence.** The decisive clue is that **both** utilizations are low: 14% of bandwidth *and* 3% of
compute. Every regime in this module saturates *something*. When nothing is saturated, the GPU is
idle, and the timeline says exactly why:

```
~450 kernels per step, each 6-30 us of work
with 8-15 us gaps between them

gap fraction ≈ 11 us gap / (18 us work + 11 us gap)  ≈  38% of wall time doing nothing
```

Compounding it: kernels of 6–30 µs are *too small to reach peak bandwidth* even while running. HBM
needs enough concurrent requests in flight to hide its own latency, and a kernel that finishes in
6 µs barely gets started. So the 14% figure understates the problem — you lose time to gaps *and*
run inefficiently in between.

Cross-check that the diagnosis explains the whole 6× gap: 38% idle alone accounts for ~1.6×, and
small-kernel inefficiency plus per-step Python/host overhead across 450 launches accounts for the
rest. The story is consistent.

**Fix, in priority order.**

1. **Capture the decode step in a CUDA/HIP graph.** This is the canonical fix: it replays hundreds of
   launches as one host-side submission, eliminating per-kernel launch latency and the Python
   overhead between them. Decode is ideal for it — the shapes are static across steps, which is
   exactly graph capture's requirement.
2. **`torch.compile`** to fuse the many small elementwise kernels into fewer larger ones. Fewer
   kernels means fewer gaps *and* larger kernels that reach a higher fraction of bandwidth. Note this
   contradicts the advice in case A — because here the problem is the *count* of kernels, not the
   bytes they move. Same optimization, different justification.
3. **Then re-measure against the 215 tokens/s ceiling.** Once the gaps are gone you should be
   bandwidth-bound for real, at which point case A's playbook (more batch, fewer weight bytes)
   applies and not before.

**Reject: increasing the batch size, and quantizing the weights.** Both are correct treatments for a
genuinely bandwidth-bound decode loop, and both are useless here: the GPU is *idle* 38% of the time.
Making each step move fewer bytes does not fill a gap where no kernel is running. Fix the utilization
first, then optimize the thing that is actually saturated.

> This case is the reason [README §5.3](../README.md#53-measured-results--run-them-yourself) says a
> large gap between your measured tokens/s and the calculator's ceiling **is the finding**. The
> calculator tells you what physics permits; the profiler tells you why you are not getting it.

---

## Synthesis — the two measurements to ask for first

**1. Tokens per forward pass** (batch × new tokens per step).

This single number determines the regime before you look at anything else. Under ~100 you are
bandwidth-bound and the levers are bytes and batch; over a few thousand you are compute-bound and the
levers are FLOPs and fusion. Every optimization in this module is correct in one of those worlds and
wasted in the other, so asking anything else first risks a confident answer to the wrong question.

**2. Achieved FLOP rate *and* achieved bandwidth, as percentages of peak.**

Two numbers, but they are one measurement and useless apart. Together they give a three-way
classification that no single number can:

| FLOP rate | Bandwidth | Diagnosis | Action |
|---|---|---|---|
| low | **high** | bandwidth-bound (case A) | fewer bytes, more batch |
| **high** | low | compute-bound (case D) | fewer FLOPs, better GEMMs, fusion |
| low | low | **nothing is saturated** (case E) | fix utilization first: graphs, fusion, launch overhead |
| high | high | balanced, near the ridge | you are close to the machine's limit; re-examine the algorithm |

The bottom-left row is the one people miss, and it is the most common real-world finding in a
first-pass profile of a decode loop. Note also that case B is a "high/low" workload where the
right action still involved memory-bound ops — the table tells you where the bottleneck *is*, not
what every kernel in the profile is doing.
