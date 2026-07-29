# Exercise 3 — Regime diagnosis

> Paper drill. Target time: **15 minutes** for all five. This is the drill that separates people who
> memorized the formulas from people who can use them.

Five situations, each with profiler-style evidence. For **each one**, answer in this order:

1. **Regime** — prefill/compute-bound, decode/bandwidth-bound, capacity-bound, or something else.
2. **Evidence** — which specific observation settles it. Name it; don't hand-wave.
3. **Fix** — the one or two optimizations that actually help, in priority order.
4. **Reject** — one plausible-sounding optimization that would *not* help here, and why.

At least two of these five have a tempting wrong answer. Don't pattern-match on keywords.

Reference numbers for the machine, where you need them: MI300X, 5.3 TB/s HBM, 192 GB, 1307.4 TFLOP/s
dense bf16, ridge point 247 FLOP/byte.

---

## Case A — The chatbot

A customer-support assistant. Median prompt 50 tokens, median response 500 tokens. Serving one
request at a time (batch 1) because the traffic is bursty and low-volume.

```
model                      13 B params, fp16  -> 26 GB weights
measured                   1 request at a time, 160 tokens/s
profiler, decode steps     achieved FLOP rate ......  4% of peak
                           achieved HBM bandwidth ... 78% of peak
                           top kernels: 7 GEMM shapes, all with M = 1
```

Before recommending anything, compute the batch-1 ceiling for this model and compare it to the
measured 160 tokens/s. That comparison changes what "the fix" even means.

---

## Case B — The training run

Pretraining, microbatch 8 × sequence 4096 = 32,768 tokens per optimizer step.

```
profiler, one step         achieved FLOP rate ...... 61% of peak
                           kernel time breakdown:
                             GEMMs .................. 71%
                             flash_attn_fwd/bwd ..... 7%
                             layer_norm ............. 8%
                             add / mul / silu ....... 9%
                             to_copy (dtype casts) .. 5%
```

An engineer proposes spending the next two weeks fusing the norms, activations, and residual adds.
Another says "we're at 61% of peak and compute-bound, there's nothing to gain from touching
memory-bound ops."

Who is right?

---

## Case C — The document summarizer

Users paste long documents. Median prompt **25,000 tokens**, median output 100 tokens. Complaints are
all about the wait before text starts appearing.

```
model                      8 B params, fp16  -> 16 GB weights
                           d = 4096, 32 layers, 32 heads, 8 KV heads
measured                   time-to-first-token ..... 2.6 s
                           inter-token latency ..... 4 ms  (fine)
profiler, prefill          attention kernels ....... 55% of prefill time
                           peak transient allocation  ~40 GB above weights
                           achieved FLOP rate ...... 20% of peak
```

Two things to explain before proposing anything: where the 40 GB transient comes from (compute it),
and why a *compute-bound* workload is only reaching 20% of peak FLOP rate.

---

## Case D — The high-throughput API

A batch-inference endpoint. Continuous batching keeps **256 sequences** decoding concurrently.

```
model                      34 B params, fp16 -> 68 GB weights
measured                   6,100 tokens/s aggregate
profiler, decode steps     achieved FLOP rate ...... 71% of peak
                           achieved HBM bandwidth .. 30% of peak
```

A vendor pitches int4 weight quantization: *"4× smaller weights, so up to 4× the decode
throughput."* Your director wants a recommendation by Friday.

---

## Case E — The one that doesn't add up

Decode serving, batch 32, 8 B model in fp16 (16 GB of weights), 2048-token contexts.

```
calculator ceiling         weights + 32 x 0.27 GB KV = 24.6 GB per step
                           5300 / 24.6 = 215 steps/s  ->  215 tokens/s per sequence
measured                   36 tokens/s per sequence   (6x below the ceiling)
profiler, one decode step  achieved HBM bandwidth .. 14% of peak
                           achieved FLOP rate ...... 3% of peak
                           timeline: ~450 kernels per step, each 6-30 us,
                                     with visible 8-15 us gaps between them
```

Both utilization numbers are low. What is the bottleneck, and what would you change?

---

## Synthesis question

Across all five cases, write down the **two measurements** you would ask for first, every time,
before offering any performance opinion at all. Justify each in one sentence.

---

Full working, including the reasoning for the rejected options:
[`../solutions/03_regime_diagnosis_solution.md`](../solutions/03_regime_diagnosis_solution.md).
