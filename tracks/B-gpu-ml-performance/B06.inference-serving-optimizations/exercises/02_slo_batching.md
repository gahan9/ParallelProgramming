# Exercise 2 — Turn an SLO into a batch-size cap

**Competency:** the single most common serving decision, and the one people get wrong by optimizing
throughput instead of goodput.

---

## The scenario

You own a chat endpoint. Product has written down a real SLO:

| Budget | Value | Why product cares |
|---|---|---|
| **TTFT** (p95) | ≤ **2000 ms** | The user must see *something* within two seconds. |
| **TPOT** (p95) | ≤ **12 ms** | ≈ 83 tokens/s — comfortably faster than reading speed. |

Offered load is **20 requests/s**. Prompts average ~550 tokens; outputs are lognormal with a mean
around 165 tokens and a long tail past 1000.

Your engine already does continuous batching with chunked prefill (512-token chunks). The one knob
you control today is `--max-num-seqs`: the **admission cap** on concurrent sequences.

### The per-step latency curve

This is the curve you would get from a decode-latency sweep on your own engine. Here it is generated
from the cost model in [`../sim/serving_sim.py`](../sim/serving_sim.py) at a nominal 2000-token
context per sequence — an analytic curve, not a hardware measurement, but it has the shape a real
sweep has:

| Batch size `B` | Decode step latency (ms) | Decode throughput (tok/s) |
|---:|---:|---:|
| 1 | 5.09 | 197 |
| 2 | 5.17 | 387 |
| 4 | 5.34 | 749 |
| 8 | 5.69 | 1406 |
| 16 | 6.38 | 2509 |
| 24 | 7.06 | 3398 |
| 32 | 7.75 | 4128 |
| 48 | 9.13 | 5259 |
| 64 | 10.50 | 6093 |
| 96 | 13.26 | 7242 |
| 128 | 16.01 | 7996 |

---

## Tasks

### 2.1 — Read the curve

The step latency is very nearly affine in `B`: `t_step(B) = a + b·B`. Fit `a` and `b` from any two
rows and say, in one sentence each, what physical cost each term corresponds to. Then explain the
number that makes batching worth doing at all: from `B = 1` to `B = 128`, by what factor does
throughput rise, and by what factor does per-step latency rise?

### 2.2 — The naive cap

One decode step produces one token for every sequence in the batch, so TPOT ≈ `t_step(B)`. Using
only the curve and the 12 ms TPOT budget, compute the **largest integer `B`** that fits. Show the
inequality.

### 2.3 — Now spend the TTFT budget

TTFT = queue wait + prefill time. A 550-token prefill at 47 µs/token is ~26 ms of GPU work.

- At 20 req/s, what fraction of every second is spent on prefill work alone?
- Chunked prefill splits a prompt into 512-token chunks, and each chunk rides along with a decode
  step. What does that do to TTFT, and what does it do to the *worst* inter-token gap seen by
  sequences already decoding? Which of the two is the reason chunked prefill exists?
- Your TPOT budget is 12 ms but the worst inter-token gap in a chunked-prefill engine is set by the
  chunk size. What chunk size would you pick if the SLO were on **p99 inter-token latency** rather
  than on mean TPOT?

### 2.4 — Predict, then check

Write down your predicted answers *before* running anything:

- (a) the batch cap you would configure;
- (b) whether goodput at that cap is closer to 100%, ~70%, or ~30% of offered load;
- (c) what happens to goodput if you set the cap to 128 instead.

Now run the sweep:

```bash
python ../sim/serving_sim.py --policies continuous_chunked --rate 20 \
    --slo-ttft-ms 2000 --slo-tpot-ms 12 --sweep-batch 4,8,16,24,32,48,64,128
```

### 2.5 — Explain the gap

The cap from §2.2 (pure latency curve) and the cap that maximizes simulated goodput are **not the
same number**, and the simulated one is smaller. Explain why — name at least three effects the
static latency curve leaves out. This is the heart of the drill: a latency curve gives you a
*necessary* condition, not the answer.

### 2.6 — The uncomfortable question

At your chosen cap, simulated goodput is well under 100% of offered load: a meaningful share of
requests miss the SLO no matter how you set the knob. Your manager asks for "more batching." Write
the three-sentence reply that a Principal Engineer would send — including the one number that shows
batch size is no longer the lever, and what the actual next step is.

---

## What a complete answer looks like

- §2.2 shows an explicit inequality, not a number pulled from the table.
- §2.5 names queueing delay, per-request TPOT variance vs the p95 aggregate, prefill interference,
  and context growth over a request's lifetime.
- §2.6 distinguishes *tuning* (free) from *capacity* (costs money) and picks the right one.
