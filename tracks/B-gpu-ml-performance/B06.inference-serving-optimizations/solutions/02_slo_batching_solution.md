# Solution 2 — Turn an SLO into a batch-size cap

Reproduce the simulated half with:

```bash
python ../sim/serving_sim.py --policies continuous_chunked --rate 20 \
    --slo-ttft-ms 2000 --slo-tpot-ms 12 --sweep-batch 4,8,16,24,32,48,64,128
```

---

## 2.1 — Read the curve

Fit from `B = 8` and `B = 128`:

```
b = (16.008 - 5.688) / (128 - 8) = 10.320 / 120 = 0.086 ms per sequence
a = 5.688 - 0.086 × 8           = 5.000 ms
t_step(B) = 5.00 + 0.086·B  ms
```

**What each term is.**

- `a = 5.00 ms` — the **fixed cost of one step**: stream every model weight from HBM once, plus
  kernel-launch and scheduler overhead. It does not depend on the batch at all. One sequence pays it
  alone; 128 sequences split it 128 ways.
- `b = 0.086 ms/sequence` — the **marginal cost per sequence**: read that sequence's own KV cache
  (which nobody else can reuse) and do its share of the arithmetic. This is the only part batching
  cannot amortize.

**Why batching is worth doing:**

| From B=1 to B=128 | Factor |
|---|---:|
| Throughput | 197 → 7996 tok/s = **40.6×** |
| Per-step latency | 5.09 → 16.01 ms = **3.1×** |

You buy 40× the throughput for 3× the latency. That ratio *is* the business case for batching, and
it is a direct consequence of `a ≫ b`: decode is bandwidth-bound on weights, so the expensive part is
already paid before the first sequence is considered. (This is the B05 result — decode is
bandwidth-bound, prefill is compute-bound — showing up as an engineering knob.)

Note the diminishing returns: the last doubling, 64 → 128, buys only 31% more throughput
(6093 → 7996 tok/s) for 52% more latency, because by then `0.086·B` dominates `a`. The bargain
expires exactly when the marginal term overtakes the fixed term, i.e. around `B ≈ a/b ≈ 58`.

---

## 2.2 — The naive cap

One decode step emits one token for every sequence in the batch, so a sequence's inter-token interval
*is* the step time: `TPOT ≈ t_step(B)`.

```
5.00 + 0.086·B ≤ 12
        0.086·B ≤ 7.00
              B ≤ 81.4
```

**B = 81.** Round down, never up: 82 would put the *typical* request over budget before any queueing,
variance, or prefill interference is accounted for.

---

## 2.3 — Spending the TTFT budget

**Prefill duty cycle.** A 550-token prompt at 47 µs/token ≈ **25.9 ms** of GPU work. At 20 req/s:

```
20 × 25.9 ms = 518 ms of prefill work per second  ->  ~52% duty cycle
```

Over half of every second is consumed by prefill before a single output token is produced. This is
the number people forget when they reason about decode batching in isolation: your decode steps only
get the other 48% of the GPU, which is why the achieved throughput in the sweep (≈2044 tok/s) is far
below the 4128 tok/s the `B = 32` row of the latency curve promises.

**What chunked prefill does to each metric.**

- **TTFT: slightly worse.** A 550-token prompt becomes two chunks (512 + 38), and each chunk pays the
  `a = 5.00 ms` fixed step cost again. You pay one extra step's overhead per extra chunk.
- **Worst inter-token gap for sequences already decoding: much better.** Without chunking, a prefill
  occupies a whole step and every decoding sequence sees a gap of `5.00 + 0.047 × prompt_len` ms — for
  a 4000-token prompt that is a **~193 ms freeze** in the middle of somebody else's answer. With a
  512-token budget the gap is capped near `5.0 + 0.047 × 512 ≈ 29 ms`.

**Which of the two is the reason chunked prefill exists: the second.** Chunked prefill is a
*tail-latency* mechanism. It trades a few milliseconds of TTFT for a bounded worst-case inter-token
gap. The simulator shows exactly this — plain continuous batching reaches `itl_max_ms ≈ 293 ms` while
chunked stays at `≈ 30 ms`, for identical throughput. Sarathi-Serve calls this "stall-free
scheduling" ([Agrawal et al., 2024](https://arxiv.org/abs/2403.02310)).

**If the SLO were on p99 ITL instead of mean TPOT:** the chunk size becomes the *primary* knob,
because it directly sets the worst-case stall. For a 12 ms p99 ITL budget:

```
5.00 + 0.047 · chunk + 0.086 · B ≤ 12
```

At `B = 32` (2.75 ms of decode marginal cost) the chunk budget is `(12 − 5.00 − 2.75)/0.047 ≈ 90`
tokens. That is a *very* small chunk, and it makes the tradeoff visible: an aggressive ITL SLO forces
tiny chunks, tiny chunks mean many steps per prompt, many steps mean more fixed-cost overhead per
prompt, and TTFT degrades. There is no setting that wins both — you are choosing a point on a curve,
which is the theme of the whole module.

---

## 2.4 / 2.5 — Predict, then explain the gap

Simulated sweep (`--rate 20`, TTFT ≤ 2000 ms, TPOT ≤ 12 ms, chunked prefill, seed 1234):

| max_batch | tok/s | p95 TTFT (ms) | mean TPOT | p95 TPOT | goodput req/s | goodput % |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 684 | 54586 | 5.63 | 6.80 | 0.31 | 7.7 |
| 8 | 1184 | 22991 | 6.34 | 7.76 | 1.00 | 14.3 |
| 16 | 1659 | 7714 | 7.96 | 10.48 | 3.06 | 31.3 |
| 24 | 1907 | 2829 | 9.66 | 13.46 | 7.82 | 69.7 |
| **32** | **2012** | **781** | **10.89** | **15.61** | **8.25** | **69.7** |
| 48 | 2044 | 242 | 11.56 | 17.00 | 7.30 | 60.7 |
| 64 | 2044 | 242 | 11.56 | 17.00 | 7.30 | 60.7 |
| 128 | 2044 | 242 | 11.56 | 17.00 | 7.30 | 60.7 |

**The shape to internalize.** Raw throughput rises monotonically and then flattens at 2044 tok/s.
**Goodput does not** — it peaks at `max_batch = 32` and then *falls* by 12% while throughput is flat.
Anyone tuning on the throughput dashboard would happily set 128 and ship a worse product.

- Below the peak the batch cap starves the engine: at `B = 4` p95 TTFT is **55 seconds**. Queueing
  delay, not compute, is destroying the SLO.
- Above the peak p95 TPOT crosses 12 ms and stays there. The extra admitted sequences make every
  step slower for everyone already in flight.

**Answer to (a):** `max_batch = 32`. **(b):** ~70% of offered load. **(c):** at 128, goodput *drops*
to 60.7% — you gain 1.6% throughput and lose 12% of your SLO-conforming traffic.

### Why 81 (curve) and 32 (simulated) differ

The latency curve gives a **necessary** condition, not the answer. It omits:

1. **Queueing delay.** The curve prices a step; it says nothing about how long a request waits before
   its first step. TTFT = queue wait + prefill, and queue wait is a *system* property that depends on
   arrival rate and service-time variance — invisible in any static sweep.
2. **Prefill stealing decode capacity.** §2.3: ~52% of the GPU goes to prefill at this load. The
   curve's throughput column assumes decode gets the whole machine, so it over-promises by ~2×.
3. **p95 is not the mean.** The SLO is on p95 TPOT, but the curve produces the *typical* step time.
   In the sweep, `B = 48` has mean TPOT 11.56 ms (inside budget!) and p95 TPOT 17.00 ms (outside).
   Tuning against the mean ships a service that fails one request in twenty.
4. **Context grows over a request's lifetime.** The curve is quoted at a nominal 2000-token context.
   A request that generates 1000 tokens ends with a longer context — and therefore a slower step —
   than it started with. Long requests are systematically slower than the curve predicts.
5. **Length variance.** A batch of 32 is not 32 identical sequences. The long tail of output lengths
   means some slots are occupied by 1000-token generations that hold KV and marginal cost for a long
   time.

**The transferable rule:** a latency sweep bounds the batch size from above. The *operating* point
comes from a load test against the real arrival process, scored on goodput. Never ship the sweep's
number.

---

## 2.6 — The reply to "more batching"

> The batch cap is already past its useful point: goodput peaks at 32 concurrent sequences, and
> raising it to 128 gains 1.6% throughput while dropping SLO-conforming traffic from 70% to 61%,
> because p95 TPOT crosses the 12 ms budget. At the best setting we complete 11.8 req/s against
> 20 req/s offered and only 70% of those meet the SLO, so this is a capacity shortfall, not a tuning
> problem — a second replica is the fix that moves goodput, and it is the only one that does.
>
> Before we buy it, two cheap things are worth a week: prefix caching (every request carries the same
> system prompt, so we are re-prefilling it thousands of times a day) and fp8 KV, which raises
> concurrency without touching per-step latency. If prefix cache hit rate is high, the ~52% of GPU
> time currently spent on prefill drops sharply and the capacity gap may close without new hardware.

**The one number that shows batch size is no longer the lever:** throughput is flat at 2044 tok/s
from `max_batch = 48` upward, and the engine completes only 11.8 of the 20 req/s offered. The engine is saturated; the admission cap has stopped being the
constraint. **The right next step** is either more capacity (replicas) or less work per request
(prefix caching) — not a bigger batch.

**Why this is the Principal-Engineer answer and "let's batch more" is not:** it distinguishes *tuning*
(free, already exhausted) from *capacity* (costs money, is the actual answer), quantifies both, and
proposes the cheap experiment before the expensive commitment.
