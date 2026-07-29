# Solution 1 — Order the optimization ladder

Reference answer for [`../exercises/01_optimization_order.md`](../exercises/01_optimization_order.md).

**Headline: this is two problems wearing one hat, and they have different answers.** The cost problem
is solved on the cheap rungs — fixing the cache key alone is worth roughly a third of the bill. The
latency problem is *not* solvable on the cheap rungs, and the honest answer there is "the model is
1,600 ms of a 800 ms budget; either the model or the SLO has to move." Distillation is the wrong tool
for either, for a reason that has nothing to do with distillation: **the base model is being replaced
in two quarters.**

---

## 1. Ceilings, computed before proposing anything

Baseline: demand = 40 qps × 320 tokens = **12,800 tokens/s**. Usable per GPU = 1,800 × 0.45 = 810.
`ceil(12800/810) = 16 GPUs` → **$37,376/month** (16 × $3.20 × 730). Hardware utilization against peak
is `12,800 / (16 × 1,800)` = **44%**.

| Candidate | Ceiling | Verdict before any work |
|---|---|---|
| **Right-sizing the fleet** | 44% utilization *looks* like $20k/month of waste — but peak is ~2× average, so peak demand is 25,600 tokens/s against 28,800 of peak capacity. The headroom is **mostly justified**. Real slack: one GPU (15 × 1,800 = 27,000 > 25,600) ≈ **$2,336/month** | Small, but it is a config change. Take it. |
| **Caching** | bounded by the true repeat rate. The measured 5% would cap this at ~$1.9k/month — but the measurement is wrong (§4). The product statement that **30% of requests are catalogue templates** puts the real ceiling near **$11-12k/month** | Largest single item. Investigate first. |
| **Batching** | currently batch-1, so the multiplier could be large — but it is **unmeasured**, and it *increases* latency | High cost ceiling, negative latency effect. |
| **Quantization** | ×1.5-1.8 on throughput would take 16 GPUs to ~10, ≈ **$13k/month**; latency effect depends on whether this model is memory-bound | Real, but rung 3, and needs a gate. |
| **Distillation** | claimed 2.5× ⇒ ~$22k/month | Highest ceiling, and still the wrong project (§3). |

**Latency ceilings are the important half, and they are brutal:** the budget is 800 ms and the model
alone is 1,600 ms. Preprocessing + postprocessing + queueing total 250 ms, so **even eliminating
every non-model millisecond leaves you at 1,600 ms — 2× over SLO.** No amount of caching, batching, or
right-sizing can meet this SLO. That single calculation reframes the entire meeting, and it should be
the first slide.

## 2. The ordering for *this* situation

1. **Fix the cache key and build the exact-match cache** — the ceiling is the largest, the risk is
   zero (identical outputs), and it is model-agnostic so it survives the platform migration.
2. **Right-size by one GPU** — a day of work, $2,336/month, no risk. Do it while the cache work
   proceeds.
3. **Add batching to the serving path** — large unmeasured cost ceiling and it removes the hackathon
   architecture, but it makes latency *worse*, so it must be sequenced after the latency decision in
   step 4, not before.
4. **Escalate the latency SLO as a product decision** — see §5. This is not an engineering task and it
   is on the critical path for everything below it.
5. **Then, and only then, quantization** — the cheapest model-touching option, with a real chance of
   helping latency if the model is memory-bound, and it needs a quality gate.
6. **Not distillation.** See §3 and §6.

**Where this differs from the generic ladder:** generically, batching (rung 2) precedes caching
(rung 1) in ease of implementation. Here caching jumps ahead because its ceiling is ~3× larger and
because batching actively conflicts with the open latency problem. The generic ordering is a prior;
the facts update it.

## 3. The argument against starting with distillation

What I would actually say, and the tone matters because the proposal represents real work:

> *"The analysis in this proposal is good and I want to come back to it. But two facts kill it for
> this quarter. First, the base model is being replaced in two quarters — a student distilled from
> the current teacher has to be re-distilled and re-evaluated the moment that lands, so we would be
> buying a 14-week asset with a ~26-week life. Second, the '2.5× faster' figure comes from a
> published result on a different model and different hardware; we have no measurement on ours. If we
> want to know what distillation is worth here, the cheap version of that experiment is to measure
> whether a small off-the-shelf model can match our teacher on our traffic — a week, not a quarter."*

**What is wrong with the evidence specifically:** a speedup from another model/hardware/serving stack
is not transferable. Speedup depends on which resource binds (memory bandwidth vs compute), on batch
size, on sequence shape, and on the runtime. Also note "3× smaller" and "2.5× faster" are *different
claims* — compression ratio is not speedup (§3.5 of the module), and the proposal conflates them.

## 4. Why the 5% cache hit rate is a measurement bug

The product statement and the measurement contradict each other: **30% of requests are re-renders of
8,000 catalogue templates**, but the log shows 5% exact repeats. Both cannot be true unless something
is making identical requests look distinct.

**The bug is stated in the problem: a per-request ID is embedded in the cache key.** Every request is
therefore unique by construction, and the 5% is measuring incidental collisions, not reuse.

**What I would measure:** recompute the repeat rate over one week of the log after **normalising** the
key — hash the image bytes (or use the template ID) plus the parameters that actually affect the
output, and *exclude* request ID, timestamp, and session token. If normalised repeat rate comes back
near 30%, the cache is the biggest item on this list. If it comes back near 5%, my top recommendation
is wrong and I move batching to the front.

Modelled cost at a 30-35% hit rate (`cost_model.py`, analytic): demand drops to ~8,300 tokens/s,
`ceil(8300/810) = 11` GPUs → **$25,696/month, saving ~$11,700/month (31%)**. Roughly three weeks of
work, no quality risk. On the simulator, note that hit rate is a property of traffic skew, not cache
size: at low skew a bigger cache buys almost nothing, which is why the *key fix*, not the *cache
size*, is the whole project.

## 5. Cost and latency are different problems

| | Cost problem | Latency problem |
|---|---|---|
| Current | $37,376/month | p95 1,850 ms against an 800 ms SLO |
| Top recommendation | fix the cache key, then batch | escalate: the SLO cannot be met without changing the model or the SLO |
| Why | removes work entirely; no quality risk | 1,600 ms of model in an 800 ms budget is arithmetic, not effort |

**Where a fix for one hurts the other:**
- **Batching improves cost and worsens latency.** Requests wait to form a batch. With the SLO already
  violated by 2×, adding queueing delay is indefensible until the latency question is settled — which
  is exactly why I sequenced it after step 4.
- **Caching improves both, but not the metric you are judged on.** A 30% hit rate makes those requests
  ~150 ms, but p95 is determined by the 65-70% that still hit the model. Cache is a cost win and a
  *median* latency win; it does not move p95 at all. Saying this out loud prevents a predictable
  disappointment three weeks in.
- **Quantization can improve both** if the model is memory-bound — which is the reason it is the right
  next model-touching step rather than distillation.

**On the SLO itself:** the right conversation is with product. Is 800 ms a real requirement or an
aspiration someone typed in? If captions render asynchronously in the UI, 1.8 s may be fine and this
is a cost project only. If a user is waiting on it, the SLO stands and the model has to change. **You
cannot order the ladder correctly until someone answers that**, which makes it the highest-priority
item on the list and it costs nothing but a meeting.

## 6. What I would decline

- **Distillation this quarter.** 14 engineer-weeks (~$84k at $6k/week) against a base model with about
  two quarters left. Even taking the 2.5× claim at face value, the asset is obsolete before it
  amortises, and the claim is not measured on our stack.
- **Building a semantic cache.** It has been floated in the past for exactly this kind of workload.
  It is a quality risk (false hits return a caption for a *similar* image) that requires the full
  gate, and we have not yet earned the exact-match cache. Revisit after.
- **Buying more GPUs to fix latency.** More replicas reduce queueing, which is 100 ms of an
  1,850 ms p95. It cannot work, and it makes the cost problem worse.

## 7. The "it depends on X" pivot

**X is the normalised repeat rate in the request log** — the one measurement in §4.

- **Near 30%:** caching is the top item, worth ~31% of the bill, and the quarter is: fix the key,
  build the cache, right-size, then batch. Quantization becomes a next-quarter item.
- **Near 5%:** caching is worth ~$1.9k/month and is not worth three weeks. Batching moves to first
  (largest remaining ceiling, zero quality risk), and quantization becomes the second item —
  a materially different plan.

Either way distillation stays declined, because that decision rests on the model-replacement
timeline, not on the cache.

**What I would measure next, in priority order:** (1) normalised repeat rate over one week; (2)
whether the model is memory- or compute-bound at batch 1, which determines whether quantization can
help latency at all; (3) the achievable batching multiplier at the latency target. All three are days
of work, and together they turn every remaining question in this document into arithmetic.
