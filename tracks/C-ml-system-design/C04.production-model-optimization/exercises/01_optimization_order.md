# Exercise 1 — Order the optimization ladder

> Reference answer: [`../solutions/01_optimization_order_solution.md`](../solutions/01_optimization_order_solution.md)
> — write yours first.

## The situation

You own an **image-captioning API** used inside a larger product. It is both too slow and too
expensive, and the team has already written a distillation proposal.

Everything you know:

**Traffic and cost**
- **40 qps** average, with a **~2× daily peak**.
- ~320 compute-weighted tokens per request.
- Served on **16 GPUs** at **$3.20/GPU-hour** ≈ **$37,400/month**.
- Per-GPU throughput measured at **1,800 tokens/s** at the current latency target.
- The fleet is sized so that **average** load uses about **45%** of peak throughput, leaving room for
  the daily peak.
- Analysis of one week of the request log: about **5%** of requests are exact repeats of a request
  seen in the previous hour. The popularity distribution is close to flat — many distinct images.

**Latency**
- **SLO: p95 under 800 ms. Actual p95: 1,850 ms.** This is the reason there is a meeting.
- Breakdown of a p95 request: 60 ms preprocessing, **1,600 ms model**, 90 ms postprocessing,
  ~100 ms queueing.
- The serving path is **synchronous, one request per forward pass**. There is no batching. Someone
  built it that way in a hackathon and it shipped.

**Product**
- About **30% of requests are for images the product itself generated** from a catalogue of 8,000
  templates, so the *same* images recur constantly — but they arrive with a per-request ID embedded
  in the cache key.
- Caption quality is user-visible and the product team is nervous about any change to it.
- The base vision model is **due to be replaced in the next two quarters** with a newer one.

**Team**
- Two engineers, available for about a quarter.
- The distillation proposal estimates **14 engineer-weeks** and claims a "3× smaller, 2.5× faster"
  student, based on a published result for a different model.

## Your task

**1. Compute the ceiling for each candidate before proposing anything.** For at least three
candidate optimizations, state the *maximum* it could save (cost or latency) given the facts above.
An optimization whose ceiling is below your target is eliminated without further analysis.

**2. Order the ladder for this specific situation.** Not the generic ordering from the module — the
ordering *for these facts*, with a one-line justification each. Where your ordering differs from the
generic one, say why.

**3. Justify not starting with distillation.** Give the argument you would actually make in the
meeting, in a form that does not antagonise the person who wrote the proposal. Address the
"3× smaller, 2.5× faster" claim specifically — what is wrong with the evidence behind it?

**4. The 5% cache hit rate.** The team concluded from it that caching is not worth building. Explain
why that conclusion is probably wrong, name the specific bug in their measurement, and say what you
would measure to prove it.

**5. Separate the latency problem from the cost problem.** They are not the same problem and they do
not have the same answer. For each, state your top recommendation, and note where a fix for one makes
the other worse.

**6. Say what you would decline to do**, and why. At least one item.

**7. The "it depends on X" pivot.** Name the one fact that would most change your ordering, and say
how.

## Constraints on your answer

- You may use `../tools/cost_model.py` and `../tools/cache_simulator.py`. State the inputs you used
  and label the outputs as model results.
- Your recommendation must fit two engineers and one quarter. An ordering that cannot be executed is
  not a plan.
- One page.
