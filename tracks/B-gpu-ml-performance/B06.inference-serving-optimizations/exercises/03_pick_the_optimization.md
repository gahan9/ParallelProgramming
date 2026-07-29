# Exercise 3 — Triage: five symptoms, five levers

**Competency:** the actual job. An on-call page is a *symptom*; your value is in mapping it to a
mechanism and then to a fix, and in knowing what measurement would prove you wrong.

---

## How to answer

For **each** symptom below, write four things:

1. **Mechanism** — what is physically happening inside the engine. One or two sentences.
2. **Primary fix** — the single highest-leverage change, and roughly what you expect it to buy.
3. **Confirming measurement** — the *one* metric or experiment that would confirm this diagnosis
   before you ship anything.
4. **Falsifier** — what you would see if you were wrong, and which of the other symptoms it would
   turn out to be instead.

A diagnosis you cannot falsify is a guess wearing a lab coat. The falsifier is the part interviewers
are actually listening for.

---

## The symptoms

### 3.1 — "TTFT is fine, TPOT is terrible at high load"

Dashboards: p95 TTFT well inside budget and stable. p95 TPOT climbs from 8 ms at low traffic to
35 ms at peak. GPU is busy. No OOMs. Users describe it as "starts instantly, then types slowly."

### 3.2 — "GPU sits at 30% utilization with a full queue"

Dashboards: hundreds of requests queued, TTFT in the tens of seconds, and the GPU utilization
counter hovers around 30%. Scaling out to more replicas helps roughly linearly — which is a hint,
not a solution, because it means each GPU is being wasted identically.

> Two *different* mechanisms can produce this symptom. Name both, and say what single dashboard
> panel would tell them apart.

### 3.3 — "OOM, but only with long contexts"

The service is stable for days, then crashes with an out-of-memory error. Every crash correlates
with a document-summarization burst: prompts of 30k+ tokens. Halving the admission cap makes it go
away — and halves throughput for everyone, including the 99% of traffic with short prompts.

### 3.4 — "Throughput is fine but p99 is awful"

Aggregate tokens/s is at target and mean TPOT looks healthy at 9 ms. But p99 inter-token latency is
280 ms, and support tickets say the output "freezes for a moment" mid-answer. Nobody can reproduce
it on demand.

### 3.5 — "Cost per token is too high"

Finance has the number and it is roughly 3× the competitor's published price. Latency SLOs are
being met comfortably. The traffic is a customer-support assistant: every request carries the same
1800-token system prompt plus the conversation so far, and answers average 60 tokens.

---

## Then: rank them

You have one engineer for one quarter. Rank the five fixes by **expected value** — impact divided by
implementation and operational risk — and justify the top and the bottom of your ranking. Explicitly
state which one you would *not* do, and why "we could also add speculative decoding" is the wrong
answer to at least three of these five.

---

## Evidence you can generate

```bash
# Symptoms 3.1, 3.2, 3.4: batching policy, occupancy, and the ITL tail
python ../sim/serving_sim.py

# Symptom 3.3: what a 30k-token context does to the memory budget
python ../sim/kv_cache_planner.py --preset gqa-8b --gpu mi300x --context 32768 --mean-context 4100

# The lever that is NOT the answer to most of these
python ../sim/speculative_model.py --alpha 0.6 --k 4
```

Pay attention to the `itl_p99_ms` / `itl_max_ms` columns versus `tpot_mean_ms`. One of the five
symptoms is *invisible* in mean TPOT and obvious in the ITL tail; finding which is most of the
lesson.
