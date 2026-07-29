# Exercise 1 — Design the serving stack

**Time:** 60–90 minutes · **Deliverable:** ~2 pages including one diagram and one latency table.

---

## The product

**"AskDocs"** — an in-app assistant for a B2B document-management SaaS. A user selects a document
set and asks a natural-language question; the assistant returns a written answer with citations
back to the source passages.

### Stated requirements

| Requirement | Value |
|---|---|
| Perceived latency SLO | **p95 time-to-first-token ≤ 800 ms**; full answer typically 4–12 s |
| Availability SLO | 99.9% monthly, measured as "a useful response was returned" |
| Traffic | 4 000 daily active users; peak 90 requests/second; 8× diurnal swing; occasional 3× spikes when a large customer runs a bulk workflow |
| Model | A 13B instruction-tuned model, ~26 GB in bf16, plus a 400 MB embedding model for retrieval |
| Retrieval | Per-tenant vector index; typical query retrieves 20 passages from a tenant index of 10⁴–10⁷ chunks |
| Tenancy | 340 enterprise tenants. Contractual requirement: **no tenant's document content may be exposed to another tenant** |
| Freshness | A newly uploaded document must be answerable within 5 minutes |
| Cost | Current spend is \$41k/month on GPUs. Finance has asked for a 30% reduction without an SLO regression |
| Cold start | Measured: 195 s from pod scheduled to first token, of which 140 s is pulling weights from object storage |

---

## Your task

Produce a design document with the following sections. Be specific and numeric.

### 1. Serving pattern (and what you rejected)

Choose from: online synchronous, async queue + worker, streaming, batch/offline, edge. State which
you chose **for each stage** of the request (retrieval and generation may differ), and give the
decision criteria from §2.1 of the module README that drove each choice. Explicitly name at least
one pattern you rejected and why.

### 2. Architecture diagram

Draw the request path from client to GPU and back. Include the gateway, router, caches, the vector
store, the model replicas, the fallback tier, and observability. Mark which components are on the
**critical path** for time-to-first-token and which are not.

### 3. Latency budget

Produce a per-hop table that sums to **less than 800 ms** with at least 30% headroom, at **p99 per
hop**. Every hop in your diagram that is on the TTFT critical path must appear. State which numbers
are measured, which are assumed, and where you would go to get the real value for each assumption.

### 4. Scaling design

- Which metric does the autoscaler consume? Justify it against the two obvious wrong answers.
- What are the floor, ceiling, target value, and scale-down cooldown?
- Size the warm pool. Show the arithmetic: you know the cold start (195 s), the peak rate (90 rps),
  and the diurnal shape. What ramp rate are you defending against, and what does the pool cost?
- Finance wants 30% off. Where does it come from? Show the tradeoff you are accepting, with numbers
  from `../tools/capacity_autoscale.py`.

### 5. Caching

What do you cache, at what layer, keyed on what, with what TTL and what invalidation? There is a
correctness trap here involving both model versions and tenancy — make sure your cache key survives
both a model rollback and a security review.

### 6. Multi-tenancy

340 tenants, one model. Do you pack tenants onto shared replicas or isolate them? What are the
per-tenant controls? What specifically prevents tenant A's content reaching tenant B — name every
place in your design where that could happen.

### 7. Failure behaviour

Write the fallback chain with a stated quality level for each step. For each dependency, give the
timeout, the retry policy, and the circuit-breaker behaviour. Then compute the resulting end-to-end
availability with `../tools/degradation_planner.py` and say whether you hit 99.9%. If you do not,
say what you would change.

### 8. What you are trading away

Three to five sentences. What is worse in your design than in an alternative you considered, and why
you accepted that.

---

## Constraints

- Do not name a product where a decision is required. "We'll use vLLM" is not a serving-pattern
  choice; "streaming generation over SSE, because TTFT is the SLO" is.
- Every latency number needs a source: measured, vendor-published, or assumed-and-flagged.
- If you need information the brief does not give you, **say what you need and who would have it**
  rather than silently assuming. That is a real skill and it is graded.

---

## Self-check before reading the solution

- [ ] Does my latency table actually sum to less than 800 ms, at p99 per hop, with headroom?
- [ ] Is my scaling metric something other than CPU or GPU utilization, and can I say why?
- [ ] Does my cache key contain the model version **and** the tenant?
- [ ] Does every dependency have a timeout, and is it derived from that dependency's p99?
- [ ] Does my fallback chain end in something that is never a 500?
- [ ] Did I compute the availability rather than asserting it?
- [ ] Did I say where the 30% cost reduction comes from and what it costs in SLO?

Reference answer: [`../solutions/01_design_the_serving_stack_solution.md`](../solutions/01_design_the_serving_stack_solution.md).
