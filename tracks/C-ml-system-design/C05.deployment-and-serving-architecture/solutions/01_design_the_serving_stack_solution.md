# Solution 1 — Design the serving stack (AskDocs)

A defensible answer, with the reasoning that produces it. Yours may differ; what should not differ
is that the arithmetic closes and every decision has a stated driver.

---

## 1. Serving pattern — and what was rejected

The request has **three stages with different characteristics**, so it gets three answers. This is
the first thing a strong answer does: it refuses to treat "the request" as one thing.

| Stage | Pattern | Driver |
|---|---|---|
| Retrieval (embed query → vector search → fetch passages) | **Online synchronous** | On the TTFT critical path, must complete before generation starts; the work is ~30 ms, well inside budget |
| Generation | **Streaming (SSE)** | The SLO is *time-to-first-token*, and full answers run 4–12 s. A blocking response would violate the perceived-latency requirement by an order of magnitude while being no faster in total |
| Document ingestion (chunk → embed → index) | **Async queue + worker** | Freshness requirement is 5 minutes, not 5 milliseconds. Nobody is blocked; embedding a large document is minutes of GPU work; a queue lets ingestion spikes become longer waits instead of an outage |

**Rejected, with reasons:**

- **Batch/offline scoring of answers** — the input space is unbounded (arbitrary questions over
  arbitrary document subsets). Precomputation is impossible. *Partial* precomputation is not: chunk
  embeddings are precomputed at ingest, which is exactly why retrieval is 30 ms rather than seconds.
- **Edge / on-device** — 26 GB of weights does not run on a B2B web client, the documents live
  server-side anyway, and the "no cross-tenant exposure" requirement is easier to enforce with
  server-side isolation than with a client we do not control.
- **WebSocket instead of SSE** — the stream is one-directional and short-lived. SSE is simpler,
  survives more corporate proxies, and reconnects natively. Choose WebSocket only if you later need
  client→server messages mid-generation (interruption, for instance).

> **Verify the streaming path end to end through the real ingress.** The most common way to build a
> streaming service that is not a streaming service is a gateway, CDN or corporate proxy that
> buffers the response body. Test through the production path, not against the pod.

---

## 2. Architecture

```
                       ON THE TTFT CRITICAL PATH
   ┌──────────────────────────────────────────────────────────────┐
   │                                                              │
Client ──▶ Edge/TLS ──▶ Gateway ──▶ AskDocs app ──▶ Router ──▶ LLM replicas (GPU)
             │           authn        │              │  version/tenant  │
             │           quota        │              │  canary split    │ SSE
             │                        ▼              └──▶ Fallback tier │ stream
             │                 Semantic cache                           │
             │                 (tenant+version keyed)                   ▼
             │                        │                             Client
             │                        ▼
             │                 Embed model (400 MB, CPU or shared GPU)
             │                        ▼
             │                 Vector store (per-tenant index)
             │                        ▼
             │                 Passage store (object storage + hot cache)
             │
             └──▶ Observability: traces (request ID from gateway), metrics,
                  prediction log (query, retrieved IDs, model version, tenant,
                  fallback level, TTFT, total tokens)

                       OFF THE CRITICAL PATH
   Upload ──▶ Ingest queue ──▶ Chunker ──▶ Embed workers ──▶ Vector index upsert
                                                             (5-minute freshness SLO)
```

**On the TTFT critical path:** edge, gateway, app, semantic cache lookup, query embedding, vector
search, passage fetch, router, replica queue wait, prefill, first token.
**Not on it:** ingestion, the full generation after the first token, prediction logging (fire and
forget), and analytics.

That distinction is the point of the diagram. Anything you can move off the critical path is worth
more than anything you can make 10% faster on it.

---

## 3. Latency budget — p99 per hop, target ≤ 800 ms TTFT

| Hop | p99 | Source |
|---|---:|---|
| Client → edge (TLS, geography) | 45 ms | **Measured** from RUM, 95th-percentile region |
| Edge → gateway (authn, quota) | 8 ms | **Measured** |
| Gateway → app | 2 ms | **Measured**, same zone |
| Semantic cache lookup (miss path) | 4 ms | **Measured** (Redis, same zone) |
| Query embedding (400 MB model) | 12 ms | **Measured** |
| Vector search, 10⁷-chunk tenant index | 35 ms | **Assumed** — largest tenant, HNSW, `ef_search` tuned. *Get the real number by replaying production queries against the largest tenant index; ask the retrieval owner.* |
| Passage fetch (20 chunks, hot cache) | 18 ms | **Assumed** — *measure the cold-cache path too; it is the risk* |
| Prompt assembly | 5 ms | **Measured** |
| App → router → replica (network) | 4 ms | **Measured** |
| Replica queue wait | 90 ms | **Target, enforced by the autoscaler** — this is the number the scaling design must hold |
| Batch formation | 15 ms | **Assumed** — server config; scheduler wait for batch slot |
| Prefill → first token (13B, ~3k prompt tokens) | 210 ms | **Measured** on the target GPU |
| SSE first flush + network back | 45 ms | **Measured** |
| **Total** | **493 ms** | |
| **Budget** | **800 ms** | |
| **Headroom** | **307 ms (38%)** | |

**Notes a reviewer should want:**

- Queue wait (90 ms) is the only line the autoscaler controls, and it is the second-largest term.
  That is why §4 exists and why the scaling signal must be queue depth.
- Prefill (210 ms) is the largest term and belongs to Track B
  ([B06](../../../B-gpu-ml-performance/B06.inference-serving-optimizations/README.md)) /
  [C04](../../C04.production-model-optimization/README.md) — prompt compression, prefix
  caching for the repeated system prompt, and chunked prefill are the levers, not anything in this
  module.
- The two assumed retrieval numbers (35 + 18 = 53 ms) are the weakest part of the budget. If the
  largest tenant's index is slower than assumed, headroom absorbs it — up to about 6× the assumed
  value before the budget breaks. Say that explicitly rather than hoping.

---

## 4. Scaling design

**Signal: pending requests per replica** (queue depth), exported by the inference server and
consumed by KEDA (or KServe/Knative concurrency, which is close enough).

Rejected: **CPU utilization** measures the wrong device — the CPU marshals bytes while the GPU
works, and sits near 15% at saturation. **GPU utilization** reports "a kernel was resident during
the sampling window", which a continuous-batching serving loop satisfies at *any* request rate; it
saturates long before the system does and carries almost no information about how far behind you
are.

| Parameter | Value | Reasoning |
|---|---|---|
| Target | 4 pending requests/replica | Chosen so queue wait ≈ 4 × per-request service slice ≈ 90 ms, matching the budget line |
| Floor | 6 replicas | Covers the overnight trough with the fallback headroom intact; never zero on a user-facing path |
| Ceiling | 26 replicas | Peak 90 rps ÷ (~5 rps/replica at target utilization), plus 30% burst room |
| Scale-down cooldown | 10 minutes | A 195 s cold start means thrashing is expensive; be eager up, lazy down |
| Scale-up step | unlimited up to ceiling | There is no reason to be gradual about a decision that takes 195 s to take effect |

**Warm-pool arithmetic.** Defend against the *ramp*, not the *spike*:

```
  observed morning ramp        R = 12 rps per minute
  cold start                   T = 195 s = 3.25 min
  per-replica capacity         C = 5 rps at target utilization

  warm replicas needed = R · T / C = 12 × 3.25 / 5 = 7.8  ->  8
```

But the brief also mentions **3× bulk-workflow spikes**, which are step changes, not ramps. A pool
of 8 cannot absorb a step from 30 to 90 rps (that needs ~12 additional replicas immediately). Lab 2
panel 1 shows exactly this: small warm pools barely move SLO violations against a step spike.

**So split the problem, which is the real answer:**

- **Warm pool of 4** for the diurnal ramp — this is the cheap, effective part.
- **Bulk workflows are announced**: they are triggered by a customer action in the product, so the
  application can call a pre-scale hook *before* the workflow starts, giving the autoscaler its 195 s
  in advance. **Converting an unpredictable spike into a scheduled one is worth more than any warm
  pool**, and it costs one API call.
- Anything not covered by either: admission control. Shed bulk-workflow requests before interactive
  ones — they are asynchronous by nature and a 30-second delay is invisible.

**The 30% cost reduction.** Four sources, largest first:

| Lever | Saving | What it costs |
|---|---:|---|
| Attack the cold start: bake weights into the image / node-local NVMe cache (140 s of the 195 s is the object-store pull) | Enables a smaller warm pool and a lower floor → ~12% | One engineering week; a larger image to distribute |
| Semantic cache on the query+tenant+version key, target 20% hit rate | ~15% | Staleness bounded by TTL; complexity in invalidation |
| Raise target utilization 70% → 78% | ~8% | Queue wait rises ~15 ms; consumes headroom |
| Batch-window tuning (+5 ms formation for larger batches) | ~5% | 5 ms of TTFT |

The first two are free of SLO cost and get most of the way there. Verify with
`python ../tools/capacity_autoscale.py --cold-start-s 55 --slo-ms 800` versus `--cold-start-s 195`:
in the shipped simulation, cutting the cold start improves SLO violations several-fold at
essentially flat cost, which is the argument to take to finance.

---

## 5. Caching

| Layer | What | Key | TTL | Invalidation |
|---|---|---|---|---|
| Exact-match answer cache | Full answer for an identical question over an identical document set | `sha256(tenant_id, doc_set_hash, normalized_query, model_version, prompt_template_version, retrieval_config_version)` | 1 h | On document change in the set; on model or template deploy |
| Semantic cache | Near-duplicate questions (embedding similarity ≥ 0.97) | same tuple, plus the embedding-model version | 15 min | as above |
| Passage cache | Chunk text by chunk ID | `tenant_id + chunk_id + chunk_version` | 24 h | On re-ingest |
| Prefix/KV cache | The shared system prompt inside the replica | server-managed | replica lifetime | on model reload |

**The two traps in the key, both mandatory:**

1. **`model_version`.** Without it, a rollback keeps serving the bad model's cached answers until the
   TTL expires. Users experience the incident continuing after you fixed it, and you will spend the
   first twenty minutes of the postmortem confused.
2. **`tenant_id`.** A semantic cache without the tenant in the key can serve tenant A's answer —
   containing A's document content — to tenant B. That is not a performance bug, it is a contractual
   and regulatory breach. It must be structurally impossible, not merely avoided: use per-tenant
   cache namespaces so a missing key component fails closed rather than leaking.

Also include `prompt_template_version` and `retrieval_config_version`. Anything that changes the
output must be in the key, or your cache is serving a model you are no longer running.

---

## 6. Multi-tenancy

**Pack, do not isolate — with hard controls.** 340 tenants against a 26 GB model: dedicating
replicas would need 340 × ~2 replicas minimum, which is absurd. Per-tenant traffic is spiky and low,
which is precisely the regime where packing wins.

| Control | Mechanism |
|---|---|
| Rate limit | Per-tenant requests/minute **and** tokens/minute at the gateway. Tokens matter more: one tenant with 30k-token prompts can consume the cluster while staying under a request limit |
| Concurrency cap | Max in-flight requests per tenant, so no tenant can occupy every batch slot |
| Queue fairness | Weighted fair queueing by tenant, not FIFO. A FIFO queue lets one bulk workflow starve 339 tenants |
| Cost attribution | GPU-seconds and tokens per tenant, logged per request |
| Burst allowance | Token bucket, so normal usage is unaffected and sustained abuse is throttled |

**Where cross-tenant exposure could happen — every place, enumerated:**

1. **Vector search** returning another tenant's chunks → per-tenant physical index (not a metadata
   filter on a shared index; a filter is one query-builder bug away from a breach).
2. **Semantic/answer cache** → tenant in the key, enforced by per-tenant namespaces.
3. **Passage store** → tenant-scoped object paths plus IAM, so a wrong chunk ID cannot resolve.
4. **Batching on the GPU** → requests from different tenants share a batch. This is *safe* for a
   stateless forward pass, but **not** if you use cross-request prefix sharing on anything other
   than the tenant-independent system prompt. Prefix caching must be scoped by tenant.
5. **Logs and traces** → the prediction log contains document content. Tenant-scoped retention and
   access control, and redaction before it reaches any shared analytics system.
6. **Error messages** → a stack trace or a "similar question" hint must never echo another tenant's
   content.

That enumeration is the deliverable. "We're multi-tenant safe" is not a design.

---

## 7. Failure behaviour

**Fallback chain:**

| Level | Behaviour | Quality | When |
|---|---|---|---|
| 1 | Full RAG answer from the 13B model | 1.00 | normal |
| 2 | Cached answer for a semantically similar question (≤ 1 h old), labelled as such in the UI | 0.70 | primary pool saturated or erroring |
| 3 | Retrieved passages with no generated answer — "here are the 5 most relevant passages" | 0.45 | generation unavailable but retrieval healthy |
| 4 | Keyword search results over the document set | 0.25 | vector store unavailable |
| 5 | Explicit "we can't answer right now", with the document set intact and a retry affordance | 0.10 | everything down. **Never a 500.** |

Level 3 is the one people miss and it is the best one on this list: retrieval and generation fail
*independently*, and passages-without-an-answer is genuinely useful for a document-search product.
Design the fallback around what the user actually wanted, not around what your stack happens to
still be running.

**Dependency policies:**

| Dependency | Timeout | Retry | Breaker |
|---|---|---|---|
| Vector store | 120 ms (p99 35 ms + 3.4×) | 1 retry, full jitter, 20 ms base | Open after 20 failures in 10 s; probe every 5 s |
| Passage store | 80 ms | 1 retry | Open after 20/10 s |
| Embedding model | 60 ms | 1 retry | Open after 10/10 s |
| LLM replica | 4 s to first token | **No retry** — go straight to level 2. A retry doubles the most expensive resource in the system exactly when it is scarce | Open after 5 consecutive timeouts per replica; eject the replica |
| Semantic cache | 15 ms | none — cache misses are free | Open fast; a cache must never be able to slow the system down |

Retry budget: retries ≤ 10% of requests, enforced globally. Every request carries a
client-supplied idempotency key; ingestion is idempotent on `(tenant, document_hash)`.

**Availability check.** Modelling the serial path as edge 99.999% × gateway 99.99% × router 99.95%
× app 99.95%, and the fallback chain as primary 99.5% / cache 98% / retrieval-only 99.5% / keyword
99.9% / static 99.99%:

```bash
python ../tools/degradation_planner.py --config askdocs.json --shared-fate 0.25
```

Result shape (run it to get your exact numbers): the fallback chain lifts the model path well past
four nines, at which point **the serial path is the binding constraint** — exactly the pattern the
built-in demo shows. The serial product alone is ~99.889%, which is **below the 99.9% target**. So:

- Remove the router as a separate hop, or make it a library inside the app tier. It buys 99.95%
  worth of risk for a routing table lookup.
- Run the gateway multi-zone active-active.
- With those two, the serial path clears 99.95% and the target holds with margin.

That is the value of doing the arithmetic: the fix is in a component nobody was worried about, and
you would not have found it by staring at the GPU.

---

## 8. What this design trades away

- **Packing tenants** maximizes utilization and accepts weaker isolation than dedicated hardware.
  The enumerated controls in §6 are what makes that acceptable, and they are load-bearing — if a
  contract later demands physical isolation for one large tenant, that tenant gets a dedicated pool
  and a different price.
- **A warm pool plus a floor of 6** means paying for idle GPUs every night. That is deliberate: the
  simulation says the alternative is a cold-start outage for the first user each morning.
- **Caching accepts staleness** up to an hour. For a document-QA product where documents change
  slowly, that is a good trade; for a product over live data it would not be.
- **The pre-scale hook couples the application to the serving layer.** It is an abstraction leak, and
  it is worth it: it converts the only unpredictable spike in the system into a scheduled one.
- **38% headroom in the latency budget is expensive.** It is also what lets two assumed retrieval
  numbers be wrong by 6× without a user noticing. Headroom is not waste; it is the price of an
  estimate you have not measured yet.

---

## What a reviewer should push back on

- The two assumed retrieval latencies are the weakest numbers in the budget. Replay production
  queries against the largest tenant index before committing to the SLO.
- "5 rps per replica" is a placeholder. Measure it at your real prompt-length distribution — it
  moves the ceiling, the floor, and the whole cost model.
- The fallback chain has never been exercised. Schedule a game day that fails the primary pool in
  production and verify that level 2 and level 3 actually engage and are labelled correctly in the
  UI. Until then, §7's availability number is a spreadsheet, not a property of the system.
