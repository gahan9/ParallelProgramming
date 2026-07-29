# Solution 2 — Semantic search over a large document corpus

Reference design for [`../exercises/02_design_semantic_search.md`](../exercises/02_design_semantic_search.md).

**A** defensible answer, not **the** answer. **[fork]** marks places where a different choice is
equally right.

---

## 0. Assumptions register

| # | Assumption | Value | If wrong |
|---|---|---|---|
| A1 | Corpus | 200 M documents, 3,000 words average (given) | Index size |
| A2 | Traffic | 5,000 QPS peak, 300 ms p99 (given) | Fleet, reranker width |
| A3 | Freshness | 2 M edits/day, 5-minute SLA (given) | Index tiering |
| A4 | Chunk size | 300 words, 50-word overlap → 250-word stride | Chunk count, relevance |
| A5 | Bi-encoder | 768-d output, ~110 M params, 384-token window | Index size, embed cost |
| A6 | Cross-encoder reranker | ~22 M params (distilled), 512-token window | **The dominant serving cost** |
| A7 | Query mix | ~55% keyword-like, ~45% natural-language | Retrieval routing |
| A8 | Access breadth | Median user can access ~2% of the corpus; some users < 0.1% | The ACL design |
| A9 | Cost | $3.00/GPU-hour, $1.20/CPU-host-hour | Cost model |

---

## 1. Requirements

**Functional.** Return ranked, permission-filtered results with snippets, for both keyword-like
and natural-language queries, over a corpus of wikis, tickets, code, chat, and PDFs.

**Non-functional.**

| Requirement | Target |
|---|---|
| Latency | 300 ms p99 |
| Throughput | 5,000 QPS peak |
| Freshness | New/edited document findable in ≤ 5 min |
| **Deletion / permission revocation** | **Effective immediately, not at next merge** |
| Correctness | A user must never see a snippet of a document they cannot open |
| Availability | 99.9% |

That fourth row is a hard correctness requirement wearing a freshness costume, and it needs a
different mechanism from the 5-minute indexing path — see §5.

**Out of scope:** the crawl/connector layer per source system, generated answers (RAG — noted in
§10), and query autocomplete.

---

## 2. Metrics

| Layer | Metric |
|---|---|
| Business | Time-to-answer for an information need; reduction in duplicate ticket creation |
| Product | Click-through rate at rank 1–3; **abandonment rate** (searches with no click); query reformulation rate; session success |
| Offline | nDCG@10 and recall@100 on a judged set; MRR for known-item queries |
| **Guardrails** | p99 latency; **zero-result rate**, sliced by access breadth; permission-violation count (must be 0); stale-result rate; per-source-type recall |

Two of those guardrails are specific to this system and easy to miss:

- **Zero-result rate sliced by access breadth.** The ACL design (§6) fails specifically for
  narrow-access users, and an aggregate zero-result metric will not show it because those users
  are a minority. Slice it or you will not see it.
- **Query reformulation rate.** A user rephrasing is a failure signal that CTR does not capture
  — they clicked nothing, so there is no negative click to learn from.

**Ground truth on day one, with no click log.** Three sources, in order of cost:
(1) **synthetic known-item queries** — take a document, generate a query it should answer, check
it ranks first; cheap, scalable, and biased toward the generator; (2) **~500 human-judged
queries** sampled from the real query log of whatever search exists today; (3) **weak labels
from existing signals** — a ticket's resolution links, a wiki page's inbound links. Start with
(1) and (2), switch the primary metric to click-derived once you have four weeks of traffic, and
**measure the correlation between the two before trusting the transition.**

---

## 3. Architecture — hybrid retrieval with reranking

```
QUERY
  │
  ├─▶ Query understanding: normalize, detect exact-match signals
  │   (quoted strings, error codes, IDs, filters)  ── 15 ms
  │
  ├──────────────┬──────────────────────────┐
  │              │                          │
  ▼              ▼                          ▼
BM25          Dense ANN                (filters: source, date,
(sharded)     (IVF-PQ, sharded)         author, ACL partition)
top 200        top 200                  ── 25 / 40 ms
  │              │
  └──────┬───────┘
         ▼
   Reciprocal Rank Fusion + near-duplicate dedup   ── 5 ms
         ▼
   Permission filter (partition-aware, §6)          ── 20 ms
         ▼
   Fetch chunk text for top 40                      ── 20 ms
         ▼
   CROSS-ENCODER RERANK, 40 pairs                   ── 65 ms   <- the expensive stage
         ▼
   Snippet generation + highlighting                ── 20 ms
         ▼
   Response
```

**Why hybrid is the answer rather than a hedge.** Dense retrieval handles "why did the deploy
fail last Tuesday" and reliably fails on `ERR_CONN_4021`, `TICKET-88431`, or a rare surname —
exactly the queries where the user *knows* the document exists, which is the worst place to
fail. BM25 has the inverse profile. Running both always and fusing is cheaper and far more
robust than building a query classifier and routing, because the classifier's errors land
precisely on the ambiguous queries where you can least afford them. **[fork]** If you are
latency-starved, route instead of fusing — but own that you have introduced a new failure mode.

**Fusion: Reciprocal Rank Fusion**, `score(d) = Σ 1/(k + rank_i(d))` with `k ≈ 60`. It needs no
score normalization between two systems whose scores are not comparable, has one parameter, and
is hard to get wrong. **[fork]** A learned fusion model does better once you have click data;
start with RRF and revisit.

---

## 4. The arithmetic

### 4.1 Chunks — the number that surprises people

```
Words in corpus     200e6 docs x 3,000 words              = 6.0e11 words
Stride (A4)         300-word chunks, 50-word overlap      = 250 words per chunk
Chunks              6.0e11 / 250                          = 2.4e9 chunks
```

**2.4 billion vectors.** That is 48× the recommender's 50 M item index, and it changes the index
technology (§4.2), not just its size.

### 4.2 Index sizing — and why HNSW is wrong here

```
Dense, fp32, 768-d   2.4e9 x 768 x 4 B                    = 7.37 TB     infeasible
Dense, int8          2.4e9 x 768 x 1 B                    = 1.84 TB     expensive
Dense, PQ 96 B/vec   2.4e9 x 96 B                         = 230 GB      workable

HNSW graph, M=32     2.4e9 x 32 x 4 B                     = 307 GB      larger than the vectors
```

The graph overhead exceeds the compressed vectors, which is the point at which HNSW stops being
the obvious choice. **Use IVF-PQ** (or a DiskANN-style disk-resident graph): coarse centroids
plus product-quantized residuals, with negligible structural overhead.

```
IVF-PQ index         230 GB + centroids  -> 32 shards      = ~7.5 GB/shard
x 3 replicas (tail latency + AZ)                           = 96 hosts

Inverted (BM25) index, compressed postings                 = ~400 GB, comparably sharded
```

**This is a genuine, defensible difference from the recommender** (module README §3.2.2), where
50 M vectors fit in 32 GB and HNSW was correct with no compression at all. Same problem shape,
two orders of magnitude more data, different right answer. Being able to say *why* the answer
changed is the signal.

Recall cost of PQ: expect a few points of recall@100 versus exact search. Recover it with a
**re-scoring pass** — fetch full-precision vectors for the ANN top-200 and re-score exactly
before fusion. That costs one extra storage tier and a few milliseconds, and it turns PQ from a
quality compromise into a pure speed/space win.

### 4.3 Reranking is the cost centre — compute it early

```
Cross-encoder FLOPs ~= 2 x params x tokens
                     = 2 x 22e6 x 512                     = 22.5 GFLOP per (query, chunk) pair

Top-40 rerank        40 x 22.5 GFLOP                      = 0.90 TFLOP per query
At 5,000 QPS         0.90e12 x 5,000                      = 4.5 PFLOP/s
GPUs at 300 TFLOP/s achievable                            = 15 GPUs for capacity
x2 for p99 headroom and AZ redundancy                     = 30 GPUs
Cost                 30 x $3.00                           = $90/hour
```

Compare the bi-encoder query embedding:

```
2 x 110e6 x 32 tokens = 7.0 GFLOP per query x 5,000 QPS   = 35 TFLOP/s  ->  ~0.1 GPU
```

**Encoding the query is free; reranking is 150× more expensive.** That ratio is the single most
important number in this design, and it means reranking depth is the primary cost/quality dial:

| Rerank depth | GPUs (with headroom) | $/hour | Latency |
|---:|---:|---:|---:|
| 20 | 15 | $45 | ~35 ms |
| **40** | **30** | **$90** | **~65 ms** |
| 100 | 75 | $225 | ~160 ms — blows the budget |

40 is chosen because it is the largest depth that fits both the latency budget (§4.4) and a
sane cost. **[fork]** Distilling the reranker further, or using a late-interaction model
(ColBERT-style) that precomputes token embeddings, changes this table substantially and is the
right next investment.

### 4.4 Latency budget, p99

| Hop | p99 |
|---|---:|
| Edge + auth | 10 ms |
| Query understanding + embedding | 15 ms |
| BM25 retrieval (sharded scatter-gather) | 25 ms |
| Dense ANN retrieval (IVF-PQ, 32 shards) + exact re-score | 40 ms |
| RRF fusion + near-duplicate dedup | 5 ms |
| Permission filter (incl. over-fetch retry allowance) | 20 ms |
| Chunk text fetch, top 40 | 20 ms |
| **Cross-encoder rerank, 40 pairs** | **65 ms** |
| Snippet generation + highlighting | 20 ms |
| Response assembly | 10 ms |
| Reserve for jitter, GC, one retry | 45 ms |
| **Total** | **275 ms** |
| Budget | 300 ms |
| **Headroom** | **25 ms** |

The two retrievals run in parallel, so their 25 ms and 40 ms overlap — the table charges the
larger, which is why the arithmetic works. Scatter-gather over 32 shards means the stage is
paced by the **slowest** shard, so shard-level p99 must be far better than 40 ms; budget the
tail, not the mean, or this row is a lie.

### 4.5 Full re-embed — the model-upgrade cadence

```
Per chunk    2 x 110e6 params x 384 tokens        = 84.5 GFLOP
Corpus       2.4e9 x 84.5e9                       = 2.03e20 FLOP = 203 EFLOP
GPU-hours    2.03e20 / 300e12 / 3600              = 188 GPU-hours
On 64 GPUs                                        = ~3 hours
Cost         188 x $3.00                          = $564
```

$564 of compute is not the constraint; the **IVF-PQ retraining and index rebuild over 2.4 billion
vectors is**, and that is a multi-hour to day-long job. Budget a **full working day** for an
embedding-model upgrade, build the new index alongside the old, and swap atomically.

**And the trap, identical to the recommender's tower-version trap:** query and index embeddings
must come from the same model version. A mismatch produces no error and silently destroys
recall. Stamp `embedding_model_version` on every shard and on every query vector; the search
service **refuses** a mismatch. Fail closed.

---

## 5. Freshness — three index tiers

2 M edits/day = 23 edits/s ≈ 24 M new chunks/day. A single index cannot absorb that at a
5-minute SLA, so tier it:

| Tier | Contents | Structure | Rebuild | Search cost |
|---|---|---|---|---|
| **Hot** | Last 5 min: ~6,900 edits ≈ 83,000 chunks | Flat, in memory, brute force | Every 5 min | 83,000 × 768 × 2 = 128 MFLOP/query — a few ms |
| **Warm** | Last 24 h: ~24 M chunks | Small HNSW | Hourly merge | Few ms |
| **Cold** | Everything else: 2.4e9 chunks | IVF-PQ, sharded | Nightly merge | 40 ms (§4.4) |

Every query fans out to all three and the results merge before fusion. The hot tier is brute
force because at 83,000 vectors that is *faster* than any index and needs no build step — the
right structure at that size is no structure.

**Deletions and permission revocations do not wait for a merge.** A tombstone bitmap, updated
synchronously on the write path and consulted at query time, removes a document from results
immediately. The vector stays in the index until the next merge; it just cannot be returned.
This is the mechanism that satisfies the correctness requirement in §1, and it is deliberately
separate from the indexing path because it has a different deadline: **seconds, not minutes.**

---

## 6. Access control — the interesting problem

**Why the naive approaches both fail.** Post-filtering: if a user can access fraction `f` of the
corpus and you want `k` results, you must retrieve ~`k/f`. At A8's narrow case (`f = 0.001`,
`k = 40`) that is 40,000 candidates — infeasible, and the user gets an empty page. Pre-filtering:
an IVF-PQ or HNSW index is built over the whole corpus and has no notion of your ACLs; you
cannot restrict traversal to a per-user subset without rebuilding the index per user.

**The design: partition by access group, with a post-filter fast path.**

Enterprise ACLs are heavily reused — most documents belong to one of a few thousand distinct
access sets (a team, a project, an org, "everyone"). Partition the index by access group; a
query searches only the partitions the user can reach, and every result is accessible by
construction.

```
Broad-access user (f > 20%): search the global index, post-filter, over-fetch 3x
                             -> cheap, and the empty-page failure cannot occur
Narrow-access user (f < 20%): search only their partitions (typically 5-50 of them)
                             -> results are accessible by construction, no over-fetch
```

Costs, stated plainly: documents in multiple access sets are indexed multiple times (measure the
duplication factor on your real ACL distribution — 1.3–2× is typical); ACL changes require
re-partitioning, handled by the tombstone bitmap in the interim; and you now operate two query
paths, which is real complexity.

**[fork]** ACL-aware graph traversal (filter during search) is elegant and has worse and less
predictable tail latency when the filter is very selective, because the traversal wanders
looking for admissible neighbours. Defensible; be honest about the tail.

**Preventing inference, which is the part people forget.** Correct filtering is not enough — the
*response* must not leak. Three rules: never return a total result count that includes
inaccessible documents; make the latency of "no accessible results" indistinguishable from "no
results at all" (pad to a fixed floor if necessary); and never surface an inaccessible
document's title in a "did you mean" or autocomplete path. A user who can distinguish
"doesn't exist" from "exists but you can't see it" has learned something, and in an enterprise
corpus document titles are frequently the sensitive part.

---

## 7. Evaluation and rollout

**Offline:** nDCG@10 and recall@100 on the judged set; MRR on synthetic known-item queries;
**per-source-type slices** (code, chat, and PDFs behave very differently and an aggregate metric
hides a regression in one of them); latency micro-benchmarks per stage.

**Interleaving before A/B.** For ranking changes, team-draft interleaving — mix results from
both rankers into one list and attribute clicks — is far more sensitive than an A/B test and
needs one to two orders of magnitude less traffic. At 5,000 QPS you have the traffic either way,
but interleaving gets you an answer in hours rather than a week. **[fork]** A/B is still
required for anything that changes the *experience* (snippets, layout, latency), because
interleaving only measures relative ranking preference.

**Rollout:** shadow (score both, serve the old, compare) → 1% canary with automatic rollback on
p99 latency, zero-result rate **sliced by access breadth**, and any permission violation → ramp.
The previous index and model stay warm for 72 hours; rollback is a config flag and is tested
monthly.

---

## 8. Monitoring

| Signal | Alert | Catches |
|---|---|---|
| p99 latency, **per stage** | Stage over budget | Which hop regressed, not just that something did |
| Zero-result rate, sliced by access breadth | > 5% for any slice | The ACL failure that aggregates hide |
| Reranker queue depth / GPU utilization | Sustained > 80% | The capacity wall approaching |
| Index freshness lag (edit → searchable) | > 5 min | The freshness SLA |
| Tombstone propagation lag | > 30 s | A **correctness** breach, not a quality one — page immediately |
| `embedding_model_version` mismatch count | > 0 | The silent recall killer; must be zero |
| Query-mix distribution | Shift | New usage pattern the routing/fusion was not tuned for |
| Click-through at rank 1–3, reformulation rate | Trend | Relevance regression |

**Degradation ladder:** (1) reduce rerank depth 40 → 20, roughly halving the reranker cost with
a modest relevance cost; (2) skip reranking entirely and serve fused results — noticeably worse,
still useful; (3) serve BM25 only, which keeps known-item search working, the queries users are
most confident about; (4) serve from the query-result cache with a staleness banner. Note the
ordering: **the dense path degrades before the lexical path**, because a search engine that
cannot find a ticket by its ID is more visibly broken than one that answers vague questions
poorly.

---

## 9. Cost

| Component | Hosts / GPUs | $/hour | Monthly |
|---|---|---:|---:|
| **Cross-encoder reranking** | **30 GPUs** | **$90.00** | **$64,800** |
| Dense ANN (IVF-PQ) | 96 CPU hosts | $115.20 | $82,944 |
| BM25 | 40 CPU hosts | $48.00 | $34,560 |
| Query service, fusion, snippets | 20 hosts | $12.00 | $8,640 |
| **Total serving** | | **$265.20** | **$190,944** |

```
At 50% average load: 2,500 QPS -> 9.0 M queries/hour
Cost per 1,000 queries = $265.20 / 9,000                    = $0.029
```

The ANN fleet costs *more than the GPUs*, which is the same lesson as the recommender: the
expensive part of a retrieval system is usually holding and traversing the index, not running
the model. The two levers that matter are rerank depth (§4.3) and index compression — and note
that halving rerank depth saves $45/hour while a 2× better reranker saves nothing.

---

## 10. Tradeoffs, 10×, omissions

| Choice | Bought | Paid |
|---|---|---|
| Hybrid BM25 + dense | Robust across both query types | Two indexes to build, operate, and keep in sync |
| IVF-PQ instead of HNSW | 230 GB instead of 2 TB+ | A few points of recall, bought back with exact re-scoring |
| Rerank depth 40 | Fits budget and cost | Recall@40 is a hard ceiling on final quality |
| Three-tier freshness | 5-minute SLA on 2.4e9 vectors | Three structures, one merge pipeline, three failure modes |
| ACL partitioning | Correct results for narrow-access users | 1.3–2× index duplication and two query paths |

**At 10× (2 B documents, 50,000 QPS):**

- The reranker becomes the single largest line item at ~300 GPUs. This is where a
  late-interaction retriever (ColBERT-style), which moves most reranking cost offline into the
  index, stops being a research option and becomes the architecture.
- The index goes disk-resident (DiskANN-style) — 2.3 TB of PQ codes no longer fits comfortably
  in RAM across a sane number of shards.
- **Multi-tenancy of the index becomes the hard problem**, not scale: per-tenant partitions
  multiply the shard count and destroy the batching that makes the reranker affordable.
- Freshness at 20 M edits/day likely forces a streaming index rather than tiered merges.

**Deliberately left out:** generated answers (RAG) — which is this system's output feeding case
study 1's prefill, and where the prefix-cache design in that doc becomes load-bearing because a
RAG prompt is a large, partly-shared prefix; personalized ranking (using the searcher's team and
history), which is a recommender problem bolted onto a search problem; multilingual retrieval;
and the source connectors, which are 80% of the real engineering effort in an enterprise search
project and 0% of what an interview asks about.
