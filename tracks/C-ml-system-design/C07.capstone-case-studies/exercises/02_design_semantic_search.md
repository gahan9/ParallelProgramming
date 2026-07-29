# Exercise 2 — Design semantic search over a large document corpus

**Clock: 45 minutes.** Set a timer. Reference solution:
[`../solutions/02_design_semantic_search_solution.md`](../solutions/02_design_semantic_search_solution.md).

---

## The prompt

> Build search over an enterprise document corpus: **200 million documents**, averaging
> **3,000 words** each. Sources include wikis, tickets, code, chat archives, and PDFs.
>
> **5,000 queries per second** at peak. **p99 latency 300 ms.** Queries are a mix of
> keyword-like lookups ("VPN setup guide", an error code, a ticket ID) and natural-language
> questions ("why did the deploy fail last Tuesday").
>
> Documents change constantly: **2 million edits per day**, and a newly-created document must
> be findable within **5 minutes**. Access control is per-document and per-user, and it must be
> enforced correctly — a user must never see a snippet of a document they cannot open.
>
> Design the system end to end.

---

## Work the framework

Same ten steps, same clock. Reference:
[`../README.md` §2](../README.md#2-first-principles).

---

## What makes this prompt different

### Dense retrieval alone will lose, and you should say why

Embeddings are good at "why did the deploy fail last Tuesday" and reliably bad at exact-match
lookups: a specific error code, a ticket ID, a rare product name, a person's surname. A pure
dense system fails on the queries users are most confident about, which is the worst possible
place to fail — the user *knows* the document exists.

BM25 has the opposite profile. **Hybrid retrieval is not a hedge; it is the answer**, and a
strong design says which query types go to which path, how the two result lists are fused, and
what the fusion costs. Then a cross-encoder reranker over the fused top-N gives you the
precision that neither retriever has alone, at a cost you must budget.

### Chunking is a real design decision with real consequences

A 3,000-word document does not fit a single embedding usefully. So you chunk — and every
chunking choice trades something:

- **Chunk size.** Small chunks give precise matches and lose context; large chunks retain
  context and dilute the embedding.
- **Overlap.** Prevents an answer being split across a boundary; costs storage and creates
  near-duplicate results that then need deduplication.
- **Structure-awareness.** Splitting on headings/functions beats splitting on token count, but
  requires per-format parsing for wikis, code, chat, and PDFs — four different problems.
- **Granularity of the returned unit.** Do you return chunks or documents? If documents, how do
  you aggregate chunk scores? Max, mean, and top-k-mean give measurably different results.

Do the arithmetic on how many chunks 200 M documents produce. That number sizes your index and
it is bigger than people expect.

### Freshness is a hard constraint here, not a nice-to-have

Five minutes from edit to findable, at 2 million edits per day. Compare that to the
recommender's nightly index rebuild — you cannot rebuild a 200 M-document index every five
minutes, so the design needs an incremental path, and the incremental path needs a merge
strategy and a correctness story for deletions. Deletions are the part people forget: a
document that becomes inaccessible must stop being returned *immediately*, not at the next
merge.

### Access control interacts badly with ANN, and this is the interesting problem

This is the subtlest part of the prompt, and it is where a strong answer separates itself.

You cannot simply retrieve the top 100 and then filter by permission. If a user can access only
0.1% of the corpus, the top 100 will contain roughly zero results they are allowed to see, and
your search returns an empty page. But you also cannot filter *before* the ANN search, because
an HNSW graph is built over the whole corpus and does not know about your ACLs.

There are several real answers — over-fetching with a retry ladder, partitioning the index by
access group, or ACL-aware graph traversal — and each has a distinct cost. Pick one and defend
the cost. Also: **leaking through the result count or the latency is still leaking.** If a user
can tell from a 404-versus-empty distinction that a document exists, you have a problem.

---

## Questions to answer explicitly

**Requirements and scope**
1. What is the query mix, and how do you route it? Do you classify queries, or run both paths
   always?
2. What is a "result" — a chunk, a document, or a generated answer? What changes if the product
   wants generated answers?
3. What is your relevance definition, and where do labels come from?

**Retrieval design**
4. Do the arithmetic: 200 M documents × 3,000 words → how many chunks? At what chunk size?
5. Index size for the dense vectors at your chosen dimension and dtype. Show the number.
6. How do BM25 and dense results get fused? Name the method and its parameter.
7. How many candidates go to the reranker, and what does the reranker cost per query?

**Latency budget**
8. Build the per-hop budget to 300 ms p99. Include query embedding, both retrievals, fusion,
   reranking, snippet generation, and permission filtering. Leave a reserve.
9. Cross-encoder reranking is the expensive hop. Do the arithmetic: how many candidates can you
   afford, and what hardware does that imply at 5,000 QPS?

**Freshness**
10. What is the write path from "document edited" to "findable"? Name every stage and its
    latency.
11. How do deletions and permission revocations propagate, and how fast?
12. What happens when the embedding model is upgraded? Do the arithmetic on re-embedding the
    whole corpus.

**Access control**
13. Which of the three approaches above do you pick, and what does it cost?
14. How do you prove a user cannot infer the existence of a document they cannot read?

**Evaluation**
15. What is your offline metric, and where does the ground truth come from with no click log on
    day one?
16. What is the online metric, and what is your guardrail against a system that returns
    plausible-looking but wrong results?

---

## Do the arithmetic

| Quantity | Notes |
|---|---|
| Total chunks in the corpus | Documents × words ÷ chunk size, with overlap |
| Dense index size | Chunks × dimension × dtype bytes; then decide about quantization |
| Inverted index size | Rough estimate is fine; note it is much smaller than people assume |
| Query embedding cost at 5,000 QPS | A small encoder, but 5,000 times a second |
| Cross-encoder reranking cost | Candidates × cost per pair × QPS. This will dominate |
| Full re-embed cost | Chunks × encoder FLOPs — the number that decides your model-upgrade cadence |
| Cost per 1,000 queries | Split by component so the dominant one is visible |

The reranking row is the one that will surprise you, and it is the one that decides the
architecture. Compute it early.

---

## Common traps

- **All-dense retrieval.** It fails on exact-match queries, which are the ones users are most
  confident about.
- **Ignoring chunking**, or treating it as a preprocessing detail rather than a design decision
  with a measurable effect on relevance.
- **Reranking too many candidates.** Cross-encoders are ~100× the cost per item of a
  bi-encoder. Budget them like the expensive scan they are.
- **Filtering permissions after retrieval** without an over-fetch strategy — the empty-results
  bug, and it only shows up for users with narrow access.
- **No plan for embedding-model upgrades.** A new model means re-embedding everything, and a
  version mismatch between query and index silently destroys recall (the same trap as the
  recommender's tower version, §3.2.2 of the module README).
- **No relevance ground truth.** "We'll use click data" — from what, on day one?
