# C07 Solutions

Reference answers for [`../exercises/`](../exercises/). Read these **after** you have written
your own version — a design you have struggled with for 45 minutes teaches you something a
design you have read does not.

| # | Solution | Covers |
|---|---|---|
| 1 | [`01_design_video_moderation_solution.md`](01_design_video_moderation_solution.md) | Base-rate arithmetic, a cascade whose last stage is people, the two feedback loops, fail-open/fail-closed |
| 2 | [`02_design_semantic_search_solution.md`](02_design_semantic_search_solution.md) | Hybrid retrieval, why HNSW loses at 2.4 B vectors, reranking as the cost centre, ACL-partitioned search |
| 3 | [`03_review_a_design_solution.md`](03_review_a_design_solution.md) | The five planted defects, ranked by expected damage, with fixes |

---

## How to read a solution

**Score yourself first.** Use the rubric in
[`../README.md` §9.4](../README.md#94-self-review-rubric), then read the solution, then score
again. The gap between the two scores tells you which of the ten dimensions you habitually skip,
and that is more useful than the solution itself.

**These are *a* defensible answer, not *the* answer.** Places where a different choice is
equally right are marked **[fork]**. Two designs can both be correct; what cannot vary is
whether the arithmetic is present and whether the assumptions are labelled. If your design
differs from the solution and you can defend the difference with numbers, you have done the
exercise correctly.

**What to compare, in priority order.** Do not compare architectures first — compare these:

1. **Did you find the same binding constraint?** In exercise 1 it is 200 reviewers, not model
   accuracy. In exercise 2 it is cross-encoder cost, not index size. Finding the wrong binding
   constraint means every subsequent decision was optimizing the wrong thing, and that is the
   single most valuable diff to notice.
2. **Are your numbers the same order of magnitude?** Not the same — the same order. A 2× gap is
   a different assumption; a 100× gap is a mistake, and it is worth finding which one.
3. **Did you state the same failure modes?** Architectures vary; the ways systems break do not.
4. **Did you reach step 10?** Most incomplete answers stop after serving. The tradeoff and
   omission section is where the seniority signal lives.

---

## The thread running through all three

Every solution here reaches the same shape of conclusion, and it is the module's real thesis:

- **The binding constraint is rarely the model.** It is 200 reviewers, a cross-encoder budget, a
  feature store's egress, or an all-reduce on a fabric. Find it first and the design writes
  itself; guess wrong and you optimize something that does not matter.
- **The most expensive component is usually not the GPU.** Human review is 92% of the moderation
  bill; the ANN fleet costs more than the reranker GPUs; the recommender's feature store outcosts
  its ranking cluster.
- **The loop is the thing.** Every system whose output becomes its own training data degrades
  silently unless something unbiased is measuring it — a randomized audit sample, an exploration
  budget, or a control holdout. It is always the cheapest line in the budget and always the
  first one someone proposes cutting.
