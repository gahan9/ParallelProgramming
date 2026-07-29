# Exercise 3 — Build the cost case

> Reference answer: [`../solutions/03_build_the_cost_case_solution.md`](../solutions/03_build_the_cost_case_solution.md)
> — write yours first.

## The situation

You are asked to approve or reject an optimization program for an **internal document-classification
service**. Your job is to produce the business case, not the technology choice.

**The service**
- **18 qps**, steady, internal traffic only. No peak to speak of.
- ~900 compute-weighted tokens per request.
- Per-GPU throughput **2,000 tokens/s** at the latency target, **70% usable headroom**.
- GPUs cost **$2.80/hour** all-in.
- Latency is comfortably within SLO. **Nobody is complaining about speed.** This is purely a cost
  conversation.

**The proposals on the table** (multipliers measured by the team on this model and this hardware):

| Proposal | Effect | Engineering estimate |
|---|---|---|
| Exact-match cache | 40% hit rate (documents are re-submitted often) | 3 weeks |
| Better batching | ×1.25 throughput | 2 weeks |
| INT8 quantization | ×1.5 throughput | 6 weeks |
| Distilled student | ×2.5 throughput | 16 weeks |

**Organisational facts**
- A fully loaded engineer-week costs **$6,000**.
- Your planning horizon for infrastructure investment is **12 months**.
- The base model **will be replaced in roughly 7 months** by a company-wide platform migration.
  Anything coupled to the current checkpoint dies with it.
- The team of two is otherwise fully committed; this work displaces roadmap features.
- Finance has asked for "a 30% reduction in the ML serving line item."

## Your task

**1. Compute the baseline.** GPUs required, cost per 1,000 requests, cost per month. Show the
arithmetic, including the rounding to whole GPUs.

**2. Compute each proposal.** For each: GPUs required, new monthly cost, monthly saving,
engineering cost, and break-even in months. Present it as a table.

**3. Apply the model-lifetime constraint.** Re-evaluate every proposal against the fact that the base
model has ~7 months to live. Which proposals survive, and which are now obviously wrong? Be explicit
about which optimizations are **model-agnostic** (they survive the migration) and which are not.

**4. Answer the finance ask.** Can you deliver a 30% reduction? With which combination, at what
engineering cost, and by when? If you cannot, say so and say what you *can* deliver.

**5. Make the recommendation, including a "don't do it".** At least one proposal on that list should
be declined. Name it, and give the number that justifies the decision — not the intuition.

**6. Compute the opportunity cost.** These engineers are not free; the work displaces roadmap
features. State how you would frame that tradeoff to a director who does not care about GPUs, and
what information you would need from the product side to complete the argument.

**7. Sensitivity.** Which single input, if you are wrong about it, most changes your recommendation?
Give the value at which your answer flips.

**8. The uncomfortable question.** Suppose the platform migration slips by a year — a thing that
happens. Does your recommendation change? Say how you would structure the decision so that you are
not badly wrong in either world.

## Constraints on your answer

- Use `../tools/cost_model.py` to check your arithmetic; do the baseline by hand first.
- Every recommendation carries a number and a date.
- Explicitly separate **saving** (recurring) from **cost** (one-off). Mixing them is the most common
  error in these proposals and it always flatters the expensive option.
- End with the two sentences you would put in the email to Finance. They must be true, specific, and
  survivable if quoted back to you in six months.
