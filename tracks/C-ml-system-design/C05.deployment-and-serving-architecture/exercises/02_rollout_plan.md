# Exercise 2 — Write the rollout plan

**Time:** 45–60 minutes · **Deliverable:** ~1.5 pages: a stage table, a trigger table, sample-size
arithmetic, and a rollback rehearsal plan.

---

## The situation

You own the ranking model for a marketplace search results page. The current model, `rank-v7`, has
been in production for four months. Your team has trained `rank-v8`, which differs in three ways:

1. A new architecture (two-tower → cross-encoder reranking of the top 100 candidates).
2. Three new features, two of which come from a **new** feature-store table populated by a pipeline
   that has been running for six weeks.
3. Inference cost per request is **2.3× higher** and measured p99 model latency goes from 34 ms to
   79 ms.

### What you know

| Fact | Value |
|---|---|
| Offline evaluation | NDCG@10 up 4.1% on a held-out week; the offline set is logged impressions from `rank-v7` |
| Traffic | 2.1 M search sessions/day, roughly uniform across the week with a 2.4× weekday/weekend swing |
| Primary business metric | Purchase-conversion rate per session, currently 3.4% |
| Secondary metrics | Search abandonment rate (currently 22%), seller-diversity index, revenue per session |
| Latency SLO | Search results page p99 ≤ 400 ms end to end; the model's 34 ms sits inside that |
| Error budget | 99.95% monthly availability; 12 minutes of budget remain this month |
| Serving | Kubernetes, 60 GPU replicas at peak, KServe-style traffic splitting available |
| Rollback | Currently: rebuild the image with the previous model URI and redeploy. Takes 22 minutes. |
| Team | Two engineers, one data scientist. Nobody is on call overnight. |

---

## Your task

### 1. Pre-flight gate

What must be true **before** any traffic reaches `rank-v8`? List the checks, and for each say
whether it is automated or manual and what a failure means. At least one of the three differences
above should worry you before you deploy anything at all — say which and why.

### 2. Stage plan

A table with one row per stage. For each stage give:

| Stage | Traffic % | Duration | Sessions observed | What it can detect | Exit criteria |
|---|---|---|---|---|---|

Include a shadow stage, and say honestly what shadow mode can and cannot tell you here.

Justify the durations. "24 hours" is not a justification; "24 hours because the weekday/weekend
swing is 2.4× and a 4-hour bake would only sample the evening peak" is.

### 3. Sample size

Your guardrail is purchase-conversion rate (3.4% baseline). You need to be able to detect a **1%
relative regression** (i.e. 3.4% → 3.366%) at α = 0.05 with 80% power.

Use `../tools/ab_calculator.py size`. Then answer:

- How many sessions per arm?
- At 2.1 M sessions/day, how long does a **1% canary** take to reach that? Is that acceptable?
- What does that tell you about which stages can actually make a quality decision, versus which
  stages are only checking that the system does not fall over?

This is the crux of the drill. Most rollout plans state canary percentages that are far too small to
detect the regression they claim to be guarding against.

### 4. Automated rollback triggers

A table with a numeric threshold, a sustain window, and an owner for each. Cover at minimum:

- error rate
- latency (and note the wrinkle: `rank-v8` is *expected* to be slower — how does your trigger
  distinguish "expected slower" from "broken slower"?)
- a quality guardrail

For each trigger, state what happens on breach: automatic rollback, page a human, or annotate the
dashboard. Not everything should page.

### 5. Rollback mechanics

The current rollback takes 22 minutes. Your error budget is 12 minutes. Resolve that. Describe the
target rollback mechanism, how long it should take, and how you will **verify** it — an untested
rollback is a hypothesis.

### 6. The overnight problem

Nobody is on call overnight. Does that change your plan? Say how.

### 7. What could make this rollout look successful and still be wrong

Name at least three mechanisms, using the module's §3.4 and §3.6 vocabulary. The feature-store detail
and the offline-evaluation detail in the brief are both there for a reason.

---

## Self-check before reading the solution

- [ ] Does every stage have a numeric exit criterion, not "looks good"?
- [ ] Did I compute the sample size rather than picking a percentage that felt safe?
- [ ] Is my canary large enough and long enough for the guardrail to have power — and if not, did I
      say so explicitly rather than pretending?
- [ ] Does my latency trigger handle the fact that the new model is legitimately slower?
- [ ] Is the rollback faster than my remaining error budget?
- [ ] Did I plan to *rehearse* the rollback, not just document it?
- [ ] Did I flag the risk in the six-week-old feature pipeline?

Reference answer: [`../solutions/02_rollout_plan_solution.md`](../solutions/02_rollout_plan_solution.md).
