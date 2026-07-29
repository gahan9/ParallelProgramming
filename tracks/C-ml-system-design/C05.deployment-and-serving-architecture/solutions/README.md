# C05 Solutions

Reference answers with the reasoning, not just the answer key. Attempt
[`../exercises/`](../exercises/) first — these drills are about *producing* a design document, and
reading a good one teaches you very little about writing one.

| # | File | The point of the drill |
|---|------|------------------------|
| 1 | `01_design_the_serving_stack_solution.md` | Requirements → pattern → architecture → a latency budget that closes. The graded skill is arithmetic that adds up and a cache key that survives both a rollback and a security review. |
| 2 | `02_rollout_plan_solution.md` | The uncomfortable result: the canary percentages people write down are usually **orders of magnitude too small** for the guardrail they claim to be watching. A 1% canary needs 427 days to detect a 1% conversion regression. Stages that cannot decide must be labelled as such. |
| 3 | `03_ab_test_critique_solution.md` | Ten defects in one plausible-looking write-up. The headline: the p-value is 1.2 × 10⁻⁶ and the result is still uninterpretable, because a tiny p-value on a corrupted design is a precise measurement of the wrong thing. |

---

## Where reference answers stop being authoritative

These are *defensible* answers, not the only correct ones. Design has legitimate variation, and a
different answer with better-stated reasoning beats matching this one. Specifically:

- **Drill 1** — packing tenants versus isolating them is a genuine judgement call driven by
  contractual and risk factors the brief only partly specifies. Both answers can be right; only one
  of them can be *unjustified*.
- **Drill 2** — a slower, safer rollout is defensible for a marketplace and so is a faster one, if
  you say what you are accepting. What is not defensible is a stage plan whose guardrails have no
  statistical power and whose author has not noticed.
- **Drill 3** — the defect list is closer to objective. The mechanisms are not matters of taste, and
  the corrected plan has little room for variation.

## What is *not* negotiable in any answer

1. Latency budgets sum at **p99 per hop** and leave headroom.
2. The autoscaling signal is not CPU utilization and not raw GPU utilization.
3. Cache keys include the **model version** (and the tenant, where there is one).
4. Rollback is faster than the error budget it is protecting, and it has been **rehearsed**.
5. Randomization is on the **user**, and the sample size was fixed **before** launch.
6. Every design states what it is trading away.
