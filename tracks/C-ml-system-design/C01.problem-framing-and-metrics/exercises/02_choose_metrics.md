# Exercise 02 — Choose metrics for three very different cost structures

**Goal:** show that metric choice is driven by imbalance and error cost, not by habit — and that the
*same* set of candidate metrics leads to three different answers.

**Time:** 45–60 minutes. **Deliverable:** the three tables below, plus the comparison at the end.

---

## The scenarios

### A — Card-not-present fraud detection

- ~1,000,000 transactions/day; roughly **0.15%** are fraudulent (state it as an assumption; you would
  confirm with the risk team).
- A **false positive** blocks a legitimate payment: lost sale, a support contact, and measurable
  customer annoyance. Reversible, but not free.
- A **false negative** means the merchant eats the chargeback plus fees.
- **12 analysts** review the queue; each handles ~250 cases/day.
- The score is used to decide: auto-allow, send to review, or auto-block.

### B — Content recommendation on a home feed

- Every session requests a ranked list of 20 items from a candidate pool of ~50,000.
- There is no single "correct" item; relevance is graded and multiple items are acceptable.
- The only cheap feedback is implicit: impressions, clicks, dwell time, completions, explicit "hide
  this."
- Leadership wants "engagement up." The company's stated long-term goal is subscriber retention.

### C — Medical triage assistant (emergency department)

- Predicts whether a presenting patient is at high risk of deterioration within 12 hours, to
  prioritize clinician attention.
- ~4% of presentations are high-risk (state it as an assumption).
- A **false negative** may mean a patient deteriorates while waiting: potentially irreversible.
- A **false positive** consumes scarce clinician time and, at volume, causes alert fatigue — which
  itself causes misses.
- The model is an **aide**: a clinician always makes the final call, and must be able to see why.

---

## Your task

For each scenario, complete the table. Then answer the cross-scenario questions.

### Scenario A — Fraud

| Field | Your answer |
|---|---|
| Prevalence, and what it implies | |
| Which error costs more, and roughly by what ratio | |
| **Target metric** (one) | |
| Why *not* accuracy | |
| Why *not* ROC-AUC as the headline | |
| Threshold basis (cost, capacity, or both) | |
| Guardrail metrics (must not regress) | |
| Counter-metric(s) (catch gaming) | |
| Required slices | |
| Calibration needed? Why? | |
| What you deliberately would **not** report | |

### Scenario B — Recommendation

| Field | Your answer |
|---|---|
| Is this classification or ranking? What follows? | |
| **Target metric** (one), and at what `k` | |
| Metric for the candidate-generation stage vs the ranking stage | |
| Why not precision/recall on the whole pool | |
| How the target metric would be gamed within one sprint | |
| Guardrail metrics | |
| Counter-metric(s) | |
| Bias in the implicit labels, and what it does to your offline metric | |
| Required slices | |
| What you deliberately would **not** report | |

### Scenario C — Medical triage

| Field | Your answer |
|---|---|
| Prevalence, and what it implies | |
| Which error costs more, and why the ratio is hard to state | |
| **Target metric** (one) | |
| Why a fixed precision floor (or fixed alert budget) belongs in the metric | |
| Calibration: required or optional? Defend it | |
| Guardrail metrics | |
| Counter-metric(s) — including the alert-fatigue feedback loop | |
| Required slices, and why this list is longer here | |
| Failure behaviour when the model is unavailable | |
| What you deliberately would **not** report | |

---

## Cross-scenario questions

1. **One metric appears as a *candidate* in all three scenarios but should be the headline in none.**
   Name it and explain why in each case.

```
(your answer)
```

2. All three scenarios have imbalanced classes, yet only two of them should lead with PR-AUC. Which
   one should not, and what should it lead with instead?

```
(your answer)
```

3. Rank the three scenarios by **how much calibration matters**, most to least, with a one-line reason
   each.

```
(your answer)
```

4. For each scenario, give the single sentence you would use to elicit the cost ratio from the
   business owner — phrased so that someone who cannot state a number can still answer.

```
(your answer)
```

5. Each scenario has a **capacity or budget constraint** that can override the cost-optimal
   threshold. Name it for each, and say what the objective becomes once capacity binds.

```
(your answer)
```

---

## Self-check

- [ ] Every scenario has exactly **one** target metric — if you listed three, you have not chosen.
- [ ] Every target metric has at least one guardrail that would catch its most likely abuse.
- [ ] You said what you would *not* report, in all three scenarios.
- [ ] At least one scenario's threshold is set by **capacity**, not cost, and you said so.
- [ ] Calibration is required in some scenarios and optional in others, with reasons — not "always
      good to have."
- [ ] Slices are named concretely (which dimensions), not as "we'll slice the data."
- [ ] You used the phrase "it depends on" at least once and finished the sentence with a *specific*
      variable.

Reference answer with the reasoning:
[`../solutions/02_choose_metrics_solution.md`](../solutions/02_choose_metrics_solution.md).
