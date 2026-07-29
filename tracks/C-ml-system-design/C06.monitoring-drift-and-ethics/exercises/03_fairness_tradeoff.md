# Exercise 3 — The fairness tradeoff

**Time:** 45–60 minutes · **Deliverable:** ~1.5 pages: metric analysis, criterion choice with
justification, an explicit cost statement, and the words you would say to the stakeholder.

---

## The situation

You own the **candidate-screening model** for a large employer's hiring platform. It scores
applicants for a "advance to phone screen" decision. A recruiter sees the score and the decision;
they can override, and in practice they override about 6% of the time.

An audit produced the following, at the current shared threshold of 0.50. The two groups are
defined by a protected attribute; the outcome label `Y = 1` means "the candidate passed the
subsequent technical interview", which is the platform's operational definition of "should have
advanced".

| Metric | Group A (n = 41 200) | Group B (n = 9 850) |
|---|---:|---:|
| Base rate `P(Y=1)` | 0.34 | 0.21 |
| Selection rate `P(Ŷ=1)` | 0.312 | 0.148 |
| TPR (recall) | 0.681 | 0.474 |
| FPR | 0.122 | 0.062 |
| Precision (PPV) | 0.742 | 0.673 |
| Accuracy | 0.771 | 0.812 |
| ECE (calibration error) | 0.021 | 0.026 |

Additional context:

- The score is **well calibrated in both groups** — a 0.7 means roughly 70% in each.
- Group B is 19% of applicants.
- The label `Y` (passed the technical interview) is itself produced by human interviewers.
- The model does not use the protected attribute as a feature.
- Legal has asked to be kept informed but has not yet issued a requirement.
- A VP has seen the selection-rate row (0.312 vs 0.148) and asked, in a meeting, "why is the model
  rejecting group B twice as often?"

---

## Your task

### 1. What the numbers say

Compute or state, for each fairness criterion in §3.7 of the module README, whether it is satisfied
and by how much it is violated:

- demographic parity
- equal opportunity
- equalized odds
- predictive parity
- calibration within groups

Use `../tools/fairness_metrics.py` to build intuition — you can approximate this scenario with
`--base-rate-a 0.34 --base-rate-b 0.21 --n-a 41200 --n-b 9850`, though the exact numbers will
differ from the audit table.

### 2. Answer the VP's question, correctly

The VP asked why the model rejects group B twice as often. There is a technically accurate answer
involving the base rates, and there is a *complete* answer that does not stop there.

Give the complete one. Specifically: the selection-rate ratio is 0.148/0.312 = 0.47, but the
base-rate ratio is 0.21/0.34 = 0.62. **The selection gap is larger than the base-rate gap.**
Explain what that discrepancy means and what mechanism could produce it.

### 3. Choose a criterion and justify it

State which criterion you believe should govern this system, and justify it from **the nature of
the decision** — who bears the cost of each error type here — not from which number is worst.

Then state the criterion you rejected that has the strongest counter-argument, and make that
counter-argument as well as you can before rejecting it. An answer that does not steelman the
alternative is incomplete.

### 4. Quantify the intervention

For your chosen criterion:

- What intervention achieves it? (Threshold adjustment, reweighting, a different model, changing the
  label, changing the data — say which and why.)
- **What does it cost?** Give numbers: which other metrics move, in which direction, and by how
  much. Use the tool.
- **Who is affected?** Specifically: which people get a different decision than they do today, and
  in which direction. This is the part that makes the tradeoff real rather than abstract.

### 5. Question the label

`Y = "passed the technical interview"` is produced by human interviewers. Say what could be wrong
with that as a definition of "should have advanced", and what you would do to check. Then say what
follows if you *cannot* check it.

This part is worth as much as the rest combined. Every fairness criterion above is defined relative
to `Y`, so if `Y` is biased, satisfying any of them satisfies nothing.

### 6. What you would not do alone

Name the decisions in this drill that are **not yours to make**, who should make them, and what you
would bring to them so they can decide well. Be specific about what you would hand each party.

### 7. The conversation

Write what you would actually say to the VP — 4–6 sentences, spoken, no jargon. It must:

- answer their question honestly,
- convey that "make the numbers equal" is a choice with a cost, not a bug fix,
- not lecture them,
- and end with a concrete next step.

This part is graded. A technically flawless analysis that leaves the VP thinking you are being
evasive results in someone else making the decision with worse information.

---

## Self-check before reading the solution

- [ ] Did I evaluate all five criteria, not just the one the VP noticed?
- [ ] Did I explain why the selection gap exceeds the base-rate gap?
- [ ] Did I justify my criterion from the *decision*, not from the metrics?
- [ ] Did I steelman the alternative before rejecting it?
- [ ] Did I give numeric costs for the intervention, including who is affected?
- [ ] Did I seriously question the label?
- [ ] Did I identify which decisions are not mine, and what I would give each owner?
- [ ] Would my VP script land as helpful rather than evasive?

Reference answer: [`../solutions/03_fairness_tradeoff_solution.md`](../solutions/03_fairness_tradeoff_solution.md).
