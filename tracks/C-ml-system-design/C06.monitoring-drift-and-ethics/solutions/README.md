# C06 Solutions

Reference answers with the reasoning. Attempt [`../exercises/`](../exercises/) first.

| # | File | The point of the drill |
|---|------|------------------------|
| 1 | `01_design_monitoring_solution.md` | Monitoring is an *allocation* problem, not a coverage problem. The graded skill is what you refuse to alert on, and the ability to distinguish a legitimate world-change (a new customer) from a break — with the same signal. |
| 2 | `02_diagnose_the_drift_solution.md` | The honest answer is **two mechanisms at once**, and the tell is calibration error moving 3.7× while the prediction distribution barely moves. Also: the monitoring gap was a *reasonable* decision that was nonetheless wrong, and the fix is a better rule, not "monitor everything". |
| 3 | `03_fairness_tradeoff_solution.md` | Every criterion is defined relative to the label, so the label is the first thing to interrogate — and in this scenario it is the weakest link. The selection gap exceeding the base-rate gap is the quantitative tell. |

---

## Where reference answers stop being authoritative

- **Drill 1** — the alert/dashboard split depends on team size, on-call load and risk appetite. A
  different split with stated reasoning is fine. A split that produces five pages a week is not,
  because it will be muted.
- **Drill 2** — the primary diagnosis is well-determined by the evidence. The *fix* has legitimate
  variation, particularly on how aggressive to be in the first 24 hours with a 60-day feedback loop.
- **Drill 3** — the criterion choice is genuinely contested and reasonable people disagree. What is
  not negotiable is that the choice be justified from the decision, that its cost be quantified, and
  that the label be interrogated.

## What is not negotiable in any answer

1. Alert on symptoms, dashboard on clues, and stay inside the on-call budget.
2. Every alert has a runbook with concrete first steps.
3. Diagnose the drift *type* before choosing a fix; five of the seven branches in §6 end in "do not
   retrain".
4. State what you cannot know. With delayed labels, that is most of it.
5. A fairness intervention is presented with its cost and with who is affected.
6. Where the constraint is legal, the answer is "involve counsel", not a guess.
