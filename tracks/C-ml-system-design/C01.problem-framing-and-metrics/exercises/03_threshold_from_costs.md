# Exercise 03 — Cost-optimal threshold, and what the base rate does to it

**Goal:** do the threshold arithmetic by hand, verify it with the tool, then reason correctly about
what changes when the base rate shifts. This is the most commonly botched quantitative question in ML
system design interviews.

**Time:** 30–45 minutes. **Deliverable:** the computed numbers and the written explanation in Part 4.

---

## Setup

A payments company scores transactions for fraud.

| Quantity | Value |
|---|---|
| False positive cost `C_fp` | **\$8** (analyst review time + customer friction from a declined payment) |
| False negative cost `C_fn` | **\$160** (average net loss on an undetected fraudulent transaction) |
| Base rate (scenario A) | **2.0%** of transactions are fraud |
| Base rate (scenario B) | **0.2%** of transactions are fraud |
| Evaluation set | 200 fraudulent transactions in both scenarios |

The model's ranking quality is **identical** in both scenarios — only the number of legitimate
transactions differs. That is the point of the exercise: you are isolating the effect of the base
rate.

---

## Part 1 — The analytic threshold (calibrated probabilities)

Assume for this part that the model outputs a **calibrated probability** `p` of fraud.

**1a.** Write the expected cost of flagging a case and of not flagging it, in terms of `p`, `C_fp`,
`C_fn`.

```
(your answer)
```

**1b.** Derive the threshold `p*` above which flagging is the cheaper action. Show the algebra.

```
(your answer)
```

**1c.** Compute `p*` for the costs above. Give it to four decimal places.

```
p* =
```

**1d.** Does `p*` change between scenario A (2%) and scenario B (0.2%)? Answer yes/no **and justify
from your derivation**, not from intuition.

```
(your answer)
```

**1e.** Suppose finance revises `C_fn` upward to \$400. Recompute `p*`. Which direction did the
threshold move, and does that match your intuition about a more expensive miss?

```
p* =
```

---

## Part 2 — Hand-compute the cost of three operating points

These are real operating points from the module's tool at **scenario A** (2% base rate, 200
positives, 9,800 negatives). Fill in the cost column using `cost = FP × C_fp + FN × C_fn`.

| Threshold | TP | FP | FN | TN | Precision | Recall | **Total cost** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9037 | 95 | 117 | 105 | 9683 | 0.4481 | 0.4750 | |
| 0.8223 | 145 | 625 | 55 | 9175 | 0.1883 | 0.7250 | |
| 0.6420 | 178 | 2764 | 22 | 7036 | 0.0605 | 0.8900 | |
| flag nothing | 0 | 0 | 200 | 9800 | — | 0.0000 | |
| flag everything | 200 | 9800 | 0 | 0 | 0.0204 | 1.0000 | |

**2a.** Which of the three thresholds is cheapest?

```
(your answer)
```

**2b.** The first row (0.9037) is the **F1-optimal** threshold. By what percentage does choosing it
instead of the cost-optimal row increase your total cost?

```
(your answer)
```

**2c.** Verify one row with the tool and confirm your arithmetic:

```bash
python ../tools/metric_explorer.py confusion --tp 145 --fp 625 --fn 55 --tn 9175 \
    --fp-cost 8 --fn-cost 160
```

**2d.** Now run the full sweep. It uses `--fp-cost 1 --fn-cost 20`, which is the **same 1:20 ratio**
scaled down by 8. Predict what relationship its `cost` column will have to your numbers *before*
running it, then check:

```bash
cd .. && make run-exercises PYTHON=python
```

```
prediction:
observed:
```

---

## Part 3 — Scenario B: the base rate drops 10×

The tool's scenario B output (200 positives, 99,800 negatives — the same model, same seed) reports:

| | Scenario A (2%) | Scenario B (0.2%) |
|---|---:|---:|
| ROC-AUC | 0.9180 | 0.9181 |
| PR-AUC | 0.4111 | 0.0969 |
| Cost-optimal **score** threshold | 0.8223 | 0.9240 |
| Recall at that threshold | 0.7250 | 0.3550 |
| Precision at that threshold | 0.1883 | 0.1074 |
| Total cost at that threshold (1:20 units) | 1725 | 3170 |
| Cost of flagging nothing (1:20 units) | 4000 | 4000 |

**3a.** ROC-AUC is unchanged to three decimal places while PR-AUC fell roughly 4×. Explain both facts
in one sentence each.

```
(your answer)
```

**3b.** In Part 1d you argued about `p*` in probability space. Yet the cost-optimal **score**
threshold *rose* from 0.8223 to 0.9240. Reconcile these two facts. (Hint: write the decision rule as
a likelihood ratio `f₁(x)/f₀(x) > (C_fp/C_fn) × ((1−π)/π)` and see which term contains the
prevalence `π`.)

```
(your answer)
```

**3c.** In scenario B the best achievable cost is 3170 while doing nothing at all costs 4000. Compute
the fraction of the "do nothing" cost that the model saves in each scenario, and state what you would
tell a product owner about scenario B.

```
scenario A saving =
scenario B saving =
what you would say:
```

**3d.** The 200 positives are the same in both scenarios, so recall is measured on 200 cases. Using
`SE = sqrt(p(1−p)/n)`, compute the standard error of the scenario-B recall estimate (0.3550) and give
the approximate 95% interval. Can you distinguish a recall of 0.36 from 0.40 on this evaluation set?

```
SE  =
95% CI ~
verdict:
```

---

## Part 4 — Write the explanation

In **150–250 words**, answer the interview question directly:

> "You've set a cost-optimal threshold. Our fraud rate then drops by 10× because the payments team
> shipped a new bot filter upstream. What do you do, and what changes?"

Cover, at minimum: what changes in probability space vs score space; what happens to precision,
recall, and alert volume; whether the model is still worth running; and the operational check you
would run first. Then add the "it depends on X" pivot.

```
(your answer)
```

---

## Part 5 — Capacity check

Scenario A's cost-optimal point has recall 0.7250 and precision 0.1883. Assume 1,000,000
transactions/day at the 2% base rate.

**5a.** Compute alerts per day at that operating point.

```
positives/day =
true alerts/day =
total alerts/day =
```

**5b.** With 12 analysts at 250 cases/day, is the cost-optimal threshold operationally feasible? If
not, name the four options from README §5 and pick one, with a reason.

```
(your answer)
```

---

## Self-check

- [ ] Your derivation in 1b shows the algebra, not just the final formula.
- [ ] You answered 1d from the derivation (the formula contains no `π`), and 3b explains why the
      *score* threshold still moved.
- [ ] Your Part 2 costs are exactly 8× the tool's `--fp-cost 1 --fn-cost 20` column.
- [ ] You computed the F1-vs-cost penalty as a percentage.
- [ ] Part 3c has both fractions and a recommendation a product owner could act on.
- [ ] Part 3d states an interval and a verdict about whether a 4-point delta is resolvable.
- [ ] Part 5 reaches a number that is larger than analyst capacity, and you said what you would do.

Reference answer with the reasoning:
[`../solutions/03_threshold_from_costs_solution.md`](../solutions/03_threshold_from_costs_solution.md).
