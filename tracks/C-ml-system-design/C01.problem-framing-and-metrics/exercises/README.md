# C01 Exercises

Three design drills. Do them yourself before reading [`../solutions/`](../solutions/) — the whole
value is in producing an argument you can defend, and a solution you have read is an argument you
have borrowed.

These are markdown, not code, because the deliverable of problem framing is **a document a team can
disagree with**. Write your answers directly in the exercise files (or copy them somewhere private);
each file has explicit deliverables and a self-check list.

| # | File | Skill it drills |
|---|------|-----------------|
| 1 | [`01_frame_the_problem.md`](01_frame_the_problem.md) | Turn a vague ask into a framed task: decision, inputs, output, horizon, metrics, requirements, non-ML alternative |
| 2 | [`02_choose_metrics.md`](02_choose_metrics.md) | Pick target + guardrail + counter-metrics for three very different cost structures |
| 3 | [`03_threshold_from_costs.md`](03_threshold_from_costs.md) | Cost-optimal threshold arithmetic, and what moves when the base rate moves |

Interview mapping: (1) first-principles problem formulation, (2) tradeoff analysis and communication,
(3) quantitative reasoning under a cost model. All three are asked, in some form, in every ML system
design interview.

---

## Checking your work with the tools

Exercise 3 is the only one with numeric answers, and it is checkable:

```bash
cd ..
make run-exercises PYTHON=python      # drop PYTHON= on Linux/macOS if python3 is your default
```

That runs both base-rate scenarios from exercise 3 plus the baselines for each, so you can compare a
hand calculation against the sweep. The tools are standard-library Python 3 — no `pip install`.

You can also check any single confusion matrix you compute by hand:

```bash
python ../tools/metric_explorer.py confusion --tp 145 --fp 625 --fn 55 --tn 9175 \
    --fp-cost 8 --fn-cost 160
```

---

## Sample tool output you will be comparing against

Reference runs use a fixed seed (`--seed 7`), so **these numbers reproduce exactly on any machine** —
unlike the GPU tracks, there is no hardware variance here. The synthetic scores are drawn from two
overlapping Gaussians two standard deviations apart, which is a deliberately *mediocre* model: good
enough to be useful, bad enough that the threshold choice matters.

### Base rate 2% (`--positives 200 --negatives 9800`)

```text
n = 10000  positives = 200  negatives = 9800  prevalence =  0.0200
  ROC-AUC =  0.9180
  PR-AUC  =  0.4111   (random baseline = prevalence =  0.0200)

  thresh      TP     FP     FN     TN  precision  recall      F1        cost
  0.9037     95    117    105   9683     0.4481  0.4750  0.4612     2217.00 <- max F1
  0.8223    145    625     55   9175     0.1883  0.7250  0.2990     1725.00 <- min cost

  flag nothing (reject all) cost = 4000.00 | flag everything cost = 9800.00
```

### Base rate 0.2% (`--positives 200 --negatives 99800`)

```text
n = 100000  positives = 200  negatives = 99800  prevalence =  0.0020
  ROC-AUC =  0.9181
  PR-AUC  =  0.0969   (random baseline = prevalence =  0.0020)

  thresh      TP     FP     FN     TN  precision  recall      F1        cost
  0.9449     42    205    158  99595     0.1700  0.2100  0.1879     3365.00 <- max F1
  0.9240     71    590    129  99210     0.1074  0.3550  0.1649     3170.00 <- min cost

  flag nothing (reject all) cost = 4000.00 | flag everything cost = 99800.00
```

> [!NOTE]
> Look at what did and did not change between those two blocks. **ROC-AUC is identical to three
> decimal places (0.9180 → 0.9181)** — the model's ranking ability never changed, only the class
> balance. **PR-AUC fell 4×** and the cost-optimal recall collapsed from 0.725 to 0.355. And in the
> 0.2% case the best achievable cost (3170) is barely below the cost of *doing nothing at all*
> (4000). Same model, same seed; the system went from clearly worth running to marginal. Explaining
> that gap is exercise 3.

> [!TIP]
> Costs scale linearly, so only the **ratio** matters. The sweep above uses `--fp-cost 1 --fn-cost 20`
> while exercise 3 states \$8 and \$160 — the same 1:20 ratio, so every cost in the table is exactly
> 8× the number shown, and the optimal threshold is identical. If your hand calculation disagrees with
> the tool by a constant factor, this is why.

---

## What "good" looks like

A strong answer to any of these is not the one with the most metrics. It is the one that:

1. **States the decision and the action first**, before any metric appears.
2. **Names the cost asymmetry explicitly** and derives the operating point from it, rather than
   defaulting to 0.5 or to F1.
3. **Compares against a trivial baseline**, so the reader knows whether the number is impressive.
4. **Says what it would not report, and why** — dropping a metric on purpose is a stronger signal
   than listing ten.
5. **Includes the "it depends on X" pivot** for every contested choice, because that is the sentence
   that turns an opinion into engineering judgement.
6. **Specifies failure behaviour.** What happens when the model is unavailable is part of the design,
   not an operational detail to sort out later.
