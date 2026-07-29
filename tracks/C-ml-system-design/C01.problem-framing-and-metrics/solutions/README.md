# C01 Solutions

Reference answers for [`../exercises/`](../exercises/). Each one gives the **reasoning**, not just the
answer — for design drills the reasoning *is* the answer, and an interviewer is grading the argument.

| # | Solution | The one thing to take away |
|---|---|---|
| 1 | [`01_frame_the_problem_solution.md`](01_frame_the_problem_solution.md) | The framing forces out the horizon, the label definition, the capacity limit, and the fallback — none of which came from a model |
| 2 | [`02_choose_metrics_solution.md`](02_choose_metrics_solution.md) | Three imbalanced problems, three different headline metrics; the cost asymmetry decides, not the imbalance |
| 3 | [`03_threshold_from_costs_solution.md`](03_threshold_from_costs_solution.md) | `p*` depends only on the cost ratio; the *score* threshold and the whole business case still move with the base rate |

---

## How to read these

There is rarely one correct answer to a design question, and these solutions are not claiming to be
the only defensible ones. What they *are* claiming is that a strong answer must contain certain
**moves**, and you can grade yourself on whether you made them:

| Move | What it looks like |
|---|---|
| Decision before metric | The first paragraph names an action, not a score |
| Explicit cost asymmetry | A stated (or bounded) FP/FN ratio, and an operating point derived from it |
| Baseline comparison | Every number sits next to what a trivial policy already achieves |
| Named proxy gaps | "We assume offline X predicts online Y; we verify by Z" |
| Guardrails for every target | For each optimization target, the abuse and the metric that catches it |
| Slices declared up front | Named dimensions, with minimum sample sizes |
| Capacity reconciliation | The cost-optimal point checked against what humans/systems can absorb |
| Failure behaviour | What happens when the model is unavailable, written down |
| The "it depends on X" pivot | Every contested choice has its deciding variable named |

If your answer made seven of those nine moves, you would do well in the interview even where your
specific choices differ from these.

---

## Where the numbers come from

Exercise 3's tables are **real output** from [`../tools/metric_explorer.py`](../tools/metric_explorer.py)
with `--seed 7`, so they reproduce exactly:

```bash
cd .. && make run-exercises PYTHON=python
```

Nothing in these solutions is a benchmark measurement or an industry statistic. Every number is either
(a) arithmetic from the stated assumptions, or (b) reproducible tool output from a seeded synthetic
dataset. Where a claim is a common industry *pattern* rather than a measurement, it is labelled as a
pattern.
