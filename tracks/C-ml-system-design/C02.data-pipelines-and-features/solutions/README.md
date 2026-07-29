# C02 Solutions

Reference answers for [`../exercises/`](../exercises/). Each gives the **reasoning**, because for design
and debugging drills the reasoning is what is being graded.

| # | Solution | The one thing to take away |
|---|---|---|
| 1 | [`01_find_the_leak_solution.md`](01_find_the_leak_solution.md) | The dangerous leaks have correct-looking timestamps, innocent names, and only moderate separation — no automated check finds them |
| 2 | [`02_design_a_pipeline_solution.md`](02_design_a_pipeline_solution.md) | The regime per feature follows from one question: does the decision change inside the batch window? |
| 3 | [`03_skew_postmortem_solution.md`](03_skew_postmortem_solution.md) | Matching online and offline *scores* proves the model is deployed correctly and says nothing about whether the *features* are right |

---

## How to grade yourself

| Move | What it looks like |
|---|---|
| Verdict plus mechanism | Not "this leaks" but "the model learns X, which is always zero at decision time" |
| Comfort with "I need to ask" | A real feature audit ends in questions with named owners |
| Regime chosen by rule | Batch unless the decision changes inside the batch window; streaming priced honestly |
| Gates with a failure policy | Per table: block and page, or reject rows and alert on the rejection rate |
| Hypotheses ranked by likelihood ÷ cost-to-check | Cheap decisive checks first; each with a specific confirm/eliminate criterion |
| Measurement separated from system | "The number was wrong" and "production throws away a real gain" need different fixes |
| Power arithmetic before "no effect" | A standard error computed before any conclusion about a flat metric |
| Structural prevention | Detections and shared definitions, never "be more careful" |
| The "it depends on X" pivot | Every contested choice names its deciding variable |

---

## Where the numbers come from

Tool output quoted in these solutions is real, from fixed seeds, and reproduces exactly:

```bash
cd .. && make run-exercises PYTHON=python
```

Everything else is arithmetic from the stated assumptions in the exercise. **No benchmark measurements
and no industry statistics are invented anywhere in this module.** Where a claim is a common
practitioner pattern rather than a measurement — for example the ordering of "what usually moves the
metric" in README §2 — it is labelled as a pattern.
