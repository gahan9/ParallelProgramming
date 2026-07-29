# C04 Exercises

Three design drills. Write your answers **before** opening
[`../solutions/`](../solutions/) — the reference answers include the reasoning and the tradeoff
commentary, and reading them first turns a design exercise into a reading exercise.

These are markdown, not code. The skill being tested is **defending a decision under constraints**,
which is exactly what a design interview measures and exactly what you cannot look up during one.

## How to work them

```bash
make -C .. run-exercises                  # the scenarios below, as tool invocations
make -C .. run-exercises PYTHON=python    # Windows
```

Or run a scenario directly, e.g.:

```bash
python ../tools/cost_model.py --qps 18 --tokens-per-request 900 --per-gpu-tokens-per-second 2000
```

The tools are **analytic models** whose multipliers are placeholders. Use them to check arithmetic
and explore sensitivity; never quote their output as a measurement.

| # | File | Skill |
|---|------|-------|
| 1 | [`01_optimization_order.md`](01_optimization_order.md) | Order the ladder for a *specific* situation and justify not starting with distillation |
| 2 | [`02_quality_gate.md`](02_quality_gate.md) | Design a quantization rollout gate: eval set, slices, thresholds, canary, rollback trigger |
| 3 | [`03_build_the_cost_case.md`](03_build_the_cost_case.md) | Break-even analysis, and stating plainly when the answer is "don't do it" |

## What a good answer looks like

1. **Compute the ceiling before proposing the work.** If idle capacity is 40% of the bill, no model
   change can beat 40% — say so before you talk about quantization.
2. **Show the arithmetic.** "$0.0889 per 1k requests, break-even 0.8 months" beats "significant
   savings" every time.
3. **Name the quality risk and how you would detect it.** Every rung below "serve it better" changes
   outputs; an answer with no gate is incomplete regardless of how good the cost analysis is.
4. **Give the "it depends on X" pivot.** State the fact that would change your recommendation, and to
   what.
5. **Be willing to recommend doing nothing.** In these drills at least one correct answer is "no",
   and saying so with a number behind it is the strongest possible response.

The most common failure is a technically excellent answer that never states what it costs or how it
could be wrong.
