# C03 Exercises

Three design drills. Write your answers **before** opening
[`../solutions/`](../solutions/) — the reference answers include the reasoning, and reading them
first turns a design exercise into a reading exercise.

These are markdown, not code, because the skill being tested is *judgement under stated
constraints*. That is what a design interview measures, and it is the thing you cannot look up
during one.

## How to work them

```bash
make -C .. run-exercises     # the scenarios below, as tool invocations
make -C .. run-exercises PYTHON=python    # Windows
```

Or run a scenario directly, e.g.:

```bash
python ../tools/checkpoint_planner.py --mtbf-hours 0.75 --checkpoint-seconds 240 --shards 16
```

The tools are **analytic models**. Use them to check arithmetic and to explore sensitivity — never
quote their output as a measurement.

| # | File | Skill |
|---|------|-------|
| 1 | [`01_diagnose_slow_training.md`](01_diagnose_slow_training.md) | Ordered triage: from a symptom list to a root cause, with the reasoning for the order |
| 2 | [`02_checkpoint_strategy.md`](02_checkpoint_strategy.md) | Quantitative checkpoint design under MTBF, write cost, and spot interruption |
| 3 | [`03_repro_audit.md`](03_repro_audit.md) | Reproducibility audit; separating bit-exact from statistical, and pricing each |

## What a good answer looks like

Every drill is graded the way a design review grades you:

1. **State your assumptions.** Any number you were not given, you assume out loud.
2. **Show the arithmetic.** "Checkpoint more often" is not an answer; "T* = sqrt(2·C·M) ≈ 21 min,
   so every ~1,700 steps" is.
3. **Give the tradeoff.** Every recommendation costs something. Name it.
4. **Give the "it depends on X" pivot.** Say what fact would change your answer, and to what.
5. **Say what you would measure next.** A design that cannot be falsified is a wish.

An answer that is confidently wrong about a number is recoverable. An answer with no numbers at all
is not.
