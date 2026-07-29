# C03 Solutions

Reference answers for [`../exercises/`](../exercises/). Read them **after** writing your own —
these include the reasoning and the tradeoff commentary, which is the part that transfers.

| # | Solution | Headline |
|---|----------|----------|
| 1 | [`01_diagnose_slow_training_solution.md`](01_diagnose_slow_training_solution.md) | Two overlapping problems: input-pipeline starvation from the re-export, and a straggler tail that grows with rank count. The loss bump at restart is a third, separate bug. |
| 2 | [`02_checkpoint_strategy_solution.md`](02_checkpoint_strategy_solution.md) | `T*` = 65.7 min synchronous → 5.2 min sharded+async, overhead 14.8% → 3.6%. Spot wins on cost by ~54%, and the deciding variable is **restart time**, not the checkpoint interval. |
| 3 | [`03_repro_audit_solution.md`](03_repro_audit_solution.md) | The +1.3 claim is *unsupported*, not merely unverified: the eval set changed and there is one seed. Fix the comparison before fixing determinism. |

## How these are written

Each solution follows the structure the exercises ask for: assumptions stated, arithmetic shown,
tradeoff named, "it depends on X" pivot given, and a next measurement proposed. Where a number
appears it is either given in the problem or derived — and where it comes from one of the
[`../tools/`](../tools/) models, that is labelled explicitly, because a model output is not a
measurement.

## A note on grading yourself

The most common gap between a competent answer and a strong one is **not** knowledge — it is
ordering and honesty:

- A competent answer lists the right causes. A strong one lists them in the order it would actually
  check them, and says why that order.
- A competent answer recommends improvements. A strong one names the thing it would *decline* to do
  and the money that saves.
- A competent answer gives a number. A strong one gives the number, its sensitivity, and the fact
  that would change it.

If your answer had no numbers, or recommended everything, redo it before reading further.
