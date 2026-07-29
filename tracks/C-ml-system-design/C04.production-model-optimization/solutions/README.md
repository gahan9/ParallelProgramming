# C04 Solutions

Reference answers for [`../exercises/`](../exercises/). Read them **after** writing your own — the
value is in the reasoning and the tradeoff commentary, not the conclusions.

| # | Solution | Headline |
|---|----------|----------|
| 1 | [`01_optimization_order_solution.md`](01_optimization_order_solution.md) | Two different problems wearing one hat. Caching (once the key is fixed) solves ~31% of the cost; nothing on the cheap rungs solves the latency SLO, and distillation is wrong here because the base model dies in two quarters. |
| 2 | [`02_quality_gate_solution.md`](02_quality_gate_solution.md) | The proposed evidence is four separate failures at once. The safety-critical queues are 0.9% of volume, so the uniform eval set is structurally incapable of detecting a regression there — that is the whole problem to solve. |
| 3 | [`03_build_the_cost_case_solution.md`](03_build_the_cost_case_solution.md) | Cache + batching delivers ~50% for 5 engineer-weeks and survives the platform migration. Distillation loses ~$24,500 against a model with ~7 months to live: decline it. |

## How these are written

Assumptions stated, arithmetic shown, tradeoff named, "it depends on X" pivot given, next measurement
proposed. Numbers are either given in the problem or derived; where one comes from a
[`../tools/`](../tools/) model, that is labelled — a model output is not a measurement, and the
multipliers in those models are placeholders.

## Grading yourself

The gap between a competent answer and a strong one, in this module specifically:

- A competent answer picks the right optimization. A strong one **computes the ceiling first** and
  eliminates candidates before analysing them.
- A competent answer mentions evaluation. A strong one gives **numeric thresholds per slice** and a
  rollback trigger an on-call engineer could execute without judgement.
- A competent answer recommends action. A strong one recommends **declining** something, with the
  number that justifies it.
- A competent answer treats cost and latency as one problem. A strong one notices they are two, with
  different answers, and that some fixes trade one against the other.

If your answer would have approved everything on the list, redo it.
