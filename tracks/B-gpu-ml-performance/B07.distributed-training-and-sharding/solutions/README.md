# B07 Solutions — Distributed Training & Sharding

Reference answers with full working for the three drills in
[`../exercises/`](../exercises/). Read your own answer first; these are written to be
compared against, not copied from.

| Solution | Drill | The one thing to take away |
|---|---|---|
| [`01_fit_the_model_solution.md`](01_fit_the_model_solution.md) | [Fit the model](../exercises/01_fit_the_model.md) | `16Ψ/N` is a hard floor — sometimes the answer is "no configuration exists" |
| [`02_choose_parallelism_solution.md`](02_choose_parallelism_solution.md) | [Choose parallelism](../exercises/02_choose_parallelism.md) | Name the binding constraint first; the axis follows from it |
| [`03_comm_budget_solution.md`](03_comm_budget_solution.md) | [Comm budget](../exercises/03_comm_budget.md) | "Communication-bound" is about *overlap*, not about a percentage threshold |

## Grading yourself

You have the right answer if you can defend all four of these:

1. **The binding constraint is named and quantified.** "Memory" is not an answer;
   "state is 2608 GiB and activations are 256 GiB, so state is the 10× problem" is.
2. **Every axis choice cites a number.** "TP stays inside a node" must come with the
   bandwidth comparison that makes it true on *your* hardware, not with tradition.
3. **The failure mode you accepted is stated.** Every configuration is a bet. Say
   what you bet against and what symptom would tell you that you lost.
4. **You said what the model does not know.** Fragmentation, the logit tensor, the
   transient attention matrix inside a recompute, algorithm switching in
   NCCL/RCCL, and stragglers are all outside these tools. A Principal-Engineer-grade
   answer names at least one.

## Three results worth remembering past this module

**1. Sometimes the answer is "buy more GPUs."** Drill 1's model cannot be trained on
8×192 GiB under *any* parallelism configuration, and the proof is one inequality:
`16Ψ/N ≤ HBM − reserve`. Doing that arithmetic in the first five minutes of a
design review saves weeks.

**2. Pipeline parallelism does not reduce activation memory.** Under 1F1B the
worst-loaded stage holds `~S` in-flight micro-batches of `L/S` layers, so the `S`
cancels ([arXiv:2205.05198](https://arxiv.org/abs/2205.05198) eq. 5 has no `p` in
it). This is the single most common wrong answer in distributed-training interviews.

**3. The same axis means different things in training and serving.** Tensor
parallelism in training is a *memory and bandwidth* decision dominated by 64 MiB
activation all-reduces. In decode it is a *latency* decision dominated by 192 KiB
messages that are 98% launch overhead — and the mechanism by which it helps is
dividing the per-GPU weight *read*, not adding FLOPs.

## Verifying the arithmetic

```powershell
# from the module directory
python tools/memory_planner.py --explain          # every formula with its citation
make run-exercises                                # the drills' inputs
```

Each solution shows its tool invocation inline so you can reproduce every table.
Remember what the tools are: **closed-form analytic models, not measurements.** They
are right about ratios and orders of magnitude and silent about fragmentation,
algorithm selection, stragglers, and everything else that decides whether a real job
finishes.
