# B07 Exercises — Distributed Training & Sharding

Three drills that turn the module into decisions. All of them are **arithmetic and
judgement**, not code — which is exactly what a distributed-training design review
or interview asks of you.

| Drill | File | What it tests |
|---|---|---|
| 1 | [`01_fit_the_model.md`](01_fit_the_model.md) | Memory budgeting: does it fit, and what is the *minimum* intervention? |
| 2 | [`02_choose_parallelism.md`](02_choose_parallelism.md) | Axis selection across four scenarios, plus the interconnect each demands |
| 3 | [`03_comm_budget.md`](03_comm_budget.md) | Communication cost as a fraction of step time; comm-bound or not? |

## How to work these

1. **Do the arithmetic by hand first.** Every formula you need is derived in the
   module README §2. If you reach for the tool before you have a number, you are
   practising the wrong skill.
2. **Then check with the tools.** They are analytic models, so if your hand
   arithmetic and the tool disagree, one of you has a bug — find out which.
3. **Write the justification, not just the answer.** "ZeRO-3 with selective
   recompute" is worth nothing without "because state is 2608 GiB and activations
   are 256 GiB, so I need to shard both, and the cheapest way to do that is…".

```powershell
# From the module directory (one level up):
make run-exercises           # prints each drill's inputs
python tools/memory_planner.py --explain     # every formula with its citation
python tools/memory_planner.py --help
python tools/collective_cost.py --help
python tools/pipeline_bubble.py --help
```

## Shared assumptions unless a drill says otherwise

| Quantity | Value | Source |
|---|---|---|
| Precision | bf16 compute, fp32 master weights | [Mixed Precision Training](https://arxiv.org/abs/1710.03740) |
| Optimizer | Adam (two fp32 moments) | 16 bytes/param total, [ZeRO](https://arxiv.org/abs/1910.02054) |
| GPU | 192 GiB HBM (MI300X-class) | AMD Instinct MI300X spec |
| Framework reserve | 10 GiB per GPU | rule of thumb; calibrate against one real run |
| Activation formula | `s*b*h*(34 + 5*a*s/h)` per layer | [arXiv:2205.05198](https://arxiv.org/abs/2205.05198) eq. 1 |
| Node size | 8 GPUs, fully connected intra-node | [B01 §3.5](../../B01.accelerator-architecture/) — 7 × 128 GB/s Infinity Fabric per GPU |

Reference answers with full working: [`../solutions/`](../solutions/). Read them
*after* you have written yours down — the solutions include the reasoning you are
being graded on, and a few honest caveats about where the analytic model lies.
