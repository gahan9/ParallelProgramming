# Drill 1 — Fit the model

**Goal:** given a model and a fixed pile of GPUs, decide whether training is
possible at all, and if so find the *minimum viable* parallelism configuration —
the one that fits while disturbing the least.

## The setup

You have been handed a GPT-3-175B-shaped model to pre-train:

| Quantity | Value |
|---|---|
| Parameters `Ψ` | 175 × 10⁹ |
| Layers `L` | 96 |
| Hidden `h` | 12288 |
| Attention heads `a` | 96 |
| Sequence `s` | 2048 |
| Micro-batch `b` | 1 |
| Precision / optimizer | bf16 + fp32 master + Adam |

Your cluster: **one node, 8 GPUs, 192 GiB HBM each**, fully connected intra-node.
Reserve 10 GiB per GPU for framework overhead and fragmentation.

## Tasks

**(a) State memory.** Compute the total resident parameter + gradient + optimizer
bytes for this model, and express it as "how many 192 GiB GPUs of pure state."
Show the bytes-per-parameter breakdown.

**(b) Activation memory.** Using `s*b*h*(34 + 5*a*s/h)` per layer, compute:
   - the `5*a*s/h` factor,
   - activation bytes for one layer,
   - activation bytes for all 96 layers.
   Then say which of state or activations is the bigger problem here.

**(c) Rule out the cheap options.** For each of DDP, ZeRO-1, ZeRO-2, and ZeRO-3
on these 8 GPUs, compute the per-GPU total and state *why it fails* in one line.
Do not just write "too big" — name the term that is too big.

**(d) The uncomfortable question.** Is there **any** configuration of ZeRO stage,
tensor parallelism, pipeline parallelism, and activation recomputation that fits
this model on 8 GPUs? Prove your answer with an inequality, not by enumeration.

> Hint: the most aggressive sharding of state you can achieve on `N` GPUs divides
> it by `N`. Start there and see whether anything is left for activations.

**(e) Minimum GPU count.** Find the smallest GPU count `N` (a power of two) at
which the model fits, and give the configuration. Which recomputation mode did you
need, and how much headroom do you have?

**(f) The comfortable answer.** At `N = 32`, give the configuration you would
actually run and justify it against the next-cheapest alternative. Consider both
memory *and* the communication each axis implies.

## Check your work

```powershell
python tools/memory_planner.py --preset gpt3-175b --gpus 8
python tools/memory_planner.py --preset gpt3-175b --gpus 8  --suggest
python tools/memory_planner.py --preset gpt3-175b --gpus 16 --suggest --top 4
python tools/memory_planner.py --preset gpt3-175b --gpus 32 --suggest --top 5
```

## What a good answer looks like

- Every number derived, not quoted.
- Part (d) answered with an inequality that closes the question for *all*
  configurations, not a table of the ones you happened to try.
- Part (f) picks a configuration **and names what it is trading**, e.g. "selective
  recompute costs ~2.7% FLOPs, which I prefer to a TP group I would then have to
  keep inside one node."
- One sentence on what the analytic model is *not* telling you.
