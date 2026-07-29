# Drill 3 — The communication budget

**Goal:** compute the all-reduce time for a real DDP step, express it as a fraction
of step time, and decide — with a defensible criterion — whether the setup is
communication-bound.

## The setup

| Quantity | Value |
|---|---|
| Model | 1.3 × 10⁹ parameters |
| Gradient dtype | bf16 (2 bytes) |
| World size `P` | 32 GPUs (4 nodes × 8) |
| Parallelism | pure DDP (no ZeRO, no TP, no PP) |
| Achievable per-rank bandwidth `B` | 25 GB/s (the inter-node tier is the limit) |
| Per-ring-step latency `α` | 20 µs |
| Measured **compute-only** step time | 400 ms |
| Gradient accumulation | 1 micro-batch per optimizer step |

## Tasks

**(a) Message size.** How many bytes does the gradient all-reduce carry? Give it in
both bytes and MiB.

**(b) Ring all-reduce time.** Using
`T = 2(P-1)α + (2(P-1)/P) · S/B`, compute the latency term, the bandwidth term, and
the total. State which term dominates and by how much.

**(c) Fraction of step time.** Express `T` as a fraction of the 400 ms step. Then
compute the two bracketing efficiencies:
   - **fully exposed** (no overlap at all): `efficiency = 400 / (400 + T)`
   - **perfectly overlapped**: what is the ceiling, and what has to be true for it?

**(d) The verdict — and be careful here.** The tool prints
`COMMUNICATION-BOUND` when comm exceeds 50% of step time. Do you agree with that
label for this setup? Defend your position. (There is a defensible answer on each
side; what is being tested is whether you know *what the label depends on*.)

**(e) Three fixes, quantified.** Propose three interventions and compute the effect
of each on the comm-to-compute ratio:
   1. Gradient accumulation of 4 micro-batches per optimizer step.
   2. Switching from DDP to ZeRO-2.
   3. Moving the DP group inside a node (300 GB/s) — and explain why this is
      usually not actually available to you.

**(f) Bucket sizing.** Compute the latency/bandwidth crossover `S* = P·α·B`. Then:
   - What does a 1 MiB bucket cost, in total, for this gradient?
   - What is the smallest bucket size you would choose, and why not larger?
   - PyTorch DDP defaults to a 25 MB bucket. Is that a good default *here*?

**(g) Scaling.** You are asked to go from 32 to 256 GPUs on the same fabric. What
happens to the bandwidth term, the latency term, and the global batch size? Which
of the three is most likely to end the project?

## Check your work

```powershell
python tools/collective_cost.py --gpus 32 --bw 25 --latency 20 --params 1.3 --step-ms 400 --buckets 1,8,16,25,100
python tools/collective_cost.py --gpus 256 --bw 25 --latency 20 --params 1.3 --step-ms 400
python tools/collective_cost.py --gpus 32 --bw 25 --latency 20 --layer-ratio --hidden 4096 --seq 4096 --grad-accum 1
python tools/collective_cost.py --gpus 32 --bw 25 --latency 20 --layer-ratio --hidden 4096 --seq 4096 --grad-accum 4
```

## What a good answer looks like

- Part (b) shows both terms separately. If you only report the total you have not
  demonstrated that you know which regime you are in.
- Part (d) argues from **overlap**, not from the threshold. The number 50% is a
  heuristic in a script; the physics is "can this traffic hide behind the backward
  pass?"
- Part (f) gives a bucket size with a reason on *both* sides — why not smaller
  (latency) and why not larger (nothing left to overlap with).
- Part (g) identifies that the two communication terms behave completely
  differently under scaling, and that neither is necessarily the real limit.
