# Solution 3 — The communication budget

> Every figure is from the analytic ring model in
> [`../tools/collective_cost.py`](../tools/collective_cost.py). It is a *ring-only*
> model: below the crossover it overstates cost, because real NCCL/RCCL switch to
> tree and one-shot algorithms.

## (a) Message size

```
S = 1.3e9 params * 2 bytes (bf16) = 2.6e9 bytes
  = 2.6e9 / 2^20 = 2479.6 MiB ~= 2480 MiB = 2.42 GiB
```

## (b) Ring all-reduce time

```
latency term    = 2*(P-1)*alpha       = 2 * 31 * 20e-6      =   1.24 ms
bandwidth term  = (2*(P-1)/P) * S/B   = 1.9375 * 2.6e9/25e9 = 201.5  ms
total T                                                     = 202.7  ms
```

**The bandwidth term dominates by 162×.** This is the normal state of affairs for a
whole-model gradient buffer: 2.4 GiB is three orders of magnitude above the
crossover `S* = 15.26 MiB` (part f). Buying lower latency here would be a wasted
purchase; the only levers are volume, bandwidth, and overlap.

## (c) Fraction of step time, and the two brackets

```
T / step            = 202.7 / 400 = 50.7%
fully exposed       = 400 / (400 + 202.7) = 66.4% efficiency
perfectly overlapped= 100% ceiling
```

The overlap ceiling is only reachable if the traffic fits *inside* the backward
pass, because that is the only window in which gradients exist and compute is still
running. Backward is roughly `4/6` of the step's FLOPs:

```
T_backward ~= 400 * 2/3 = 267 ms
T_comm     =  202.7 ms   <   267 ms      -> theoretically hideable, with ~64 ms slack
```

So: hideable in principle, but with only 24% margin, and the last bucket can never
overlap with anything (nothing is left to compute after the final gradient).
Realistically expect 80–90% efficiency, not 100%.

## (d) The verdict — and why the label is not the answer

The tool prints `COMMUNICATION-BOUND` because 50.7% > 50%. **That threshold is a
heuristic in a script, and here it is misleading.** The physics is not "is comm more
than half the step," it is **"does comm fit inside the backward pass, and does your
implementation actually overlap it?"**

The defensible verdict: **communication-*sensitive*, not communication-bound.**
`202.7 ms < 267 ms` of backward compute, so a correctly bucketed, overlapped DDP can
hide almost all of it. But the margin is thin enough that anything which breaks
overlap turns the job comm-bound immediately — for example:

- a bucket size below `S*` (see part f),
- gradient clipping or a norm computed *before* the reduce, forcing a full barrier,
- a `.item()` / `.cpu()` call in the training loop that synchronises,
- a straggler that shifts when gradients become available.

**How you would settle it for real:** profile one step and look at whether the
RCCL/NCCL kernels are concurrent with backward compute kernels. If they are
serialised, you are comm-bound regardless of what the arithmetic says; if they
overlap, you are not. This is the difference between a model and a measurement, and
it is the answer a reviewer wants to hear.

## (e) Three fixes, quantified

**1. Gradient accumulation of 4.** The all-reduce fires once per optimizer step, so
its cost is unchanged while compute quadruples:

```
comm/compute = 202.7 / (4 * 400) = 12.7%      (was 50.7%)  -> 4x improvement
```

Zero extra memory, no code restructuring, and it also moves you toward a larger
global batch (watch the critical batch size). **This is the first thing to try.**

**2. ZeRO-2.** Wire volume is *identical* — an all-reduce already *is* a
reduce-scatter plus an all-gather (§2.4), so ZeRO-2 rearranges which halves each
rank keeps, not how many bytes cross the link:

```
comm time: 202.7 ms      (unchanged)
memory:    16 Psi -> 2 Psi + 14 Psi/32 = 2.42 + 1.06 = 3.48 GiB (was 19.4 GiB)
```

So ZeRO-2 is **not a communication optimisation directly** — it is a free memory
optimisation that *enables* one, because the freed memory buys a bigger micro-batch,
which improves the ratio exactly as accumulation does. Turn it on regardless; there
is no downside.

**3. Move the DP group intra-node (300 GB/s).**

```
bandwidth term = 201.5 * 25/300 = 16.8 ms
total          ~= 18.0 ms  ->  4.5% of the step        -> 11x improvement
```

**Why this is usually not available:** a `dp=32` group cannot be intra-node when
nodes hold 8 GPUs. What you can actually do is a **hierarchical all-reduce** —
reduce-scatter within each node (300 GB/s), all-reduce the 8× smaller partials
across the 4 node leaders (25 GB/s), then all-gather within each node. That cuts the
inter-node volume by the node size:

```
inter-node bytes ~= 2*(4-1)/4 * (2.6e9/8) = 487e6 bytes -> ~19.5 ms at 25 GB/s
```

This is what NCCL/RCCL's topology-aware algorithms already attempt, which is why a
measured all-reduce often beats this flat ring model on multi-node clusters. Verify
with `rccl-tests` rather than assuming either direction.

## (f) Bucket sizing

```
S* = P * alpha * B = 32 * 20e-6 * 25e9 = 16.0e6 bytes = 15.26 MiB
```

```powershell
python tools/collective_cost.py --gpus 32 --bw 25 --latency 20 --params 1.3 --step-ms 400 --buckets 1,8,16,25,100
```

```text
 bucket MiB  count   time ms  latency %  overlap?  bandwidth-bound?
------------------------------------------------------------------
          1   2480    3276.7       93.9  yes       no
          8    310     585.9       65.6  yes       no
         16    155     393.7       48.8  yes       yes
         25    100     325.5       38.1  yes       yes
        100     25     232.5       13.3  yes       yes
```

- **A 1 MiB bucket costs 3277 ms** — 16× the ideal 201.5 ms. Every one of the 2480
  buckets pays the full `1.24 ms` ring latency: `2480 × 1.24 = 3075 ms` of pure
  launch overhead for the same bytes. This single row is the entire justification
  for gradient bucketing.
- **Smallest sensible bucket: not `S*` itself.** At exactly `S*` the two terms are
  equal *by construction*, so you pay 2× the ideal (393.7 vs 201.5 ms). Pick a
  few multiples of `S*`: at 100 MiB you are within 15% of ideal and still have 25
  buckets — enough granularity to overlap with the backward pass. **Answer: ~64–100
  MiB here.**
- **Why not larger still?** Because a bucket cannot start reducing until it is full,
  so one giant bucket serialises *all* communication after *all* backward compute —
  the ideal 201.5 ms becomes 201.5 ms of fully exposed time. The optimum is the
  smallest bucket that is comfortably bandwidth-bound, which is exactly the tension
  the tool's two right-hand columns encode.
- **Is PyTorch's 25 MB default good here?** It is *safe* but not great: 23.8 MiB is
  only 1.6× `S*`, giving 325.5 ms — 62% over ideal, with 38% of the time in latency.
  On this deliberately slow, wide (32-rank, 25 GB/s, 20 µs) fabric you should raise
  it. On a typical 8-rank 300 GB/s / 5 µs setup, `S* = 12 MB` and 25 MB is a
  sensible 2×. **The default is tuned for common hardware; `S* = P·α·B` tells you
  when your hardware is not common.**

## (g) Scaling from 32 to 256 GPUs

```powershell
python tools/collective_cost.py --gpus 256 --bw 25 --latency 20 --params 1.3 --step-ms 400
```

```text
DDP gradient all-reduce for 1.3B params @ 2B: 2480 MiB -> 217.4 ms
  fraction of a 400 ms step: 54.3%
latency/bandwidth crossover S* = P*alpha*B = 122.07 MiB
```

| Term | at P=32 | at P=256 | Behaviour |
|---|---:|---:|---|
| bandwidth `2(P-1)/P · S/B` | 201.5 ms | 207.0 ms | **+2.7%** — essentially flat, as `2(P-1)/P → 2` |
| latency `2(P-1)α` | 1.24 ms | 10.2 ms | **8× worse**, still only 5% of the total |
| crossover `S* = P·α·B` | 15.26 MiB | **122.07 MiB** | 8× larger — your 25 MB buckets are now *latency-bound* |
| global batch | 32 units | **256 units** | 8× larger |

Three observations, in increasing importance:

1. **The bandwidth term barely moves.** Ring all-reduce is rank-independent for
   large messages — this is why the algorithm won.
2. **Bucket sizing must be revisited.** `S*` grew 8×, so the bucket that was fine at
   32 ranks is now below the crossover: the 25 MiB row costs 1227 ms at 256 ranks
   versus 325 ms at 32. **Scaling out silently invalidates your bucket tuning.**
3. **The global batch grew 8×, and that is what ends the project.** Neither
   communication term is the real limit; the statistics are. If 32 ranks was already
   near the critical batch size, 256 ranks does not train the model 8× faster — it
   takes 8× fewer optimizer steps per epoch and converges worse. The honest answer
   to "can we scale to 256?" is "not with pure data parallelism," and the fix is to
   spend the extra ranks on a different axis (TP/PP/ZeRO) rather than on more batch.
