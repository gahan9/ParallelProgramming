# Solution 1 — Fit the model

> All figures below come from the analytic model in
> [`../tools/memory_planner.py`](../tools/memory_planner.py). Nothing here was
> measured on hardware. Published figures are attributed inline.

## (a) State memory

Per-parameter accounting for bf16 compute + fp32 master + Adam:

| Tensor | dtype | bytes |
|---|---|---:|
| parameter | bf16 | 2 |
| gradient | bf16 | 2 |
| fp32 master weight | fp32 | 4 |
| Adam `m` | fp32 | 4 |
| Adam `v` | fp32 | 4 |
| **total** | | **16** |

```
state = 175e9 * 16 = 2.80e12 bytes
      = 2.80e12 / 2^30 = 2607.7 GiB
      = 2607.7 / 192 = 13.6 GPUs of pure state
```

**Thirteen and a half GPUs before a single activation exists.** This is the ZeRO
paper's `16Ψ` figure ([arXiv:1910.02054](https://arxiv.org/abs/1910.02054)).

## (b) Activation memory

```
5*a*s/h      = 5 * 96 * 2048 / 12288 = 983040 / 12288 = 80
per layer    = s*b*h * (34 + 80)
             = 2048 * 1 * 12288 * 114
             = 25,165,824 * 114 = 2.869e9 bytes = 2.672 GiB
all 96 layers= 96 * 2.672 = 256.5 GiB
```

**State (2607.7 GiB) is ~10× the activations (256.5 GiB).** This matters for the
choice of fix: the axes that shard *state* (ZeRO, TP, PP) are the ones that will
move the needle here, and recomputation alone can never be enough. Compare
Scenario C of drill 2, where the ratio is inverted by three orders of magnitude —
the same model family, a different sequence length, and a completely different
answer.

## (c) Ruling out the cheap options (8 GPUs, 182 GiB usable after reserve)

Component sizes, since they recur: `2Ψ = 326.0 GiB` (params), `2Ψ = 326.0 GiB`
(grads), `12Ψ = 1955.8 GiB` (optimizer).

| Strategy | Per-GPU state | + activ | Total | Why it fails |
|---|---:|---:|---:|---|
| DDP | 2607.7 | 256.5 | **2874.2** | Nothing is sharded; every rank holds all 13.6 GPUs' worth |
| ZeRO-1 | 326.0 + 326.0 + 244.5 = 896.4 | 256.5 | **1162.9** | The *replicated* params + grads alone are 652 GiB |
| ZeRO-2 | 326.0 + 40.7 + 244.5 = 611.2 | 256.5 | **877.7** | The replicated **parameters** alone are 326 GiB > 192 |
| ZeRO-3 | 2607.7/8 + 6.8 transient = 332.8 | 256.5 | **599.3** | Even *perfectly* sharded state is 326 GiB > 182 usable |

The ZeRO-3 row is the informative one: it shows the wall is not "we forgot to
shard something," it is the total divided by 8.

## (d) Does *anything* fit on 8 GPUs? No — and here is the proof

The three axes that reduce state do so by their degrees, and those degrees multiply
to at most the GPU count:

```
state_per_gpu >= 16*Psi / (tp * pp * dp_shard)   and   tp * pp * dp <= N
=> best possible state_per_gpu = 16*Psi / N
```

Recomputation does not touch state at all, so the bound is unconditional. With
`N = 8`:

```
16 * 175e9 / 8 = 350e9 bytes = 325.96 GiB
usable = 192 - 10 (reserve) - activations(>= 0) <= 182 GiB
325.96 > 182   ->  no configuration exists
```

**Answer: it is impossible.** Not "hard," not "needs tuning" — the arithmetic
closes the question for every combination of ZeRO stage, TP, PP, and recompute
mode. This is why memory budgeting comes *before* strategy selection: the answer
to "which parallelism?" was "buy more GPUs."

Verify:

```powershell
python tools/memory_planner.py --preset gpt3-175b --gpus 8 --suggest
```

```text
strategy                        axes                      state    activ  reserve    total  fits 192 GiB?
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
nothing fits: add GPUs, shorten the sequence, or cut the micro-batch
```

## (e) Minimum GPU count

Invert the bound from (d), leaving room for the cheapest activation mode
(full recompute: `2sbhL = 2*2048*12288*96 = 4.83e9 bytes = 4.5 GiB`):

```
16*Psi/N + 4.5 + 10 <= 192
16*Psi/N            <= 177.5 GiB
N                   >= 2607.7 / 177.5 = 14.7   ->  N >= 15  ->  N = 16 (power of two)
```

At `N = 16`, ZeRO-3 + **full** recomputation:

```
state       = 2607.7/16 = 163.0  + 6.8 transient (ZeRO-3 prefetch) = 169.8 GiB
activations = 4.5 GiB   (full recompute)
reserve     = 10 GiB
total       = 184.3 GiB     headroom = 7.7 GiB (4.0%)
```

Selective recomputation would need `34*s*b*h*L = 76.5 GiB`, giving 256.3 GiB —
**does not fit**. So at 16 GPUs you are forced into full recomputation and its
~33% extra FLOPs. And 4% headroom is not headroom: one fragmentation event or one
larger logit buffer and you OOM. Slightly better at the same 16 GPUs is
`tp=2` or `pp=2`, which also divides the ZeRO-3 transient buffer:
`166.4 + 4.5 + 10 = 180.9 GiB`.

**16 GPUs is technically the minimum and practically a bad idea.**

## (f) The configuration to actually run, at N = 32

```
state       = 2607.7/32 = 81.5 + 6.8 transient = 88.3 GiB
activations = 76.5 GiB  (selective recompute)
reserve     = 10 GiB
total       = 174.8 GiB     headroom = 17.2 GiB (9%)
```

**Primary answer: FSDP / ZeRO-3 across all 32 ranks + selective activation
recomputation + sequence parallelism if you enable any TP.**

Justification against the alternatives:

| Alternative | Total | Why not |
|---|---:|---|
| ZeRO-3 + **full** recompute | 102.8 GiB | Much more headroom, but ~33% extra FLOPs instead of ~2.7% ([arXiv:2205.05198](https://arxiv.org/abs/2205.05198) for GPT-3). Paying 30% of your cluster for headroom you do not need is the wrong trade — keep this as the fallback if fragmentation bites. |
| `tp=2` + ZeRO-3(dp=16) + selective | 133.1 GiB | **Arguably the better engineering answer**: it halves activations *and* the transient buffer, giving 59 GiB of headroom, at the cost of one intra-node all-reduce pair per layer. Choose this if the two ranks are on the same node and you have measured that TP traffic overlaps acceptably. |
| `pp=2` + ZeRO-3 + selective | 171.4 GiB | Saves almost nothing (PP does not touch activations) and adds a bubble that needs `M ≥ 9` micro-batches for 90% efficiency. All cost, no benefit at this scale. |
| Pure TP (`tp=8` per node) | see §3.3 | Caps at 8 (GPUs per node); crossing nodes with TP costs 2.9× the compute in communication. |

**What the model is not telling you** — the honest caveats:

1. **No fragmentation, no allocator behaviour.** The 10 GiB reserve is a fudge
   factor. Calibrate it against one real run before trusting 9% headroom.
2. **No vocabulary/logit tensor.** For GPT-3's 50257-token vocabulary the fp32
   logits at `s=2048` are ~412 MB per copy, and the softmax needs more than one.
   That belongs in the reserve.
3. **Micro-batch 1 with `s=2048` and `dp=32` gives a global batch of ~65K tokens**,
   against the ~3.2M tokens GPT-3 actually used
   ([arXiv:2005.14165](https://arxiv.org/abs/2005.14165)). You will need heavy
   gradient accumulation — which costs no extra memory and *improves* the
   communication ratio (see drill 3), but it means your real step time is ~50×
   what a single micro-batch suggests.
4. **Nothing here is a throughput prediction.** Fitting is necessary, not
   sufficient. The next question is whether FSDP's `3Ψ` of traffic overlaps on your
   fabric — that is drill 3's subject.
