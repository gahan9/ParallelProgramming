# Solution 2 — Choose the parallelism

> All memory and communication figures are **analytic models**
> ([`../tools/`](../tools/)). Hardware peaks (192 GiB HBM, ~5.3 TB/s on MI300X) are
> vendor specifications; published measurements are attributed.

The four scenarios differ in their **binding constraint**, and that is the whole
lesson. Scenarios A and C differ only in `s` and get completely different answers.

| Scenario | Binding constraint | Answer |
|---|---|---|
| A — 8B, too slow | throughput, then batch statistics | `dp=64` + ZeRO-2 + gradient accumulation 8 |
| B — 65B, 5× too large | state memory | `tp=8` intra-node × `dp=8` + ZeRO-1 + SP + selective recompute |
| C — 8B at 128K | activation memory (quadratic term) | FlashAttention + `cp=8` × `dp=8` + ZeRO-2 + selective recompute |
| D — 175B serving | decode latency + weight memory | `tp=8` in one node, replicate engines for QPS, **no PP** |

---

## Scenario A — "It fits, but it trains too slowly"

**1. Binding constraint: throughput.** Confirm it is not memory:

```
state       = 8e9 * 16 = 128e9 bytes = 119.2 GiB
activations = 34*s*b*h*L (selective) = 34*4096*1*4096*32 = 18.3e9 = 17.0 GiB
total       = 119.2 + 17.0 + 10 = 146.2 GiB  <=  192 GiB    -> fits, as claimed
```

(Without recomputation it is `4096*4096*(34 + 160) = 3.25e9` per layer = 3.03 GiB,
×32 = 97 GiB, total 226 GiB — it does *not* fit. The word "selective" in the problem
statement is load-bearing.)

**2. Axes: `dp = 64`, `tp = 1`, `pp = 1`.** There is no reason to split a model that
fits. Add **ZeRO-2**, which is free (§3.2): state drops from 119.2 to
`2Ψ + 14Ψ/64 = 14.9 + 1.6 = 16.5 GiB`, freeing ~103 GiB. Spend that on micro-batch
4, which improves per-GPU arithmetic efficiency.

**3. Tier and bandwidth requirement.** The gradient all-reduce is
`8e9 × 2 = 16e9 bytes = 14.9 GiB`, and `dp=64` necessarily spans all 8 nodes, so it
runs on the 50 GB/s tier:

```
T = 2*(63/64) * 16e9 / 50e9 + 2*63*15e-6 = 630 ms + 1.9 ms = 632 ms
```

Against a 700 ms step that is **90%** — and the backward pass is only ~2/3 of the
step (~467 ms), so **632 ms of traffic cannot hide behind it.** Pure DDP at
accumulation 1 fails here even with perfect bucketing.

The fix is **gradient accumulation**, driven by the batch target:

```
target global batch = 2e6 tokens / 4096 tokens per sequence = ~488 sequences (call it 512)
sequences per optimizer step per rank = 512 / 64 = 8
```

With accumulation 8 the all-reduce fires **once per 8 micro-batches**:

```
comm / compute = 632 ms / (8 * 700 ms) = 11.3%     -> comfortably hideable
```

Minimum viable bandwidth, from the same relation with a 20% budget:
`B ≥ 2Ψ_bytes / (0.2 × accum × T_compute) = 16e9 × 1.97 / (0.2 × 5.6 s) ≈ 28 GB/s`.
The 50 GB/s tier clears it; a 10 GB/s tier would not, and would force either
accumulation ≥ 32 or a hierarchical all-reduce.

**4. Failure mode accepted: the critical batch size.** Data parallelism buys
throughput by growing the global batch, and past the critical batch size extra
tokens per step stop buying convergence — you get *fewer optimizer steps* for the
same tokens. Symptom: loss-versus-wall-clock improves while loss-versus-tokens gets
worse; the fix (LR scaling, longer warmup) has limits.

**5. What would change the answer.** Inter-node bandwidth: below ~28 GB/s the DP
all-reduce stops hiding at accumulation 8, and you would move to hierarchical
all-reduce or ZeRO-2 with a smaller world size. Also: if the critical batch size
were 500K rather than 2M tokens, 64-way DP would be statistically wasteful and
you would use fewer, better-fed GPUs.

**Axis deliberately not used:** tensor parallelism. Splitting a model that fits adds
~4 all-reduces per layer on the critical path in exchange for nothing.

---

## Scenario B — "The model is 5× too large for one GPU"

**1. Binding constraint: state memory, with activations a close second.**

```
state       = 65e9 * 16 = 1.04e12 bytes = 968.9 GiB   (5.05 GPUs)
activations = 4096*8192*(34 + 5*64*4096/8192) = 4096*8192*194 = 6.51e9 = 6.06 GiB/layer
            = x 80 layers = 485 GiB          (also over budget on its own)
```

**2. Axes: `tp=8` × `dp=8` = 64, ZeRO-1 on the DP mesh, sequence parallelism,
selective recomputation.**

```
state:  params 2Psi/8   = 15.1 GiB
        grads  2Psi/8   = 15.1 GiB
        opt    12Psi/8  = 90.8, /8 over dp with ZeRO-1 = 11.4 GiB
                          total 41.6 GiB
activ:  selective + TP/SP = 85 GiB / 8 = 10.6 GiB
total:  41.6 + 10.6 + 10 = 62.2 GiB per GPU        -> 130 GiB of headroom
```

That headroom is not waste: spend it on a larger micro-batch, which improves
arithmetic efficiency and amortises the DP all-reduce.

**Why not the simpler answer — pure FSDP over 64 ranks?** It fits on memory
(`968.9/64 + 3.0 transient + 85 = 106 GiB`) but fails on throughput:

```
ZeRO-3 traffic  = 3 * Psi_bytes = 3 * 65e9 * 2 = 390e9 bytes per rank per step
at 50 GB/s      = 7.8 s
compute/micro-batch = 6 * 65e9 * 4096 / 400e12 = 4.0 s
ratio           = 1.95x   -> cannot hide even with perfect overlap
```

And gradient accumulation only helps one third of that traffic: the parameter
all-gathers are needed for *every* micro-batch's forward and backward, while only
the gradient reduce-scatter amortises. **This is the concrete case where "just use
FSDP" is the wrong answer**, and it is decided by one bandwidth number.

**3. Tier and bandwidth requirement.**

- **TP on the intra-node tier.** 4 all-reduces/layer × 80 layers = 320 messages of
  `s·b·h·2 = 64 MiB`. At 300 GB/s: `320 × 0.461 ms = 148 ms` against
  `6·65e9·4096/(8 × 400e12) = 499 ms` of compute → **30% overhead, on the critical
  path** (it cannot be overlapped away). At 50 GB/s the same traffic costs
  `320 × 2.559 ms = 819 ms` → **164% of compute**, i.e. a 2.6× slower step. So the
  requirement is roughly **≥ 200 GB/s** to stay under a ~45% penalty, which is
  exactly why TP is node-local.
- **DP on the inter-node tier.** The all-reduce is over the *sharded* gradient
  (`65e9/8 × 2 = 16.25e9` bytes) across `dp=8`: `2·(7/8)·16.25e9/50e9 = 569 ms` per
  optimizer step, amortised over accumulation.

**4. Failure mode accepted:** TP degree is capped at `min(GPUs per node, heads) = 8`,
and a TP group is a fate-sharing unit — lose one GPU and the whole group's step
fails. The 30% TP overhead is structural, not tunable. Symptom of trouble: step
time scaling worse than linearly when you widen TP.

**5. What would change the answer.** If inter-node bandwidth were ~300 GB/s, pure
FSDP over 64 becomes competitive and much simpler — worth revisiting on every
hardware generation. If nodes had 4 GPUs instead of 8, `tp=4` would leave state at
83 GiB and you would add `pp=2` across nodes to cover the gap.

---

## Scenario C — "We need 128K context"

**1. Binding constraint: activation memory, and specifically its quadratic term.**

```
5*a*s/h   = 5 * 32 * 131072 / 4096 = 5120        (was 160 at s=4096)
per layer = 131072 * 1 * 4096 * (34 + 5120) = 2.767e12 bytes = 2577 GiB PER LAYER
all 32    = 82,464 GiB per GPU
```

Eighty-two *thousand* gibibytes. The `34` is now 0.66% of the total — this is
entirely an attention-score problem. Sharding cannot close a four-order-of-magnitude
gap: `tp=8` still leaves 10,308 GiB.

**2. The answer, in the order that matters:**

1. **Tiled (FlashAttention-style) attention — a prerequisite, not an
   optimisation.** It removes the `5as/h` term *structurally* by never
   materialising the `s × s` score matrix. Note what the analytic model hides
   here: even with full recomputation the *transient* score matrix during a naive
   recompute is `s² × 2 bytes = 131072² × 2 = 34.4 GB per head`. No amount of
   recomputation or sharding survives that. If you answered "full recompute" and
   stopped, the tool agreed with you and the hardware would not.
2. **Selective recomputation** — with the quadratic term gone,
   `34·s·b·h = 17.0 GiB` per layer, ×32 = 544 GiB. Still too big alone.
3. **Context / ring parallelism `cp = 8`** — shard the sequence itself
   ([Ring Attention, arXiv:2310.01889](https://arxiv.org/abs/2310.01889)):
   `544 / 8 = 68 GiB`.
4. **ZeRO-2** for the state: `119.2 → 16.5 GiB` (free).

```
total = 16.5 (state) + 68 (activations) + 10 (reserve) = 94.5 GiB per GPU   -> fits
axes  = cp=8 (intra-node) x dp=8 (across nodes)
```

Why context parallelism rather than full recomputation everywhere? Full recompute
would fit (`2sbhL = 32 GiB`) but costs ~33% of your FLOPs. Context parallelism
costs communication that **overlaps with attention compute**, which is the better
trade at long context.

**3. Tier and bandwidth requirement.** Ring attention passes K/V blocks around the
`cp` ring: `2 · (s/cp) · h · 2 bytes = 2 · 16384 · 4096 · 2 = 268 MB` per ring step.
Intra-node at 300 GB/s that is ~0.9 ms per step, ~7 ms for the full ring — trivially
hidden behind attention compute at 128K tokens. Notably, at 50 GB/s it is ~43 ms and
*still* plausibly hidden, so **context parallelism tolerates crossing nodes far
better than tensor parallelism does** — because its communication is overlappable
and its compute per byte is enormous at long sequence.

**4. Failure mode accepted:** causal masking makes the ring's work **uneven** — the
rank holding early tokens attends over less history than the rank holding late
tokens, so a naive contiguous split leaves ranks idle. Production implementations
use striped/zig-zag token assignment to rebalance. Symptom: per-rank attention time
increasing monotonically with rank index in a profile.

**5. What would change the answer.** Sequence length, obviously — but state the
threshold: drop context parallelism once `34·s·b·h·L` (or `2sbhL` with full
recompute) fits with margin. At `s = 8192` that is 34 GiB selective / 2 GiB full,
so `cp` would be pure overhead. Also: GQA shrinks `h_kv` and therefore the ring's
K/V traffic, making longer contexts cheaper to shard.

---

## Scenario D — "Low-latency serving of a large model"

This is an inference problem, and the axes change meaning (§3.8).

**1. Binding constraint: decode latency, plus weight memory.** Decode is
**memory-bandwidth-bound**: producing one token requires reading *every* weight.

```
weights (bf16) = 175e9 * 2 = 350e9 bytes = 326.0 GiB      (> 192 GiB: does not fit)
one GPU-equivalent read at 5.3 TB/s = 326.0 * 2^30 / 5.3e12 = 66 ms per token
                                                     -> 2.6x over a 25 ms SLO
```

**2. Axes: `tp = 8` inside one node; replicate whole engines with `dp` for QPS;
`pp = 1`.**

```
per-GPU weight read with tp=8 = 40.8 GiB -> 40.8 * 2^30 / 5.3e12 = 8.3 ms per token
at a realistic 70% of peak HBM bandwidth  -> ~11.8 ms
```

**This is the actual mechanism by which tensor parallelism cuts latency: it divides
the per-GPU weight read.** It is not "more FLOPs"; decode barely uses the FLOPs.

**No pipeline parallelism.** A request must traverse every stage, so its latency is
the *sum* of stage latencies. PP buys throughput at the cost of the exact metric in
the SLO. Use it only if the SLO were throughput-shaped.

**3. Tier and bandwidth requirement — and the trap.** The decode all-reduce carries
`b·h·2 = 8 × 12288 × 2 = 192 KiB`:

```
python tools/collective_cost.py --gpus 8 --bw 300 --latency 5 --bytes 196608
-> single message 0.19 MiB: 0.071 ms (latency 70.0 us + wire 0.001 ms)
```

The wire time is **1 microsecond**; the other 70 µs is pure ring latency. With
2 all-reduces × 96 layers the ring model charges 13.6 ms per token — over half the
SLO. Reality is better because NCCL/RCCL switch to one-shot/tree algorithms for
small messages (single-digit-to-low-tens of µs), giving ~2–4 ms. Either way the
conclusion is the same and it is the opposite of the training conclusion:

> **Decode-phase tensor parallelism is bounded by interconnect *latency* and
> algorithm selection, not bandwidth.** A fabric with excellent bandwidth and
> mediocre small-message latency will look great in training and disappoint in
> serving. Measure it: `rccl-tests all_reduce_perf -b 8 -e 1M`.

Prefill is the opposite: its messages are `s·b·h·2 = 96 MiB`, firmly
bandwidth-bound. **You need a fabric that is good at both**, and the two phases
should be benchmarked separately.

**Latency budget estimate:** ~8–12 ms weight read + ~3 ms collectives + kernel
launch and sampling overhead ≈ **12–18 ms**, inside the 25 ms SLO with margin.

**4. The KV cache constraint (the part people forget).** With `TP = t`, rank `r`
owns a subset of attention heads, so it must own K and V for *exactly those* heads —
otherwise every decode step needs a cache reshuffle across ranks.

```
KV bytes per token (MHA, bf16) = 2 (K,V) * L * h * 2 = 2 * 96 * 12288 * 2 = 4.72 MB
8 sequences x 4096 tokens = 32,768 tokens -> 154.6 GB total
per rank with tp=8                        -> 19.3 GB = 18.0 GiB
per-GPU total: 40.8 (weights) + 18.0 (KV) + overhead ~= 60 GiB of 192 GiB
```

Plenty of room to raise the batch — which is the throughput knob you *do* have.
**The hard cap: `tp ≤ number of KV heads`.** GPT-3-175B is multi-head (96 heads), so
`tp=8` is fine; a GQA model with 8 KV heads cannot run `tp=16` without replicating
KV and giving up the memory saving. Paging and batching of that cache is
[B06](../../B06.inference-serving-optimizations/)'s subject
([PagedAttention, arXiv:2309.06180](https://arxiv.org/abs/2309.06180)); the
partitioning theory is Pope et al.
([arXiv:2211.05102](https://arxiv.org/abs/2211.05102)).

**5. Failure mode accepted:** the TP all-reduce sits on the critical path of every
layer, so fabric jitter lands directly in p99 time-per-output-token; and `tp` cannot
grow past the node without latency collapsing. **What would change the answer:** a
throughput SLO at batch 256 would favour lower TP plus replication (and possibly PP);
a GQA model changes both the TP cap and the cache arithmetic.

**Axis deliberately not used across all four scenarios:** expert parallelism — every
model here is dense, so there are no experts to place, and MoE's all-to-all would be
pure cost.
