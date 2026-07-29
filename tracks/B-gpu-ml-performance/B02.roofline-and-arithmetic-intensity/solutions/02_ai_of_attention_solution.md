# Solution 2 — The arithmetic intensity of attention

Reference answers for [`../exercises/02_ai_of_attention.md`](../exercises/02_ai_of_attention.md).

Device R: `P_m = 1200 TFLOP/s` (bf16 matrix cores), `B = 4.0 TB/s`, 192 GB HBM.
bf16 = 2 bytes. `s` = sequence length, `d` = head dimension. One head, forward only.

---

## Part A — FLOPs

**A1.** A matrix product of shape `(m × k) · (k × n)` costs `2mnk` FLOPs (one multiply and one
add per MAC).

```
Q · Kᵀ : (s × d) · (d × s)  ->  2·s·s·d = 2s²d
P · V  : (s × s) · (s × d)  ->  2·s·s·d = 2s²d
softmax: 5 FLOPs per element of the s × s matrix  ->  5s²

F(s, d) = 4s²d + 5s²  =  s²(4d + 5)
```

**A2.** At `s = 8192`, `d = 128`:

```
GEMM part    = 4 · 8192² · 128 = 3.436e10 FLOPs
softmax part = 5 · 8192²       = 3.355e8  FLOPs   ->  0.97% of the total
```

Dropping the softmax costs you ~1% — fine for an envelope estimate, and the reason
`F ≈ 4s²d` is the number everyone quotes. You must **not** drop it when `d` is small
(the softmax share is `5 / (4d + 5)`, so at `d = 8` it is 13%), and you must not drop it
when reasoning about *time* rather than FLOPs: `exp` runs on the transcendental unit at a
fraction of FMA throughput, and the row-wise max and sum impose reductions and
synchronization that the FLOP count does not capture at all.

---

## Part B — Bytes

**B1. Unfused.** Count elements crossing HBM per stage, then multiply by 2 bytes:

| Stage | reads | writes | elements |
|---|---|---|---|
| `Q · Kᵀ` | `Q` (`sd`), `K` (`sd`) | `S` (`s²`) | `2sd + s²` |
| softmax | `S` (`s²`) | `P` (`s²`) | `2s²` |
| `P · V` | `P` (`s²`), `V` (`sd`) | `O` (`sd`) | `s² + 2sd` |
| **total** | | | **`4sd + 4s²`** |

```
bytes_unfused = 2 · (4sd + 4s²) = 8sd + 8s²
```

**B2. Fused.** `Q`, `K`, `V` are each read once and `O` is written once. The `s × s`
intermediate is produced and consumed in on-chip memory, tile by tile, and never lands.

```
elements = 3sd (Q,K,V) + sd (O) = 4sd
bytes_fused = 2 · 4sd = 8sd
```

**B3.**

```
bytes_unfused / bytes_fused = (8sd + 8s²) / (8sd) = 1 + s/d
```

At `d = 128`: `s = 512` → **5×** more traffic; `s = 8192` → **65×** more traffic.
The ratio grows without bound in `s`. That is the entire FlashAttention argument in one line.

---

## Part C — Arithmetic intensity

**C1.**

```
I_unfused = s²(4d + 5) / (8sd + 8s²)
          = s²(4d + 5) / (8s(d + s))
          = s(4d + 5) / (8(d + s))
```

**C2.** Take `s → ∞` with `d` fixed:

```
lim I_unfused = (4d + 5) / 8  ≈  d/2
```

At `d = 128` that ceiling is `517/8 = 64.6 FLOP/byte`. **Longer sequences do not help** because
the unfused implementation's traffic is dominated by the `s × s` score matrix, which grows at
exactly the same `O(s²)` rate as the FLOPs. Numerator and denominator scale together, so the
ratio saturates at a constant set by the head dimension alone. No context length rescues it.

**C3.**

```
I_fused = s²(4d + 5) / (8sd) = s(4d + 5) / (8d)
```

This is **linear in `s`** and unbounded: the FLOPs are `O(s²d)` while the traffic stays `O(sd)`.
At `d = 128`, `I_fused = 517s/1024 ≈ 0.505·s`.

**C4.** `d = 128`, ridge point 300 FLOP/byte (Part D1):

| `s` | `I_unfused` | `I_fused` | unfused verdict | fused verdict |
|---|---|---|---|---|
| 512 | 51.7 | 258 | memory-bound | memory-bound (just below the ridge) |
| 1024 | 57.4 | 517 | memory-bound | compute-bound |
| 2048 | 60.8 | 1034 | memory-bound | compute-bound |
| 4096 | 62.7 | 2068 | memory-bound | compute-bound |
| 8192 | 63.6 | 4136 | memory-bound | compute-bound |
| 32768 | 64.4 | 16544 | memory-bound | compute-bound |

The unfused column is flat and stuck. The fused column doubles every time `s` doubles.
(Applying the ±2× "near the ridge" dead band from Solution 3, `s = 512` and `s = 1024` fused
both land in the ambiguous band; from `s = 2048` on, fused attention is unambiguously
compute-bound.)

---

## Part D — Where fusion changes the classification

**D1.** `I_ridge = P_m / B = 1200e12 / 4.0e12 = **300 FLOP/byte**`.

**D2.** Fused, `d = 128`:

```
s(4d + 5) / (8d)  ≥  300
517·s / 1024      ≥  300
s                 ≥  307,200 / 517  =  594.2
```

So **`s ≥ 595`** (in practice: any sequence of 1K or longer). Fused attention is compute-bound
for essentially every sequence length used in production.

**D3.** Unfused, `d = 128`:

```
517·s / (8(128 + s))  ≥  300
517·s                 ≥  2400·(128 + s)
517·s                 ≥  307,200 + 2400·s
-1883·s               ≥  307,200
s                     ≤  -163
```

No positive solution — the inequality can never be satisfied. Consistent with C2: the
supremum of `I_unfused` is 64.6, which is 4.6× *below* the ridge. **Unfused attention is
memory-bound at every sequence length, on this machine, forever.** No amount of tuning the
three kernels changes that; only removing the HBM round-trip does. That is why FlashAttention
is an algorithmic change, not an optimization.

**D4.** `s = 8192`, `d = 128`, `F = 3.470e10 FLOPs`:

| | `I` | attainable rate | time | limited by |
|---|---|---|---|---|
| unfused | 63.6 | `63.6 × 4.0 TB/s = 254 TFLOP/s` | 136 µs | HBM bandwidth |
| fused | 4136 | `min(1200, 4136×4) = 1200 TFLOP/s` | 28.9 µs | matrix cores |

**Speedup ceiling from fusion alone: `1200 / 254 = 4.7×`** — and it grows with `s`. Note what
the model is claiming: not that fused code is better written, but that the unfused version is
forbidden by physics from exceeding 254 TFLOP/s on this hardware.

---

## Part E — The other reason to fuse

**E1.** Score matrix footprint in bf16:

```
one head, s = 8192 : 8192² × 2 B = 134.2 MB
batch 8 × 32 heads : 256 × 134.2 MB = 34.4 GB
```

That is **18% of Device R's 192 GB** for the intermediates of a *single* attention layer — and
training needs them retained (or recomputed) for the backward pass. Stack 80 layers and the
unfused formulation is not slow, it is impossible. Fusion is what makes long context fit at
all; the speedup is a bonus.

**E2.** Fused attention has `FLOPs = O(s²d)` and `HBM traffic = O(sd)`, so intensity grows
linearly in `s`. Longer context therefore drives the operator *deeper into the compute-bound
regime*, which is where a GPU is at its best: matrix cores near peak, bandwidth no longer the
constraint. Long context is, from the hardware's point of view, a *gift*.

The unfused implementation destroys that gift by making traffic `O(s²)` as well. Traffic then
grows exactly as fast as work, intensity saturates, and the operator stays pinned to the
bandwidth roof no matter how long the context gets. Same math, same hardware — one data-movement
decision converts a favourable asymptotic into a flat one.

---

## Part F — Stretch

**F1. Causal masking.** Assume tiles entirely above the diagonal are skipped, so roughly half
the score matrix is computed:

```
F_causal      ≈ 2s²d + 2.5s²                    (half of everything)
bytes_unfused ≈ 8sd + 4s²                        (half the score traffic)
bytes_fused   =  8sd                             (unchanged - it never wrote them anyway)

I_unfused(causal) = (2s²d + 2.5s²) / (8sd + 4s²) → (2d + 2.5)/4 ≈ d/2   as s → ∞
I_fused(causal)   = (2s²d + 2.5s²) / (8sd)       = s(2d + 2.5)/(8d)
```

**Unfused is unchanged asymptotically** (`≈ 64.6` at `d = 128`) — halving the FLOPs and halving
the dominant byte term cancels exactly. **Fused is halved**, because its traffic (`Q`, `K`, `V`,
`O`) did not shrink while its FLOPs did. Redoing D2:

```
s(2d + 2.5)/(8d) ≥ 300  ->  s ≥ 2400d/(2d + 2.5) = 307,200/258.5 = 1188.4  ->  s ≥ 1189
```

The crossover exactly doubles, from ~595 to ~1189. Useful sanity check on a real pattern: any
optimization that removes FLOPs without removing bytes *lowers* your intensity and pushes you
back toward memory-bound. Skipping work is not free.

**F2. Decode / incremental inference.** One new query token against a KV cache of length `s`:

| Stage | FLOPs | elements crossing HBM |
|---|---|---|
| `q · Kᵀ` | `2sd` | `d` (q) + `sd` (K) + `s` (scores) |
| softmax | `5s` | `2s` |
| `p · V` | `2sd` | `s` (p) + `sd` (V) + `d` (o) |
| **total** | **`4sd + 5s`** | **`2sd + 4s + 2d`** |

```
I_decode = (4sd + 5s) / (2·(2sd + 4s + 2d))  →  4sd / 4sd = 1.0 FLOP/byte   for s, d » 1
```

At `s = 8192`, `d = 128` the exact value is **0.994 FLOP/byte** — about **300× below the ridge**.
Decode attention is as memory-bound as an elementwise kernel.

**Fusion does not save you here** because there is no large intermediate to eliminate: the score
"matrix" is a length-`s` vector, and the traffic *is* the KV cache itself. Every generated token
must read all of `K` and `V` once. That is irreducible for a single sequence, which is why
decode throughput tracks HBM bandwidth almost linearly and why nobody quotes matrix-core
utilization for the decode phase.

**F3. Grouped-query attention.** With `g` query heads sharing one KV head, the shared `K`/`V`
traffic is paid once for `g` heads' worth of work:

```
FLOPs    ≈ g·(4sd + 5s)
elements ≈ 2sd  (K, V read once)  +  g·(2d + 2s)
I_gqa    ≈ g·4sd / (2·2sd) = g   FLOP/byte
```

**Intensity scales linearly with `g`.** GQA with `g = 8` moves decode attention from ~1 to
~8 FLOP/byte — an 8× reduction in time for the same output, achieved purely by making fewer
bytes serve more arithmetic. Still far from the ridge, but the win is real and it is a *roofline*
win, not a FLOP win.

The same reasoning, applied to a different tensor, explains inference batching: batching `n`
requests amortizes the *weight* reads of every GEMM across `n` sequences, so the projection and
MLP layers gain intensity proportional to batch size. Note the distinction — **GQA amortizes KV
reads across query heads; batching amortizes weight reads across sequences**. Each KV cache is
still private per sequence, so batching does *not* raise attention's own intensity. Two levers,
two different tensors; conflating them is a common interview stumble.
