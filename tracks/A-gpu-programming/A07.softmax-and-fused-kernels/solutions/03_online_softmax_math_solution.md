# Solution 3 — Online softmax: derivation, proof, and byte counts

Reference answers for [`../exercises/03_online_softmax_math.md`](../exercises/03_online_softmax_math.md).
Notation as in the exercise: `m_k = max(x_0..x_{k-1})`, `s_k = Σ_{i<k} exp(x_i − m_k)`, with the
empty prefix `(m_0, s_0) = (−∞, 0)`.

---

## Part 1 — The max trick is free

**1.** Multiply numerator and denominator by `exp(m)`:

```
exp(x_i − m)            exp(x_i)·exp(−m)          exp(x_i)
──────────────────  =  ────────────────────────  =  ────────────────
Σ_j exp(x_j − m)        exp(−m)·Σ_j exp(x_j)       Σ_j exp(x_j)
```

`exp(−m)` is a nonzero constant common to both, so it cancels. The shift is **exactly** free in
real arithmetic — softmax is invariant to adding any constant to all its inputs. (That invariance
is also why softmax has one redundant degree of freedom, and why logits are only defined up to a
per-row additive constant.)

**2.** Any `m` works mathematically, but only `m = max(x)` guarantees `x_i − m ≤ 0` for **every**
`i`, hence `exp(x_i − m) ∈ (0, 1]`. Every exponential is then bounded by 1 — overflow becomes
structurally impossible rather than merely unlikely.

**3.** With `m = max(x)`, at least one term is `exp(0) = 1` and every term is `≤ 1`, so

```
1 ≤ Σ_j exp(x_j − m) ≤ C
```

The normalizer is bounded by the row length. Even `C = 2^31` is nowhere near `3.4e38`, so the
denominator cannot overflow — and it cannot be zero either, which is what makes the division
safe. **The max trick converts an unbounded quantity into one bounded by `C`.**

**4.** `exp(−100)` in fp32 is a subnormal (`≈3.7e−44`), and on most GPUs subnormals are flushed
to zero, so the small scores get probability exactly `0`. That answer is **not wrong**: the true
probability is ~`1e−44`, and `0` is the nearest fp32 value that matters for any downstream use.
Underflow here is benign precisely because it can only hit terms that are negligible relative to
the maximum term, which is pinned at `1`. Overflow, by contrast, destroys the *whole row*. This
asymmetry is why "subtract the max" and not "subtract the mean" is the standard fix.

---

## Part 2 — The online update

**1.** `m_{k+1} = max(m_k, x_k)`.

**2.** Split off the new term, then insert `exp(m_k)·exp(−m_k) = 1` into each remaining term:

```
s_{k+1} = Σ_{i≤k} exp(x_i − m_{k+1})

        = Σ_{i<k} exp(x_i − m_{k+1})               + exp(x_k − m_{k+1})

        = Σ_{i<k} exp(x_i − m_k)·exp(m_k − m_{k+1}) + exp(x_k − m_{k+1})
                  └──────────────┘ └──────────────┘
                   already have    the ONE new factor

        = s_k · exp(m_k − m_{k+1}) + exp(x_k − m_{k+1})
```

The inserted factor is `exp(m_k − m_{k+1})`: it converts a sum shifted by the *old* max into the
same sum shifted by the *new* max. That single line is the whole of online softmax.

**3.** `m_{k+1} = max(m_k, x_k) ≥ m_k`, so `m_k − m_{k+1} ≤ 0` and
`exp(m_k − m_{k+1}) ∈ (0, 1]`. The update is therefore a **down-scaling** of the accumulator plus
a term that is itself `≤ 1`: no intermediate value can exceed `k+1`. (It can underflow to 0 when
a vastly larger score arrives — which is correct: the old contributions genuinely are negligible.)

**4.** The factor is exactly `1` whenever `x_k ≤ m_k`, i.e. whenever the new element is not a new
record. For a randomly ordered row the expected number of running-maximum updates is the harmonic
number `H_C = ln C + γ ≈ ln C + 0.577`:

| `C` | expected max updates | fraction of elements |
|---|---:|---:|
| 256 | ≈ 6.1 | 2.4 % |
| 4096 | ≈ 8.9 | 0.22 % |
| 131072 | ≈ 12.4 | 0.009 % |

So the rescale is a no-op for ~99.8 % of elements at `C = 4096`. It is *not* free to branch around
it on a SIMT machine, though: if any lane in the wavefront takes the update, all lanes pay for it.
Branchless (always multiply by a factor that is usually `1.0`) is the right choice.

---

## Part 3 — Equivalence and the monoid

**1. Induction.** Base case `k = 0`: `(m_0, s_0) = (−∞, 0)` matches the definitions for an empty
prefix. Inductive step: assume `(m_k, s_k)` equals the definitions for the first `k` elements.
Part 2 derived `s_{k+1}` from `s_k` using nothing but the definition and an exact algebraic
identity, and `m_{k+1} = max(m_k, x_k)` is the definition of the max over `k+1` elements. Hence
`(m_{k+1}, s_{k+1})` is also correct, and by induction `(m_C, s_C)` is exactly what the 3-pass
algorithm computes. **The online algorithm is exact, not an approximation.**

(Floating point rounds differently in the two schemes — the online version performs `O(C)` extra
multiplies by factors `≤ 1` — but because every factor is `≤ 1` there is no error amplification;
both schemes are accurate to `O(√C · ε)` in practice.)

**2. Merge operator.** For partials `(m_a, s_a)` and `(m_b, s_b)`:

```
m = max(m_a, m_b)
s = s_a · exp(m_a − m) + s_b · exp(m_b − m)
```

**3. Monoid proof.**

*Commutative:* the expression is symmetric under swapping `a` and `b`.

*Associative:* expand `((a ∘ b) ∘ c)`. With `m_ab = max(m_a, m_b)` and
`s_ab = s_a e^{m_a − m_ab} + s_b e^{m_b − m_ab}`, and `m = max(m_ab, m_c) = max(m_a, m_b, m_c)`:

```
s = s_ab·e^{m_ab − m} + s_c·e^{m_c − m}
  = s_a·e^{m_a − m_ab}·e^{m_ab − m} + s_b·e^{m_b − m_ab}·e^{m_ab − m} + s_c·e^{m_c − m}
  = s_a·e^{m_a − m} + s_b·e^{m_b − m} + s_c·e^{m_c − m}
```

The result is symmetric in `a, b, c`, so `(a ∘ b) ∘ c = a ∘ (b ∘ c)`.

*Identity:* `(−∞, 0)`. Then `m = max(−∞, m_b) = m_b` and
`s = 0·e^{−∞ − m_b} + s_b·e^{0} = s_b`.

*Why the curriculum cares:* an associative, commutative operator with an identity is exactly what
a **parallel reduction** requires (A04). It means you may partition the row across lanes, waves,
blocks, tiles, or GPUs in any way, combine the partials as a tree in any order, and pad the ragged
tail with the identity — and still get the identical answer. FlashAttention exploits precisely
this: K/V tiles may be visited in any order, and FlashDecoding/split-K splits one row's reduction
across several blocks and merges the `(m, s, O)` triples afterwards.

**4. The NaN.** If both partials are empty, `m_a = m_b = m = −∞` and `exp(−∞ − (−∞)) = exp(NaN)`.
One comparison prevents it:

```cpp
const float mn = fmaxf(m, m2);
s = (mn == -INFINITY) ? 0.0f : s * expf(m - mn) + s2 * expf(m2 - mn);
m = mn;
```

Note that `-ffast-math` (which implies `-ffinite-math-only`) permits the compiler to assume no
infinities exist and may delete this guard. Do not build masked softmax with fast-math.

---

## Part 4 — Numerical thresholds

| Format | Largest finite value | `exp(x)` overflows above `x ≈` | significand bits (incl. implicit) |
|---|---|---|---|
| fp32 (`float`) | 3.4028e38 | **88.72** | 24 |
| fp16 (`half`) | 65504 | **11.09** | 11 |
| bf16 | 3.3895e38 | **88.72** | 8 |

**1.** **fp16 is unusable for a naive softmax**: attention logits of ~11 already overflow, and
pre-softmax logits in trained transformers routinely reach tens. bf16 keeps fp32's exponent range,
so it does not overflow — but its 8-bit significand (relative precision ~`2^-8 ≈ 4e-3`) makes it
risky for the *accumulation*: summing thousands of terms in bf16 loses the small ones entirely
(and fp16 cannot even represent integers above 2048 exactly, so an fp16 normalizer saturates for
`C > 2048`). **Rule: store in fp16/bf16, reduce in fp32.**

**2.** `1/sqrt(d)` keeps the *variance* of `q·k` at `O(1)` instead of `O(d)`, so typical logits
shrink by `√d` (≈11× at `d = 128`). It is a distributional argument about the average case, not a
bound: attention-sink tokens and "massive activations" produce logits of 20–50 in real models, and
one outlier is all it takes. Scaling reduces the probability of overflow; the max trick removes it.

**3.** Exponentials in fp16 with an fp32 normalizer is **safe in range** (every `exp(x − m)` lies
in `(0, 1]`, comfortably inside fp16) and is what FlashAttention actually does for the probability
tile. What degrades as `C` grows is **precision, not range**: each term carries ~`5e-4` relative
error, and the tail terms below fp16's smallest subnormal (`6e-8`) become exactly zero, biasing
long-tail probabilities toward 0. Acceptable for inference; for training, keep the normalizer and
the `P·V` accumulation in fp32 (which is why tensor cores accumulate in fp32).

---

## Part 5 — HBM bytes at `R = 8192`, `C = 4096`, fp32

**1.** One row = `4096 × 4 B` = **16 KiB**. The tensor = `8192 × 16 KiB` = **128 MiB**
(134,217,728 B).

**2.**

| Variant | reads | writes | traversals | total bytes |
|---|---:|---:|---:|---|
| safe 3-pass | 3 | 1 | **4** | **512 MiB** |
| online, statistics in one pass | 2 | 1 | **3** | **384 MiB** |
| online + row cached on-chip | 1 | 1 | **2** | **256 MiB** |

**3.** At 2.0 TB/s (`2.0e12 B/s`):

| Variant | bytes | lower-bound time | speedup vs 3-pass |
|---|---|---:|---:|
| safe 3-pass | 536,870,912 | **268 µs** | 1.00× |
| online 2-pass | 402,653,184 | **201 µs** | **1.33×** |
| cached 1-pass | 268,435,456 | **134 µs** | **2.00×** |

**4.** Arithmetic intensity of the cached variant. Counting generously — the online update does
two `exp` per element and the normalize pass one more, so ~3 `exp` at ~8 FLOP each, plus a
compare, an add and a multiply ≈ **27 FLOP per element** — against 8 bytes moved (one 4-byte read,
one 4-byte write):

```
AI ≈ 27 / 8 ≈ 3.4 FLOP/byte        (a lean count of 1 exp gives ≈ 1.4 FLOP/byte)
```

The ridge point is ~35 FLOP/byte, so softmax sits **10–25× to the left of it: firmly
memory-bound.** No amount of cheaper arithmetic (`__expf`, fast-math) changes the runtime while
the kernel is bandwidth-limited; removing a pass does.

**5. The catch.** At `C = 128` a row is 512 B. All of a block's re-reads then hit L1 (or at worst
L2), so passes 2 and 3 never touch HBM and the pass-count model badly overstates the penalty. What
decides the runtime instead is: kernel launch overhead (a `R × 128` softmax may be only
microseconds of work), the `__syncthreads()` barriers in the block reduction, occupancy (one small
row per block wastes most of a CU), and the transcendental unit's `exp` throughput.

The model starts to predict reality when a row stops fitting in the cache level that serves the
re-reads:

- **L1 / LDS (~32 KB per CU):** rows above ~8K fp32 elements spill to L2.
- **Last level (say 256 MB, shared by all resident blocks):** with ~2000 concurrent blocks that is
  ~128 KB per block, so rows above ~32K fp32 elements genuinely re-read from HBM.

Below those sizes, expect the measured 3-pass → 1-pass gap to be **well under** the modelled 2×.
Which leads to the honest conclusion: for standalone softmax, the online formulation is a modest
win. Its real value is that it makes softmax **fusable into a tiled matmul at all** — see Part 6.

---

## Part 6 — The bridge to FlashAttention

**1.** Attention computes `O = softmax(S)·V`. Because the merge operator is associative, the
softmax *and* the product with `V` can both be accumulated tile-by-tile: for each K/V tile compute
the score tile in registers/SRAM, update `(m, s)`, and accumulate an unnormalized `O`. The `N × N`
probability matrix is therefore never written to HBM — it exists only as an on-chip tile. This is
the entire idea: the max trick made softmax *safe*, the online recurrence makes it *streamable*,
and streamability makes it *fusable*.

**2.** Keep the unnormalized accumulator `O_k = Σ_{i<k} exp(x_i − m_k)·v_i`. It carries the same
shift as `s_k`, so it needs the same rescale:

```
m_{k+1} = max(m_k, x_k)
s_{k+1} = s_k · exp(m_k − m_{k+1}) + exp(x_k − m_{k+1})
O_{k+1} = O_k · exp(m_k − m_{k+1}) + exp(x_k − m_{k+1}) · v_k        <- one extra rescale
output  = O_C / s_C                                                   <- normalize once, at the end
```

Tile form: replace the scalar `x_k` with a score tile and the scalar product with a small GEMM;
nothing else changes. The triple `(m, s, O)` is again a monoid, which is why split-K decoding can
merge per-block triples after the fact.

**3.** Traffic. Standard attention materializes `S` and `P`, giving `Θ(N² + N·d)` HBM accesses.
FlashAttention keeps tiles in SRAM of size `M` and needs `Θ(N²·d²/M)` HBM accesses
(Dao et al. 2022, Theorem 2) — for typical `d = 64…128` and `M` in the hundreds of KB, that is a
several-fold reduction *and* it removes the `O(N²)` **memory footprint**, which is what actually
unlocked long context windows. The reported end-to-end wins (~2–4× on attention, ~15 % on
BERT-large training in the original paper) come from exactly the traffic argument this module
measured on a standalone softmax.

---

## Bonus — log-softmax and cross-entropy

From the safe form, `softmax(x)_i = exp(x_i − m) / s` with `s = Σ_j exp(x_j − m)`:

```
log_softmax(x)_i = (x_i − m) − log(s)          and     logsumexp(x) = m + log(s)
```

Cross-entropy for target `t` is then `−log_softmax(x)_t = (m − x_t) + log(s)`.

Why compute it this way rather than as `log(softmax(x)_t)`: the intermediate probability can
underflow to exactly `0` (any `x_t − m < −87` in fp32), and `log(0) = −inf` — a finite,
well-conditioned loss turns into `inf`, then the gradient into `NaN`. The log-space expression is
a difference of two well-scaled finite numbers, needs no division, reuses the `(m, s)` pair the
softmax already computed, and yields the famously clean gradient `∂L/∂x = softmax(x) − onehot(t)`.
Every framework's `log_softmax` / `cross_entropy` is implemented exactly like this, which is why
you should never hand a framework `log(softmax(x))`.
