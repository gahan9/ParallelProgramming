# Solution 2 — Loss scaling drill

Every number below is derived from four fp16 constants, which Lab 1 prints from the bit
layout rather than a table:

```
max finite      6.5504e+04      = (2 - 2^-10) · 2^15
min normal      6.1035e-05      = 2^-14
min subnormal   5.9605e-08      = 2^-24
ulp(1)          9.7656e-04      = 2^-10
```

---

## Part 1 — Establish that there is a problem

**1 — Fraction below fp16's min normal (`6.1035e-5`), i.e. subnormal.**

Whole buckets below `1e-4` that are entirely under the threshold:

```
1e-6 … 1e-5   25 %
1e-7 … 1e-6   10 %
1e-8 … 1e-7    4 %
< 1e-8         1 %
              ---
              40 %   (lower bound, no assumptions)
```

The `1e-5 … 1e-4` bucket straddles the threshold. Assuming values are log-uniform inside
a decade, the fraction of that bucket below `6.1035e-5` is
`log10(6.1035e-5 / 1e-5) = 0.786`, so `0.786 × 30 % = 23.6 %` more.

**Total ≈ 64 %.** Roughly **two thirds of the gradient tensor is subnormal in fp16** —
representable, but with fewer than 10 mantissa bits and progressively fewer as the value
shrinks. This is not a tail case; it is the bulk of the tensor.

**2 — Fraction that rounds to exactly zero.**

The threshold is **half** the min subnormal, `2^-25 = 2.98e-8`, because round-to-nearest
maps anything below that to zero (values between `2.98e-8` and `5.96e-8` round *up* to the
min subnormal).

```
< 1e-8              1 %      (entirely below)
1e-8 … 1e-7  :  log10(2.98e-8 / 1e-8) = 0.474  ->  0.474 × 4 % = 1.9 %
                   ---
                  ≈ 2.9 %
```

About **3 % of the gradients become exactly 0.0**.

**3 — What a zeroed gradient does.**

It is not a small update — it is **no update at all**, plus a corrupted optimizer state.
The weight does not move, Adam's `m` decays toward zero for that coordinate, and `v`
shrinks too. Summing over steps does **not** rescue it: the loss of information happens
*before* accumulation, in the cast. Every step independently rounds that coordinate's
gradient to zero, so you are adding `0 + 0 + 0 + …`. That is the crucial asymmetry —
gradient *noise* averages out over steps, gradient *underflow* does not, because it is a
deterministic function of the magnitude. Whole features can simply stop learning, and the
loss curve will look plausible while it happens.

**4 — The same census in bf16 (min normal `1.18e-38`).**

**Zero buckets underflow.** The smallest bucket, `< 1e-8`, is thirty orders of magnitude
above bf16's min normal. No loss scaling, no dynamic scale, no skip-on-overflow logic, no
`GradScaler` in the training loop. That is the entire argument for bf16 in one line — and
it is a *range* argument, which is why it could only be won by spending exponent bits.

---

## Part 2 — Size the scale factor

**5 — The largest usable power-of-two scale.**

```
S · |g|max <= 65504    ->    S <= 65504 / 0.5 = 131008
largest power of two <= 131008  ->  S = 2^16 = 65536      (2^17 = 131072 > 131008)
```

Powers of two are the only sensible choice because **scaling by a power of two is exact**:
it adds a constant to the exponent field and leaves the mantissa untouched, so the scaling
itself introduces *zero* rounding error and the unscaling (`g / S`) is exact too. Any
other factor rounds twice — once on the way in, once on the way out — and quietly adds
noise to every gradient in the model in exchange for nothing.

**6 — Where the histogram lands at `S = 2^16`.**

Every magnitude moves up by 16 binades (a factor of 65536):

| original | scaled | status in fp16 |
|---|---|---|
| `1e0` | `6.55e4` | normal, just inside `65504`… by 0.1% |
| `1e-4` | `6.55e0` | normal |
| `1e-6` | `6.55e-2` | normal |
| `1e-8` | `6.55e-4` | **normal** (was 3 orders below min normal) |

Every bucket in the table is now normal. What is *still* lost: anything originally below
`6.1035e-5 / 65536 = 9.3e-10` is still subnormal, and anything below `4.5e-13` is still
exactly zero. The `< 1e-8` bucket is open-ended, so part of it remains degraded — loss
scaling shifts the window, it does not widen it.

**7 — `S = 2^20` overflows.**

```
2^20 · 0.5 = 524288  >  65504   ->   inf
```

The largest gradients become `inf`; any `inf − inf` or `0 · inf` downstream becomes `NaN`.
After unscaling you have `inf/NaN` gradients, so the optimizer must detect them and
**skip the entire step**. If it does not, see question 10.

**8 — The usable window.**

fp16's normal numbers span 30 binades (`2^-14` up to the `2^15` binade), i.e. a dynamic
range of about `2^30 ≈ 1.07e9`, or **~9 decades**. Including subnormals gets you to
`2^-24`, i.e. 40 binades ≈ 12 decades, at degraded precision.

A single scale factor **shifts** that window; it cannot stretch it. So if a tensor's own
dynamic range `|g|max / |g|min` exceeds ~9 decades, **no value of `S` works** — some
elements overflow or others underflow whatever you choose. Our tensor spans
`0.5 / 1e-8 = 5e7` ≈ 7.7 decades, which just fits, and that is why one global scale is
enough here. When it is not enough, the answers are per-tensor scales (fp8 delayed
scaling), per-block scales (the MX formats' one exponent per 32 values), or a wider
exponent field (bf16).

---

## Part 3 — Make it dynamic, and get the order right

**9 — Dynamic loss scaling in five lines.**

```
loss_scaled = loss * S
loss_scaled.backward()                       # grads are now S times too large
if any(!isfinite(g) for g in grads):         # overflow detected
    S = S / 2 ; skip_optimizer_step()        # BACKOFF FACTOR = 2
else:
    unscale(grads, S) ; clip(grads) ; step()
    if steps_since_last_overflow >= N: S = S * 2   # GROWTH INTERVAL = N
```

The two hyper-parameters are the **backoff factor** (usually 2 — halve on overflow) and
the **growth interval** `N` (usually 1000–2000 steps — double after that many clean
steps). The loop is a hill-climb that keeps `S` just below the overflow threshold, because
that is where the fewest gradients underflow.

**10 — Why you must skip the step on overflow.**

`inf`/`NaN` gradients poison the *state*, not just the step. `w = w − lr · NaN` makes the
weight `NaN`; Adam's `m` and `v` become `NaN` and stay `NaN` forever because they are
running averages of a `NaN`. Every subsequent forward pass produces `NaN` loss. The run is
dead and can only be recovered from a checkpoint — and since the loss goes to `NaN`
several steps later, the *cause* looks like it was somewhere else entirely. Skipping the
step costs you one batch out of thousands; not skipping costs you the run.

**11 — Correct order: unscale → check finiteness → clip → step.**

Clipping is defined on the *true* gradient norm, so it must see unscaled gradients.

- **Clip before unscale:** the measured norm is `S × ‖g‖`, i.e. ~65536× the real value.
  Against a threshold of 1.0 the clip fires on **every step** and multiplies all gradients
  by `1/(S·‖g‖)`, so the effective learning rate collapses by roughly a factor of `S`.
  Training does not crash — it just stops making progress, and every diagnostic you look
  at (loss finite, gradients finite, clip active) looks *normal*.
- The mirror-image bug, comparing the *unscaled* norm against a threshold you forgot to
  scale, makes the clip effectively `S×` looser, so it **never fires** and never protects
  you from the exploding-gradient step it was added for. That is the "clipping is doing
  nothing" symptom: it is present, configured, and inert.

Both are silent. This is why the ordering lives inside the framework's `GradScaler`
(`scaler.unscale_(optimizer)` before `clip_grad_norm_`, then `scaler.step()`) rather than
in user code.

**12 — Forward-pass `inf` in an attention score: loss scaling does not help.**

Loss scaling multiplies the *loss*, so it only affects the backward pass. A forward
activation that overflows fp16 overflows regardless of `S`. Fixes, in order of preference:

1. **bf16** for that tensor — it has fp32's range, and this failure is a range failure.
2. The **max-subtraction trick** in softmax (A07): subtract the row max before `exp()` so
   every argument is `<= 0` and `exp() <= 1`. Lab 1 section 3 shows fp16 `exp()` escaping
   the format at `x = 11.10`, which is an entirely ordinary logit.
3. Keep the attention logits in fp32, and/or scale `QK^T` by `1/sqrt(d)` *before* the
   matmul accumulates rather than after.

---

## Part 4 — The memory bill, and what bf16 does not fix

**13 — Mixed precision with Adam, 7e9 parameters.**

| item | bytes/param |
|---|---:|
| 16-bit compute weights | 2 |
| fp32 master weights | 4 |
| fp32 Adam `m` | 4 |
| fp32 Adam `v` | 4 |
| 16-bit gradient | 2 |
| **total** | **16** |

`16 × 7e9 = 112 GB`.

**14 — Pure fp32 training, same model.**

| item | bytes/param |
|---|---:|
| fp32 weights | 4 |
| fp32 gradient | 4 |
| fp32 `m` | 4 |
| fp32 `v` | 4 |
| **total** | **16** |

`16 × 7e9 = 112 GB` — **identical**. So no: mixed precision does **not** halve your memory
for the parameter state. It cannot, because it *adds* a 16-bit copy on top of an fp32 copy
you still need.

What mixed precision actually buys:

- **Speed** — matrix cores at 2–8× the fp32 rate, and half the bytes for every operand
  read.
- **Activation memory** — activations are stored in 16 bit and, at large batch × long
  sequence, activations (not parameters) dominate the footprint. This is the real memory
  win, and it is why the claim survives despite the table above.

Being able to say this precisely in a design review is worth more than the arithmetic:
the "half the memory" claim is true of *activations*, false of *parameter state*.

**15 — What bf16 is worse at.**

Precision near 1.0: `ulp(1)` is `7.81e-3` in bf16 against `9.77e-4` in fp16 — bf16 is
**8× coarser**, exactly the ratio of `2^-7` to `2^-10`. It is acceptable because the fp32
master copy absorbs the updates: the bf16 value is only ever a *rounded snapshot* used for
one forward/backward pass, and the accumulated learning lives in fp32 where `ulp(1)` is
`1.19e-7`. Remove the master copy and bf16 training stops converging — the two design
choices are a package, not independent options.

**16 — The review comment.**

> Switching bf16 → fp16 trades a *range* property for a *precision* property, and the
> range is the one that kills runs: ~64% of our gradients are already below fp16's min
> normal and ~3% would round to exactly zero, so this change is only safe if it also adds
> dynamic loss scaling with skip-on-overflow and puts the unscale before the grad-norm
> clip. The extra 3 mantissa bits buy nothing we can spend, because the update is applied
> to the fp32 master weights either way. What eval regression are we trying to fix?
