# Solution 1 — Format selection drill

The method, before the answers: **range first, precision second, throughput third.** A
range failure produces `inf`/`NaN` and destroys the run; a precision failure degrades
quality gradually and is caught by an eval; throughput is the thing you were buying. Any
answer that starts with throughput has skipped the two ways it can go wrong.

---

## Scenario A — Pre-training a 70B LLM on 512 GPUs

**A1 — 16-bit compute format: bf16.**

Gradients span `1e-9 … 1e0`, and some activations reach `1e4`.

- bf16 range: min normal `1.18e-38`, max finite `3.39e38`. Both extremes are ~30 orders
  of magnitude inside the format. Nothing underflows, nothing overflows.
- fp16 range: min normal `6.10e-5`, max finite `65504`. The gradient tail at `1e-9` is
  **four orders below fp16's min normal** and only ~1.5 orders above the point where it
  rounds to exactly zero. It survives only as a subnormal with fewer than 10 mantissa
  bits, or not at all.

So fp16 needs **loss scaling** (multiply the loss by `S` before `backward()` to shift the
whole gradient histogram up into the normal range, then unscale before the optimizer
step) plus **dynamic adjustment** of `S`, plus **skip-on-overflow** logic. bf16 needs none
of it. That is the whole reason bf16 displaced fp16 for training once hardware supported
it — same 8 exponent bits as fp32, so the same dynamic range.

**A2 — Master weights and optimizer moments: fp32.**

Lab 1 section 4 prints the reason. `ulp(1)` is `7.81e-3` in bf16, so an update smaller
than about `3.9e-3` applied to a weight of order 1 rounds straight back to the original
weight and is **silently discarded**. A learning rate of `1e-3` with gradients of order 1
is exactly that regime. Accumulate the updates in fp32 (`ulp(1) = 1.19e-7`) and cast down
to bf16 only for the next forward pass. Adam's `m` and `v` are running averages with the
same problem, plus `v` involves squared gradients — `(1e-9)² = 1e-18`, which is fine in
fp32 and zero in fp16.

**A3 — Bytes per parameter, and what it forces.**

| item | bytes/param |
|---|---:|
| bf16 compute copy | 2 |
| fp32 master weights | 4 |
| fp32 Adam `m` | 4 |
| fp32 Adam `v` | 4 |
| **total** | **14** |

`14 × 70e9 = 980 GB`. Aggregate capacity is `512 × 192 GB = 98 TB`, so it fits *in
total* — by a factor of 100. But a single GPU holds 192 GB and 980 GB does not fit on
one, so **the optimizer state and master weights must be sharded across ranks**
(ZeRO-2/3, FSDP). Notice the split: 12 of the 14 bytes are optimizer state and master
weights, only 2 are the weights you compute with. That ratio is why optimizer-state
sharding is the first thing every large-scale training stack does, and why the "model is
only 140 GB in bf16" intuition is off by 7×.

**A4 — Before fp8 GEMMs are safe you need:**

1. **Per-tensor (or finer) scaling with amax tracking.** E4M3 tops out at `448` and its
   min normal is `1.56e-2`; nothing in a transformer naturally lives in that window. Each
   tensor has to be scaled into it, the scale factor tracked in fp32, and the scale
   updated as the distribution drifts (delayed scaling with an amax history, or per-block
   scaling as in the MX formats).
2. **fp32 accumulation and a higher-precision fallback for sensitive tensors.** Keep the
   accumulator in fp32 (never fp8), and keep the layers you have *measured* to be
   sensitive — typically the first and last layers, LayerNorm, and the softmax — in
   bf16/fp32. Plus, before shipping: a task-level eval, not a loss curve (§6).

---

## Scenario B — Serving a 13B model on one 80 GB GPU, latency-critical

**B1 — Weight memory.**

| scheme | bytes/weight | total |
|---|---:|---:|
| fp16 | 2 | **26.0 GB** |
| int8 | 1 | **13.0 GB** |
| int4 | 0.5 | **6.5 GB** |
| int8 + fp16 scales, g=128 | 1 + 2/128 = 1.0156 | **13.2 GB** |
| int4 + fp16 scales, g=128 | 0.5 + 2/128 = 0.5156 | **6.70 GB** |

The scale overhead at `g = 128` is 1.6% of an int8 payload and 3.1% of an int4 one.
Small — but it is not zero, and at `g = 32` it is four times larger (exercise 3).

**B2 — Which resource sets tokens/second.**

Batch-1 decode reads every weight exactly once per token and does ~2 FLOPs with it:

```
intensity(fp16) = 2 FLOP / 2 bytes = 1 FLOP/byte
intensity(int8) = 2 FLOP / 1 byte  = 2 FLOP/byte
ridge point     = 1000e12 / 3e12   = 333 FLOP/byte
```

Two orders of magnitude below the ridge: decode is **hopelessly memory-bound**. The
ceiling is bandwidth ÷ weight bytes:

| scheme | upper bound on tokens/s |
|---|---:|
| fp16 (26 GB) | 3e12 / 26e9 ≈ **115** |
| int8 (13 GB) | ≈ **231** |
| int4 (6.5 GB) | ≈ **462** |

**B3 — Why "half the bytes" and "2× faster matrix cores" are different arguments.**

Halving the weight bytes halves the time to *stream the weights*, which is the entire
critical path here — so tokens/second roughly doubles, exactly as B2's table says. The
matrix-core argument ("int8 does 2× the TOPS of fp16") is about the **compute ceiling**,
and at batch 1 you are nowhere near it: the arithmetic units are idle waiting for
weights. That argument only starts to matter at large batch or during **prefill**, where
the same weights are reused across many tokens and intensity climbs. Two mechanisms, two
regimes; quoting the wrong one in a design review is a classic tell.

**B4 — The risk, and the evidence.**

The risk is a **quality regression that no throughput metric can see**: int4 weight-only
quantization degrades some capabilities far more than others, and the loss/perplexity
number is nearly insensitive to it.

"Perplexity looked fine" is **not** sufficient evidence. Perplexity is an average over a
generic corpus, dominated by easy tokens; the damage from quantization concentrates in
rare, high-information tokens and in long-context or multi-step reasoning behaviour. You
need a **task-level eval that matches your product**, run before and after on the same
prompts and the same decoding parameters, with a decision threshold agreed in advance —
plus a spot check of the outlier-sensitive layers. See README §6 for the full rule.

---

## Scenario C — A loss reduction over 1e6 elements, then a LayerNorm

**C1 — fp16 loss accumulation: stagnation, not overflow.**

The true sum is `1e6 × 1e-2 = 1e4`, comfortably inside fp16's `65504`, so overflow is not
the failure. Underflow is not either — `1e-2` is a normal fp16 number. The failure is
**stagnation of the running sum**:

For a running sum in the binade `[2^e, 2^(e+1))`, `ulp = 2^(e-10)`. A term of `1e-2` is
lost entirely once `1e-2 < ulp/2 = 2^(e-11)`, i.e. once `2^e > 20.48`, i.e. once the sum
reaches **32**. From there the sum stops growing: it converges to ~32 instead of 1e4, a
relative error of about **99.7%**. Lab 1 section 5 shows the same effect on a dot product.

**C2 — Accumulator widths: fp32 for both** (fp64 if the reduction is a loss you compare
across runs, though fp32 with pairwise/tree summation is standard).

The distinction the question is after: **storage precision and accumulator precision are
independent choices.** Inputs can live in bf16 (halving the bytes read, which is what
makes the kernel fast) while the accumulator is fp32 (which is what makes it correct).
Every matrix core is built on exactly this split. So no — the accumulator choice does not
follow from the storage choice, and conflating them is the single most common precision
bug in hand-written kernels.

**C3 — LayerNorm via `E[x²] − E[x]²`.**

With values of order 10 and variance of order 1: `E[x²] ≈ 101`, `E[x]² ≈ 100`, and
`ulp(101)` in fp32 is `7.6e-6`, so the difference of 1 keeps ~5 significant digits.
fp32 survives *at this scale*. Push the values to order `1000` and the same formula gives
`E[x²] ≈ 1e6` with `ulp = 0.0625`, so a variance of 1 is computed with ~6% error — and in
fp16 it fails outright.

This is Lab 1 section 7 (catastrophic cancellation). The fix is algebraic, not numeric:
**subtract the mean first** (two-pass) or use Welford's online update, which is what
framework LayerNorm kernels do. Extra precision only moves the cliff; the unstable
formula stays unstable.

**C4 — The 6th-decimal-place change is not a bug.**

fp32 addition is not associative, so changing the block count changes the summation tree
and therefore the last few bits. Lab 1 section 6 measures the spread directly. The
regression test must assert `|got − ref| / |ref| <= tol` against a **fp64 reference**,
with `tol` derived from the accumulation (`~sqrt(n) · eps` for random rounding), and must
never assert bit equality across launch configurations. Bit-exactness is a legitimate
requirement in some settings (debugging, some certification regimes) but it costs you a
fixed reduction order — i.e. performance — and it must be an explicit decision.

---

## Scenario D — KV cache for a Llama-3-8B-shaped model

**D1 — Size, per token first.**

```
per token per layer : 2 (K and V) × 8 kv_heads × 128 head_dim = 2048 values
per token           : 2048 × 32 layers                        = 65,536 values
                    : 65,536 × 2 bytes (fp16)                 = 128 KiB per token
per sequence        : 128 KiB × 8192                           = 1 GiB
batch 32            : 1 GiB × 32                               = 32 GiB
```

**128 KiB per token** is the number worth memorising: it lets you size any context/batch
combination in your head.

**D2 — The cache is the bottleneck.**

fp16 weights are `8e9 × 2 = 16 GB` (≈14.9 GiB) and **fixed**. The cache is 32 GiB and
**grows linearly with batch × context**. At this operating point the cache is more than
twice the weights, and at 128k context it dwarfs them. So the highest-leverage
quantization target in a long-context serving stack is the **KV cache**, not the weights —
which is the opposite of where most quantization effort is spent. Halving it to int8/fp8
either halves the memory or (more usefully) **doubles the batch size** you can serve,
which is a throughput win on top of the memory win.

**D3 — Format: int8 or fp8 per-token/per-channel, keys and values treated differently.**

Empirically the **key** cache should be quantized **per channel** and the **value** cache
**per token**, because their outlier structure differs — keys have a few consistently
large channels, values vary token to token (KIVI, arXiv:2402.02750, README §8). A common
production compromise is fp8 E4M3 for both (cheap, no zero-point, hardware conversion),
keeping the most recent tokens in fp16.

**D4 — Both memory and latency, by the same mechanism.**

Decode reads the entire cache for every generated token, so the cache is *streamed* once
per token: it sits in the same memory-bound regime as the weights in B2. Halving its
bytes halves that read time (latency) and halves its footprint (memory → larger batch →
throughput). One change, both benefits — which is why it is often the first thing to
quantize.

---

## Scenario E — The portability trap

**E1 — Two different E4M3s.**

| | exponent bias | max finite | specials |
|---|---:|---:|---|
| **OCP E4M3** (NVIDIA Hopper onward; AMD's newer targets, e.g. `gfx12x`) | 7 | **448** | no infinities; one NaN pattern; ±0 |
| **E4M3FNUZ** (AMD CDNA 3, `gfx942`) | 8 | **240** | no infinities; single NaN = `0x80`; no −0 |

**E2 — Silent numerical disaster, not a crash.**

The bias differs by exactly 1, so the *same 8 bits decode to values a factor of 2 apart*.
Bit pattern `0_0111_000` is `1.0` under OCP (exponent field 7, bias 7) and `0.5` under
FNUZ (bias 8). Load an FNUZ checkpoint as OCP and every weight is **2× too large** — and
because it is a uniform factor, it does not produce `NaN` immediately; it propagates
through the network as `2^L` amplification and looks like a diverged or badly tuned model.
Worse, `0x80` means `NaN` under FNUZ and `−0.0` under OCP, so a legitimately-NaN element
silently becomes a harmless zero, or vice versa. There is no error message anywhere.

**E3 — What to store instead.**

Store the checkpoint in **bf16 or fp32** (a format with one universally agreed encoding)
plus, if you want reproducibility of the quantized run, the **scale factors and an
explicit format tag** (`e4m3`, `e4m3fnuz`, …). Quantize at load time for the hardware you
are actually running on. Cost: 2–4× the checkpoint bytes and a conversion pass at load.
That is cheap insurance against a class of bug that produces no diagnostic at all — and it
is why ONNX, safetensors, and the OCP MX specification all insist on naming the exact
encoding rather than saying "fp8".

---

## The deliverable table

| scenario | format | range justification | precision justification | throughput argument | risk accepted |
|---|---|---|---|---|---|
| A compute | bf16 | grads `1e-9`, acts `1e4` both ≫30 orders inside `1.18e-38 … 3.39e38` | `ulp(1)=7.8e-3` is fine for a *product*, not for an *update* | 2 bytes + matrix-core path | 8× coarser than fp16 near 1.0; mitigated by fp32 master |
| A master/opt | fp32 | n/a | `ulp(1)=1.19e-7` keeps `1e-3` updates | none (not on the hot path) | 12 of 14 bytes/param → must shard |
| B weights | int4 g=128 (or int8) | scale maps `[-127,127]` onto each group's max | int8 step ≈ max/127 per group | 6.7 GB vs 26 GB → ~4× tokens/s ceiling | quality regression invisible to perplexity |
| C accumulate | fp32 | sum `1e4` fits everywhere | fp16 stalls at 32 → 99.7% error | irrelevant: accumulate wide, store narrow | non-reproducibility across block counts |
| D KV cache | fp8/int8, K per-channel, V per-token | values are already normalised, small range | 3–4 mantissa bits survives eval in practice | 32 GiB → 16 GiB; 2× batch, ~2× decode read | long-context quality; needs an eval |
| E checkpoint | bf16 + explicit tag | one unambiguous encoding | lossless w.r.t. the fp8 you will cast to | 2× bytes at rest, quantize on load | none — this is the insurance |
