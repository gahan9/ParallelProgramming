# Solution 1 — Size a model from its spec

Reference working for [`../exercises/01_size_a_model.md`](../exercises/01_size_a_model.md). Every
figure here is a **derived estimate** from the module's closed-form formulas, reproducible with:

```bash
cd ..
python tools/model_calculator.py -L 48 -d 6144 --heads 48 --kv-heads 8 \
    --ffn 16384 --glu --vocab 64000 --seq 8192 --gpu mi300x
```

Recap of the spec: `L = 48`, `d = 6144`, `h = 48`, `kv_heads = 8`, `d_ffn = 16384` (SwiGLU),
vocab 64,000, untied embeddings, `N = 8192`, fp16, MI300X (5.3 TB/s, 192 GB, 1307.4 TFLOP/s).

Useful constant to compute once and reuse everywhere: `d² = 6144² = 37,748,736 ≈ 37.75 M`.

---

## 1. Parameters

### (a) Head dimensions

```
d_head = d / h        = 6144 / 48 = 128
d_kv   = kv_heads × d_head = 8 × 128 = 1024
group size g = h / kv_heads = 48 / 8 = 6
```

GQA shrinks **`W_K` and `W_V` only** — from `d × d` to `d × d_kv`. `W_Q` and `W_O` are untouched,
because every query head still needs its own query vector and the output projection still consumes
the full concatenated `d`-wide head output. This is the detail people most often get wrong: GQA is
*not* a 6× reduction in the attention block, only in two of its four matrices.

### (b) Parameters in one layer

```
W_Q      : d × d          = 6144 × 6144  =  37.75 M
W_K, W_V : 2 × d × d_kv   = 2 × 6144 × 1024 =  12.58 M
W_O      : d × d          = 6144 × 6144  =  37.75 M
                            attention subtotal =  88.08 M   (2.33 d²)

FFN (SwiGLU, three matrices):
  gate, up : 2 × d × d_ffn = 2 × 6144 × 16384 = 201.33 M
  down     : 1 × d_ffn × d = 16384 × 6144     = 100.66 M
                            FFN subtotal      = 301.99 M   (8.00 d²)

layer total = 88.08 + 301.99 = 390.07 M  =  10.33 d²
```

### (c) Total parameters

```
blocks     = 48 × 390.07 M = 18,723 M = 18.72 G
embeddings = 2 × 64,000 × 6144 = 786.4 M      (untied: input embedding + lm_head)

TOTAL = 18.72 + 0.79 = 19.51 G  ->  ~19.5 G parameters
```

Sanity check on the magnitude: a "20 B-class" model. Note the embeddings are only 4% of the total
here — with a 64k vocabulary and `d = 6144` they are comparatively small. (Contrast Llama-3-8B, where
a 128k vocabulary and `d = 4096` makes embeddings **13%** of the model. Vocabulary size matters more
for small models.)

### (d) The `12d²` shortcut

```
12 d² = 12 × 37.75 M = 452.99 M      predicted
actual                 390.07 M      -> the shortcut is 1.16x too HIGH
```

The two competing deviations:

| Choice | Effect vs textbook | Magnitude |
|---|---|---|
| **GQA** (8 KV heads instead of 48) | attention `4d²` → `2.33d²` | **−1.67 d²** |
| **SwiGLU** (3 matrices instead of 2) | FFN `2md²` → `3·d·d_ffn` | with `d_ffn/d = 2.67`, gives `8.00 d²` — **exactly the textbook `8d²`** |

So SwiGLU contributes *nothing* here, because the designer already compensated: `d_ffn = 16384` is
`2.67d`, not `4d`. Three matrices of width `2.67d` equal two matrices of width `4d`. **GQA wins
uncontested**, and the shortcut overshoots by 16%.

Compare with Llama-3-8B, where `d_ffn/d = 3.5` so SwiGLU gives `10.5 d²` and *over*-shoots the
textbook `8d²` by `+2.5d²`, nearly cancelling GQA's `−1.5d²` and landing at `13.00 d²`. **Same two
features, opposite net effect, purely because of the `d_ffn/d` ratio.** The lesson: know why `12d²`
works so you can tell when it won't.

---

## 2. Weight memory

```
(a) fp16 : 19.51 G × 2 B = 39.02 GB
(b) int8 : 19.51 G × 1 B = 19.51 GB
```

**(c) Fit:**

- **MI300X (192 GB):** yes, comfortably. 39 GB of weights leaves ~153 GB for KV cache and
  activations — this model is *capacity-rich* on this GPU, which (see 6c) is the good problem to have.
- **H100 SXM (80 GB):** the weights fit at 39 GB, leaving ~41 GB for KV. Serviceable, but you get
  roughly a quarter of the MI300X's KV headroom, so a quarter of the concurrency.

---

## 3. FLOPs

### (a) Forward FLOPs per token (blocks only)

```
2 × 18.72 G = 37.45 GFLOP per token
```

### (b) Training on 1.5 T tokens

```
C ≈ 6 N D = 6 × 19.51e9 × 1.5e12 = 1.76e23 FLOPs
```

Using the `6 × params × tokens` heuristic (forward `2×`, backward `4×`). It ignores:

1. **Activation recomputation / gradient checkpointing** — commonly adds roughly another forward
   pass, pushing the real figure toward `8ND`.
2. **The attention `4N²d` term**, which at `N = 8192` is a real ~20% of the per-layer FLOPs (see 3c).
3. The `lm_head`, the optimizer step, and — most importantly for wall-clock — the gap between peak
   and *achieved* FLOP/s, typically 30–50% on a large distributed run.

### (c) Prefill FLOPs for one 8192-token prompt

```
linear term    = (2 × 18.72 G + 2 × 64,000 × 6144) × 8192
               = (37.45 G + 0.79 G) × 8192
               = 38.23 GFLOP/token × 8192           = 313.2 TFLOP

attention term = 4 × L × N² × d
               = 4 × 48 × 8192² × 6144
               = 4 × 48 × 67,108,864 × 6144         =  79.2 TFLOP

TOTAL PREFILL                                       = 392.4 TFLOP
attention share = 79.2 / 392.4                      =  20.2%
```

So attention is a real but minority cost at 8k context — matching the crossover prediction:

```
N_crossover = p_layer / (2d) = 390.07e6 / (2 × 6144) = 31,744 tokens
```

At `N = 8192` we are well below 31,744, so the linear term should dominate — and it does, 4:1.

### (d) Decode FLOPs for one token at 8192 context

```
linear    = 38.23 GFLOP           (identical to prefill's per-token cost)
attention = 4 × L × N × d = 4 × 48 × 8192 × 6144 = 9.66 GFLOP
TOTAL                                            = 47.9 GFLOP per token
```

### (e) The comparison, and the punchline

```
prefill : 392.4 TFLOP / 8192 tokens = 47.9 GFLOP per token
decode  :                             47.9 GFLOP per token
```

**They are the same** — decode does no less arithmetic per token than prefill. Yet decode is 100×
slower per token, because the *FLOPs* are not what you pay for: prefill amortizes one 39 GB weight
read over 8192 tokens (4.8 MB of weight traffic per token), while decode pays the entire 39 GB for
**one** token. Identical arithmetic, ~8000× the bytes per token.

---

## 4. KV cache

```
(a) per token = 2 × L × d_kv × bytes
              = 2 × 48 × 1024 × 2 B = 196,608 B = 196.6 KB

(b) per sequence at N = 8192
              = 196.6 KB × 8192 = 1.611 GB

(c) under MHA, d_kv = d = 6144:
    per token = 2 × 48 × 6144 × 2 B = 1.180 MB
    per seq   = 1.180 MB × 8192      = 9.664 GB

    ratio = 9.664 / 1.611 = 6x  =  the group size g = h / kv_heads = 48/8
```

The ratio is exactly `g`, and it must be: the *only* term that changed in the formula is `d_kv`,
which went from `d` to `d/g`.

Concurrency consequence on the MI300X: `(192 − 39.02) / 1.611 = 94` sequences with GQA versus
`152.98 / 9.664 = 15` with MHA. **GQA is worth 6× the concurrent users.**

---

## 5. The prediction

### (a) Single-stream decode ceiling, weights only

```
                 5.3 TB/s        5300 GB/s
tokens/s  ≤  ───────────────  =  ──────────  =  135.8 tokens/s
              39.02 GB           39.02 GB

units: (GB/s) / (GB) = 1/s   ✓
```

### (b) Including the KV read at full context

```
bytes per step = 39.02 GB (weights) + 1.611 GB (KV) = 40.63 GB
tokens/s ≤ 5300 / 40.63 = 130.4 tokens/s
```

That is a **4% reduction** from 135.8. At batch 1 the KV term is not worth carrying on a whiteboard —
quote 136 and move on. But note the term is `batch × KV_per_seq`: at batch 94 it would be 151 GB and
would utterly dominate the 39 GB weight read. **The KV term is negligible at batch 1 and decisive at
high batch**, so "can I drop it?" depends entirely on the batch size you are asked about.

### (c) int8 weights

```
tokens/s ≤ 5300 / 19.51 = 271.7 tokens/s      (2.0x)
```

**Rule:** in the bandwidth-bound regime, throughput is inversely proportional to bytes read, so
halving the weight precision doubles the token rate. This is the entire performance case for
quantization on the decode path — and it is why it is *only* a decode optimization: at batch 512 you
would be compute-bound and int8 weights would buy little (see [exercise 3, case
D](../exercises/03_regime_diagnosis.md#case-d--the-high-throughput-api)).

### (d) H100 SXM

```
tokens/s ≤ 3350 / 39.02 = 85.9 tokens/s
```

**HBM bandwidth alone determined it.** The H100's compute peak (989.4 TFLOP/s) never entered the
calculation, because at batch 1 we use 0.4% of it. The MI300X is 1.58× faster here purely on the
5300/3350 bandwidth ratio.

---

## 6. Regime check

### (a) Arithmetic intensity of a `d × d` projection, fp16

```
AI = T·d / (d + 2T)   FLOP/byte

T = 1     : 1 × 6144 / (6144 + 2)     = 1.0 FLOP/byte     -> MEMORY-BOUND (247x short)
T = 8192  : 8192 × 6144 / (6144 + 16384) = 2234 FLOP/byte -> COMPUTE-BOUND (9x over)
```

Same matrix, same kernel, a **2200× swing** in intensity driven only by token count. The ridge point
of 247 sits between them, which is why the same model has two personalities.

### (b) Batch size for compute-bound decode

For small `T`, `AI ≈ T`, so you need roughly `T ≈ 247` — a couple of hundred concurrent sequences.
Solving exactly (`T·d/(d+2T) = 247` → `T = 247d/(d − 494)` = `247 × 6144 / 5650`) gives **268**.

The approximation `batch ≈ ridge point` is good to ~8% here and is the version to use on a
whiteboard. It only breaks when `d` is small enough that `2T` is comparable to `d`.

### (c) Bandwidth-bound or capacity-bound?

**Bandwidth-bound.** Two numbers already computed:

- Weights are **39 GB of 192 GB** — only 20% of capacity, leaving 153 GB for KV.
- That headroom supports **94 concurrent sequences** at full 8k context (4(c)), which is well past
  the **268** needed to reach the ridge point... but not past it.

So the honest answer is: capacity is *not* the binding constraint (unlike the 70 B model in
[exercise 2](../exercises/02_kv_cache_budget.md), where weights eat 72% of HBM), but capacity still
caps the batch at 94 — below the 268 that would make you compute-bound. **You will be
bandwidth-bound at every batch size this GPU can actually hold.** Therefore: quantize the weights,
shrink the KV cache to raise the reachable batch, and do not spend money on FLOPs.

That two-step conclusion — "capacity is comfortable, yet capacity still keeps me bandwidth-bound" —
is the sophisticated version of this answer, and it is what a senior interviewer is listening for.
