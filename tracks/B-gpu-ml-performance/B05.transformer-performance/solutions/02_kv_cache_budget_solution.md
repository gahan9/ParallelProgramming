# Solution 2 — The KV-cache budget drill

Reference working for [`../exercises/02_kv_cache_budget.md`](../exercises/02_kv_cache_budget.md).
Reproduce the analytical parts with:

```bash
cd ..
python tools/model_calculator.py --preset llama2-70b --seq 32768 --batch 4 --gpu mi300x
python tools/model_calculator.py --preset llama2-70b --seq 32768 --batch 8 --kv-dtype fp8 --gpu mi300x
python tools/model_calculator.py --preset llama2-70b --seq 32768 --kv-heads 64 --gpu mi300x
```

Recap: `L = 80`, `d = 8192`, `h = 64`, `d_head = 128`, fp16 weights = 138 GB, `N = 32,768`,
MI300X (192 GB, 5.3 TB/s), 10 GB runtime reserve.

---

## 1. The budget

### (a) Available for KV

```
192 GB  capacity
−138 GB weights
− 10 GB runtime reserve (activations, workspace, graph pools, fragmentation)
──────
  44 GB for KV cache
```

Note what just happened: **the weights consumed 72% of the GPU** before a single request arrived.
This is the defining characteristic of serving a large model on one accelerator, and it is why the
KV cache — not the weights — is what limits how many users you can serve.

### (b) The formula

```
KV bytes per sequence = 2 × layers × N × d_kv × dtype_bytes
                      = 2 × 80 × 32,768 × d_kv × 2 B

  2            K and V are two separate tensors
  layers = 80  every layer keeps its own cache
  N = 32,768   one cache entry per token in the context
  d_kv         width of one of K or V = kv_heads × d_head   <- the only free variable
  bytes = 2    fp16
```

Everything except `d_kv` is fixed by the model and the product requirement. `d_kv` is the entire
design space, and it is what MHA/GQA/MQA choose between.

---

## 2. Concurrency under two attention schemes

### (a) MHA — `kv_heads = 64`

```
d_kv       = 64 × 128 = 8192            (= d, as MHA always is)
per token  = 2 × 80 × 8192 × 2 B = 2,621,440 B = 2.62 MB
per seq    = 2.62 MB × 32,768         = 85.90 GB
concurrency = 44 GB / 85.90 GB        = 0.51  ->  ZERO sequences
```

### (b) GQA — `kv_heads = 8`

```
d_kv       = 8 × 128 = 1024
per token  = 2 × 80 × 1024 × 2 B = 327,680 B = 327.7 KB
per seq    = 327.7 KB × 32,768   = 10.74 GB
concurrency = 44 GB / 10.74 GB   = 4.10  ->  4 concurrent sequences
```

### (c) The ratio

**8×**, exactly the group size `g = h / kv_heads = 64/8`. The only term that changed in the formula
is `d_kv`, from `d` down to `d/8`; every other factor is identical. That is worth saying out loud in
an interview, because it shows you are reading the formula rather than recalling a fact.

### (d) Why the MHA answer is alarming

**You cannot serve even one request.** A single 32k-context sequence needs 85.9 GB of KV against a
44 GB budget — the *first* user OOMs. This is not a throughput problem or a tuning problem; the
product as specified is impossible on this hardware with this attention scheme.

Options if the model is MHA and cannot be changed:

1. **Shorten the advertised context.** At the 44 GB budget, MHA supports one sequence up to
   `44 GB / 2.62 MB ≈ 16,700` tokens. So 16k context, one user at a time. Probably not a product.
2. **Shard across GPUs (B07).** Tensor parallelism splits both the weights *and* the KV cache across
   devices. 4-way TP on 4 MI300X: weights 34.5 GB/GPU, budget `192 − 34.5 − 10 = 147.5` GB/GPU, KV
   per sequence per GPU `85.9/4 = 21.5` GB → 6 sequences. Costs 4× the hardware and two all-reduces
   per layer.
3. **Quantize the KV cache** to fp8 (42.95 GB/seq → exactly 1 sequence) or int4 (21.5 GB/seq → 2
   sequences). Buys single digits, not a fix.
4. **Uptrain the checkpoint into GQA.** Ainslie et al. (2023) show this needs only a small fraction
   of the original pretraining compute — which is precisely why the paper exists and why the industry
   converted en masse. This is the real answer.

The honest summary for a design review: *"With MHA this product cannot ship on one GPU at any
concurrency. GQA conversion is the fix, and it is a known, cheap, published procedure."*

---

## 3. Throughput consequences (GQA, batch 4)

### (a) Bytes and step time

```
bytes per decode step = weights + batch × KV_per_seq
                      = 138 GB + 4 × 10.74 GB
                      = 138 + 42.95  =  180.9 GB

step time ≥ 180.9 GB / 5300 GB/s = 34.13 ms
```

Note the shape of that sum: at batch 4 the KV term (43 GB) is already **24%** of the traffic. The KV
cache is not just a capacity problem — at long context it becomes a *bandwidth* problem too.

### (b) Throughput

```
steps/s              = 1 / 0.03413 s   = 29.3 steps/s
per-sequence tokens/s= 29.3 (one token per sequence per step)
aggregate tokens/s   = 4 × 29.3        = 117.2 tokens/s
```

### (c) Why concurrency is nearly free for the individual user

```
batch 1 ceiling  = 5300 / 138 = 38.4 tokens/s per user
batch 4 actual   =              29.3 tokens/s per user   (a 24% slowdown)
                                117.2 tokens/s aggregate (3.05x the throughput)
```

For a product manager: *"Serving four users at once costs each of them about a quarter of their
speed, but nearly quadruples what the GPU delivers overall. The reason is that most of the work in
generating a token is reading the model's 138 GB of weights out of memory, and all four users share
that one read. The only cost that isn't shared is each user's own conversation history."*

---

## 4. Doubling the concurrency (4 → 8)

| Option | New concurrency | Cost |
|---|---:|---|
| **(a)** fp8 KV cache | `44 / 5.37 = 8.19` → **8** | Accuracy of cached K/V. Well-studied and usually mild for K/V specifically, but it is a numerics decision (B04), and fp8 KV needs kernel support in your attention implementation. |
| **(b)** int8 weights | weights 69 GB → budget `192−69−10 = 113` GB → `113/10.74 = 10.5` → **10** | Weight-quantization accuracy loss. **Also doubles the per-token ceiling** to `5300/69 = 76.8` tokens/s. Best single lever. |
| **(c)** MQA (1 KV head) | `d_kv = 128`, `1.34 GB/seq` → `44/1.34` = **32** | Requires retraining/uptraining, and MQA is measurably worse in quality than GQA at moderate `g`. 8× is more than asked for. |
| **(d)** Halve context to 16k | `5.37 GB/seq` → **8** | Not an engineering change — it is a product capability reduction. Free in compute, expensive in positioning. |
| **(e)** 2-way tensor parallel | weights 69 GB/GPU, KV 5.37 GB/seq/GPU → budget `192−69−10 = 113` → **21** | 2× the hardware. Adds two all-reduces per layer (`2 × B × d × bytes` each), which eats into the latency win. Also roughly halves the per-token weight-read time. |

### (f) Recommendation

**Option (b), int8 weight quantization**, is the recommendation, and the reason is that it is the
only option on the list that improves **both** axes at once:

- **Concurrency**: frees 69 GB of HBM, raising the KV budget from 44 GB to 113 GB → 10 sequences
  (2.6× the target).
- **Per-token latency**: halves the weight bytes each step must read, so the batch-1 ceiling doubles
  from 38.4 to 76.8 tokens/s.

Every other option trades one thing for another. (a) and (d) buy concurrency at no latency benefit;
(c) buys concurrency at a quality cost and requires retraining; (e) buys both but costs a second GPU.

The mechanism behind (b)'s double benefit is that weight bytes appear in *both* governing
expressions — the capacity equation (`capacity − weights − reserve`) and the bandwidth equation
(`weights + batch × KV`). Shrinking one number improves both. That is the structural insight worth
stating.

In practice you would do **(a) and (b) together**: int8 weights *and* fp8 KV cache gives
`(192−69−10)/5.37 = 21` concurrent sequences at double the per-token ceiling — a ~5× concurrency
improvement, on one GPU, for two numerics decisions.

---

## 5. The subtle bit — free concurrency

### (a) Step time at fp8 KV, batch 8

```
bytes per step = 138 GB + 8 × 5.37 GB = 138 + 42.95 = 180.9 GB
step time      = 180.9 / 5300         = 34.13 ms
```

**Identical to question 3(a).** Halving the per-sequence cache and doubling the sequence count leaves
the total KV bytes unchanged at 42.95 GB, so the step reads exactly the same 180.9 GB.

### (b) Consequences

```
                    batch 4, fp16 KV     batch 8, fp8 KV
step time                34.13 ms            34.13 ms      (unchanged)
per-sequence tokens/s        29.3                29.3       (unchanged)
aggregate tokens/s          117.2               234.4       (2.00x)
```

**Throughput exactly doubled and no individual user got slower.** That is as close to a free lunch as
this subject offers, and it exists only because decode is bandwidth-bound: the metric that matters
is bytes-per-step, and we held it constant while doubling the tokens produced per step.

### (c) The general condition

Adding concurrency is free in per-user latency when the added KV traffic is negligible against the
weight read:

```
batch × KV_per_seq  ≪  weight_bytes
```

While that inequality holds, `step_time ≈ weight_bytes / bandwidth` is a constant, so per-user
tokens/s is flat and aggregate throughput scales linearly with batch. As `batch × KV_per_seq`
approaches and then exceeds `weight_bytes`, each additional sequence starts paying for its own cache
read and per-user latency degrades proportionally.

For this model: `weight_bytes = 138 GB` and `KV_per_seq = 10.74 GB`, so at `batch = 13` the KV traffic
(140 GB) has grown to *equal* the weight read — the step now takes twice as long as the weights alone
would explain, and every further sequence costs full price in per-user latency. **This is the real
reason long-context serving is hard**, and it is a *different* constraint from running out of
capacity: you can hit the bandwidth wall while still having HBM to spare, and vice versa.

---

## Why the calculator disagrees with question 1(a)

The calculator reports **5** concurrent sequences where you computed **4** (and **10** where you
computed **8**). The difference is the **10 GB runtime reserve**:

```
calculator : (192 − 138) / 10.74      = 5.03  ->  5
you        : (192 − 138 − 10) / 10.74 = 4.10  ->  4
```

The calculator prints exactly what it says it prints — its own output labels the number *"ignores
activations, fragmentation, runtime overhead."* It is the theoretical ceiling, and it is the right
thing for a tool to report, because the reserve depends on your runtime, your batch scheduler, and
your allocator.

**Provision at the calculator's number and you will OOM in production.** The tool gives you the
ceiling; the reserve is engineering judgment you add on top. Knowing the difference between those two
numbers — and being able to say which one you are quoting — is the point of question 1(a) and of
[module README pitfall 4](../README.md#pitfall-4--sizing-the-kv-cache-without-the-runtimes-overhead).
