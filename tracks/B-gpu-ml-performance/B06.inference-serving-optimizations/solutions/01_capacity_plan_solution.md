# Solution 1 — Capacity plan: how many requests fit?

All arithmetic below is exact for the stated configuration. Nothing here is measured on hardware.
Reproduce with:

```bash
python ../sim/kv_cache_planner.py --preset gqa-70b --gpu mi300x --context 8192 \
    --mean-context 2050 --compute-batch-cap 128
```

---

## 1.1 — KV bytes per token

Derive it from what attention actually needs to remember. For each token already in the context,
each layer must keep one **key** vector and one **value** vector, per KV head:

```
kv_bytes_per_token = 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes
                   = 2 × 80 × 8 × 128 × 2
                   = 327,680 bytes
                   = 320 KiB per token per sequence
```

**Where the 2 comes from:** K and V are two separate cached tensors of identical shape. It is not a
fudge factor and it is not about bf16 being two bytes — the dtype is the last term.

**Why `num_attention_heads` (64) does not appear:** with grouped-query attention, 64 query heads
share 8 KV heads (8 queries per group). Queries are *recomputed* every step from the current token
and never cached; only K and V are cached, so only `num_key_value_heads` sizes the cache. This is
exactly why GQA is a serving optimization and not just a quality tradeoff — see the sweep in §1.5.

**Sanity check on the magnitude:** 320 KiB per token means a single 8192-token conversation carries
**2.5 GiB** of state. That is the number people are surprised by, and it is why KV cache — not
weights — is what limits concurrency once the weights fit at all.

---

## 1.2 — The HBM budget

The GB/GiB trap first: the datasheet says **192 GB**, meaning 192 × 10⁹ bytes. That is
192 × 10⁹ / 1024³ = **178.81 GiB**. If you plan in "192 GiB" you have silently invented 27 GB of
memory you do not own — larger than the entire fragmentation headroom you were about to reserve.

| Component | Bytes | GiB | % of HBM |
|---|---:|---:|---:|
| HBM (per GPU) | 192.00 × 10⁹ | 178.81 | 100.0 |
| Weights, 70e9 × 2 B (fp16) | 140.00 × 10⁹ | 130.39 | 72.9 |
| Activations + workspace + graphs | 4.29 × 10⁹ | 4.00 | 2.2 |
| Fragmentation headroom (5% of what's left) | 2.39 × 10⁹ | 2.22 | 1.2 |
| **KV cache pool** | **45.32 × 10⁹** | **42.21** | **23.6** |

**The observation that should change your plan:** the weights consume **73%** of HBM before a single
request arrives. Only a quarter of a 192 GB GPU is available to hold conversations. Any discussion of
"how do we serve more users" that does not start here is not a capacity plan.

---

## 1.3 — Max concurrency, two allocators

### (a) Contiguous preallocation

Reserve the full context at admission:

```
bytes/seq = 320 KiB × 8192 = 2.684 × 10⁹ B = 2.50 GiB
max seqs  = 45.32e9 / 2.684e9 = 16.88  ->  16 concurrent sequences
```

### (b) Paged allocation, 16-token blocks

A 2050-token sequence occupies `ceil(2050 / 16) = 129` blocks = 2064 token slots:

```
bytes/seq = 320 KiB × 2064 = 676.3 × 10⁶ B = 0.63 GiB
max seqs  = 45.32e9 / 676.3e6 = 67.0   ->  67 concurrent sequences
```

**Ratio: 67 / 16 = 4.2×.** No kernel was made faster. No memory was added. The only change is *not
reserving space for a length the request never reaches.*

### Bounding the waste of each scheme

| Scheme | Waste per sequence | As % of the sequence's allocation | Bounded by |
|---|---:|---:|---|
| Contiguous | 320 KiB × (8192 − 2050) = **1.87 GiB** | **75.0%** | nothing — it grows with `max_context` |
| Paged | 320 KiB × (2064 − 2050) = **4.4 MiB** | **0.68%** | ≤ 15 tokens = **4.69 MiB**, always |

The asymmetry is the entire argument. Contiguous waste is `(max_len − actual_len) / max_len`, which
is driven by a number the *product* chooses (the advertised context window) and grows without bound
as that number grows. Paged waste is at most `block_size − 1` tokens in the final block — a constant,
independent of context length.

**Why "bound" and not "average":** internal fragmentation is whatever is left in the last block.
Averaged over a length distribution it is about `(block − 1)/2` tokens; the *worst* case is
`block − 1`. Quoting the bound is the honest way to size a pool, because you cannot control which
sequences show up. This bounded-waste property is what the vLLM paper means by "near-zero waste in
KV cache memory," and it reports internal fragmentation under 4% against 60–80% waste in reservation-
based systems ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180) — their measurements, their
workloads, not ours).

**External fragmentation** disappears for a second reason: every block is the same size, so a freed
block always fits the next request. A contiguous allocator with variable-size buffers can hold plenty
of *total* free bytes and still fail an allocation — the classic problem paging solved in operating
systems in the 1960s.

---

## 1.4 — Which resource binds?

```
KV-cache limit (HBM)     : 67 sequences
Latency limit (TPOT SLO) : 128 sequences
```

**HBM binds.** Configure `--max-num-seqs 67` (round to 64 for a friendlier block/graph geometry).
Everything above 67 is unreachable; the SLO cap of 128 is not doing any work today.

**What the team should work on next — and should not.** Every millisecond shaved off the decode
kernel buys exactly **zero** additional concurrent users, because you cannot admit user 68. Work
that pays, in order of leverage:

1. **Weight bytes.** 73% of HBM is weights. fp8 weights halve that: the KV pool grows from 42.21 GiB
   to **104.14 GiB** and concurrency to **165** — a 2.5× capacity gain from a *weight* change, not a
   KV change. This is the counter-intuitive result of the whole drill: the biggest KV-capacity lever
   was not in the KV cache.
2. **KV bytes.** fp8 KV → **134** sequences (§1.5).
3. **More GPUs.** TP=8 → **1912** (§1.5), at 8× the cost.
4. **Prefix sharing.** If requests share a system prompt, those blocks are shared rather than
   duplicated — free concurrency with no accuracy risk at all.

Doing (1) and (2) together gives **330** sequences, at which point the *latency* cap of 128 binds and
the entire conversation changes: you would then stop buying memory and start buying per-step speed.

---

## 1.5 — Two levers, quantified

### fp8 KV cache

`kv_bytes_per_token` halves to 160 KiB, so concurrency doubles: **67 → 134** sequences (2.0×). The
KV pool is unchanged; each sequence simply costs half as much. Cost: it is an **accuracy** decision,
not a memory trick — quantized K and V feed straight into the attention softmax, and long-context
retrieval is the first thing to degrade. Ship it only behind a task-level eval (see B04).

### Tensor parallelism = 8

Two things shrink at once. Weights shard 8 ways (130.39 → 16.30 GiB), which grows the KV pool to
**150.59 GiB**; and the 8 KV heads shard one-per-rank, cutting `kv_bytes_per_token` to 40 KiB.
Concurrency: **67 → 1912** sequences (29×).

### The trap: TP beyond the KV-head count

| TP | KV heads per GPU | KV pool (GiB) | Max seqs | Gain over previous |
|---:|---:|---:|---:|---|
| 1 | 8 | 42.21 | 67 | — |
| 8 | 1 | 150.59 | 1912 | 29× |
| 16 | **1 (replicated)** | 158.33 | 2010 | **1.05×** |

With `num_key_value_heads = 8`, TP=8 already puts one KV head on each rank. At TP=16 there is nothing
left to split: the KV cache must be **replicated** across pairs of ranks. You doubled the GPU count
and bought **5%** more KV capacity (the only gain is from further sharding the weights). Beyond
`kv_heads`, TP stops being a memory strategy and becomes purely a latency strategy — and one that
pays more collective-communication tax per step. This is the fact that makes people distrust
"just add more GPUs."

**Ship first: fp8 KV.** It is one flag, one eval gate, and 2× capacity on hardware you already own.
TP=8 is a 29× capacity gain but costs 8 GPUs, adds an all-reduce to every layer, and turns a
single-GPU failure domain into an eight-GPU one. Do it when you need the latency or when the model
genuinely does not fit — not as a capacity fix.

---

## 1.6 — Raising max context to 128k

| Allocator | Bytes per sequence | Max seqs |
|---|---:|---:|
| Contiguous (reserve 131,072 tokens) | 40.00 GiB | **1** |
| Paged (unchanged: mean is still 2050) | 0.63 GiB | **67** |

The contiguous allocator collapses to a **single** concurrent request, and it does so without a
single user actually sending a 128k prompt. The *advertised* limit alone destroyed the service.
Paged allocation is completely unaffected — it charges for tokens that exist, so a 16× larger
advertised window costs nothing until somebody uses it. **Ratio: 67×.**

**The conclusion that matters:** long-context support is a **serving-architecture** feature, not a
model feature. A model whose weights support 128k is easy; a *service* that offers 128k without
destroying concurrency for the 99% of traffic that needs 2k requires on-demand block allocation.
Under contiguous reservation, "we support 128k context" and "we serve more than one user" are
mutually exclusive statements.

This is also the trap in §3.3 of exercise 3: the crash is not caused by long contexts, it is caused
by an allocator (or an admission controller) that treats the advertised maximum as the unit of
reservation.
