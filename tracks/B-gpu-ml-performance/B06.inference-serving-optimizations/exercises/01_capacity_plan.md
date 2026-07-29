# Exercise 1 — Capacity plan: how many requests fit?

**Competency:** memory-budget arithmetic, and the discipline of naming the *binding* resource
instead of guessing at a batch size.

---

## The scenario

You are asked to serve a **70B-parameter GQA model** on a **single AMD Instinct MI300X**
(192 GB HBM, datasheet) with a **maximum context of 8192 tokens**.

Model shape (this is what you would read out of the model's `config.json`):

| Property | Value |
|---|---|
| Parameters | 70 × 10⁹ |
| Layers (`num_hidden_layers`) | 80 |
| Query heads (`num_attention_heads`) | 64 |
| **KV heads** (`num_key_value_heads`) | **8** |
| Head dim (`hidden_size / num_attention_heads`) | 128 |
| Weight dtype | fp16 |
| KV dtype | fp16 |

Operating assumptions, which you must carry through the whole calculation:

- Activations, workspace, HIP/CUDA graph buffers: reserve **4 GiB**.
- Allocator fragmentation and traffic-spike headroom: reserve **5%** of what remains.
- The **mean actual sequence length is 2050 tokens** — a quarter of the maximum. (Real traffic
  looks like this: a long tail of long conversations, a fat head of short ones.)
- KV block size for the paged allocator: **16 tokens**.
- Separately, your latency budget has already told you that batches above **128** sequences break
  the TPOT SLO. (Exercise 2 is where that number comes from.)

---

## Tasks

### 1.1 — KV bytes per token

Derive, from first principles (not from a memorised table), the formula for KV-cache bytes per
token per sequence, and evaluate it for this model. Say explicitly where the factor of 2 comes from
and why `num_attention_heads` does *not* appear.

### 1.2 — The HBM budget

Produce the full budget table: HBM → weights → activations → headroom → KV pool. Give each line in
GiB and as a percentage of HBM. Note the GB-vs-GiB trap: the datasheet says 192 **GB** (10⁹ bytes).

### 1.3 — Max concurrency, two allocators

Compute the maximum number of concurrent sequences under:

- **(a) contiguous preallocation** — one buffer per sequence, reserved for the full 8192 tokens at
  admission time (this is what pre-vLLM systems did, and what Orca's "reserve max tokens" does);
- **(b) paged allocation** — 16-token blocks, allocated on demand, so a sequence holds only
  `ceil(actual_len / 16) × 16` tokens.

Report the ratio. Then bound the waste of *each* scheme: for (a), how many bytes per sequence are
reserved and never used; for (b), what is the *worst case* internal fragmentation per sequence, and
why is that a bound rather than an average?

### 1.4 — Name the binding resource

You now have a memory-derived concurrency limit and a latency-derived batch cap of 128. Which one
binds? State the concurrency you would actually configure, and then answer the question that
matters: **what should the team work on next?** Be specific — "optimize the kernel" is not an answer
if the kernel is not the limit.

### 1.5 — Two levers, quantified

For the *binding* resource you identified, quantify these two changes and say which you would ship
first and why:

- moving the KV cache to **fp8**;
- serving with **tensor parallelism = 8** across a full MI300X node.

For the TP case, watch for the trap: what happens to the KV cache when `num_key_value_heads` is
smaller than the TP degree? Work out `kv_heads = 8, tp = 8` and then `tp = 16`.

### 1.6 — The stress test

Your product team wants to raise the maximum context to **128k tokens**. Without changing anything
else, what is the concurrency under (a) and under (b)? What does the difference tell you about
whether long-context support is a *model* feature or a *serving-architecture* feature?

---

## What a complete answer looks like

- Every number carries units, and the GB/GiB conversion is done once, explicitly.
- The contiguous-vs-paged ratio is explained by *reservation*, not by "paging is more efficient".
- §1.4 ends with a concrete engineering recommendation, not a description.
- §1.5 notes that TP below the KV-head count stops helping the KV cache.

Check yourself with:

```bash
python ../sim/kv_cache_planner.py --preset gqa-70b --gpu mi300x --context 8192 \
    --mean-context 2050 --compute-batch-cap 128
```

and then re-run it with `--kv-dtype fp8`, `--tp 8`, and `--context 131072`.
