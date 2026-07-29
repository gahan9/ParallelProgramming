# Exercise 2 — The KV-cache budget drill

> Paper drill. Target time: **10 minutes**. This is the single most common quantitative question in
> an LLM-serving interview, because it is the calculation that decides how much hardware a product
> needs.

## The situation

You are asked to serve a 70 B-parameter decoder-only model on **one AMD MI300X** (192 GB HBM,
5.3 TB/s). Product wants to advertise a **32,768-token context**.

Given, so you can focus on the cache rather than re-deriving §2.1:

| Field | Value |
|---|---|
| layers `L` | 80 |
| hidden size `d` | 8192 |
| attention heads `h` | 64 (so `d_head` = 128) |
| fp16 weights on disk/HBM | **138 GB** |
| target context `N` | 32,768 |
| KV cache dtype | fp16 |
| GPU | MI300X: 192 GB, 5.3 TB/s |

The serving runtime needs a **10 GB reserve** for activations, workspace, CUDA/HIP graph pools, and
allocator fragmentation. Provisioning without it is [module README pitfall
4](../README.md#pitfall-4--sizing-the-kv-cache-without-the-runtimes-overhead).

---

## Questions

### 1. The budget

**(a)** How many GB are available for KV cache after the weights and the reserve?

**(b)** Write the KV-cache formula and identify every term for this model.

### 2. Concurrency under two attention schemes

**(a)** **MHA** (`kv_heads = 64`). Compute `d_kv`, the KV bytes per token, the KV bytes for one
32k-context sequence, and the **maximum concurrent sequences**.

**(b)** **GQA with 8 KV heads.** Same four quantities.

**(c)** State the ratio between the two answers and explain, in one sentence, exactly what in the
formula produced it.

**(d)** Your MHA answer should be alarming. Say precisely what it means for the product as specified,
and what the deployment options are if the model is MHA and cannot be changed.

### 3. Throughput consequences

Use the GQA configuration from 2(b) for the rest of the drill.

**(a)** Bytes read per decode step at your maximum concurrency, and the resulting step time on
5.3 TB/s.

**(b)** Aggregate tokens/s and per-sequence tokens/s at that batch.

**(c)** Compare per-sequence tokens/s at this batch against the batch-1 ceiling
(`5300 ÷ 138`). Explain to a product manager, in two sentences, why running many users at once
costs each individual user so little.

### 4. Double the concurrency

Product now wants **twice** the concurrent users at the same context length, same GPU, same latency
target. Evaluate each option below. For each, give the new concurrency **as a number**, and state
what it costs.

**(a)** Quantize the **KV cache** to fp8.

**(b)** Quantize the **weights** to int8, leaving the KV cache in fp16.

**(c)** Switch to **MQA** (1 KV head).

**(d)** Halve the advertised context to 16,384.

**(e)** Add a second MI300X with 2-way tensor parallelism.

**(f)** Which do you recommend, and why? Name the one option that improves *both* concurrency and
per-token latency, and explain the mechanism.

### 5. The subtle bit

For option 4(a) — fp8 KV cache at double the batch — compute the bytes read per decode step and
compare it to your answer in 3(a).

**(a)** What do you notice about the step time?

**(b)** Therefore what happened to aggregate throughput, and to per-sequence throughput?

**(c)** Generalize: under what condition does adding concurrency cost *nothing* in per-user latency?
State the condition as an inequality involving the weight bytes and the total KV bytes.

---

## Check your work

```bash
cd ..
python tools/model_calculator.py --preset llama2-70b --seq 32768 --batch 4 --gpu mi300x
python tools/model_calculator.py --preset llama2-70b --seq 32768 --batch 8 --kv-dtype fp8 --gpu mi300x
python tools/model_calculator.py --preset llama2-70b --seq 32768 --kv-heads 64 --gpu mi300x
```

The calculator reports a *higher* concurrency than you should have computed. Work out why before
reading the solution — the discrepancy is deliberate and it is the lesson of question 1(a).

Full working:
[`../solutions/02_kv_cache_budget_solution.md`](../solutions/02_kv_cache_budget_solution.md).
