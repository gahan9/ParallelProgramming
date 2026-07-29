# Drill 2 — Choose the parallelism

**Goal:** four scenarios, four different right answers. For each one, state the
parallelism combination, the **interconnect requirement**, and the failure mode you
are accepting.

## Your hardware, for all four scenarios

| Tier | Bandwidth (achievable, per GPU) | Latency |
|---|---|---|
| Intra-node (8 GPUs, fully connected) | ~300 GB/s | ~5 µs |
| Inter-node (RoCE/InfiniBand) | ~50 GB/s | ~15 µs |
| GPU | 192 GiB HBM, ~400 TFLOP/s achieved bf16 dense | |

64 GPUs total = 8 nodes × 8 GPUs. Assume bf16 + Adam (16 bytes/param) throughout.

For **each** scenario answer these five questions:

1. What is the **binding constraint** — memory (state? activations?), throughput,
   or latency?
2. Which axes do you use, and at what degree? Give the full `dp × tp × pp` (+ `cp`
   or `ep` if relevant) decomposition that multiplies to your GPU count.
3. **Which tier does each axis sit on**, and what is the minimum bandwidth that
   makes it viable? Justify with a number.
4. What is the failure mode you have accepted, and what symptom would tell you it
   has arrived?
5. What would change your answer? Name one parameter and its threshold.

---

## Scenario A — "It fits, but it trains too slowly"

An 8B model (`L=32, h=4096, a=32, s=4096`) trains on a *single* GPU at micro-batch 1
with selective activation recomputation — verify that yourself, it is close. Loss
curves are healthy. Management wants it 40× faster and has given you all 64 GPUs.

Additional facts: the single-GPU step time is **700 ms**, and the critical batch
size for this dataset is believed to be around **2M tokens**.

## Scenario B — "The model is 5× too large for one GPU"

A 65B model (`L=80, h=8192, a=64, s=4096`). Its state alone is ~969 GiB, about 5×
one GPU's HBM. You have the same 64 GPUs and need to pre-train from scratch.

## Scenario C — "We need 128K context"

The same 8B model from Scenario A, but sequence length 131072 instead of 4096.
Everything else is unchanged.

> Before you answer: compute the activation memory per layer at `s = 131072`. The
> number is genuinely shocking and it drives the whole answer.

## Scenario D — "Low-latency serving of a large model"

Serve the 175B model from Drill 1 for interactive chat. The SLO is a
**time-per-output-token under 25 ms at batch size 1–8**. Weights in bf16 are ~326
GiB, so this does not fit on one GPU either. You have full nodes available.

> This is an *inference* question. Re-read module §3.8 before answering: the axis
> that helps training throughput is not the axis that helps decode latency, and the
> KV cache adds a constraint that does not exist in training.

---

## Check your work

```powershell
# Scenario A: does 8B really fit on one GPU?
python tools/memory_planner.py --preset llama3-8b --gpus 1 --seq 4096 --recompute selective

# Scenario B memory shape
python tools/memory_planner.py --params 65 --layers 80 --hidden 8192 --heads 64 `
  --seq 4096 --gpus 16 --suggest --top 4

# Scenario C: the quadratic term bites
python tools/memory_planner.py --preset llama3-8b --gpus 8 --seq 131072
python tools/memory_planner.py --preset llama3-8b --gpus 8 --seq 131072 --suggest --top 4

# Scenario A: is DDP's all-reduce affordable on each tier?
python tools/collective_cost.py --gpus 64 --bw 50  --latency 15 --params 7 --step-ms 700
python tools/collective_cost.py --gpus 8  --bw 300 --latency 5  --params 7 --step-ms 700

# Scenario D: the decode-phase message is tiny
python tools/collective_cost.py --gpus 8 --bw 300 --latency 5 --bytes 32768
```

## What a good answer looks like

- Each scenario's answer starts by naming the binding constraint. Scenarios A and
  C differ *only* in `s`, and that single change moves the constraint from
  throughput to activation memory — if your two answers look similar, re-read them.
- Every "keep it inside a node" claim is backed by a bandwidth number, not by
  convention.
- Scenario D distinguishes prefill from decode. They have different bottlenecks.
- You name at least one axis you deliberately *did not* use, and why.
