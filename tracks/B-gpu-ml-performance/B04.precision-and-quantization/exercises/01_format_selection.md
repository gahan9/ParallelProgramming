# Exercise 1 — Format selection drill

No code. Decide, and **justify from range, precision, and throughput** — in that order,
because range failures are catastrophic (inf/NaN), precision failures are gradual, and
throughput is what you were trying to buy in the first place.

Check yourself against
[`../solutions/01_format_selection_solution.md`](../solutions/01_format_selection_solution.md).

Answers of the form "bf16, it's what everyone uses" score zero. Every answer needs a
number: a magnitude that fits (or doesn't) in the format's range, an `ulp` compared
against the smallest quantity you must resolve, or a byte count.

---

## Reference numbers you are allowed to look up (from Lab 1's table)

| format | bits | E | M | max finite | min normal | ulp(1) |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 32 | 8 | 23 | 3.40e38 | 1.18e-38 | 1.19e-7 |
| tf32 | 32 (19 used) | 8 | 10 | 3.40e38 | 1.18e-38 | 9.77e-4 |
| bf16 | 16 | 8 | 7 | 3.39e38 | 1.18e-38 | 7.81e-3 |
| fp16 | 16 | 5 | 10 | 65504 | 6.10e-5 | 9.77e-4 |
| fp8 E4M3 (OCP) | 8 | 4 | 3 | 448 | 1.56e-2 | 1.25e-1 |
| fp8 E5M2 | 8 | 5 | 2 | 57344 | 6.10e-5 | 2.50e-1 |
| int8 (symmetric) | 8 | — | — | 127 × scale | — | 1 × scale |

---

## Scenario A — Pre-training a 70B-parameter LLM on 512 GPUs

Gradients during training span roughly `1e-9` to `1e0` in magnitude, and a handful of
activation tensors reach `1e4` in the forward pass of an unstable layer.

1. Which 16-bit format for the **forward/backward compute**, and why does the other
   16-bit format need extra machinery to survive?
2. What precision holds the **master weights** and the optimizer moments, and why can
   the answer not be the same as (1)? (Lab 1 section 4 printed the reason — quote the
   number.)
3. Compute the **bytes per parameter** for: bf16 compute copy + fp32 master + fp32 Adam
   `m` + fp32 Adam `v`. Multiply by 70e9. Does that fit on 512 × 192 GB even before
   activations? What does that tell you about which of these four things gets sharded?
4. You are offered fp8 E4M3 for the GEMMs. State the *two* things you must add to the
   pipeline before that is safe.

## Scenario B — Serving a 13B model on one 80 GB GPU, latency-critical

Weights are 13e9 parameters. Decode is one token at a time, batch 1–8.

1. Compute the weight memory in fp16, int8, and int4 (ignore scales first, then add
   per-group fp16 scales at `g = 128` and redo it).
2. Decode at batch 1 reads every weight once per token and does ~2 FLOPs per weight.
   Is that arithmetic intensity above or below the ridge point of a GPU with 3 TB/s and
   1000 TFLOP/s (bf16)? Which resource sets your tokens/second?
3. From (2): if you halve the weight bytes, what happens to tokens/second, and why is
   that a *different* argument from "int8 matrix cores are 2× faster"?
4. Name the quality risk you have taken on, and the *evidence* you would need before
   shipping it. ("Perplexity looked fine" — is that enough? Justify.)

## Scenario C — A loss reduction over 1e6 elements, then a LayerNorm

The loss is a mean over 1e6 per-element losses, each of order `1e-2`. LayerNorm needs
the mean and variance of a 8192-element row whose values are of order 10.

1. If you accumulate the loss sum in fp16, estimate the relative error and say what
   goes wrong first — underflow, overflow, or stagnation of the running sum. (Lab 1
   section 5 measured the shape of this.)
2. Choose the accumulator width for the loss sum and for the LayerNorm statistics.
   Does your answer depend on the *storage* precision of the inputs? Explain the
   distinction in one sentence.
3. The LayerNorm is computed with `E[x²] − E[x]²`. Given values of order 10 and a
   variance of order 1, is fp32 enough? Which lab section is this, and what is the
   algebraic fix?
4. Your reduction result changes in the 6th decimal place when you change the block
   count from 256 to 1024. Is that a bug? What must the regression test assert instead?

## Scenario D — KV cache for a Llama-3-8B-shaped model

32 layers, 8 key/value heads, head dimension 128, context 8192, batch 32.

1. Derive the KV cache size in bytes for fp16. (Per token per layer you store `K` and
   `V`, each `n_kv_heads × head_dim` values.) Express it per token first — that number
   is the one worth memorising.
2. Compare it against the 16 GB the fp16 *weights* occupy. Which one is the memory
   bottleneck at this batch and context, and what does that imply about where to spend
   your quantization effort?
3. Choose a format for the cache. Why is the K cache usually treated differently from
   the V cache? (One sentence; the citation is in README section 8.)
4. The cache is written once and read on every subsequent token. Does quantizing it
   help latency, memory, or both? Be precise about the mechanism.

## Scenario E — The portability trap

You train with fp8 E4M3 on an H100 and export the fp8 tensors to a checkpoint. A
colleague loads that checkpoint on an MI300X (`gfx942`).

1. The two chips do not implement the same E4M3. Name the difference and give the two
   maximum representable values.
2. What exactly goes wrong: silent numerical drift, NaNs, or nothing at all? Justify
   from the bit patterns.
3. What is the correct thing to store in a checkpoint instead, and what does that cost?

---

## Deliverable

A table with one row per scenario: **format chosen · the range number that justifies it ·
the precision number that justifies it · the throughput argument · the risk you accepted**.
If any cell is empty, you have not finished the decision.
