# Exercise 1 — Size a model from its spec

> Paper drill. Target time: **12 minutes** with no calculator beyond mental arithmetic and a scratch
> pad. Show the formula before each number.

You have been handed the config for a model your team is about to deploy. It is a plausible modern
architecture — GQA plus a SwiGLU FFN — but the specific numbers here are **synthetic**, invented for
this drill. Do not go looking for a published model that matches them.

## The spec

| Field | Value |
|---|---|
| layers `L` | 48 |
| hidden size `d` | 6144 |
| attention heads `h` | 48 |
| KV heads | 8 (GQA) |
| FFN inner width `d_ffn` | 16384, **SwiGLU** (gate + up + down, three matrices) |
| vocabulary | 64,000 |
| embeddings | **untied** (separate input embedding and `lm_head`) |
| max context `N` | 8192 |
| serving dtype | fp16 for weights and KV cache |
| target GPU | AMD MI300X — 5.3 TB/s HBM, 192 GB, 1307.4 TFLOP/s dense bf16 |

---

## Questions

### 1. Parameters

**(a)** Derive `d_head` and `d_kv`, and state which parameter matrices GQA shrinks.

**(b)** Compute the parameters in one layer, broken out as: Q projection, K+V projections, O
projection, FFN. Express the layer total both in absolute terms and as a multiple of `d²`.

**(c)** Compute the total parameter count of the model, including embeddings. Round to the nearest
0.1 G.

**(d)** The textbook shortcut says a layer is `12d²`. What does it predict here, and by what factor
is it wrong? Explain *which two* architectural choices in this spec pull in opposite directions and
which one wins.

### 2. Weight memory

**(a)** How much HBM do the weights occupy in fp16?

**(b)** In int8?

**(c)** Does the fp16 model fit on one MI300X (192 GB) with room to serve? On one H100 SXM (80 GB)?

### 3. FLOPs, both regimes

**(a)** Forward FLOPs per token for the transformer blocks (exclude the `lm_head`), from the
`2 × params` rule.

**(b)** FLOPs to train this model on 1.5 T tokens. State the heuristic you used and two things it
ignores.

**(c)** Total FLOPs to **prefill** one full 8192-token prompt. Include both the linear term and the
attention term, and report what fraction of the total the attention term is.

**(d)** FLOPs to **decode** one token, for a sequence that already has 8192 tokens of context.

**(e)** Compare (c) and (d) as *FLOPs per token*. Then explain why decode is nevertheless the slow
regime, in one sentence.

### 4. KV cache

**(a)** KV cache bytes per token per sequence.

**(b)** KV cache for one sequence at the full 8192-token context.

**(c)** What would (b) have been under MHA? State the ratio and where it comes from.

### 5. The prediction

**(a)** Compute the ceiling on **single-stream** decode tokens/s on an MI300X, counting only the
weight read. Show the division with units.

**(b)** Redo it including the KV-cache read for a full 8192-token context. How much did it move the
answer, and is that term worth carrying in a whiteboard estimate?

**(c)** Redo (a) with int8 weights. State the general rule this demonstrates in one sentence.

**(d)** Redo (a) for an H100 SXM (3.35 TB/s). Which single hardware number determined the answer?

### 6. Regime check

**(a)** The MI300X ridge point is 247 FLOP/byte. Compute the arithmetic intensity of a single `d × d`
projection in fp16 for `T = 1` token and for `T = 8192` tokens. Place each in a regime.

**(b)** Roughly what batch size would make decode compute-bound on this machine? You do not need the
exact algebra — reason from the small-`T` approximation.

**(c)** Your capacity planner wants to know: is this model's serving cost going to be dominated by
bandwidth or by capacity? Support your answer with two numbers you have already computed.

---

## Check your work

Only after you have your own answers:

```bash
cd ..
python tools/model_calculator.py -L 48 -d 6144 --heads 48 --kv-heads 8 \
    --ffn 16384 --glu --vocab 64000 --seq 8192 --gpu mi300x
```

Then try `--dtype int8`, `--gpu h100`, and `--kv-heads 48` to confirm questions 2b, 5c, 5d and 4c.

Full working: [`../solutions/01_size_a_model_solution.md`](../solutions/01_size_a_model_solution.md).
