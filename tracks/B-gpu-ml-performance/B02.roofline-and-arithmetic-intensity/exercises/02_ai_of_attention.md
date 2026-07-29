# Exercise 2 — The arithmetic intensity of attention

No code. Derive it symbolically, then evaluate. Check your work against
[`../solutions/02_ai_of_attention_solution.md`](../solutions/02_ai_of_attention_solution.md).

This is the single most valuable roofline derivation in modern ML, because it produces the
justification for FlashAttention as an inequality rather than an anecdote. If you can do
this on a whiteboard you can defend almost any fusion decision.

---

## Setup

**Operator.** Scaled dot-product attention, forward pass, one head:

```
S = Q · Kᵀ        Q, K, V ∈ R^(seq × dhead)
P = softmax(S)    S, P ∈ R^(seq × seq)
O = P · V         O ∈ R^(seq × dhead)
```

Write `s` for sequence length and `d` for head dimension. All tensors are **bf16 (2 bytes)**.
Ignore the `1/√d` scale and any mask. Charge `exp` as 1 FLOP, and charge the softmax at
**5 FLOPs per element** of `P` (max compare, subtract, exp, accumulate, divide).

**Machine — "Device R"** (same as Exercise 1):

| Property | Value |
|---|---|
| bf16 matrix-core peak, `P_m` | **1200 TFLOP/s** |
| HBM bandwidth, `B` | **4.0 TB/s** |
| HBM capacity | **192 GB** |

**Two implementations to compare:**

- **Unfused** — four kernels. `QKᵀ` writes the full `s × s` score matrix to HBM; softmax reads
  and writes it; `PV` reads it again. This is what a naive PyTorch implementation does.
- **Fused** — one kernel (FlashAttention-style). `Q`, `K`, `V` are read once, `O` is written
  once, and the `s × s` intermediate is tiled through on-chip memory and **never reaches HBM**.

---

## Part A — Derive the FLOP count

**A1.** Count the FLOPs of `Q · Kᵀ` symbolically in `s` and `d`. Then `P · V`. Then the softmax.
Give the total `F(s, d)`.

**A2.** For `s = 8192`, `d = 128`: what percentage of the total FLOPs is the softmax? Justify
dropping it in a back-of-envelope estimate — and state when you must *not* drop it.

---

## Part B — Derive the byte counts

**B1.** For the **unfused** path, count the elements crossing HBM for each of the three stages,
then the total. Convert to bytes. Show that the total is `8sd + 8s²` bytes for bf16.

**B2.** For the **fused** path, count the bytes crossing HBM. Show it is `8sd`.

**B3.** State the *ratio* of unfused to fused traffic as a function of `s` and `d`, and evaluate
it at `d = 128` for `s ∈ {512, 8192}`.

---

## Part C — Arithmetic intensity

**C1.** Derive `I_unfused(s, d)` and simplify it to a single fraction.

**C2.** Take the limit as `s → ∞` with `d` fixed. This is the surprising result: write down what
`I_unfused` converges to and explain in one sentence **why longer sequences do not help**.

**C3.** Derive `I_fused(s, d)` and simplify. How does it behave as `s → ∞`?

**C4.** Fill in the table for `d = 128` (three significant figures):

| `s` | `I_unfused` | `I_fused` | unfused verdict | fused verdict |
|---|---|---|---|---|
| 512 | | | | |
| 1024 | | | | |
| 2048 | | | | |
| 4096 | | | | |
| 8192 | | | | |
| 32768 | | | | |

---

## Part D — Where fusion changes the classification

**D1.** Compute `I_ridge` for Device R's bf16 matrix cores.

**D2.** Solve for the smallest `s` at which the **fused** kernel becomes compute-bound
(`d = 128`). Show the algebra, not just the number.

**D3.** Solve the same inequality for the **unfused** kernel. What happens, and what is the
one-sentence conclusion an engineer should draw?

**D4.** State the attainable rate (in TFLOP/s) for both implementations at `s = 8192`, `d = 128`,
and the resulting speedup ceiling from fusion alone.

---

## Part E — The other reason to fuse

**E1.** Compute the HBM footprint of the `s × s` score matrix in bf16 at `s = 8192`, for one head.
Then for a realistic layer: batch 8, 32 heads. Compare to Device R's 192 GB of HBM.

**E2.** Attention FLOPs grow as `O(s²d)` while the fused traffic grows as `O(sd)`. Explain why
that asymptotic gap makes long-context models *more* attractive for the hardware, not less —
and what the unfused implementation does to that argument.

---

## Part F — Stretch: the cases that break the clean answer

**F1. Causal masking.** With a causal mask, roughly half the score matrix is never needed.
Assume a good implementation skips those tiles. What happens to `I_unfused` and to `I_fused`?
Redo D2 for the causal case.

**F2. Decode / incremental inference.** Now `s_q = 1` (one new query token) against a KV cache of
length `s`. Derive the arithmetic intensity. What does it converge to for large `s`, and why does
**fusion not save you** in this regime?

**F3. Grouped-query attention.** `g` query heads share one KV head. Re-derive the decode
intensity from F2. What does `g` do to it, and what does that tell you about why GQA and larger
inference batch sizes exist?
