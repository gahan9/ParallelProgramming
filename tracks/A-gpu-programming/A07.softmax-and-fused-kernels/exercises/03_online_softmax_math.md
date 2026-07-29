# Exercise 3 — Online softmax: derive it, prove it, then count the bytes

No code. Work this on paper, then check
[`../solutions/03_online_softmax_math_solution.md`](../solutions/03_online_softmax_math_solution.md).

This is the drill that separates "I can write a softmax kernel" from "I understand why
FlashAttention exists." Interviewers ask part 2 and part 5 almost verbatim.

## Notation

A single row of scores `x[0..C-1]`. Define, for the first `k` elements (`1 ≤ k ≤ C`):

```
m_k = max(x_0, ..., x_{k-1})                  the running max
s_k = Σ_{i<k} exp(x_i - m_k)                  the running normalizer, shifted by m_k
```

The final answer is `y_i = exp(x_i - m_C) / s_C`.

---

## Part 1 — The max trick is free

1. Show in one line that `exp(x_i - m) / Σ_j exp(x_j - m) = exp(x_i) / Σ_j exp(x_j)` for
   **any** constant `m` — not just the max.
2. Given that any `m` works, why is `m = max(x)` the *right* choice? State the property
   of every exponent `x_i - m` that this choice guarantees.
3. What is the largest possible value of `Σ_j exp(x_j - m)` when `m = max(x)`, and the
   smallest? Use this to argue the normalizer can never overflow.
4. Underflow: `exp` returns exactly `0` in fp32 once the argument drops below about
   `-87.3`. If a row contains one score at `+50` and the rest at `-50`, what does the
   safe formula produce for the small scores, and is that answer *wrong*?

---

## Part 2 — Derive the online update

You are streaming the row one element at a time and hold only the pair `(m_k, s_k)`.
Element `x_k` arrives.

1. Write `m_{k+1}` in terms of `m_k` and `x_k`.
2. Starting from the definition of `s_{k+1}`, derive the update

   ```
   s_{k+1} = s_k · exp(m_k - m_{k+1}) + exp(x_k - m_{k+1})
   ```

   Show every step. The whole derivation is one multiply-and-divide of `exp(m_k)` —
   name the factor you insert and where it cancels.
3. Prove that the rescale factor `exp(m_k - m_{k+1})` is always in `(0, 1]`. Why does
   that make the update overflow-proof?
4. When does the rescale factor equal exactly `1`, and what does the update reduce to
   then? (This is the common case — quantify how often it fires for a random row.)

---

## Part 3 — Prove equivalence with the 3-pass algorithm

1. By induction on `k`, prove that the streaming recurrence produces exactly
   `(m_C, s_C)` as defined above — i.e. the online algorithm is not an approximation.
2. Now the parallel version. Two threads each processed part of the row and hold
   `(m_a, s_a)` and `(m_b, s_b)`. Write the merge operator.
3. Prove the merge is **associative** and **commutative**, and identify its **identity
   element**. Why does the curriculum care — what does a monoid buy you on a GPU?
4. Where exactly does the merge produce `NaN`, and what single comparison prevents it?

---

## Part 4 — Numerical thresholds

Fill in this table. `ln(MAX)` is the largest `x` for which `exp(x)` is finite.

| Format | Largest finite value | `exp(x)` overflows above `x ≈` | mantissa bits |
|---|---|---|---|
| fp32 (`float`) | | | |
| fp16 (`half`) | | | |
| bf16 | | | |

1. Attention logits before scaling routinely reach tens or hundreds. Which of the three
   formats is unusable for a naive softmax, and which is *also* risky for the safe one?
2. Why does `1/sqrt(d_head)` scaling reduce — but not remove — the need for the max trick?
3. A colleague proposes computing the exponentials in fp16 to halve the bandwidth, and
   accumulating the normalizer in fp32. Is that safe? What breaks first as `C` grows?

---

## Part 5 — Count the HBM bytes

Setup: `R = 8192` rows, `C = 4096` columns, **fp32**, one block per row. Assume nothing
is cached across passes (the honest worst case for long rows).

1. How many bytes is one row? The whole tensor?
2. For each variant, count the **row traversals** (reads + the write) and turn that into
   total bytes for the tensor:

   | Variant | reads | writes | traversals | total bytes |
   |---|---|---|---|---|
   | safe 3-pass | | | | |
   | online, statistics in one pass | | | | |
   | online + row cached on-chip | | | | |

3. At an achieved bandwidth of **2.0 TB/s**, what is the lower-bound runtime of each
   variant? Express the online-2-pass and cached-1-pass numbers as a speedup over the
   3-pass version.
4. Arithmetic intensity: softmax does roughly one `exp`, one compare, one add and one
   multiply per element. Taking `exp` as ~8 FLOP, compute FLOP/byte for the cached
   variant. Compare against a ridge point of ~35 FLOP/byte. Memory- or compute-bound?
5. **The catch.** At `C = 128` (a small vocabulary or head dim) the row is 512 B and
   every pass hits L2 or even L1. Does the pass count still decide the runtime? What
   *does*? State the row size at which you would expect the pass-count model to start
   predicting reality on a GPU with a 256 MB last-level cache and 8192 resident rows.

---

## Part 6 — The bridge to FlashAttention

1. In attention, softmax is followed immediately by `P · V`. Explain why that lets you
   avoid ever writing the `S × S` probability matrix to HBM.
2. FlashAttention processes K/V in tiles and keeps a running output accumulator `O`
   alongside `(m, s)`. When a new tile raises the max, both `s` **and** `O` must be
   rescaled. Write the update for `O`.
3. What does this buy asymptotically: state the HBM traffic of standard attention and of
   FlashAttention in terms of sequence length `N`, head dim `d`, and on-chip tile size.

## Bonus

Derive `log_softmax(x)_i = (x_i - m) - log(s)` from the safe softmax, and explain in one
sentence why cross-entropy is computed from that expression rather than from
`log(softmax(x)_i)`.
