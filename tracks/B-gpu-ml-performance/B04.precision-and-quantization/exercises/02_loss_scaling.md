# Exercise 2 — Loss scaling drill

No code — arithmetic on paper (a calculator is fine). Check yourself against
[`../solutions/02_loss_scaling_solution.md`](../solutions/02_loss_scaling_solution.md).

This is the drill that turns "fp16 needs loss scaling" from a slogan into something you
can size, debug, and defend in review.

---

## Setup

A gradient tensor from one backward pass, bucketed by magnitude:

| `|g|` bucket | fraction of elements |
|---|---:|
| `1e-1` … `1e0` | 0.1 % |
| `1e-2` … `1e-1` | 1 % |
| `1e-3` … `1e-2` | 8 % |
| `1e-4` … `1e-3` | 21 % |
| `1e-5` … `1e-4` | 30 % |
| `1e-6` … `1e-5` | 25 % |
| `1e-7` … `1e-6` | 10 % |
| `1e-8` … `1e-7` | 4 % |
| `< 1e-8` | 1 % |

The largest gradient in the tensor is `|g|max = 0.5`.

fp16 facts you need (Lab 1 prints all of them):

```
max finite      6.5504e+04
min normal      6.1035e-05      (below this you are in subnormals: fewer than 10 mantissa bits)
min subnormal   5.9605e-08      (below half of this, a value rounds to exactly 0)
ulp(1)          9.7656e-04
```

---

## Part 1 — Establish that there is a problem

1. What fraction of the elements is **below fp16's min normal**, i.e. representable only
   as a subnormal with degraded precision?
2. What fraction rounds to **exactly zero** in fp16? (Careful: the threshold is half the
   min subnormal, not the min subnormal.)
3. A gradient that becomes exactly zero is not "a small update" — say precisely what it
   does to the corresponding weight, and why summing it over thousands of steps does
   *not* rescue it.
4. Now do the same census for **bf16** (min normal `1.18e-38`). How many buckets
   underflow? This is the entire argument for bf16 in one line.

## Part 2 — Size the scale factor

5. Loss scaling multiplies the loss by `S` before `backward()`, so every gradient is
   scaled by `S`. Find the largest `S` that keeps `S · |g|max` finite in fp16. Then round
   it **down to a power of two** and say why powers of two are the only sensible choice.
6. With your `S`, where does the `< 1e-8` bucket land? Which buckets are now normal
   numbers? Which, if any, are still lost?
7. Suppose instead you pick `S = 2^20`. Compute `S · |g|max` and state exactly what the
   optimizer sees on that step.
8. Derive the **usable window**: for a tensor with dynamic range `|g|max / |g|min`, how
   many binades of headroom does fp16 give you in total, and can a single `S` ever cover
   a range wider than that? (fp16 has 30 binades between min normal and max finite.)

## Part 3 — Make it dynamic, and get the order of operations right

9. Write the dynamic loss-scaling loop in five lines of pseudo-code: what you do on a
   step whose gradients contain `inf`/`NaN`, and what you do after `N` consecutive clean
   steps. Name the two hyper-parameters.
10. On an overflow step you must **skip the optimizer update**. What specifically breaks
    if you unscale and step anyway?
11. You also use gradient clipping at global norm 1.0. Loss scaling and clipping both
    touch the gradients. Give the correct order, and describe the bug produced by the
    wrong one — including why it can look like "clipping is doing nothing".
12. Your gradients are clean but the **forward pass** produces `inf` in an attention
    score. Does loss scaling help? What does?

## Part 4 — The memory bill, and what bf16 does not fix

13. Mixed precision keeps an fp32 master copy. For a 7e9-parameter model with Adam,
    tabulate bytes per parameter for: 16-bit compute weights, fp32 master weights, fp32
    `m`, fp32 `v`, and the 16-bit gradient. Total the bytes and the whole-model GB.
14. Compare against pure-fp32 training (fp32 weights + fp32 grad + fp32 `m` + `v`). Mixed
    precision is often sold as "half the memory". From your two totals, is it?
15. bf16 avoids loss scaling. Name the thing it is *worse* at than fp16, quote the
    `ulp(1)` values for both, and explain why the fp32 master copy makes that acceptable.
16. One-sentence review comment you would leave on a PR that switches a training run
    from bf16 to fp16 "for the extra precision".
