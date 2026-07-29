# A09 Exercise 3 — SOLUTION: the byte arithmetic of attention

Setup: `B = 8`, `H = 32` (so `B*H = 256` heads), `d = 128`, fp16 (`e = 2` bytes).
All figures are **forward pass only**, and "bytes" means bytes crossing the HBM
boundary in either direction. Binary units throughout (1 MiB = 2²⁰ bytes).

---

## Q1 — Traffic table

**The two formulas.** Per head:

```
S matrix                   =  e * N^2                      = 2N^2
endpoints (Q, K, V, O)     =  4 * e * N * d                = 8Nd
naive score traffic        =  4 * e * N^2                   = 8N^2
    (write S by QK^T, read S in softmax, write P, read P by PV)

naive total  =  8Nd + 8N^2
fused total  =  8Nd                    (S never reaches HBM)
```

**Per head:**

| N | `S` matrix | endpoints (Q,K,V,O) | naive total | fused total | ratio |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2 MiB | 1 MiB | **9 MiB** | **1 MiB** | 9× |
| 4096 | 32 MiB | 4 MiB | **132 MiB** | **4 MiB** | 33× |
| 16384 | 512 MiB | 16 MiB | **2064 MiB** (2.02 GiB) | **16 MiB** | 129× |

**Scaled to all `B*H = 256` heads:**

| N | naive total | fused total |
|---:|---:|---:|
| 1024 | 2.25 GiB | 256 MiB |
| 4096 | 33 GiB | 1 GiB |
| 16384 | **516 GiB** | 4 GiB |

**Closed form.**

```
naive / fused = (8Nd + 8N^2) / (8Nd) = 1 + N/d
```

Check: `N/d = 1024/128 = 8 → 9×`; `4096/128 = 32 → 33×`; `16384/128 = 128 → 129×`.

The payoff is governed by the **ratio `N/d`**, not by `N` alone — and note what is
*absent*: the dtype cancels out entirely (halving element size halves both sides).
Since `d` is fixed by the architecture (64–128 in practice) while `N` has grown from
512 to 128k+, this ratio has been growing by orders of magnitude across model
generations. That is precisely why fusing attention went from a nice optimization to
a non-negotiable one.

---

## Q2 — Where does it stop fitting?

**1. A single head's `S` in a 256 MiB cache.**

```
2N^2 <= 256 MiB = 268,435,456 B
  N^2 <= 134,217,728
  N   <= 11,585
```

So a single head's score matrix is cache-resident up to `N ≈ 11.6k`.

**2. The honest version — 256 heads sharing the cache.** Each head's fair share is
`256 MiB / 256 = 1 MiB`:

```
2N^2 <= 1 MiB = 1,048,576 B
  N^2 <= 524,288
  N   <= 724
```

**`N ≈ 724`.** Past a *seven-hundred-token* context, the score matrices of
concurrently running heads no longer fit on chip, so every one of the naive path's
four `S` passes really does go to HBM. The generous single-head answer (11.6k) is the
number people quote when they want the cache to look useful; the 724 is the one that
predicts what you measure.

**3. Capacity, not bandwidth.** PyTorch's naive path materializes `S` for *all* heads
in one tensor, and softmax allocates its output too, so peak live memory is about
`2 * B*H * 2N^2`:

```
N = 16384:  B*H * 2N^2 = 256 * 512 MiB = 128 GiB per tensor
            two of them              = 256 GiB  >  192 GiB  ->  OOM
```

Solving for the limit with two tensors live:

```
2 * 256 * 2N^2 <= 192 GiB = 206,158,430,208 B
           N^2 <= 201,326,592
           N   <= 14,189
```

So somewhere around `N ≈ 14k` the naive path stops running *at all* — before any
bandwidth argument is needed, and that is with nothing else on the GPU (no weights,
no activations, no optimizer state, so the real ceiling is much lower). The fused
path at the same `N` needs 4 GiB of traffic and stores no score matrix, so its
footprint is just Q, K, V, O.

> This is the OOM row you see in `triton/benchmark_attention.py`. It is not a bug in
> the harness; it is the result.

---

## Q3 — Why the asymptotics change

**1. Arithmetic intensity.** Useful compute per head is `4*N^2*d` flops (two matmuls
of `2*N^2*d` each). Dividing by the traffic from Q1:

```
naive:  AI = 4N^2 d / (8Nd + 8N^2) = N*d / (2*(N + d))
fused:  AI = 4N^2 d / (8Nd)        = N / 2
```

**2. The limits.**

```
naive:  N -> infinity  =>  AI -> d/2   = 64 FLOP/byte   (d = 128)
fused:  N -> infinity  =>  AI -> N/2   -> unbounded
```

The naive implementation's arithmetic intensity **saturates at `d/2` and never
improves, no matter how long the sequence gets**. That single fact is the whole
paper: attention is not intrinsically memory-bound, the *unfused implementation* is.
Adding sequence length adds flops and bytes in the same proportion, so you can never
compute your way out of it. The fused version's intensity grows linearly in `N`,
which means longer contexts make it *more* efficient, not less.

Sanity check at `N = 4096, d = 128`: naive `= 4096*128 / (2*4224) = 62.1` FLOP/byte,
versus `d/2 = 64` — already within 3% of its asymptote at a context length people
consider short.

**3. Against a 200 FLOP/byte ridge point** (≈1000 TFLOP/s fp16 ÷ ≈5 TB/s — round
numbers for the arithmetic, not a spec quote; substitute your own GPU's):

- At `N = 4096` the naive path sits at ~62 FLOP/byte, well left of the ridge:
  **memory-bound**, and it will still be memory-bound at `N = 1,000,000`.
- The fused path crosses the ridge when `N/2 >= 200`, i.e. **`N >= 400`**. For every
  sequence length anyone actually trains on, the fused kernel is **compute-bound** —
  which is why the remaining work on attention kernels is about matrix-core
  utilization, pipelining, and precision, not about bandwidth.

**4. One sentence.** Fusing changes no flops at all; it removes the `O(N²)` HBM
round-trips that were the actual bottleneck, moving the kernel from the bandwidth
roof to the compute roof — and once you are compute-bound, the hardware's peak is the
only thing left between you and the limit.

---

## Q4 — Stretch: causal block accounting

With `BLOCK_M = BLOCK_N = b` and `T = N/b` blocks per axis, query block `i`
(0-indexed) visits `i + 1` key blocks, so:

```
visited / total = (1 + 2 + ... + T) / T^2 = T(T+1)/2 / T^2 = (T+1)/(2T)  ->  1/2
```

Roughly **half** the blocks, approaching exactly half as `T` grows. The measured
speedup is always somewhat less, for three separate reasons:

1. **The diagonal block is full-price.** It computes a complete `b × b` tile of
   scores and then throws away just under half of them via the mask. Its loads,
   `tl.dot`, and exponentials are all paid in full.
2. **Load imbalance.** Program `pid_m = 0` walks one key block; the last walks `T`.
   The kernel finishes when the slowest program finishes, so unless the scheduler can
   pack the short programs onto freed compute units (it partly can — there are
   `B*H*T` programs total), you pay a tail. This is exactly why FlashAttention-2
   changed the parallelization and work partitioning
   ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691)).
3. **Fixed costs don't shrink.** Loading the Q tile, storing the O tile, the final
   division, and kernel-launch overhead are per-program constants independent of how
   many key blocks were skipped.

A practical corollary for reviewers: if someone reports that turning on causal
masking made their kernel *exactly* 2× faster, ask to see the measurement — and if it
made no difference at all, they masked without skipping.
