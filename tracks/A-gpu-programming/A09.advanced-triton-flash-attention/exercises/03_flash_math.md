# A09 Exercise 3 — The byte arithmetic of attention (paper drill)

No GPU needed. This is the reasoning an interviewer actually probes: *can you show,
in bytes, why FlashAttention exists?* Work it out before reading
[`../solutions/03_flash_math_solution.md`](../solutions/03_flash_math_solution.md).

## The setup

One transformer layer's self-attention, forward pass only:

| Symbol | Meaning | Value |
|---|---|---|
| `B` | batch size | 8 |
| `H` | heads | 32 |
| `d` | head dimension | 128 |
| `N` | sequence length | 1024, 4096, 16384 |
| `e` | bytes per element (fp16) | 2 |

Two implementations:

- **Naive / unfused** — three kernels. `S = QK^T` writes an `(N, N)` score matrix to
  HBM per head; softmax reads `S` and writes `P`; `O = PV` reads `P`. Q, K, V are
  read once and O is written once.
- **Fused / FlashAttention** — one kernel. Q, K, V are read once, O is written once,
  and `S` is never written to HBM at all (assume K/V block re-reads are served by
  cache).

Count a byte every time it crosses the HBM boundary in either direction.

## Q1 — Traffic table

Fill this in **per head** first, then scale to the whole `B*H` batch. Show the
formula, not just the number.

| N | `S` matrix bytes | endpoint bytes (Q,K,V,O) | naive total | fused total | ratio |
|---|---|---|---|---|---|
| 1024 | | | | | |
| 4096 | | | | | |
| 16384 | | | | | |

Then answer: the ratio has a simple closed form in terms of `N` and `d`. What is
it, and what does it say about which parameter controls the payoff?

## Q2 — Where does it stop fitting?

1. A GPU has a **256 MiB** last-level cache (call it L2 / Infinity Cache). At what
   `N` does a **single head's** score matrix stop fitting?
2. That is the generous answer. The GPU runs many heads concurrently, so the honest
   question is: if `B*H = 256` heads must share those 256 MiB, at what `N` does one
   head's fair share stop holding `S`?
3. The GPU has **192 GiB** of HBM. PyTorch's naive path materializes `S` for *all*
   `B*H` heads in one tensor, and softmax needs an output tensor too. At what `N`
   does the naive path run out of memory outright — before any bandwidth argument
   is even needed?

## Q3 — Why the asymptotics change

1. Attention's useful compute is two matmuls: `2*N^2*d` flops each, so `4*N^2*d`
   flops per head. Write the **arithmetic intensity** (flops per HBM byte) for both
   implementations as a function of `N` and `d`.
2. Take the limit as `N` grows for each. One of them converges to a constant. Which,
   what constant, and why is that the whole story?
3. A GPU with ~1000 TFLOP/s of fp16 and ~5 TB/s of HBM has a roofline ridge point
   at 200 FLOP/byte. Which implementation is memory-bound at `N = 4096`, and at what
   `N` does the other one become compute-bound?
4. In one sentence: fusing does not reduce the number of *flops* at all. Explain why
   it is still the single biggest win in modern attention kernels.

## Q4 — Stretch

Causal masking lets the fused kernel skip whole K/V blocks. For `BLOCK_M = BLOCK_N`,
what fraction of blocks does a causal kernel visit compared with a full one, and why
is the measured speedup always somewhat *less* than that fraction suggests?
