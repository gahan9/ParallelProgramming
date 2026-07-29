# A06 Solutions

Reference answers with reasoning. Try the [`../exercises/`](../exercises/) yourself first — the
learning is in the struggle, not the answer key.

This directory has its own `Makefile`:

```bash
make all GPU_ARCH=gfx950      # build every solution for your GPU
make run                      # build + run all; every one must PASS
make clean
```

| # | File | Notes |
|---|------|-------|
| 1 | `01_tiled_matmul_solution.cpp` | Complete tiled kernel. The header comment explains each of the four pieces, why **both** barriers are required, and why this kernel is already bank-conflict-free at `TILE = 32`. |
| 2 | `02_ragged_dims_solution.cpp` | Zero-filled tile loads. Answers Q1–Q3: why M/N raggedness is self-contained, why a K-edge bug corrupts everything, and what an unguarded store really does (it aliases the next row before it goes out of bounds). Adds two extra shapes, including `K < TILE`. |
| 3 | `03_intensity_math_solution.md` | Full derivation: `I_naive = 0.25`, `I_tiled = T/4`, `I_regblock = BM·BN / (2(BM+BN))`, the ridge crossing, why one-thread-per-output cannot reach it, and how matrix cores + bf16 move the target to `T ≥ 320`. |

A note on what "correct" means here: `make run` stops at the first failure, because a solution that
does not PASS is a bug in the curriculum. But *fast* is a different bar — these are teaching kernels
and they are expected to sit well below rocBLAS / hipBLASLt / cuBLAS. See
[`../README.md` § 6](../README.md#6-challenges-drawbacks--tradeoffs).
