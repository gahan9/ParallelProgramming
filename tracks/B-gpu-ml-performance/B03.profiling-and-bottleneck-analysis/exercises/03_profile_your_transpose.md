# Exercise 3 — Prove *why* the padded transpose wins

**Competency:** connecting a microarchitectural counter to a source-level cause.

[A02 Lab 2](../../../A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#lab-2--transpose-coalescing--shared-memory--bank-conflicts)
builds a three-rung ladder — `transpose_naive`, `transpose_tiled`,
`transpose_tiled_padded` — and *asserts* that the ladder climbs because of
coalescing, then shared-memory staging, then bank-conflict removal. A02 measured
the bandwidth. It did not prove the mechanism.

That is this drill. **Same bandwidth numbers, but this time from counters, with
a causal chain you could defend in a review.**

---

## Step 1 — Get the bandwidths back

```bash
cd exercises
make transpose-timeline GPU_ARCH=gfx942
make parse
```

`make parse` gives you per-kernel device time for all three variants from one
run. Convert each to effective GB/s: the transpose moves `2 * n * n * 4` bytes
(one read + one write per element) with `n = 4096`.

| kernel | device time (ms) | effective GB/s | ratio vs. naive |
|---|---|---|---|
| `transpose_naive` | | | 1.00× |
| `transpose_tiled` | | | |
| `transpose_tiled_padded` | | | |

## Step 2 — Predict the counters before you collect them

Write down, for each rung, what you expect to change and in which direction.
This is the hypothesis step; skipping it turns the next step into confirmation
bias.

| Transition | Counter I expect to move | Direction | Why (one sentence) |
|---|---|---|---|
| naive → tiled | | | |
| tiled → padded | | | |

## Step 3 — Collect

```bash
make transpose-counters TRANSPOSE_KERNEL=transpose_naive
make transpose-counters TRANSPOSE_KERNEL=transpose_tiled
make transpose-counters TRANSPOSE_KERNEL=transpose_tiled_padded
```

The target collects `LDSBankConflict`, `MemUnitStalled`, `L2CacheHit`, and
`MeanOccupancyPerCU`. Read the values out of
`profiling_output/transpose/pmc_1/*_counter_collection.csv`.

| kernel | `LDSBankConflict` | `MemUnitStalled` | `L2CacheHit` | `MeanOccupancyPerCU` |
|---|---|---|---|---|
| `transpose_naive` | | | | |
| `transpose_tiled` | | | | |
| `transpose_tiled_padded` | | | | |

NVIDIA equivalent (the two sections that carry the same information):

```bash
cd ../../../A-gpu-programming/A02.memory-hierarchy-and-coalescing
make cuda SM_ARCH=sm_90
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --kernel-name regex:'transpose_tiled' --launch-skip 1 --launch-count 1 \
    ./cuda/transpose.exe
```

In the `MemoryWorkloadAnalysis` shared-memory table, the row to find is the
shared-memory **bank conflict** count; in `SpeedOfLight`, watch Memory
Throughput % move while Compute (SM) Throughput % stays near zero (transpose
does no arithmetic at all).

## Step 4 — Build the causal chain

Answer in prose, citing your numbers:

1. **naive → tiled.** Which counter changed most, and what source-level change
   caused it? Be specific about *which* of the two global accesses (the read or
   the write) was the problem in `transpose_naive`, and how staging through LDS
   fixed it without changing the arithmetic.
2. **tiled → padded.** The only source difference is `tile[TILE][TILE]` →
   `tile[TILE][TILE + 1]`. Explain, in bank terms, why one extra word per row
   changes the conflict count. Then show that your measured `LDSBankConflict`
   values are consistent with that explanation.
3. `transpose_naive` has **no shared memory at all**, so its `LDSBankConflict`
   must be ~0 — a *good-looking* value on a *bad* kernel. Explain in one sentence
   why "this counter looks fine" is never, by itself, evidence of health, and
   what you must pair it with.
4. What is `transpose_tiled_padded`'s effective bandwidth as a fraction of the
   copy ceiling you measured in A02 Lab 1? Transpose reads and writes each byte
   exactly once, the same as a copy — so what accounts for any remaining gap?
   Name one candidate and the counter that would test it.

## Step 5 — Falsify yourself

Change `TILE` from 32 to 16 in a scratch copy of A02's `transpose.cpp`, rebuild,
and re-measure. Does the padded-vs-unpadded gap grow, shrink, or vanish? Predict
first from the bank model (LDS has 32 banks of 4 bytes), then measure. If your
prediction was wrong, your model of the mechanism is wrong — fix the model, not
the memory of the prediction.

---

Reference answers, with the bank-index arithmetic worked out and the AMD↔NVIDIA
counter mapping, are in
[`../solutions/03_profile_your_transpose_solution.md`](../solutions/03_profile_your_transpose_solution.md).
