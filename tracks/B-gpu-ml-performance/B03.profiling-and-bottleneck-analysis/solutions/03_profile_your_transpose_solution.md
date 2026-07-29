# Solution 3 — Why the padded transpose wins

> **On numbers in this key.** The bandwidth figures below are the ones A02 § 5
> already publishes for **MI300X (`gfx942`, ROCm 6.2), `n = 4096`**; they are
> reproduced here as a worked example, not re-measured. Counter values are given
> as **predicted directions**, which your own run should confirm. Absolute
> numbers differ by GPU; the ratios and the mechanism do not.

---

## Step 1 — The ladder, restated

From [A02 § 5](../../../A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#5-performance-analysis):

| kernel | effective GB/s | vs. copy ceiling (~4800 GB/s) | ratio vs. naive |
|---|---:|---:|---:|
| `transpose_naive` | ~700 | ~15% | 1.00× |
| `transpose_tiled` | ~2600 | ~54% | ~3.7× |
| `transpose_tiled_padded` | ~3400 | ~71% | ~4.9× |

Transpose reads each byte once and writes it once, exactly like a copy, so the
copy kernel's ~4800 GB/s is the correct ceiling to judge all three against.
Transpose also does **zero arithmetic** — which means any SpeedOfLight report
showing meaningful Compute (SM) Throughput on these kernels is a report you have
misread.

## Step 4.1 — naive → tiled: which global access was broken

```cpp
// transpose_naive
out[x * n + y] = in[y * n + x];
```

Lanes vary `threadIdx.x`, so `x` is the fast-varying index:

- **The read `in[y*n + x]`** — consecutive lanes read consecutive addresses.
  **Coalesced. Not the problem.**
- **The write `out[x*n + y]`** — consecutive lanes write addresses `n` floats
  (16 KB at `n = 4096`) apart. Every lane lands in its own cache line: one
  64-byte transaction delivers 4 useful bytes, **1/16 efficiency**.

So the naive kernel's defect is specifically the **write**, and the counter that
shows it is `MemUnitStalled` (high) with `L2CacheHit` poor — the memory unit is
saturated moving bytes nobody uses.

**What tiling changes.** The tiled kernel does not make the strided access
disappear; it **relocates** it:

```cpp
tile[threadIdx.y][threadIdx.x] = in[y * n + x];   // coalesced global READ
__syncthreads();
out[ty * n + tx] = tile[threadIdx.x][threadIdx.y]; // coalesced global WRITE
```

Both global accesses are now stride-1. The transpose — the awkward,
against-the-grain access — happens **inside LDS**, where a scattered access costs
tens of cycles instead of a wasted HBM transaction. The arithmetic never changed;
only where the strided access lives. That is the entire ~3.7×.

Expected counter movement: `MemUnitStalled` drops sharply, `L2CacheHit` improves,
and `LDSBankConflict` goes from ~0 to **high** — you have deliberately moved the
problem into LDS, and the next rung is about paying that off.

## Step 4.2 — tiled → padded: the bank arithmetic

LDS is **32 banks of 4-byte words**; the bank holding word `w` is `w mod 32`.
A wavefront's LDS accesses are serviced in one pass **iff** they hit distinct
banks (or all read one address, which broadcasts).

The read-back is `tile[threadIdx.x][threadIdx.y]` — `threadIdx.x` varies across
lanes, so lanes walk a **column** of the tile.

**Unpadded, `tile[32][32]`.** For lanes `x = 0…31` at fixed `y = c`:

```
word index = x * 32 + c
bank       = (x * 32 + c) mod 32
           = (0 + c) mod 32              since 32x is a multiple of 32
           = c                            for EVERY lane
```

All 32 lanes hit **bank `c`** at 32 different addresses → a **32-way conflict**,
serialized into 32 passes.

**Padded, `tile[32][33]`.** The row stride is now 33 words:

```
word index = x * 33 + c
bank       = (33x + c) mod 32
           = (32x + x + c) mod 32
           = (x + c) mod 32              <- depends on x again
```

As `x` runs 0…31, `(x + c) mod 32` takes **all 32 distinct values**. One pass,
zero conflicts. The cost is one dead word per row: 32 extra floats = 128 bytes
per tile, ~3% more LDS.

**Consistency check against your measurement.** `LDSBankConflict` is reported as
*"percentage of GPU time LDS is stalled by bank conflicts"*, so it should be
clearly non-zero for `transpose_tiled` and near zero for `transpose_tiled_padded`.
If both read ~0, either your GPU's LDS conflict counter is not the one you think
it is (check `rocprofv3 --list-avail`) or the compiler vectorized the access in a
way that changed the pattern — inspect before concluding the theory is wrong.

> **AMD caveat.** A CDNA wavefront is 64 lanes, and LDS services 32 banks; the
> hardware processes the wavefront in halves. The `+1` padding trick works on
> both vendors, which is why A02 uses it unchanged — but do not assume the
> conflict *degree* maps one-to-one between a 32-lane warp and a 64-lane
> wavefront.

## Step 4.3 — why "this counter looks fine" proves nothing

`transpose_naive` allocates **no shared memory at all**, so its `LDSBankConflict`
is necessarily ~0 — the best possible reading on the worst kernel in the ladder.
A counter reading well tells you only that *that one mechanism* is not the
limiter; it is silent about every other mechanism, including the ones you have
not thought of.

**Pair every counter with a top-level roof comparison.** Effective GB/s vs. the
copy ceiling (or SpeedOfLight's Memory/Compute percentages) answers *"is there a
problem?"*. Only once that says yes do individual counters answer *"which
problem?"*. Reading counters first is how you end up reporting that a kernel at
15% of peak has excellent bank behaviour.

## Step 4.4 — the remaining 29%

`transpose_tiled_padded` reaches ~71% of the copy ceiling. Transpose moves
exactly the same bytes as a copy, so something still costs ~29%. Candidates, with
the counter that discriminates each:

| Candidate | Mechanism | Counter that tests it |
|---|---|---|
| **Reduced occupancy** | The LDS tile is a per-CU budget; fewer resident blocks means less latency hiding | `MeanOccupancyPerCU` — compare against the copy kernel's |
| **Partition camping / channel imbalance** | Output tiles are written with a large power-of-two stride, so concurrent blocks can pile onto a subset of memory channels | `MemUnitStalled` staying high despite good `L2CacheHit`; test by re-indexing blocks diagonally (the classic fix from Ruetsch & Micikevicius) |
| **Barrier + LDS round trip** | Every element makes an extra hop through LDS with a `__syncthreads()` between | Compare against a copy kernel that is artificially routed through LDS — isolates the staging cost from the transpose |

Pick **one**, change **one** thing, re-measure. The single most likely candidate
on a power-of-two matrix is partition camping, and the diagonal-block re-index is
a cheap experiment that either moves the number or eliminates the hypothesis.

## Step 5 — the `TILE = 16` prediction

Predict from the bank model **before** rebuilding. With `tile[16][16]`, lanes
`x = 0…15` at fixed `y`:

```
word index = x * 16 + y
bank       = (16x + y) mod 32
x = 0 -> y      x = 1 -> 16 + y      x = 2 -> y      x = 3 -> 16 + y   ...
```

Only **two distinct banks** across 16 lanes → an **8-way conflict**, not 32-way.
Padding to `tile[16][17]` gives `bank = (17x + y) mod 32`, and `17x mod 32` for
`x = 0…15` is `0, 17, 2, 19, 4, 21, …, 14, 31` — all distinct, conflict-free.

**Prediction: the padded-vs-unpadded gap shrinks but does not vanish** (8-way
serialization instead of 32-way). It also does not shrink by exactly 4×, because
at `TILE = 16` you have simultaneously changed occupancy, per-thread work, and
the tile's global-access width — three variables at once. That is the honest
answer, and it is also a warning: this "one experiment" is not a controlled
one-change experiment, so treat whatever you measure as suggestive rather than
conclusive.

If your measurement contradicts the bank arithmetic, fix the model — perhaps the
compiler vectorized the LDS access, or your architecture's LDS geometry differs.
Do not quietly revise what you claim to have predicted.

---

## The transferable habit

The ladder is only pedagogically useful because each rung changes **one**
mechanism and moves **one** counter:

| Rung | One change | One counter |
|---|---|---|
| naive → tiled | strided global write becomes strided LDS access | `MemUnitStalled` falls, `LDSBankConflict` rises |
| tiled → padded | LDS row stride 32 → 33 words | `LDSBankConflict` falls to ~0 |

When you cannot name the single counter your change should move, you are not
optimizing — you are trying things.
