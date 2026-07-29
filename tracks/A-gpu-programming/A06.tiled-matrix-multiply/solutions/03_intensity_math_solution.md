# Solution 3 — Arithmetic-intensity drill

## Given

- `M = N = K = 4096`, fp32 (4 bytes/element), row-major.
- **Device G**: 60 TFLOP/s fp32 · 3.0 TB/s HBM · 64 KB shared memory per CU/SM · 1024 threads/block.
- Pessimistic model: no cache hits.

---

## Q1 — Total work

Each of the `N²` output elements is a `K`-long dot product: `K` multiplies + `K` adds.

```
FLOPs = 2 · M · N · K = 2 · N³ = 2 · 4096³ = 1.374 × 10¹¹  ≈ 137.4 GFLOP
```

## Q2 — Ridge point

```
I_ridge = 60e12 FLOP/s ÷ 3.0e12 byte/s = 20 FLOP/byte
```

A kernel needs **more than 20 FLOPs of work per byte pulled from HBM** to be compute-bound on
Device G. Below that, the arithmetic units idle waiting for memory no matter how fast they are.

## Q3 — Naive kernel

Per thread (one output element):

```
FLOPs = 2K = 8192
bytes = 2K · 4 = 32,768   (a full row of A + a full column of B; the 4-byte store is noise)
I     = 8192 / 32,768 = 0.25 FLOP/byte
```

**Independent of `N`** — this is the crucial part. Making the matrix bigger does not improve the
naive kernel's intensity at all; it is a constant `0.25`, a factor of **80× below the ridge**.

Whole-kernel traffic:

```
N² threads × 8K bytes = 8N³ = 8 · 6.87e10 = 549.8 GB
```

The three matrices only occupy `3 · N² · 4 = 201.3 MB`. Traffic amplification:

```
8N³ / (12N²) = 2N/3 = 2731×
```

You move the data set **2731 times**. That single number is the entire motivation for tiling.

## Q4 — Tiled kernel, symbolically

One block, computing a `T × T` tile of `C`:

- iterations: `K / T`
- elements loaded per iteration: `T²` from A + `T²` from B = `2T²`
- elements loaded per block: `(K/T) · 2T² = 2TK`

Number of blocks: `(M/T)·(N/T) = (N/T)²`.

```
total elements = (N/T)² · 2TK = 2N²K / T = 2N³ / T
total bytes    = 8N³ / T
I              = 2N³ / (8N³/T) = T / 4   FLOP/byte
```

**Intensity scales linearly with the tile size.** Tiling did not remove work — it removed *redundant
loads*, and the tile dimension is the reuse factor.

## Q5 — Tiled kernel, numerically (`I_ridge = 20`)

| `T` | Intensity (FLOP/byte) | Total HBM traffic | Bound by |
|---|---|---|---|
| naive | 0.25 | 549.8 GB | memory (80× below ridge) |
| 16 | 4 | 34.4 GB | memory (5× below) |
| 32 | 8 | 17.2 GB | memory (2.5× below) |
| 64 | 16 | 8.6 GB | memory — but only just (1.25× below) |

Traffic falls by `64×` from naive to `T = 64`, and the kernel is *approaching* the ridge without
having crossed it.

## Q6 — Crossing the ridge, and why one-thread-per-output cannot

```
T / 4 ≥ 20   →   T ≥ 80
```

Now try to build it with one thread per output element:

- **Threads:** `80 × 80 = 6400` per block. The hardware limit is **1024**. Impossible.
- **Shared memory:** `2 · 80 · 80 · 4 = 51,200 B = 50 KB` of the 64 KB budget → exactly **one**
  resident block per CU/SM, so there is no second block to switch to when the first one stalls.
  Latency hiding is gone even if the launch were legal.

**Conclusion.** With one thread per output, the tile size is *chained* to the thread count, and the
1024-thread limit caps you at `T = 32` (`I = 8`), permanently below the ridge. To go further you
must break that chain: give each thread **several** outputs. That is register blocking, and it is
not an optional extra — it is the only way a SIMT GEMM reaches the compute-bound regime.

## Q7 — Register blocking

Per block, looping `K/BK` times and staging `BM×BK` of A plus `BK×BN` of B:

```
elements per block = (K/BK) · BK · (BM + BN) = K · (BM + BN)
bytes per block    = 4K · (BM + BN)
FLOPs per block    = 2 · BM · BN · K

I = 2·BM·BN·K / (4·K·(BM+BN)) = BM·BN / (2·(BM + BN))   FLOP/byte
```

`BK` cancels — it controls shared-memory *footprint* and pipelining depth, not global-memory
intensity. That is why real kernels pick `BK` small (8–32) and `BM`/`BN` large.

| `BM × BN` | Intensity | vs ridge 20 |
|---|---|---|
| 64 × 64 | `4096 / 256` = **16** | still memory-bound |
| 128 × 128 | `16384 / 512` = **32** | **compute-bound** |

For `BM = BN = 128`, `TM = TN = 8`, `BK = 8`:

- **Threads:** `(128/8) · (128/8) = 16 · 16 = 256` — comfortably under 1024. ✔
- **Shared memory:** `(BM + BN) · BK · 4 = 256 · 8 · 4 = 8192 B = 8 KB` → **8 blocks** fit in 64 KB. ✔
- The constraint that killed Q6 has moved: the pressure is now on **registers**
  (`acc[8][8]` alone is 64 VGPRs per thread, before operands and addresses).

Same intensity target, legal launch configuration, healthy occupancy. That is the whole payoff of
thread coarsening.

## Q8 — What becomes the bottleneck above the ridge

In the order you hit them:

| # | Resource | What it looks like | Where to look |
|---|---|---|---|
| 1 | **Shared memory / LDS bandwidth and bank conflicts** | The inner loop now reads operands from LDS every FMA; conflicts serialize it | Nsight Compute *Memory Workload → Shared*, bank-conflict counter; `rocprofv3` / Omniperf `LDSBankConflict` |
| 2 | **Register file / occupancy, then spills** | Larger `TM×TN` raises VGPR use until resident waves drop, or the compiler spills to scratch (which is HBM again) | `-Rpass-analysis=kernel-resource-usage` (hipcc), `ncu` *Launch Statistics* + *Occupancy*, scratch/local-memory traffic |
| 3 | **Instruction issue / FMA pipe — the fp32 SIMT peak itself** | You are at the roofline's flat ceiling; only a wider math unit helps | `ncu` *Compute (SM) Throughput* / pipe utilization; Omniperf VALU utilization |

Resource 3 is where **matrix cores** enter: MFMA (CDNA) and Tensor Cores (Hopper/Ada) raise the flat
ceiling itself rather than helping you reach it.

## Q9 — Critique of the no-cache model

The `0.25 FLOP/byte` figure is a *lower bound on efficiency*, not a measurement.

- **A** is read as a warp-level **broadcast**: all 32/64 lanes of a warp share the same `row`, so the
  hardware issues one address, and the row stays hot in L1 for the whole `k` loop.
- **B** is read coalesced across `threadIdx.x`, and the same `B` column block is re-read by the other
  rows of the block and by every block in that block-column, so **L2** captures a large share of it.

Measured naive traffic is therefore often one to two orders of magnitude below `8N³`. But:

1. Even a generous 100× cache benefit leaves intensity around `25`… of the *effective* kind, and the
   measured naive kernel still lands far below the ridge in practice because the caches thrash — the
   working set (`N` × tile) does not fit and eviction order is not under your control.
2. **You cannot size, guarantee, or reason about the cache.** Shared memory is a *software-managed*
   scratchpad: you decide what is resident and for how long. That determinism is why every production
   GEMM stages tiles explicitly instead of hoping for cache hits.

The right way to use the model: it tells you the *shape* of the answer (naive is hopeless, intensity
grows linearly with tile size) and gives you a target. Then you measure.

---

## Bonus — where matrix cores move the goalposts

**1. New ridge point.**

```
I_ridge = (8 × 60e12) / 3.0e12 = 480e12 / 3.0e12 = 160 FLOP/byte
```

Making the math unit 8× faster makes the *bandwidth problem 8× worse*. Every matrix-core generation
pushes the ridge further right, which is why memory optimization matters **more** on newer hardware,
not less.

**2. bf16 operands (2 bytes).**

```
total bytes = 2 · (2N³/T) = 4N³/T
I           = 2N³ / (4N³/T) = T / 2   FLOP/byte
```

Halving the element size doubles the intensity for free — one of the real reasons low precision wins,
independently of the faster math units.

**3. Combining them.**

```
T / 2 ≥ 160   →   T ≥ 320
```

A `320 × 320` tile. Even the register-blocked form (`I = BM·BN/(BM+BN)` for 2-byte operands) needs
`BM = BN = 320` to reach 160, and a `256 × 256` tile only gets to `128`.

**What that implies.** You cannot reach the flat ceiling with the single-level tiling in this module.
Production GEMMs therefore use:

- **Multi-level blocking** — block tile → warp tile → MFMA/WMMA fragment, so each level captures
  reuse in a different memory (HBM → LDS → registers).
- **Software pipelining / double buffering** (`buffer_load` + async copies, `cp.async` on NVIDIA) so
  the next `BK` slice streams in while the current one is being multiplied.
- **L2-aware block scheduling** (tile swizzling, Stream-K) because at these tile sizes the remaining
  traffic is dominated by cross-block re-reads.
- **Autotuning** over the whole `(BM, BN, BK, TM, TN, stages, warps)` space per architecture and per
  problem shape.

That list *is* the difference between this module's kernel and rocBLAS / hipBLASLt / CUTLASS — and it
is why "use the library" is the correct production answer.
