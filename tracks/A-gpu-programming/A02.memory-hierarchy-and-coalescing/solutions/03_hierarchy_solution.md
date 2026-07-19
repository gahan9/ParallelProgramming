# Solution 3 - Memory-hierarchy sizing drill

## Given

- Shared memory / SM: **100 KB** = 102,400 bytes
- Registers / SM: **65,536**
- Max threads / SM: **2048**
- Block: **256 threads**, **48 registers/thread**, tile `float[TILE][TILE+1]`

## Answers

**Q1 — Shared bytes per block at `TILE = 32`:**

```
TILE * (TILE + 1) * sizeof(float) = 32 * 33 * 4 = 4224 bytes
```

**Q2 — Blocks from shared memory:**

```
floor(102,400 / 4224) = floor(24.24) = 24 blocks
```

**Q3 — Blocks from registers:**

```
regs/block = 48 * 256 = 12,288
floor(65,536 / 12,288) = floor(5.33) = 5 blocks
```

**Q4 — Blocks from the thread limit:**

```
floor(2048 / 256) = 8 blocks
```

**Q5 — Resident blocks = min(24, 5, 8) = 5. Limiter = REGISTERS.**
Shared memory (24) is comfortably not the bottleneck at this tile size; the register file caps us.

**Q6 — Largest padded square `TILE` that still allows ≥ 2 resident blocks (shared-mem–limited case).**
We need 2 blocks to fit in shared memory:

```
2 * TILE * (TILE + 1) * 4 <= 102,400
   TILE * (TILE + 1)      <= 12,800
```

`TILE = 112` gives `112*113 = 12,656 <= 12,800` ✓; `TILE = 113` gives `113*114 = 12,882 > 12,800` ✗.
So **`TILE = 112`** is the largest that keeps 2 blocks resident *from the shared-memory constraint
alone*. (In practice registers/threads would bind first — this isolates the shared-memory limit as
the question asks.)

**Q7 — Tradeoff.** A larger tile amortizes global traffic over more on-chip reuse (fewer, larger
coalesced transactions and more arithmetic per byte loaded) — good for arithmetic intensity. But it
consumes more shared memory and often more registers, which **lowers occupancy** (fewer resident
blocks/warps), weakening the GPU's ability to hide memory latency. The sweet spot is empirical:
sweep tile sizes and measure (Stretch in the module README).

## Bonus — AMD CDNA CU with 64 KB LDS

**Q1 (same tile):** unchanged — `4224 bytes/block`.

**Q2 (LDS-limited):**

```
floor(65,536 / 4224) = floor(15.5) = 15 blocks
```

AMD is **more shared-memory (LDS) constrained** (64 KB vs 100 KB), though at `TILE = 32` neither
platform is LDS-bound for this kernel. For portability, size tiles against the **smaller** budget
(64 KB LDS) so the kernel keeps target occupancy on both vendors; expose `TILE` as a compile-time
constant and autotune per architecture rather than hard-coding a single value.
