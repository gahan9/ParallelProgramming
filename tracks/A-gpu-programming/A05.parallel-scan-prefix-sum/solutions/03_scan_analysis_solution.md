# Solution 3 — Work vs depth: Hillis–Steele against Blelloch

All counts are for `n` a power of two, one tile, `float` data.

---

## 1. Hillis–Steele add count

Step `k` uses offset `s = 2^k`; the lanes with `tid ≥ s` each do one add, so that step does
`n − s` adds. Summing over `k = 0 … log₂n − 1`:

```
adds_HS(n) = Σ (n − 2^k) = n·log₂n − (2^log₂n − 1) = n·log₂n − (n − 1)
```

At `n = 1024`: `1024·10 − 1023 = 10240 − 1023 = **9217 adds**`.

> Worth noting separately, because it does not show up in the O() notation: the ping-pong
> variant also *copies* every element it does not add to (the `else` branch), so it performs
> `n·log₂n = 10240` shared-memory **writes** either way. Shared-memory traffic, not the
> adds, is often what you actually pay.

## 2. Blelloch add count

- Up-sweep: `n/2 + n/4 + … + 1 = n − 1` adds.
- Down-sweep: `1 + 2 + … + n/2 = n − 1` adds, plus the same number of swaps.

```
adds_Blelloch(n) = 2(n − 1)
```

At `n = 1024`: `2·1023 = **2046 adds**` (plus 1023 swaps, i.e. ~2046 extra shared
read/writes — the down-sweep is not free even though its add count is small).

## 3. Sequential baseline

`n − 1 = **1023 adds**`. Blelloch is within a factor of **2** of the sequential work — that
is what "work-efficient" means: the same asymptotic work as the best serial algorithm, up to
a constant. Hillis–Steele is `9217/1023 ≈ 9.0×` the serial work at this `n`, and that factor
grows with `n`.

## 4. Extra work factor

```
ratio(n) = [n·log₂n − (n − 1)] / [2(n − 1)]  ≈  (log₂n − 1) / 2
```

| `n` | Hillis–Steele adds | Blelloch adds | ratio | `(log₂n − 1)/2` |
|---|---:|---:|---:|---:|
| 1024 (2¹⁰) | 9,217 | 2,046 | **4.50×** | 4.5 |
| 2²⁰ | 19,922,945 | 2,097,150 | **9.50×** | 9.5 |

The penalty grows **logarithmically** — unbounded in theory, but slowly. Going from a
1 K tile to a 1 M array only doubles the extra-work factor. That slow growth is precisely
why the "wrong" algorithm stays competitive in the regimes below, and why nobody runs
Hillis–Steele across a *whole* large array: you use it inside a tile, where `log₂n` is 8–10.

## 5. Depth and barriers

| | steps (depth) | `__syncthreads()` |
|---|---:|---:|
| Hillis–Steele, ping-pong | `log₂n` = **10** | 1/step → **10** |
| Hillis–Steele, in-place + register staging | `log₂n` = **10** | 2/step → **20** |
| Blelloch (up + down) | `2·log₂n` = **20** | 1/step → **20** |

Both algorithms have `O(log n)` depth — that is the shared victory over the serial chain's
`O(n)`. But Blelloch's constant is **2×** worse, and the in-place Hillis–Steele trades its
barrier advantage away for half the LDS. A barrier is not an add: it stalls every wavefront
in the block until the slowest arrives, so barrier count is frequently the better proxy for
tile-scan cost than add count.

## 6. Thread utilization in Blelloch

At `n = 1024` the block has `n/2 = 512` threads. Under 25% active means fewer than **128**
active lanes:

- Up-sweep active counts: 512, 256, **128**, 64, 32, 16, 8, 4, 2, 1 → **7 levels** below 128.
- Down-sweep is the mirror image → **7 more**.

So **14 of the 20 levels run at under a quarter utilization**, and the very top of the tree
runs *one* lane. On a 64-lane wavefront, any level with fewer than 64 active lanes means a
single partially-active wavefront while the other 7 wavefronts of the block sit at the
barrier contributing nothing. Blelloch's O(n) work is real, but a large slice of its *depth*
is spent nearly idle. Hillis–Steele's active count is `n − s ≥ n/2`, i.e. never below **50%**.

## 7. Shared-memory footprint per tile (`n = 1024`, float)

| Variant | Words | Bytes |
|---|---:|---:|
| Hillis–Steele, ping-pong | 2048 | **8192** |
| Hillis–Steele, in-place | 1024 | **4096** |
| Blelloch + `CONFLICT_FREE_OFFSET` | 1024 + 32 = 1056 | **4224** |

Ping-pong is the occupancy risk: at 8 KB/block, a 64 KB LDS CU (CDNA) caps you at 8 resident
blocks from LDS alone, before registers are considered. The padded Blelloch tile costs 128
bytes of dead storage to remove a 32-way bank conflict — the same trade as A02's `[32][33]`
tile, and just as worthwhile.

## 8. Three reasons the O(n log n) version still wins

1. **Small `n`, especially one warp/wavefront.** Hillis–Steele maps onto `__shfl_up` with
   **no shared memory and no barriers** (Q10). Blelloch cannot exploit lockstep execution
   the same way and needs twice the steps. Below ~64 elements the extra adds are noise and
   the barriers are the whole cost.
2. **Barrier count, not arithmetic, dominates.** 10 barriers vs 20 for a 1 K tile. Each
   barrier idles whole wavefronts; each add is one instruction issued across 64 lanes. If
   you are latency-limited inside the tile, Hillis–Steele's shallower schedule wins even
   though it computes 4.5× more sums.
3. **You are memory-bound anyway (Q9).** Both algorithms move the same bytes to and from
   HBM. If HBM is the ceiling, the extra ALU work executes in the shadow of the memory
   traffic and costs nothing measurable. This is the common case for a scan.

(Two more, for completeness: Hillis–Steele's shared accesses are stride-1 and
**conflict-free without padding**; and it keeps ≥50% of its lanes busy at every step where
Blelloch drops to one.)

## 9. Arithmetic intensity of a tile scan

Per 1024-float tile: HBM traffic = 4096 B read + 4096 B written = **8192 B**.

| | adds | adds per byte |
|---|---:|---:|
| Hillis–Steele | 9,217 | **1.13** |
| Blelloch | 2,046 | **0.25** |

A data-center GPU's ridge point is on the order of **20–30 FLOP/byte** (check your part:
divide peak fp32 by peak HBM bandwidth). Both algorithms sit **one to two orders of
magnitude below it** — scan is firmly **memory-bound**, exactly like A01's vector add and
A02's transpose. Optimize *bytes* first: fuse the scan with its producer/consumer, avoid the
extra pass that phase 3 of the 3-phase decomposition costs, and only then count adds.

Caveat worth stating out loud: this intensity ignores **LDS traffic and barrier latency**,
which for a tile scan are usually the real limiter inside the block. The HBM roofline tells
you the *floor* on time; the LDS/barrier schedule tells you how close you get to it.

## Bonus

**10. Warp/wavefront-level Hillis–Steele.** `log₂(64) = **6** steps` on a 64-lane AMD
wavefront (`log₂(32) = 5` on a 32-lane NVIDIA warp), using `__shfl_up`: the operand travels
through the register file, so there is **no shared memory and no `__syncthreads()`** — lanes
in a wavefront are already synchronized by construction. That is the fact that shapes real
libraries: rocPRIM and CUB build a **hierarchy** — shuffle-based scan within a wavefront,
combine the per-wavefront totals through a tiny bit of shared memory to get a block scan,
then combine block totals across the device (decoupled look-back). A pure Blelloch tree in
shared memory is the textbook answer; a hybrid is the shipped one.

**11. `n = 1000`, not a power of two.**

- *Hillis–Steele* barely cares. The loop bound is the block size, past-the-end lanes load the
  **identity** (`0.0f`) and simply do not store their result. The lab kernels already do
  this: `buf[tid] = (idx < n) ? in[idx] : 0.0f;`.
- *Blelloch* needs a power-of-two tree, so you **round the tile up** to the next power of two
  (1024) and pad with the identity, or you special-case the last, partial tile. The lab does
  the former implicitly: `TILE_ELEMS` is a compile-time power of two and the loads guard with
  `base + ai < n`, so the tail of the last tile is identity-filled and its tile sum still
  equals the sum of the real elements.

Hillis–Steele is easier to get right — one more entry on the "work-inefficient but cheap to
be correct" side of the ledger. The general lesson: the identity element is not a
mathematical footnote, it is what makes ragged sizes work at all.
