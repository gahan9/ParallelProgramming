# Exercise 3 — Work vs depth: Hillis–Steele against Blelloch

No code. Reason on paper, then check
[`../solutions/03_scan_analysis_solution.md`](../solutions/03_scan_analysis_solution.md).

The point of the drill: "work-efficient" is an asymptotic statement, and asymptotics do not
pay the bill on a real GPU. You should be able to write down both cost models, compute the
extra-work factor, and then argue — with hardware reasons, not vibes — where the
work-*inefficient* algorithm still wins.

## Setup

Scan `n` elements, `n` a power of two, one tile that fits in shared memory / LDS.

- **Hillis–Steele:** `n` threads, one element each. Step `k` (offset `s = 2^k`, for
  `k = 0 … log₂n − 1`): every lane with `tid ≥ s` does `x[tid] += x[tid − s]`.
- **Blelloch:** `n/2` threads, two elements each. Up-sweep has `log₂n` levels with
  `n/2, n/4, …, 1` active lanes; down-sweep has `log₂n` levels with `1, 2, …, n/2` active
  lanes, each active lane doing one swap and one add.
- **Sequential CPU baseline:** the obvious loop.

## Questions

1. Count the **adds** Hillis–Steele performs, exactly, for `n = 1024`. Then give the closed
   form for general `n`.
2. Count the adds Blelloch performs (up-sweep + down-sweep) for `n = 1024`, and the closed
   form.
3. How many adds does the sequential loop do? Which of the two GPU algorithms is within a
   constant factor of it, and what is that constant?
4. Compute the **extra work factor** (Hillis–Steele adds ÷ Blelloch adds) at `n = 1024` and
   at `n = 2²⁰`. Derive the approximate closed form for the ratio. What does it tell you
   about how the penalty scales?
5. Compute the **depth** (parallel steps) and the **number of `__syncthreads()` barriers**
   for each — for Hillis–Steele, do it for both the ping-pong version (one barrier per step)
   and the in-place register-staged version (two per step).
6. **Thread utilization.** In Blelloch at `n = 1024` (512 threads), how many of the 20 tree
   levels have fewer than 25% of the block's threads active? What does that mean for a
   64-lane wavefront sitting at the barrier?
7. **Shared-memory footprint** per tile, in bytes, for: Hillis–Steele ping-pong,
   Hillis–Steele in-place, Blelloch with the `CONFLICT_FREE_OFFSET` padding (`n = 1024`,
   `float`). Which one threatens occupancy first?
8. Now the real question: name **three distinct reasons** the O(n log n) algorithm can beat
   the O(n) one on real hardware, and state the regime where each applies.
9. A tile scan reads its tile once from HBM and writes it once. At `n = 1024` floats per
   tile, compute the arithmetic intensity of *each* algorithm in adds per byte moved. Are
   either of them anywhere near compute-bound? What does that imply about which cost model
   (work or bytes) you should optimize first?

## Bonus

10. On a 64-lane AMD wavefront (or 32-lane NVIDIA warp), Hillis–Steele can be written with
    `__shfl_up` and **no shared memory and no barriers at all**. How many steps? Why does
    this specific fact push production libraries (rocPRIM, CUB) toward a *hybrid*
    (warp-level Hillis–Steele composed hierarchically) rather than a pure Blelloch tree?
11. Your input length is `n = 1000`, not a power of two. Describe what breaks in each
    algorithm and the standard fix. Which one is easier to get right?
