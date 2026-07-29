# A03 Solutions

Reference answers with reasoning. Try the [`../exercises/`](../exercises/) yourself first - the
learning is in the struggle, not the answer key.

This directory has its own `Makefile`:

```bash
make all GPU_ARCH=gfx950      # build every solution for your GPU
make run                      # build + run all; every one must PASS
make clean
```

| # | File | Notes |
|---|------|-------|
| 1 | `01_fix_divergence_solution.cpp` | `t < n/2 → 2t`, else `2(t-n/2)+1` makes every wavefront class-uniform. Explains the coalescing it trades away and when that trade flips. |
| 2 | `02_warp_reduce_solution.cpp` | Ladder starts at `warpSize/2`; cross-wavefront combine uses the computed `warps_per_block`. Includes the `__shfl_down` vs `__shfl_down_sync` portability note. |
| 3 | `03_occupancy_math_solution.md` | Full four-limiter math, why `-maxrregcount` is the wrong knob here, and the non-monotonic block-size/occupancy table. |

The two programs both print `PASSED` and a measured ratio. Solution 1's speedup and
solution 2's bandwidth are device-dependent — the invariants are that solution 1 is
meaningfully faster than the divergent kernel and that solution 2 reports `ratio 1.000x` on
**both** a 32-lane warp and a 64-lane wavefront.
