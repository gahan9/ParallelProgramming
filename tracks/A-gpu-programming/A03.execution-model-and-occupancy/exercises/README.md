# A03 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                              # list targets
make all GPU_ARCH=gfx950          # build every exercise for your GPU
make 01_fix_divergence.exe        # build just one
make run                          # build + run all (FAIL / no-speedup expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200); find it with
`/opt/rocm/bin/offload-arch`. The equivalent raw command, if you prefer to see it spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 01_fix_divergence.cpp -o 01_fix_divergence.exe && ./01_fix_divergence.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, swap __shfl_down for __shfl_down_sync,
#         and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercises reuse the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_fix_divergence.cpp` | Remap thread → element so a data-dependent branch becomes wavefront-uniform; measure the win and name what you traded away. |
| 2 | `02_warp_reduce.cpp` | Write a `__shfl_down` reduction that is correct on a 32-lane warp *and* a 64-lane wavefront. |
| 3 | `03_occupancy_math.md` | Whiteboard: occupancy as a `min()` over four limiters, with shared memory binding. |

Each drill maps to an interview competency (divergence reasoning and the cost of fixing it /
portable warp-level primitives / occupancy arithmetic and knowing which knob is live).
Solutions include the reasoning, not just the answer.

---

## What the output looks like

The programs print their own numbers — **absolute ms and GB/s depend entirely on your GPU,
so none are quoted here.** What is fixed, and what you are grading yourself on, is the
**shape**: which line PASSES, and which ratio moves.

### Exercise 1 — Fix the divergence (`01_fix_divergence.exe`)

Both kernels PASS from the start; the bug is *performance*. Before you touch TODO 1 the two
timings are the same program, so:

```text
divergent  (thread t -> element t) : <ms> ms  PASSED
remapped   (your TODO)             : <ms> ms  PASSED     <- same value as above
speedup                            :    1.00x

No real speedup yet - TODO 1 is still the identity mapping.
```

After the remap, the `speedup` line should move clearly above 1: the divergent wavefront
executes the sqrt chain *and* the Newton chain with half its lanes masked, while a
wavefront-uniform one executes exactly one of them. The theoretical ceiling is
`(cost_sqrt + cost_newton) / max(cost_sqrt, cost_newton)`, so somewhere between 1.5× and 2×
is the healthy band on current hardware — not a fixed number.

If your speedup is suspiciously close to 1.00× *after* the fix, check that `n/2` is still a
multiple of `warpSize` (otherwise one wavefront straddles the class boundary — harmless) and
that you did not accidentally leave both branches reachable per wavefront.

### Exercise 2 — Portable wavefront reduction (`02_warp_reduce.exe`)

This one FAILS out of the box, and the **ratio tells you which bug you are looking at**:

```text
warpSize     : 64 lanes
sum          : <got>  (expected <exp>, ratio 0.125x)
kernel time  : <ms>   effective BW <GB/s>
result       : FAILED
```

| ratio | meaning |
|---|---|
| `0.500x` on a 64-lane wavefront | TODO 1 only: the ladder started at 16, so lane 0 holds half the wavefront |
| `0.250x` with a 256-thread block on CDNA | TODO 2 only: 1 of 4 wavefront partials published |
| `0.125x` | both bugs (`0.5 × 0.25`) |
| `1.000x` + PASSED | done |

On a 32-lane NVIDIA warp the first bug is invisible — `warpSize/2 == 16` — which is precisely
why the hardcoded version survives in so many CUDA tutorials and breaks on AMD.

### Exercise 3 — Occupancy math (no program)

Paper drill. Check your inequalities against
[`../solutions/03_occupancy_math_solution.md`](../solutions/03_occupancy_math_solution.md).
Expected headline: shared memory binds at **3 resident blocks → 18.75% occupancy**, capping
registers does *nothing*, and doubling the block size doubles occupancy for free.

### Module labs, for comparison

`make -C .. run-hip` runs the two worked labs. Read `divergence` for
`lane-divergent ≈ 2× branchless` with `warp-uniform ≈ 1×`, and `occupancy` for a theoretical
occupancy column that barely moves across block sizes next to one that halves on every
doubling of the LDS request.
