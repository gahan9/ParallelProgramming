# A06 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                              # list targets
make all GPU_ARCH=gfx950          # build every exercise for your GPU
make 01_tiled_matmul.exe          # build just one
make run                          # build + run all (FAIL is expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200); find it with
`/opt/rocm/bin/offload-arch`. The equivalent raw command, spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 01_tiled_matmul.cpp -o 01_tiled_matmul.exe && ./01_tiled_matmul.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercises reuse the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_tiled_matmul.cpp` | Write the cooperative tile loads, both barriers, and the inner product from a skeleton; measure the GFLOP/s gain over naive. |
| 2 | `02_ragged_dims.cpp` | Make a tiled GEMM correct for dimensions that are not multiples of the tile — and work out *which* guard was actually protecting you. |
| 3 | `03_intensity_math.md` | Paper drill: FLOPs, bytes, intensity for naive vs `T = 16/32/64`, where it crosses the ridge point, and what becomes the bottleneck next. |

Each drill maps to an interview competency (shared-memory algorithm construction / boundary-condition
correctness / roofline reasoning). Solutions include the reasoning, not just the answer key.

---

## Sample sandbox output

> [!IMPORTANT]
> The numeric columns below are **illustrative placeholders**, not measurements — GFLOP/s depends
> entirely on your GPU, ROCm/CUDA version and clocks. What is *not* hardware-dependent, and is the
> actual lesson, is the **PASS/FAIL pattern**, the **ordering** of the variants, and the
> **failure signature** (how many elements are wrong and where).

### Exercise 1 — Write the tiled GEMM (`01_tiled_matmul.exe`)

Before you touch the TODOs, the tiled kernel accumulates nothing, so it produces an all-zero matrix
and a relative error of exactly 1.0:

```text
$ ./01_tiled_matmul.exe
kernel          ms/iter      GFLOP/s  max rel err result
------------ ---------- ------------ ------------ --------
naive           <ms>       <GFLOP/s>      ~1e-06  PASSED
tiled           <ms>       <GFLOP/s>     1.00e+00 FAILED

The tiled kernel is still a skeleton - finish TODO 1..4.
```

After completing TODO 1–4:

```text
$ ./01_tiled_matmul.exe
kernel          ms/iter      GFLOP/s  max rel err result
------------ ---------- ------------ ------------ --------
naive           <ms>       <GFLOP/s>      ~1e-06  PASSED
tiled          <faster>   <higher>        ~1e-06  PASSED

Both PASSED. Note the GFLOP/s ratio: same arithmetic, 32x more
reuse per loaded element.
```

> [!TIP]
> Two diagnostics worth internalising. A `max rel err` of exactly **1.00e+00** means the output is
> all zeros — you are not accumulating at all. A value in the **1e-1 … 1e-3** range means you *are*
> accumulating but from the wrong or stale data — suspect a missing barrier. A value around
> **1e-6** (roughly `√K · eps`) is just fp32 rounding: that is a pass.

### Exercise 2 — Boundary guards (`02_ragged_dims.exe`)

This is the interesting one. With the broken loads, **two of the three cases pass**:

```text
$ ./02_ragged_dims.exe
TILE = 32. A case only exercises the guards when a dimension is not
a multiple of TILE.

A) all dims multiple           512 x 512   k=512    max rel  ~1e-06  PASSED
B) M,N ragged  K aligned       500 x 260   k=512    max rel  ~1e-06  PASSED
C) K ragged                    500 x 260   k=377    max rel  ~1e-01  FAILED (130000/130000 wrong, 7120 of them in the boundary tiles, first at [0,0])

1 case(s) failed. Fix TODO 1 and TODO 2, then answer Q1-Q3.
```

Read that failure line carefully — it is the whole lesson:

- **Every** element is wrong, not just the fringe. A stale K-slice pollutes the dot product of every
  output, because the same high-`k` shared slots are read by every in-range thread.
- Case B passing is *not* luck. Work out the thread-matching argument before reading the solution;
  it explains why an M/N-edge over-read is self-contained while a K-edge one is not.

After the fix, all three (and the two extra shapes in the solution) pass with `max rel` back down in
the fp32-rounding range.

### Exercise 3 — Intensity math (no program)

Paper drill. Check your algebra against
[`../solutions/03_intensity_math_solution.md`](../solutions/03_intensity_math_solution.md).
Expected headlines: naive intensity is **0.25 FLOP/byte and independent of `N`**; tiled is
**`T/4`**; on the drill's Device G you need **`T ≥ 80`** to cross the ridge, which is **impossible**
with one thread per output element (6400 threads > the 1024 limit) — and that impossibility is
precisely why register blocking exists.

### Module lab, for comparison

```text
$ cd .. && make run-hip
== hip/matmul.exe ==
device : <your GPU>
problem: C[1024 x 1024] = A[1024 x 1024] * B[1024 x 1024]   (2.15 GFLOP per call)
computing CPU reference in double (O(M*N*K), this takes a moment)...

variant                             ms/iter      GFLOP/s   speedup  max rel err result
-------------------------------- ---------- ------------ --------- ------------ --------
naive (global reads only)            <ms>       <GFLOP/s>     1.00x      ~1e-06 PASSED
tiled 32x32 (shared memory)          <ms>       <GFLOP/s>     <n>x       ~1e-06 PASSED
regblock 64x64x16, 4x4/thread        <ms>       <GFLOP/s>     <n>x       ~1e-06 PASSED
```

All three must print `PASSED`; the ordering `naive < tiled < regblock` is the reproducible result.
