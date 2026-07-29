# A05 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                                   # list targets
make all GPU_ARCH=gfx950               # build every exercise for your GPU
make 01_exclusive_to_inclusive.exe     # build just one
make run                               # build + run all (FAIL is expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200); find it with
`/opt/rocm/bin/offload-arch`. The equivalent raw command, if you prefer to see it spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 01_exclusive_to_inclusive.cpp -o 01_exclusive_to_inclusive.exe && ./01_exclusive_to_inclusive.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercises reuse the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_exclusive_to_inclusive.cpp` | Convert exclusive ↔ inclusive scan; handle the boundary element and recover the grand total an exclusive scan discards. |
| 2 | `02_segmented_scan.cpp` | Implement a segmented scan (resets at flagged boundaries) — the primitive behind ragged/jagged ML batches. |
| 3 | `03_scan_analysis.md` | Whiteboard: work vs depth for Hillis–Steele vs Blelloch, the extra-work factor, and when work-inefficient still wins. |

Each drill maps to an interview competency (off-by-one rigor on scan variants / composing an
associative operator that carries state / cost-model reasoning under real hardware limits).
Solutions include the reasoning, not just the answer key.

---

## Sample sandbox output

The PASS/FAIL states, mismatch counts, and totals below are **deterministic** — they follow
from the fixed inputs in each drill, so you can compare against them directly. Timings and
bandwidths are hardware-dependent and are left as placeholders; run the labs to get yours.
NVIDIA output is equivalent (`CUDA error …` instead of `HIP error …`).

### Exercise 1 — Exclusive ↔ inclusive (`01_exclusive_to_inclusive.exe`)

The starter's two conversion kernels write zeros, so only the *given* Blelloch scan passes:

```text
$ ./01_exclusive_to_inclusive.exe
n = 256
given  blelloch exclusive scan : PASSED (0 mismatches)
TODO 1 exclusive -> inclusive  : FAILED (256 mismatches)
TODO 2 inclusive -> exclusive  : FAILED (255 mismatches)
TODO 3 grand total             : FAILED (got 0.0, expected 766.0)
```

> [!TIP]
> Read those two counts: TODO 1 misses **all 256** elements, TODO 2 misses **255** — index 0
> is accidentally right because the exclusive scan's first element *is* the identity. That
> one-element difference is the whole lesson of the drill.

Once all three TODOs are done:

```text
$ ./01_exclusive_to_inclusive.exe
n = 256
given  blelloch exclusive scan : PASSED (0 mismatches)
TODO 1 exclusive -> inclusive  : PASSED (0 mismatches)
TODO 2 inclusive -> exclusive  : PASSED (0 mismatches)
TODO 3 grand total             : PASSED (got 766.0, expected 766.0)
```

### Exercise 2 — Segmented scan (`02_segmented_scan.exe`)

The starter runs an unsegmented scan, so it is correct up to the first segment head and then
carries the previous segment's total forward forever:

```text
$ ./02_segmented_scan.exe
segmented inclusive scan: n = 256, 7 segments (head every 37 elements)
result : FAILED (219 mismatches)
first mismatch at i = 37 (segment head at 37): got 111.0, expected 3.0
An unsegmented scan is correct until the first head, then carries the
previous segment's total forward. That is the bug the flags must stop.
```

`219 = 256 − 37`: every element from the first head onward is wrong. After fixing the
combine to consult and propagate the flags:

```text
$ ./02_segmented_scan.exe
segmented inclusive scan: n = 256, 7 segments (head every 37 elements)
result : PASSED (0 mismatches)
```

The solution also prints each head's `value -> scan`, where the two must be equal — the
visible proof that the scan restarted.

### Exercise 3 — Work vs depth (no program)

Paper drill — no runtime output. Check your numbers against
[`../solutions/03_scan_analysis_solution.md`](../solutions/03_scan_analysis_solution.md).
Expected headline: at `n = 1024` Hillis–Steele does **9,217 adds** against Blelloch's
**2,046** — a **4.5×** extra-work factor that grows as `(log₂n − 1)/2` — yet Blelloch pays
**20** barriers to Hillis–Steele's **10** and runs 14 of its 20 tree levels below 25% thread
utilization. Both sit far below the roofline ridge point, which is why the "inefficient" one
is still the right choice inside a small tile.

### Module labs, for comparison

Running the built module labs should look like this (`<...>` = hardware-dependent):

```text
$ make run-hip
== hip/scan.exe ==
parallel scan: n = 4194304 elements (16.0 MiB), tile = 256 elements
  hillis-steele: 256 threads/tile (1 elem/thread), 8 steps
  blelloch     : 128 threads/tile (2 elems/thread), 16 steps (up+down)

Part 1 - single-tile correctness (first 256 elements)
kernel                                   scan type   result
---------------------------------------- ----------- --------
hillis-steele (ping-pong buffers)        inclusive   PASSED
hillis-steele (in-place + register)      inclusive   PASSED
blelloch (up-sweep / down-sweep)         exclusive   PASSED

Part 2 - per-tile scans of the whole array (16384 tiles, same for both)
kernel                                   time (ms)   GB/s      result
---------------------------------------- ----------- --------- --------
hillis-steele  O(n log n) work           <ms>        <GB/s>    PASSED
blelloch       O(n) work                 <ms>        <GB/s>    PASSED

Part 3 - full exclusive scan of all 4194304 elements (3-phase decomposition)
recursion levels : 3 (tile sums, then sums of those, ...)
time             : <ms>
effective BW     : <GB/s>  (useful bytes only: read n + write n)
result           : PASSED

== hip/compaction.exe ==
stream compaction (keep non-zeros): n = 4194304
kept (GPU)       : 2796202
kept (std::copy_if): 2796202
time             : <ms>  (mark + 3-phase scan + scatter)
effective BW     : <GB/s>  (useful bytes: read n + write kept)
result           : PASSED (0 value mismatches)
```

Both labs must print `PASSED`. The interesting question is not correctness but how far the
reported GB/s sits below the copy ceiling you measured in
[A02 Lab 1](../../A02.memory-hierarchy-and-coalescing/README.md#4-hands-on-labs) — see
[§5 of this module's README](../README.md#5-performance-analysis).
