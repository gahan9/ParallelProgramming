# B01 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Two are written
drills (paper and pencil beats a keyboard for these) and one is code.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                              # list targets
make all GPU_ARCH=gfx950          # build the code drill for your GPU
make 03_measure_vs_spec.exe       # build just one
make run                          # build + run (FAIL is expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350 series, `gfx942` MI300 series,
`gfx90a` MI200 series); find it with `/opt/rocm/bin/offload-arch`. The equivalent raw
command, if you prefer to see it spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 03_measure_vs_spec.cpp -o 03_measure_vs_spec.exe && ./03_measure_vs_spec.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The code drill reuses the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_derive_peaks.md` | Derive peak bandwidth and peak FLOP/s from published clocks, bus widths and unit counts — and catch the DDR-factor trap. |
| 2 | `02_spec_sheet_audit.md` | Take six *real, correctly quoted* vendor claims apart: precision, engine, sparsity, scope, peak-vs-measured. |
| 3 | `03_measure_vs_spec.cpp` | Measure achieved copy bandwidth and report it as a percentage of the peak derived from your own device properties. |

Each drill maps to an interview competency (back-of-envelope hardware reasoning /
critical reading of vendor claims / turning a spec into a measured ceiling).
Solutions include the reasoning, not just the answer.

> **Run [`../hip/device_query.cpp`](../hip/device_query.cpp) first.** Drills 1 and 3
> both want your machine's real CU count, clock and bus width in front of you.

---

## Sample sandbox output

> [!IMPORTANT]
> The transcripts below are **illustrative of the output shape**, sketched for an
> MI300X-class device — they are not a recorded run on your hardware, and the exact
> GB/s will differ by GPU, driver, container and buffer size. What is exact is the
> arithmetic: `2 × 2600 MHz × 8192 bits / 8 = 5324.8 GB/s` is the real MI300X peak
> derivation. Treat the **PASS/FAIL states and the efficiency ratio** as the lesson;
> get your own numbers by running it. NVIDIA output is equivalent (`CUDA error …`
> instead of `HIP error …`).

### Exercise 3, as shipped — two visible failures

```text
$ make 03_measure_vs_spec.exe GPU_ARCH=gfx942 && ./03_measure_vs_spec.exe
device        : AMD Instinct MI300X (gfx942:sramecc+:xnack-)
CUs           : 304, wavefront 64
buffers       : 2 x 256.0 MiB
time / iter   : 0.031 ms
achieved BW   : 0.0 GB/s
theoretical   : unavailable (driver reported clock=... kHz, bus=... bits)
result        : FAILED (66846720 mismatches)
```

Read the two failures separately, because they are two different lessons:

- **`FAILED (66846720 mismatches)`** — the starter kernel copies only
  `gridDim.x * blockDim.x = 262144` of the 67,108,864 elements. The other 66.8M are
  still the zeros from `hipMemset`. This is TODO 2: without a grid-stride loop, a
  launch does not cover the array, and the *timing* is meaningless too because it
  timed 0.4% of the work.
- **`achieved BW: 0.0` / `theoretical: unavailable`** — TODO 1 and TODO 3 return
  hardcoded zeros. Notice the program does *not* invent a plausible number when it
  cannot compute one. That is deliberate: a fabricated ceiling is worse than a
  missing one.

### Exercise 3, solved

```text
$ ./03_measure_vs_spec.exe
device        : AMD Instinct MI300X (gfx942:sramecc+:xnack-)
CUs           : 304, wavefront 64
buffers       : 2 x 256.0 MiB
time / iter   : 0.128 ms
achieved BW   : 4192.6 GB/s
theoretical   : 5324.8 GB/s  (2 x 2600 MHz x 8192 bits / 8)
efficiency    : 78.7% of peak
result        : PASSED (0 mismatches)
```

> [!TIP]
> **~79% of peak is a good result, not a bad one.** A coalesced streaming copy in the
> 70–90%-of-peak band is about as well as DRAM behaves once refresh, ECC and
> read/write turnaround are paid. Carry the **achieved** number into B02 — build your
> roofline on 4192 GB/s, not on 5324 GB/s, or every kernel you profile will look
> like a failure.

### Module labs, for comparison

The two labs are introspective and have no PASS/FAIL — their output *is* the
deliverable, and you feed it into drills 1 and 3:

```text
$ make -C .. run-hip
== ../hip/device_query.exe ==
Visible devices: 8
Device 0: AMD Instinct MI300X
  LLVM target / arch name                gfx942:sramecc+:xnack-
  Compute units (CUs)                    304
  Wavefront / warp size                  64
  ...
  -- Derived ceilings (the input to B02's roofline) --
  Theoretical peak HBM bandwidth         5324.8 GB/s
  FP32 vector-FMA estimate (lower bnd)   81.7 TFLOP/s
  Roofline ridge point (vector fp32)     15.3 FLOP/byte

== ../hip/topology.exe ==
Peer access matrix (row = source, col = destination)
  Y = direct peer addressing possible, . = must route through the host
        0   1   2   3   4   5   6   7
   0    -   Y   Y   Y   Y   Y   Y   Y
   ...
Measured unidirectional copy bandwidth from device 0 (256 MiB x 20 iters)
0 -> 1     Y            <your GB/s>    direct fabric link
```

On an MI300X node each Infinity Fabric link is specified at **128 GB/s
bidirectional** ([MI300X data
sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf),
as of 29 Jul 2026), so a *unidirectional* copy is competing for at most half of
that. Compare what you measure against **64 GB/s**, not against 128 — and against
896 GB/s only when all eight GPUs are driving all their links at once.

A fully connected `Y` matrix is the 8-GPU OAM/HGX mesh. If you see `.` entries, you
are on a partially connected board or inside a container that hides the fabric — and
that changes which parallelism strategy B07 will recommend.
