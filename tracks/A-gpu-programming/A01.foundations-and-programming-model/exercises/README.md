# A01 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                          # list targets
make all GPU_ARCH=gfx950      # build every exercise for your GPU
make 01_saxpy.exe             # build just one
make run                      # build + run all (FAIL is expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200); find it with
`/opt/rocm/bin/offload-arch`. The equivalent raw command, if you prefer to see it spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 01_saxpy.cpp -o 01_saxpy.exe && ./01_saxpy.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercises reuse the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_saxpy.cpp` | Write your first kernel: `Y = a*X + Y`, with bounds guard, error checking, verification. |
| 2 | `02_race_condition.cpp` | Diagnose and fix a data race in a histogram kernel (atomics). |
| 3 | `03_occupancy.md` | Whiteboard occupancy math: resident blocks, limiter, how to improve. |

Each drill maps to an interview competency (clean bug-free code / synchronization & races /
performance reasoning). Solutions include the reasoning, not just the answer.

---

## Sample sandbox output

Reference runs captured on **`gfx950`, ROCm 7** (`hipcc -O3`). Full sanitized transcripts:
[`outputs/.../exercises-run.txt`](../../../../outputs/track-a/A01.foundations-and-programming-model/exercises-run.txt)
and [`run-hip.txt`](../../../../outputs/track-a/A01.foundations-and-programming-model/run-hip.txt).

Reproduce:

```bash
git clone https://github.com/gahan9/ParallelProgramming
cd ParallelProgramming/tracks/A-gpu-programming/A01.foundations-and-programming-model/exercises
make run GPU_ARCH=gfx950
```

Your exact numbers will differ by GPU, driver, and `N` — the **shape and the PASS/FAIL state**
are the lesson, not the digits. NVIDIA output is line-for-line equivalent (`CUDA error …` instead
of `HIP error …`).

### Exercise 1 — SAXPY (`01_saxpy.exe`)

Before you fill in the TODOs, the kernel body is empty, so `Y` is copied back unchanged and the
CPU reference disagrees:

```text
$ hipcc -O3 --offload-arch=gfx942 -I../hip 01_saxpy.cpp -o 01_saxpy.exe
$ ./01_saxpy.exe
SAXPY FAILED (1048576 errors)
```

After a correct kernel + ceil-division launch + the two post-launch `HIP_CHECK` calls:

```text
$ ./01_saxpy.exe
SAXPY PASSED (0 errors)
```

> [!TIP]
> If you forget the launch-error check (TODO 3) and pass a bad config (e.g. `blocks = 0`), the run
> can *silently* print `FAILED` with no clue why. Adding `HIP_CHECK(hipGetLastError())` turns it
> into an explicit, located error:
>
> ```text
> HIP error 01_saxpy.cpp:44: 'hipGetLastError()' -> invalid configuration argument
> ```

### Exercise 2 — Fix the race (`02_race.exe`)

The **buggy** version (plain `bins[b]++`) loses increments, and — the tell-tale sign of a data
race — the totals **change between runs** and never sum to `N = 1048576`:

```text
$ ./02_race.exe
bin  0: gpu=48123 cpu=65536
bin  1: gpu=51002 cpu=65536
bin  2: gpu=49871 cpu=65536
... (more mismatches) ...
Histogram FAILED (data race?)

$ ./02_race.exe          # run again -> different wrong numbers
bin  0: gpu=47760 cpu=65536
...
Histogram FAILED (data race?)
```

After replacing `bins[b]++` with `atomicAdd(&bins[b], 1)`, counts are correct and **deterministic**
across runs:

```text
$ ./02_race.exe
Histogram PASSED

$ ./02_race.exe          # stable every time
Histogram PASSED
```

### Exercise 3 — Occupancy (no program)

Paper drill — no runtime output. Check your worked numbers against
[`../solutions/03_occupancy_solution.md`](../solutions/03_occupancy_solution.md). The expected
answer: register-limited at **6 resident blocks → 75% occupancy**, limiter = registers.

### Bandwidth sanity check (from Lab 2, for comparison)

When you run the module's `vector_add` you should see something shaped like this — a large fraction
of your GPU's peak HBM bandwidth, and `PASSED`:

```text
$ make run-hip GPU_ARCH=gfx950
n = 16777216, block = 256, grid = 65536
kernel time      : 0.647 ms
effective BW     : 311.0 GB/s
result           : PASSED
```

> [!NOTE]
> The `effective BW` line is the number to reason about, not `kernel time`. Compare it to peak
> HBM bandwidth for your chip (order-of-thousands of GB/s on `gfx950`). Landing in the low hundreds
> on a tiny vector add under profiler-free timing is normal once setup and PCIe are in the picture;
> the coalescing lab (Lab 3) is where you should see near-peak streaming bandwidth.
