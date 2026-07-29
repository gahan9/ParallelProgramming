# A04 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                          # list targets
make all GPU_ARCH=gfx950      # build every exercise for your GPU
make 01_max_reduce.exe        # build just one
make run                      # build + run all (FAIL is expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200); find it with
`/opt/rocm/bin/offload-arch`. The equivalent raw command, if you prefer to see it spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 01_max_reduce.cpp -o 01_max_reduce.exe && ./01_max_reduce.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercises reuse the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_max_reduce.cpp` | Re-target a reduction to a new operator; get the **identity element** right for partial tiles. |
| 2 | `02_fix_sync.cpp` | Diagnose a shared-memory race from missing `__syncthreads()`; reason about barriers and divergence. |
| 3 | `03_reduction_analysis.md` | Roofline drill: theoretical minimum reduction time vs measured, and where the gap goes. |

Each drill maps to an interview competency (operator/identity generalization · barrier &
race reasoning · quantitative performance bounds). Solutions include the reasoning, not just the
answer.

---

## Sample sandbox output

The **PASS/FAIL states below are exact** — they are decided by the code, not by your hardware.
Timings and bandwidths are *not* reproduced here because they differ by GPU, driver, and ROCm
version; run the labs to get yours. Placeholders are written as `<...>`.

### Exercise 1 — Max reduction (`01_max_reduce.exe`)

The starter still sums, and still uses `0.0f` as the identity. Note *which* cases fail and which do
not — that pattern is the diagnostic:

```text
$ ./01_max_reduce.exe
A: mixed signs, n = 1048576                    expected     <~100.0>  got   <a huge sum>  FAILED
B: all negative, n = 1000003 (identity!)       expected    -1.000000  got   <a huge sum>  FAILED
C: single element, n = 1                       expected   -42.500000  got    -42.500000   PASSED

2 of 3 cases failed.
```

Case C passes with a *summing* kernel, because summing one element and taking the max of one element
are the same thing. A single-element test can never distinguish two reduction operators — worth
remembering next time a test suite looks green.

After fixing **only TODO 2** (the operator) and leaving `IDENTITY 0.0f`, case A starts passing while
B and C fail. That flip is the lesson:

```text
A: mixed signs, n = 1048576                    expected     <~100.0>  got     <~100.0>     PASSED
B: all negative, n = 1000003 (identity!)       expected    -1.000000  got      0.000000    FAILED
C: single element, n = 1                       expected   -42.500000  got      0.000000    FAILED

2 of 3 cases failed.
```

> [!TIP]
> `0.0f` beats every negative input, so the idle lanes — the ones with no data — win the max and
> report `0`. The identity for `max` is `-INFINITY` (or `-FLT_MAX` if you must avoid infinities,
> e.g. because a downstream kernel does arithmetic on the result). Every reduction operator has
> one: `0` for `+`, `1` for `*`, `+INFINITY` for `min`, `false` for `||`.

With both TODOs done:

```text
A: mixed signs, n = 1048576                    expected     <~100.0>  got     <~100.0>     PASSED
B: all negative, n = 1000003 (identity!)       expected    -1.000000  got     -1.000000    PASSED
C: single element, n = 1                       expected   -42.500000  got    -42.500000    PASSED

0 of 3 cases failed.
```

Note the comparison is exact (`==`), not a tolerance: `max` is associative *and* order-independent
in floating point, so a correct GPU tree must match the CPU bit for bit. A sum cannot make that
promise — which is the subject of exercise 3's bonus.

### Exercise 2 — Find the missing barriers (`02_fix_sync.exe`)

Broken (both TODOs outstanding). The exact shape of the failure varies:

```text
$ ./02_fix_sync.exe
n = 4194304, grid = 16384, 20 trials
reference (fp64) : <the exact sum>
run-to-run       : DIFFERENT between trials (race!)
worst rel error  : <typically 1e-3 .. 1e-1>
result           : FAILED (<k>/20 trials outside tolerance)
```

Fixed:

```text
run-to-run       : identical every trial
worst rel error  : <~1e-7>
result           : PASSED (0/20 trials outside tolerance)
```

> [!WARNING]
> On some GPUs the broken version prints `PASSED` on a lucky run — the wavefronts happened not to
> drift. A data race is *undefined behavior*, not a bug that reliably reproduces. Never conclude
> "the barrier was not needed" from a green run; conclude it from the memory model. Use
> `compute-sanitizer --tool racecheck` (NVIDIA) to get a deterministic verdict.

The file header also describes a third trap — a `__syncthreads()` placed **inside** `if (tid < s)`.
It is documented but deliberately not shipped as runnable code, because whole wavefronts never reach
that barrier and the kernel usually hangs rather than failing cleanly.

### Exercise 3 — Reduction roofline drill (no program)

Paper drill. Check your arithmetic against
[`../solutions/03_reduction_analysis_solution.md`](../solutions/03_reduction_analysis_solution.md).
Expected headline: at `N = 2^28` the reduction must move ~1.07 GB, so **~203 µs is the datasheet
floor** and ~224 µs the honest floor against the measured copy ceiling — meaning a 0.265 ms
measurement is already **~84% of achievable**, and further kernel micro-optimization is the wrong
place to spend a day.

### Module lab, for comparison

`make -C .. run-hip` prints the nine-stage ladder table. Expect stage 1 slowest; a large jump at
stage 3 once divergence and bank conflicts are gone; another at stage 7 once each thread has enough
work to keep the memory system busy; and stages 7–9 clustered together because all three are
bandwidth-bound. The determinism probe at the end should show the atomic stage (9) producing
different low-order bits across repeats while the two-launch stage (8) stays bitwise identical.
