# A01 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code. Build with the module `Makefile` pattern, e.g.:

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
