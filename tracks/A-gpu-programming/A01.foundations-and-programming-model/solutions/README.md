# A01 Solutions

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
| 1 | `01_saxpy_solution.cpp` | Kernel + ceil-division launch + error checks. |
| 2 | `02_race_condition_solution.cpp` | Race fixed with `atomicAdd`; explanation of why. |
| 3 | `03_occupancy_solution.md` | Full worked occupancy math + limiter analysis. |
