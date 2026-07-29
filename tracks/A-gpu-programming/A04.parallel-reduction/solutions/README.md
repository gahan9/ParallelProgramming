# A04 Solutions

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
| 1 | `01_max_reduce_solution.cpp` | `fmaxf` + `-INFINITY` identity; why `-FLT_MAX` is sometimes the better choice, and how NaN behaves. |
| 2 | `02_fix_sync_solution.cpp` | Both barriers restored, with the memory-model argument for each and the rule for where a barrier may *not* go. |
| 3 | `03_reduction_analysis_solution.md` | Full roofline arithmetic: 203 µs floor, 84% of achievable, where the remaining 16% goes, and the atomics code-review call. |
