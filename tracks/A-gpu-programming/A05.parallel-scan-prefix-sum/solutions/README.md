# A05 Solutions

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
| 1 | `01_exclusive_to_inclusive_solution.cpp` | `incl[i] = excl[i] + in[i]` (no boundary case) vs `excl[i] = i ? incl[i-1] : identity` (two traps: the identity at 0, and the in-place shift is a race). Grand total = `excl[n-1] + in[n-1]`. |
| 2 | `02_segmented_scan_solution.cpp` | Carry the head flag with the value: `if (have && sf[tid] == 0) { sv[tid] += add_v; sf[tid] \|= add_f; }`. Includes why the pair `(value, flag)` forms an associative operator. |
| 3 | `03_scan_analysis_solution.md` | Full work/depth math, the `(log₂n − 1)/2` extra-work law, barrier and utilization counts, LDS footprints, arithmetic intensity, and the three regimes where Hillis–Steele wins. |
