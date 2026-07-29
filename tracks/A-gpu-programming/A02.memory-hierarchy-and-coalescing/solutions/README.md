# A02 Solutions

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
| 1 | `01_coalesce_solution.cpp` | Coalesced grid-stride loop; adjacent lanes → adjacent addresses. ~5–6× BW. |
| 2 | `02_bank_conflict_solution.cpp` | `+1` column pad → column reads hit 32 distinct banks. |
| 3 | `03_hierarchy_solution.md` | Full shared-mem / register / thread budget math + portability note. |
