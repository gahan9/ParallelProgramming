# A08 Solutions

Reference answers with the reasoning. Try [`../exercises/`](../exercises/) first — the
learning is in the struggle, not the answer key.

Nothing to compile; run them directly or from the module root:

```bash
python 01_fused_relu_add_solution.py
python 02_masked_mean_solution.py

cd .. && make run-solutions          # both, must all PASS
```

| # | File | Notes |
|---|------|-------|
| 1 | `01_fused_relu_add_solution.py` | `tl.maximum(x + y, 0.0)` between the loads and the store. Adds a tail case (`n` not a multiple of `BLOCK_SIZE`, sums exactly 0.0). Explains why the measured speedup is below the 40% traffic model. |
| 2 | `02_masked_mean_solution.py` | Bounds mask vs. data mask; `other=` as the reduction's identity; the empty row guarded by clamping the **denominator**, not just selecting the result. Adds a ragged-width case. |
| 3 | `03_triton_vs_cuda_solution.md` | Real line counts for the vector-add pair, a control-surrendered table with the one row that justifies keeping a C++ kernel, how to *see* each compiler decision, and a full review comment for a Triton rewrite that is 6% slower. |

Both scripts must print `PASSED` on every check. If one does not on your machine, that
is a curriculum bug worth reporting — not an exercise.
