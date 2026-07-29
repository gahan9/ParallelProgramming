# B03 Solutions

Reference answers for [`../exercises/`](../exercises/). Read these **after** you
have written your own diagnosis — the value of this module is in forming a
hypothesis and being wrong in a way you can measure, and you only get that once
per exercise.

```bash
make                            # list targets
make reproduce GPU_ARCH=gfx950  # timeline + counter runs for all four kernels
make parse                      # phase/kernel breakdown
make clean
```

| # | File | Answers |
|---|------|---------|
| 1 | `01_diagnose_mystery_solution.md` | The four bottlenecks, the counter that proves each, and the fix. |
| 2 | `02_amdahl_solution.md` | Full arithmetic for the synthetic step and for A01's real numbers. |
| 3 | `03_profile_your_transpose_solution.md` | Bank-index arithmetic and the causal chain up A02's ladder. |

---

## A note on the numbers in these keys

**These keys contain no invented profiler output.** Where a specific measured
value appears, it is either:

- **Real and attributed** — the `rocprofv3` figures in
  `02_amdahl_solution.md` come from the run documented in
  [A01 § 4](../../../A-gpu-programming/A01.foundations-and-programming-model/README.md#reading-the-rocprofv3-output),
  and the transpose bandwidths in `03_profile_your_transpose_solution.md` come
  from [A02 § 5](../../../A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#5-performance-analysis)
  (MI300X, `gfx942`, ROCm 6.2); or
- **Derived from the source code** — e.g. `kernel_gamma`'s masked-lane fraction,
  which is arithmetic on the trip counts you can read in the file, not a
  measurement; or
- **A labelled prediction** — stated as the *direction and rough magnitude* a
  counter should move, which is exactly what you check your own run against.

Absolute values on your GPU will differ. The **ratios, directions, and the chain
of reasoning** are the transferable part, and they are what these keys grade.

## The habit these keys are trying to build

Every answer below has the same skeleton, and so should yours:

```
observation  ->  candidate causes  ->  the counter that discriminates between them
             ->  verdict  ->  predicted fix  ->  re-measure to confirm the counter moved
```

The last step is the one people skip. If the time improved but the counter you
blamed did not move, you fixed something else by accident and you still do not
understand the kernel.
