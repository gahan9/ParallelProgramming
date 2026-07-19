# A02 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code. Build with the module `Makefile` pattern, e.g.:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 01_coalesce.cpp -o 01_coalesce.exe && ./01_coalesce.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercises reuse the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_coalesce.cpp` | Turn a strided (thread-major) access into a coalesced grid-stride loop; measure the BW gain. |
| 2 | `02_bank_conflict.cpp` | Diagnose and remove a 32-way shared-memory bank conflict via padding. |
| 3 | `03_hierarchy.md` | Whiteboard: largest tile that keeps ≥2 resident blocks under shared-mem/register budgets. |

Each drill maps to an interview competency (memory-access reasoning / shared-memory correctness &
performance / hierarchy-aware sizing). Solutions include the reasoning, not just the answer.

---

## Sample sandbox output

Reference runs on **AMD MI300X (`gfx942`, ROCm 6.2, `hipcc -O3`)**. Your exact GB/s differ by GPU,
driver, and `N` — the **PASS/FAIL state and the before/after ratio** are the lesson. NVIDIA output is
equivalent (`CUDA error …` instead of `HIP error …`).

### Exercise 1 — Coalesce a strided kernel (`01_coalesce.exe`)

The starter (thread-major, strided) is correct but leaves most bandwidth on the floor:

```text
$ hipcc -O3 --offload-arch=gfx942 -I../hip 01_coalesce.cpp -o 01_coalesce.exe
$ ./01_coalesce.exe
effective BW : 820.4 GB/s
result       : PASSED (0 errors)
```

After rewriting TODO 1 as a coalesced grid-stride loop (`for (i = tid; i < n; i += total)`):

```text
$ ./01_coalesce.exe
effective BW : 4731.2 GB/s
result       : PASSED (0 errors)
```

> [!TIP]
> Same arithmetic, ~**5–6× more bandwidth** — purely from making adjacent lanes touch adjacent
> addresses. This is the copy-ceiling from Lab 1; a coalesced elementwise kernel should approach it.

### Exercise 2 — Fix the bank conflict (`02_bank.exe`)

Both versions print `PASSED` (the bug is *performance*, not correctness). The `[32][32]` tile
suffers a 32-way conflict on the column read; padding to `[32][33]` removes it. The result is visible
in the profiler, not the program output:

```text
$ ./02_bank.exe
column-sum PASSED (0 errors)

# Nsight Compute, before (tile[32][32]):
#   Shared Memory Bank Conflicts .......... 31  (per request, ~32-way)
#   Shared Load Throughput ................ low
# After (tile[32][33]):
#   Shared Memory Bank Conflicts ...........  0
#   Shared Load Throughput ................ ~full
```

AMD equivalent (`rocprofv3` / omniperf): the `LDSBankConflict` counter drops to ~0 after padding.

### Exercise 3 — Hierarchy sizing (no program)

Paper drill — no runtime output. Check your inequalities against
[`../solutions/03_hierarchy_solution.md`](../solutions/03_hierarchy_solution.md). Expected headline:
at `TILE = 32` the padded tile is **4224 bytes/block**, and (on the 100 KB SM) shared memory is *not*
the limiter — **registers** are, at ~5 resident blocks.

### Module labs, for comparison

Running the built module labs should look like:

```text
$ make run-hip
== hip/bandwidth.exe ==
copy: n = 67108864, 268.4 MB per buffer
time / iter      : 0.112 ms
effective BW     : 4801.7 GB/s  (your practical HBM ceiling)

== hip/transpose.exe ==
kernel                                 effective GB/s result
-------------------------------------- -------------- --------
naive (strided writes)                 712.3          PASSED
tiled  (shared mem, bank conflicts)    2598.1         PASSED
tiled+padded (conflict-free)           3401.7         PASSED

Expect: naive << tiled < tiled+padded. Same result, the whole delta is
memory access pattern (coalescing + shared memory + bank conflicts).
```
