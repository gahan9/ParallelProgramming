# Exercise 1 — Diagnose the mystery kernels

**Competency:** turning profiler output into a named bottleneck, with evidence.

`../hip/mystery_kernels.cpp` (and its CUDA mirror) contains four kernels named
`kernel_alpha`, `kernel_beta`, `kernel_gamma`, `kernel_delta`. All four are
*correct* — every one prints `PASSED`. They differ only in performance, and each
one is limited by a **different** cause. One of the four is essentially optimal.

The program prints measurements and nothing else. Naming the four bottlenecks is
your job.

---

## Step 0 — Establish the ceiling

You cannot say a kernel is "good" or "bad" without a reference. Run A02's
bandwidth lab once and write the number down:

```bash
cd ../../../A-gpu-programming/A02.memory-hierarchy-and-coalescing
make run-hip GPU_ARCH=gfx942       # note the copy kernel's GB/s
```

> **My measured copy ceiling:** ________ GB/s

For a purely elementwise kernel that reads 4 bytes and writes 4 bytes per
element, that copy number is the best achievable effective bandwidth. Anything
materially below it needs an explanation.

## Step 1 — Measure, then hypothesize (before profiling)

```bash
cd ../                      # the B03 module directory
make run-hip GPU_ARCH=gfx942
```

Fill in the measured columns, then commit to a hypothesis **in writing** before
you open the profiler. Being wrong here is fine and instructive; skipping it is
not.

| kernel | time (ms) | effective GB/s | % of ceiling | my hypothesis |
|---|---|---|---|---|
| `kernel_alpha` | | | | |
| `kernel_beta` | | | | |
| `kernel_gamma` | | | | |
| `kernel_delta` | | | | |

> **Trap warning.** Effective GB/s is computed the same way for all four
> kernels (8 bytes per element). For at least one of them that number is
> actively misleading about the bottleneck. Decide which, and say why.

## Step 2 — Level 1: the timeline

```bash
cd exercises
make timeline GPU_ARCH=gfx942
make parse
```

Answer from the timeline alone, before touching a counter:

1. What fraction of the traced span is GPU device-busy time?
2. Rank the four kernels by total device time. Does the ranking match your
   hypothesis table?
3. `kernel_delta` — look at its `Grid_Size_X` and `Workgroup_Size_X` columns in
   `*_kernel_trace.csv`. What is the total thread count, and how does it compare
   to your GPU's compute-unit count (`rocminfo | grep -c 'Compute Unit'`, or the
   CU count from the datasheet)?

## Step 3 — Level 2: counters, one kernel at a time

Only now, and only for kernels the timeline says are worth it:

```bash
make counters KERNEL=kernel_alpha
make counters KERNEL=kernel_beta
make counters KERNEL=kernel_gamma
make counters KERNEL=kernel_delta
```

Each run writes `profiling_output/pmc_1/*_counter_collection.csv` — one row per
counter per dispatch. Copy the values into this table:

| kernel | `MeanOccupancyPerCU` | `VALUUtilization` | `MemUnitStalled` | `L2CacheHit` |
|---|---|---|---|---|
| `kernel_alpha` | | | | |
| `kernel_beta` | | | | |
| `kernel_gamma` | | | | |
| `kernel_delta` | | | | |

NVIDIA equivalent — the same four questions, one section each:

```bash
cd ..
make profile-ncu KERNEL=kernel_beta SM_ARCH=sm_90
# reads: SpeedOfLight (memory vs compute %), MemoryWorkloadAnalysis, Occupancy
```

> If a counter is unavailable on your architecture, `rocprofv3 --list-avail`
> prints what your GPU does support. Do not substitute a counter you have not
> looked up — record that it was unavailable and reason from the ones you have.

## Step 4 — The deliverable

For **each** of the four kernels, write the four-row evidence block from
[`README.md`](README.md#what-evidence-looks-like):

```
kernel_?:
  Verdict            :
  Primary counter    :
  Corroborating number:
  Predicted fix + delta:
```

Then answer these three:

1. Which kernel is already at its ceiling, and what is your proof that no
   optimization is available rather than merely "I couldn't find one"?
2. Three of the four kernels produce a byte-for-byte identical output array.
   Name them, then explain each one's gap from the fastest of the three in one
   sentence. (Two of those three have *character-for-character identical kernel
   bodies* — find them, and say what that proves about where their difference
   comes from.)
3. For `kernel_gamma`, compute the ratio of *useful* lane-work to *issued*
   lane-work from the source, and check it against the counter you measured.
   (Hint: read the `reps` expression and count what a 32-lane group executes
   versus what it needs.)

## Step 5 — Prove one fix

Pick the kernel with the largest gap between its measured throughput and its
ceiling. Change **one** thing. Re-run `make run-hip` and `make counters`. Did the
counter you blamed actually move? If the time improved but the counter did not,
your diagnosis was wrong and you got lucky — say so.

---

Reference answers, including the vendor-by-vendor counter mapping and the
arithmetic for question 3, are in
[`../solutions/01_diagnose_mystery_solution.md`](../solutions/01_diagnose_mystery_solution.md).
