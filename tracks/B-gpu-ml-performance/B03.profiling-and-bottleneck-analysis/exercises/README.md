# B03 Exercises

Three drills that turn "it's slow" into a named, measured bottleneck. Do them
before opening [`../solutions/`](../solutions/). Write down the *evidence*, not
just the verdict — an unsupported diagnosis is a guess, and guesses are what
profiling exists to eliminate.

This directory has its own `Makefile` that runs the profiler invocations each
drill needs, so you can work without leaving it:

```bash
make                          # list targets
make timeline                 # drill 1, level 1 (rocprofv3 --summary --sys-trace)
make counters KERNEL=kernel_beta   # drill 1, level 2 (rocprofv3 --pmc)
make parse                    # phase/kernel breakdown of the CSVs
make transpose-timeline       # drill 3, on A02's transpose ladder
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200);
find it with `/opt/rocm/bin/offload-arch`. The equivalent raw commands, spelled
out, are in each drill file.

| # | File | Skill |
|---|------|-------|
| 1 | `01_diagnose_mystery.md` | Profile four kernels; name each bottleneck and cite the counter that proves it. |
| 2 | `02_amdahl.md` | From a phase breakdown, compute the end-to-end ceiling of a 10× kernel win and decide whether to do it. |
| 3 | `03_profile_your_transpose.md` | Show from counters *why* A02's padded transpose beats the unpadded one. |

Each drill maps to an interview competency: evidence-based diagnosis /
prioritization under Amdahl's law / connecting a microarchitectural counter to a
source-level cause.

---

## Ground rules for every drill

These are the rules that separate a measurement from a story:

1. **Hypothesis first.** Before you run the profiler, write one sentence:
   *"I think kernel X is limited by ___ because ___."* Then measure to confirm or
   kill it. A profiler run with no hypothesis produces a wall of counters and no
   decision.
2. **Timeline before counters.** Establish that the thing you are about to
   micro-analyze actually matters. Drill 2 is the formal version of this rule.
3. **Never profile a debug build or the first iteration.** `-O3` (the module
   Makefile already does this) and skip the warmup dispatches
   (`--kernel-iteration-range` on AMD, `--launch-skip` on NVIDIA).
4. **One change at a time.** If you change two things and it gets faster, you
   have learned nothing about which one did it.
5. **Sanitize before sharing.** Profiler logs carry hostnames, pids, and absolute
   paths. See the `sed` recipe in the module README, section 4.

## What "evidence" looks like

A complete answer for one kernel has four parts:

| Part | Example shape |
|---|---|
| Verdict | "`kernel_X` is limited by uncoalesced global access." |
| Primary counter | Name the counter and its value, e.g. `MemUnitStalled` high while `VALUBusy` is low. |
| Corroborating number | The derived quantity: effective GB/s vs. the copy ceiling you measured in A02 Lab 1. |
| Predicted fix + expected delta | "Make the innermost index stride-1; expect to approach the ceiling." |

If you cannot fill all four rows, you have a hypothesis, not a diagnosis.
