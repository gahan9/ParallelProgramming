# B01 Solutions

Reference answers for [`../exercises/`](../exercises/). **Try the drills first** — the
two written ones especially, because the whole skill is doing the arithmetic yourself
rather than recognizing it when you see it.

Each solution explains the *reasoning*, not just the result, and links the vendor
document every figure came from with a retrieval date.

| # | File | Kind |
|---|------|------|
| 1 | [`01_derive_peaks_solution.md`](01_derive_peaks_solution.md) | Written — every peak derived, cross-checked against the published figure |
| 2 | [`02_spec_sheet_audit_solution.md`](02_spec_sheet_audit_solution.md) | Written — six real claims audited, plus the paragraph to put in a design doc |
| 3 | [`03_measure_vs_spec_solution.cpp`](03_measure_vs_spec_solution.cpp) | HIP — achieved bandwidth as a percentage of the device's derived peak |

## Building the code solution

```bash
make                       # list targets
make all GPU_ARCH=gfx950   # build for your GPU
make run                   # build + run (must PASS)
make clean
```

Unlike `exercises/`, `make run` here stops at the first failure: a solution that does
not PASS is a bug in the curriculum, not a learning opportunity.

## The three changes in the code solution

| TODO | Starter behaviour | Fix |
|------|-------------------|-----|
| 1 | `theoretical_peak_gbps` returns `0.0` | `2 × clock_hz × (bus_bits / 8)`, returning `0.0` only when the driver genuinely reports zero |
| 2 | `copy_kernel` copies one element per thread, so a 1024×256 launch covers 262,144 of 67,108,864 elements | grid-stride loop: `for (i = tid; i < n; i += gridDim.x * blockDim.x)` |
| 3 | `achieved_gbps` and `efficiency` hardcoded to `0.0` | `2 × bytes / seconds` (a copy reads *and* writes every element), then `100 × achieved / peak` guarded against a zero peak |

The subtle one is TODO 3's factor of 2. A copy of `N` floats moves `2 × 4N` bytes
across the memory bus, not `4N` — forget it and you will report exactly half your
real bandwidth and conclude your kernel is broken when it is fine. Every effective-
bandwidth calculation in this curriculum counts *bytes that crossed the bus*, which
means reads **plus** writes.

## What the numbers mean afterwards

The solution prints three things, and the third is the one you keep:

```
achieved BW   : <measured>            <- the real ceiling
theoretical   : <derived from props>  <- the datasheet ceiling
efficiency    : <ratio>%              <- the gap you can never close
```

A coalesced streaming copy in the **70–90% of peak** band is healthy. Take the
**achieved** figure into B02 and build the roofline's memory ceiling on it. If you
build it on the theoretical peak instead, every kernel you subsequently profile will
look like a failure, and you will waste days optimizing code that is already at 95%
of what the silicon can do.

If your driver reports a zero memory clock or bus width (common under some
virtualization and container configurations), the solution says so and prints no
efficiency rather than a fabricated one. That behaviour is deliberate and is worth
copying into your own tools: **an honest "unavailable" is more useful than a
plausible-looking wrong number.**
