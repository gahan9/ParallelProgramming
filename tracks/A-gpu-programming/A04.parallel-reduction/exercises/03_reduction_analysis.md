# Exercise 3 — Reduction roofline drill

No code. Work the arithmetic on paper, then check
[`../solutions/03_reduction_analysis_solution.md`](../solutions/03_reduction_analysis_solution.md).

The point of this drill is the habit every performance engineer needs: **before** you optimize a
kernel, compute the number it can never beat. If you are already at 85% of that number, stop
optimizing and go find a different bottleneck.

## Setup

You are summing a float32 tensor on an AMD MI300X:

| Quantity | Value |
|---|---|
| Elements `N` | `2^28` = 268,435,456 |
| Element type | `float` (4 bytes) |
| Datasheet peak HBM bandwidth | 5.3 TB/s (= 5.3 × 10¹² bytes/s) |
| Measured copy ceiling (A02 Lab 1) | 4.8 TB/s |
| Your measured reduction time | 0.265 ms |
| Kernel launch overhead | ~5 µs per launch |
| Block size | 256 threads |

Use decimal GB/TB (10⁹ / 10¹²) throughout so your numbers are comparable to vendor datasheets.

## Questions

1. **Bytes moved.** How many bytes must an *optimal* reduction read? How many does it write? What
   is the total, and why is the write term negligible?
2. **Theoretical minimum time.** Using the datasheet peak, what is the fastest this reduction could
   possibly run? Repeat using the *measured* copy ceiling. Which of the two is the honest target,
   and why?
3. **Fraction of peak.** Your measured 0.265 ms is what fraction of the datasheet bound? Of the
   copy-ceiling bound?
4. **Arithmetic intensity.** A reduction performs `N − 1` additions. Compute FLOP/byte and place the
   kernel on the roofline. Is it memory-bound or compute-bound? Would a 2× faster FPU help at all?
5. **Explain the gap.** List at least four concrete reasons a real reduction lands below the copy
   ceiling. Rank them by how much you would expect each to cost at this `N`.
6. **Multi-pass overhead.** The two-pass scheme writes one partial per block and then re-reads them.
   For a stage-6 style launch (one block per 512 elements) how many partials are there, how many
   extra bytes does the second pass move, and what percentage of the total traffic is that? Redo it
   for a cascade with a fixed grid of 2048 blocks.
7. **When does overhead win?** At what `N` does the fixed cost of two kernel launches (~10 µs) equal
   the time spent moving the data at peak bandwidth? What does that tell you about reducing small
   tensors in a training loop?
8. **Precision.** You switch the input to bfloat16 (2 bytes) but keep a float32 accumulator. What
   happens to the theoretical minimum time, and what happens to the *accuracy* of the result? Which
   of the two effects would you mention first in a design review?

## Bonus

Your colleague proposes replacing the second kernel launch with `atomicAdd` and reports a 6 µs
improvement. Training runs then stop being bit-reproducible across restarts. Write the two-sentence
recommendation you would put in the code review, and name the condition under which you *would*
accept the atomic version.
