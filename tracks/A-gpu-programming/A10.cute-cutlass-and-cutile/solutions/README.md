# A10 Solutions

> **NVIDIA-only, optional module.** Needs `nvcc` and (for drill 1) an SM70+ GPU. AMD readers
> should skip A10 — see [`../README.md` §1](../README.md#if-you-are-on-amd-read-this-instead).

Reference answers for [`../exercises/`](../exercises/). Try the drills first; these explain the
*reasoning*, which is the part that transfers.

```bash
make                          # list targets
make all SM_ARCH=sm_80        # build every solution
make run                      # build + run all (every one must PASS)
make clean
```

`make run` here stops on the first failure — unlike `exercises/`, a solution that does not PASS
is a curriculum bug.

| # | Solution | The point |
|---|---|---|
| 1 | `01_wmma_gemm_solution.cu` | The four WMMA calls and the two address expressions, with the leading-dimension and majorness traps called out inline. |
| 2 | `02_layout_algebra_solution.md` | Every answer is one rule — inner product of coordinate and stride — applied mechanically. Includes the full padding-vs-swizzle bank arithmetic. |
| 3 | `03_library_choice_solution.md` | Four defensible choices, each with the objection a reviewer will raise and the measurement that settles it. |

---

## What each solution is really teaching

### 1 — WMMA GEMM

`diff ../exercises/01_wmma_gemm.cu 01_wmma_gemm_solution.cu` — the answer is six lines. Three
things are worth more than the code:

- **The leading dimension is the full matrix's stride, not the tile's.** `lda = K`, `ldb = N`,
  `ldc = N`. Passing `WMMA_K`/`WMMA_N` compiles, runs, and produces a wrong answer that looks
  structurally plausible. Fragments are opaque, so you cannot print your way out of it — the CPU
  reference is your only detector, which is why the harness has one.
- **`mma_sync(acc, aFrag, bFrag, acc)`** — accumulator as both input and output. The accumulator
  fragment is declared *outside* the K loop for exactly this reason.
- **Every guard around an MMA must be warp-uniform.** In the solution, `rowTile` derives from
  `threadIdx.y` (the warp index) and `colTile` from `blockIdx.x`; nothing depends on
  `threadIdx.x`. A guard on `threadIdx.x` would let part of a warp skip the warp-collective MMA
  — undefined behaviour, and a realistic hang.

**What the solution deliberately does *not* do:** stage tiles through shared memory, swizzle
anything, or pipeline the loads with `cp.async`. It reads A and B from global memory on every
K-step. That is why it is *fast relative to SIMT* and *slow relative to cuBLAS* — the gap between
those two statements is the entire argument for CUTLASS, and the §9 stretch goals walk you into
it deliberately.

### 2 — Layout algebra

If you got parts 1–4 right but stumbled on 6 or 7, that is the normal failure profile: evaluating
a layout is easy, *designing* one is the skill. The two headline results to remember:

- Tiled view of an 8×8 row-major matrix into 2×2 tiles of 4×4:
  **`((4,4),(2,2)) : ((8,1),(32,4))`** — mode 0 intra-tile, mode 1 tile coordinate, which is
  exactly CuTe's `zipped_divide` profile.
- An unpadded 32-wide `float` column read is a **32-way** bank conflict; both `(32,32):(33,1)`
  padding and `Swizzle<5,0,5>` fix it, but only the swizzle preserves the power-of-two,
  16-byte-aligned stride that vectorized loads, `ldmatrix`, and TMA descriptors require.

You can verify parts 1–4 mechanically: `cd .. && make run-cute CUTLASS_DIR=/path/to/cutlass`
prints the same tables via `cute::print_layout`.

### 3 — Library choice

The scoring rubric, stated plainly, because it is what interviewers are actually listening for:

1. Did you **name a measurement** rather than an intuition? ("I'd profile the step and bound the
   fusion win by the elementwise kernels' combined time" beats "fusion is usually worth it".)
2. Did you **decline the sophisticated tool** where the simple one wins? Scenario A is a trap and
   reaching for CUTLASS fails it.
3. Did you notice the **hard constraint**? Scenario D's dual-vendor requirement eliminates every
   NVIDIA-only option, full stop.
4. Did you **buy information cheaply before buying the kernel**? Fusing three elementwise kernels
   with each other takes an hour and often settles the question.

---

## Honesty note

The written solutions state performance *reasoning* (arithmetic intensity, upper bounds from
memory traffic, work-per-instruction ratios) but **no measured GFLOP/s**, because this module was
authored without an NVIDIA GPU available. Numbers you see in `../exercises/README.md` are marked
as placeholders for the same reason. Run the labs and fill in your own; the ratios are what
transfer between GPUs anyway.
