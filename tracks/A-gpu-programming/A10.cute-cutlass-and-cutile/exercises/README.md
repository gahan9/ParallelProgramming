# A10 Exercises

> **NVIDIA-only, optional module.** These drills need `nvcc` and (for drill 1) an SM70+ GPU.
> If you are on AMD, skip A10 — see [`../README.md` §1](../README.md#if-you-are-on-amd-read-this-instead)
> for the Composable Kernel / hipBLASLt / MFMA reading list that covers the same ideas.

Do these yourself before peeking at [`../solutions/`](../solutions/).

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                          # list targets
make all SM_ARCH=sm_80        # build every coding exercise for your GPU
make 01_wmma_gemm.exe         # build just one
make run                      # build + run all (FAIL is expected until solved)
make clean
```

Set `SM_ARCH` to your device (`sm_70` Volta, `sm_80` A100, `sm_89` Ada, `sm_90` Hopper,
`sm_100` Blackwell datacenter); find it with
`nvidia-smi --query-gpu=compute_cap --format=csv`. The equivalent raw command:

```bash
nvcc -O3 -arch=sm_80 -I../cuda 01_wmma_gemm.cu -o 01_wmma_gemm.exe && ./01_wmma_gemm.exe
```

(The drills reuse the module's `CUDA_CHECK` macro via `-I../cuda`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_wmma_gemm.cu` | Drive tensor cores directly: fragment loads with correct leading dimensions, warp-collective `mma_sync` accumulation, `store_matrix_sync`. Verified vs CPU. |
| 2 | `02_layout_algebra.md` | Whiteboard: `(Shape, Stride)` → linear offset (including a hierarchical layout), design a transposed view and a tiled view, compute shared-memory banks for padded vs swizzled tiles. |
| 3 | `03_library_choice.md` | Design judgement: cuBLAS vs CUTLASS/CuTe vs Triton vs CuTile across four concrete scenarios, with the measurement that would change your mind. |

Each drill maps to an interview competency: *layout-sensitive API correctness* /
*index-arithmetic fluency* / *tool-selection judgement under constraints*. Solutions include
the reasoning, not just the answer.

---

## Sample sandbox output

> [!IMPORTANT]
> **The GFLOP/s figures below are placeholders (`____`), not measurements.** This module was
> authored without an NVIDIA GPU available, so publishing a number here would be fabrication.
> The **PASS/FAIL states and the printed field layout are real** (they follow directly from the
> harness code); fill in your own throughput when you run it. Absolute GFLOP/s is
> GPU-specific anyway — what transfers is the ratio between kernels.

### Exercise 1 — WMMA GEMM (`01_wmma_gemm.exe`)

Starting state, straight after `make`. It compiles and runs; `C` is never written, so every
element mismatches:

```text
$ nvcc -O3 -arch=sm_80 -I../cuda 01_wmma_gemm.cu -o 01_wmma_gemm.exe
$ ./01_wmma_gemm.exe
problem      : 512x512x512 fp16 in / fp32 out
kernel time  : ____ ms
throughput   : ____ GFLOP/s
result       : FAILED (262144 mismatching elements)

Still FAILING? Work the TODOs in order and check:
  - leading dimensions: lda = K for A, ldb = N for B, ldc = N for C
  - the tile origins: A is (rowTile*16, k), B is (k, colTile*16)
  - mma_sync(acc, aFrag, bFrag, acc) - acc is both input and output
  - store_matrix_sync needs wmma::mem_row_major for row-major C
```

`262144 = 512 × 512`, i.e. *every* element — that is the signature of "nothing was stored",
not of a subtle layout bug.

After the four TODOs:

```text
$ ./01_wmma_gemm.exe
problem      : 512x512x512 fp16 in / fp32 out
kernel time  : ____ ms
throughput   : ____ GFLOP/s
result       : PASSED (0 mismatching elements)
```

**The instructive failure states**, and what each one means:

| Symptom | Almost certainly |
|---|---|
| `FAILED (262144 ...)` — every element wrong | TODO 4 not done: nothing stored |
| `FAILED` with a large but not total count, values plausible-looking | Wrong leading dimension, or `row_major`/`col_major` mismatched to your data |
| Results correct in the first 16 rows only | Tile origin arithmetic dropped the `rowTile` term |
| Kernel hangs, or `ncu` reports an illegal instruction | An MMA reached by only part of a warp — check that every guard is warp-uniform |
| Correct but throughput barely above the SIMT baseline | You are reading A and B from global memory each K-step. Expected here; fixing it is the §9 stretch goal. |

Compare your number against the module lab, which solves the same problem at `N = 1024`:

```text
$ cd .. && make run-cuda SM_ARCH=sm_80
== cuda/mma_tensor_core.exe ==
device           : <your GPU> (compute capability 8.0)
problem          : C[1024 x 1024] = A[1024 x 1024] * B[1024 x 1024], fp16 in / fp32 out
MMA shape        : 16x16x16 (nvcuda::wmma)

computing CPU reference (this takes a second or two)...

kernel                                    GFLOP/s     result
---------------------------------- -------------- ----------
simt_gemm (scalar FFMA pipeline)             ____     PASSED
wmma_gemm (tensor cores)                     ____     PASSED

Arithmetic intensity: 2*N^3 / (8*N^2) = N/4 = 256 FLOP/byte -> compute-bound.
```

Expect `wmma_gemm` to be **several times** `simt_gemm` — that is the §2 "64× work per issued
instruction" argument showing up in wall-clock — and expect *both* to sit well below your
GPU's datasheet peak, because neither stages tiles through shared memory.

### Exercises 2 and 3 — written drills (no program)

Paper drills. Check your work against
[`../solutions/02_layout_algebra_solution.md`](../solutions/02_layout_algebra_solution.md) and
[`../solutions/03_library_choice_solution.md`](../solutions/03_library_choice_solution.md).

Expected headlines:
- **Drill 2:** the `(4,2)` layouts differ *only* in which mode has stride 1; the tiled view of
  an 8×8 row-major matrix is `((4,4),(2,2)) : ((8,1),(32,4))`; an unpadded 32-wide `float`
  column read is a **32-way** bank conflict, and both `(32,32):(33,1)` padding and
  `Swizzle<5,0,5>` fix it — but only the swizzle keeps the power-of-two stride.
- **Drill 3:** cuBLAS wins scenario A, CUTLASS wins B, CUTLASS or Triton wins C depending on
  iteration speed, and **Triton wins D** because it is the only dual-vendor option.

You can check parts 1–4 of drill 2 mechanically by running the module's Lab 2
(`cd .. && make run-cute CUTLASS_DIR=/path/to/cutlass`), which prints the same tables with
`cute::print_layout`.
