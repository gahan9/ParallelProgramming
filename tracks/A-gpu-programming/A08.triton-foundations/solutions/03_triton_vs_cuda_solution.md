# Exercise 3 — SOLUTION: Triton vs. hand-written HIP/CUDA

This is a reasoning drill, so the value is in the argument, not the number. Your
counts will differ from the ones below; what should match is *which categories*
collapsed and *why that matters*.

---

## Part A — Count the code

Counted on the **vector-add pair**, because it is the smallest honest comparison:
[`A01/hip/vector_add.cpp`](../../A01.foundations-and-programming-model/hip/vector_add.cpp)
vs. [`triton/vector_add.py`](../triton/vector_add.py). Benchmarking, printing, and the
reference check are excluded — both versions have them, for the same reason.

| Measure | HIP/CUDA | Triton |
|---|---:|---:|
| Lines in the kernel body | 5 | 6 |
| Lines in the launch/host wrapper | 3 | 4 |
| Lines that exist **only** to manage memory | 9 | 0 |
| Lines that exist **only** to check errors | 1 standalone + a `HIP_CHECK` wrapper on 17 calls + a ~30-line `check.h` | 0 |
| Explicit barriers | 0 | 0 |
| Explicit shared-memory declarations | 0 | 0 |
| Source files to cover AMD **and** NVIDIA | 2 | 1 |

**The kernel body did not shrink.** It is five lines either way, and Triton's is
arguably one line longer. Everything that vanished is *ceremony*: three `hipMalloc`,
two H2D copies, one D2H copy, three `hipFree`, and the error-checking scaffold that
exists because each of those can fail silently.

That is exactly where the bugs live. A01 §6 lists five classic pitfalls; four of them
— ignored `hipError_t`, the missing bounds check, timing an async launch, and integer
overflow in byte-size math — are in the code Triton deleted or replaced:

| A01 pitfall | What happens in Triton |
|---|---|
| Ignored `hipError_t` | Gone: allocation is `torch.empty_like`, which raises a Python exception. |
| Missing `if (idx < n)` | Replaced by `mask=`, which is a *required argument* whenever the block can overhang, not an easily forgotten `if`. |
| Timing an async launch | Still yours to get wrong — the launch is still asynchronous, so you still need `torch.cuda.Event` / `synchronize()`. |
| `int` overflow in `count * sizeof(T)` | Gone: Python integers do not overflow and torch owns the byte math. |

Take the pair with real tiling (A06 matmul, A07 softmax) and the gap widens sharply,
because the shared-memory staging, the barriers, and the register micro-tile all
disappear as well — but note that the *reason* they disappear is that the compiler now
owns them, which is Part B.

---

## Part B — The control you surrendered

| # | Control | In Triton? | How / what you do instead |
|---|---|---|---|
| 1 | Threads per block | **Indirect** | You set `num_warps`; threads = `num_warps × warp_size`, and warp_size is 32 on NVIDIA and 64 on AMD. You choose a *count of warps*, not a count of threads. |
| 2 | Registers vs. shared memory | **No** | The compiler allocates. You influence it by changing tile shapes (`BLOCK_*`) and `num_warps`, then *inspect* the result (spill counts, occupancy) — an indirect loop, not a directive. |
| 3 | Barrier placement | **No** | Inserted by the compiler around the block operations it lowers. You cannot place one, and you also cannot forget one — the A02 `__syncthreads()` bug class is gone. |
| 4 | Instruction schedule / K-loop pipelining | **Indirect** | `num_stages` asks for a pipelining depth. You do not write the prefetch. |
| 5 | Matrix-core instruction (MFMA / Tensor Core / WMMA) | **Indirect** | `tl.dot` lowers to one *when the dtype, shape, and target permit*. You express intent; you do not select the instruction. Verify by reading the emitted assembly, not by assuming. |
| 6 | Warp/wavefront primitives (`__shfl_down`, ballot, DPP) | **Largely no** | You write `tl.sum` / `tl.max` and the compiler emits the shuffle tree. Cross-lane tricks that are not expressible as a block operation are the hard limit. |
| 7 | Async global→shared copies | **No** (not as a user-visible instruction) | The compiler emits them as part of pipelining when it decides to. |
| 8 | `__launch_bounds__` / max registers per thread | **Partially** | `triton.Config` exposes `maxnreg`, which the docs describe as corresponding to PTX `.maxnreg` and note is *not supported on all platforms* — so do not assume it is honoured on AMD. |

**Row 6 is the honest answer to "why keep the C++ kernel?"** When the algorithm needs a
lane-level primitive that has no block-level spelling — a hand-rolled warp-cooperative
scan with `DPP` row-shifts on CDNA, a ballot-driven compaction, an ISA instruction
Triton has no intrinsic for — the C++ kernel is not legacy, it is the only version that
can express the algorithm. Rows 2 and 3 are the second reason: when you are chasing the
last 10% and need to *place* a barrier or pin a value in registers, the Triton kernel
gives you a search where the C++ kernel gives you a knob.

---

## Part C — Five decisions the compiler made, and how to check them

For the softmax kernel in [`triton/fused_softmax.py`](../triton/fused_softmax.py):

| # | Compiler decision | How to see what it chose |
|---|---|---|
| 1 | How the `BLOCK`-wide row is distributed across the `num_warps` warps/wavefronts | Dump the intermediate IR (`MLIR_ENABLE_DUMP=1`) and read the layout attributes; the ROCm Triton optimization guide walks through reading `num-warps` out of the dump. |
| 2 | Vector width of the row load (one wide load per lane vs. several narrow ones) | Read the emitted assembly from the Triton cache (`~/.triton/cache`, or `TRITON_CACHE_DIR`) and look at the load instructions. |
| 3 | Whether the reduction tree uses shared memory / LDS at all, and how much | Profiler: `rocprofv3` or Nsight Compute shared-memory traffic counters — you wrote no shared memory, so anything you see there is the compiler's. |
| 4 | Register allocation, and whether anything spilled | The compiled-kernel handle exposes register and spill counts on the versions that support it (see `compiler_report` in [`triton/vector_add.py`](../triton/vector_add.py)); a non-zero spill count is a real slowdown you did not write. |
| 5 | Which tile shape / warp count / pipelining depth actually ran (matmul) | `matmul_kernel.best_config` after the first call, and the autotuner's own timings. |

The meta-point: in C++ you *state* these five things and read them back off your own
source. In Triton you *discover* them, which means a Triton kernel is not finished when
it passes — it is finished when you have looked at what the compiler produced.

---

## Part D — The review comment

> **Approve with conditions.** The code is a clear improvement in maintainability —
> 70 fewer lines, one source file instead of two, and it deletes the hand-managed
> shared-memory staging that is the most bug-prone part of the current kernel. I am not
> going to block a 6% regression on principle; I am going to block shipping it on the
> evidence we currently have, because one shape does not tell us what production will
> do. Before merge, please add:
>
> 1. **A shape sweep, not a data point.** Benchmark the shape distribution we actually
>    serve, including the small and ragged shapes. Triton kernels are tuned per shape;
>    a config that is 6% behind at the benchmark shape can be far worse at a shape the
>    autotuner never saw. If the sweep is within a few percent everywhere, the 6% is a
>    price I will pay for the maintainability.
> 2. **The first-call cost, measured in the process that will run this.** Autotuning and
>    JIT compilation are paid per process, per key. For a long-lived training job this
>    rounds to zero. For a short-lived CLI, a serverless worker, or a latency-SLO
>    inference path with cold starts, it can dominate — and if the key includes a
>    runtime shape, every new shape pays again. Tell me which of those we are.
> 3. **Both vendors, measured separately.** `num_warps` is warps on NVIDIA and
>    wavefronts on AMD, so the same number is a different thread count on each. A config
>    tuned on one vendor is not tuned on the other. If we ship on both, I want two sets
>    of numbers and, most likely, two config lists.
> 4. **A named owner and a version pin.** Triton's tuning behaviour and its supported
>    APIs move between releases. Who re-tunes when we bump the PyTorch/Triton wheel, and
>    what test catches a silent 2× regression from a compiler change? A perf test with a
>    threshold in CI, or this reverts to a manual chore nobody does.
> 5. **Keep the C++ kernel in the tree for one release**, behind a flag or at least in
>    history, with a comment saying why. It is our rollback.
>
> Two things I am explicitly *not* asking for: parity with the old kernel (6% is not
> worth the complexity we are deleting), and a micro-optimised Triton kernel (get the
> autotune config list right first — that is usually most of the gap).

The shape of that answer is the transferable part: a Triton rewrite is a trade of
**control and predictability** for **less code and one source per vendor**. Approving it
means asking for evidence about the things you gave up — shape sensitivity,
compilation latency, cross-vendor behaviour, and version stability — not re-litigating
the abstraction.
