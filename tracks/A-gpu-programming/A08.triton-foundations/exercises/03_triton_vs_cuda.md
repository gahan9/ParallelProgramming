# Exercise 3 — Triton vs. hand-written HIP/CUDA: what did you trade?

**No code to run.** This is a written drill, and it is the one that turns A08 from
"I learned a Python API" into "I can defend a design decision in a review."

Pick **one** kernel you have already written twice — once in C++ and once in Triton.
The strongest pairs in this repo:

| Pair | C++ source | Triton source |
|---|---|---|
| Softmax | [`A07/hip/softmax.cpp`](../../A07.softmax-and-fused-kernels/hip/softmax.cpp) | [`triton/fused_softmax.py`](../triton/fused_softmax.py) |
| Tiled matmul | [`A06/hip/matmul.cpp`](../../A06.tiled-matrix-multiply/hip/matmul.cpp) | [`triton/matmul.py`](../triton/matmul.py) |
| Vector add | [`A01/hip/vector_add.cpp`](../../A01.foundations-and-programming-model/hip/vector_add.cpp) | [`triton/vector_add.py`](../triton/vector_add.py) |

Write your answers in this file (or a copy of it). Aim for specifics, not adjectives.

---

## Part A — Count the code

Fill this in for the pair you chose. Count only the kernel and its launch wrapper —
not the benchmarking, printing, or CPU reference code, since those exist in both
versions for the same reason.

| Measure | HIP/CUDA | Triton |
|---|---:|---:|
| Lines in the kernel body | | |
| Lines in the launch/host wrapper | | |
| Lines that exist **only** to manage memory (`hipMalloc`, `hipMemcpy`, `hipFree`) | | |
| Lines that exist **only** to check errors | | |
| Explicit `__syncthreads()` / barriers | | |
| Explicit shared-memory declarations | | |
| Separate source files needed to cover AMD **and** NVIDIA | | |

Then answer: **which category shrank the most, and is that category where the bugs
actually live?**

---

## Part B — Name the control you surrendered

For each item below, say whether Triton still lets you control it, and if so, how
(which decorator, argument, or code shape). If it does not, say what you would have
to do instead.

1. The exact number of threads in a block.
2. Which values live in registers vs. shared memory / LDS.
3. Where the barriers go.
4. The instruction schedule and whether the K-loop is software-pipelined.
5. Whether a matrix-core instruction (MFMA / Tensor Core / WMMA) is emitted.
6. The wavefront/warp-level primitives you used in A04 (`__shfl_down`, ballot, DPP).
7. Asynchronous global→shared copies.
8. `__launch_bounds__` / max registers per thread.

> One of these is the honest answer to "why would a Principal Engineer still keep the
> C++ kernel?" Identify which, and write one sentence on the workload that forces it.

---

## Part C — What did the compiler decide for you?

List, for your chosen kernel, **five specific decisions** the Triton compiler made
that you wrote by hand in C++. For each one, note *how you would find out what it
chose*. Examples of the kind of evidence that counts:

- the register and spill counts from the compiled-kernel handle
  (`vector_add.py`'s `compiler_report`),
- the emitted assembly (`TRITON_CACHE_DIR`, or `MLIR_ENABLE_DUMP=1` per the ROCm
  Triton optimization guide),
- the config the autotuner picked (`matmul_kernel.best_config`),
- a profiler counter (`rocprofv3`, Nsight Compute) that shows shared-memory traffic
  you never wrote.

---

## Part D — The review question

You are reviewing a pull request that replaces a hand-written, currently-shipping HIP
kernel with a Triton one. It is 70 lines shorter, passes all correctness tests, and is
6% slower on the one shape in the benchmark suite.

Write the review comment you would actually post. It must take a position (approve,
approve-with-conditions, or request-changes) and name the **evidence you would require**
before the change ships. At minimum, decide what you want to know about:

- performance across the full shape distribution in production, not one shape,
- the first-call compilation cost in the process that will run this,
- what happens on the *other* vendor,
- who owns the kernel when the Triton version bumps and the tuning shifts.

---

Reference answer with the reasoning: [`../solutions/03_triton_vs_cuda_solution.md`](../solutions/03_triton_vs_cuda_solution.md).
Do Parts A–C first — the solution's numbers are for the softmax pair, and yours will
differ if you chose another.
