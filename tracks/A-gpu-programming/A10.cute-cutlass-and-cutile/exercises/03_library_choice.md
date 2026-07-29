# Exercise 03 — The engineering decision: cuBLAS, CUTLASS/CuTe, Triton, or CuTile?

**Goal.** Knowing what CuTe *is* is the easy half. The half that shows up in design reviews
and interviews is knowing **when not to use it**. For each scenario below, pick a primary
approach, justify it with tradeoffs, and — most importantly — say what you would **measure
before committing**.

Answer in prose or as a table; there is no single right answer for every scenario, but there
are answers a reviewer will push back on. Reference answers, with the pushback:
[`../solutions/03_library_choice_solution.md`](../solutions/03_library_choice_solution.md).

**Your options.**

| Option | One-line character |
|---|---|
| **cuBLAS / cuBLASLt** | Closed-source, vendor-tuned, zero implementation effort, no layout control |
| **CUTLASS / CuTe** | Total control of tiles, layouts, pipelining, epilogues; high complexity and compile cost; NVIDIA-only |
| **Triton** | Tile-level Python DSL, autotuned, far less code; less control than C++; **runs on AMD and NVIDIA** |
| **CuTile** | Tile-level model, NVIDIA-only, compiler picks tensor-core and memory instructions; newer surface (see README §3.5) |

For each scenario, state:
1. **Primary choice** and one sentence of why.
2. **Runner-up** and what would make you switch to it.
3. **Two concrete tradeoffs** you are accepting.
4. **What you measure first** — the number that would change your mind.
5. **The AMD question**: does your choice survive a requirement to also run on MI300-class
   hardware? If not, what is the plan?

---

## Scenario A — Standard fp16 GEMM in a training loop

A transformer training job on H100s. The hot GEMMs are large, dense, standard shapes
(`[8192, 4096] × [4096, 4096]`, fp16 in, fp32 accumulate), called through
`torch.nn.Linear`. A teammate proposes writing a custom CUTLASS kernel "to get closer to
peak".

## Scenario B — GEMM fused with a custom activation epilogue

Same model, but the layer is `y = swiglu(x @ W1) * scale + bias`, where `swiglu` is a
gated activation your team defined and `scale` is a per-channel tensor. Profiling shows
the GEMM is followed by three separate elementwise kernels, each re-reading the full
activation tensor from HBM. The GEMM itself is already near cuBLAS speed.

## Scenario C — A novel sparse-ish attention variant

Research code. The attention pattern is block-sparse with a data-dependent block mask
(blocks are 128×128; roughly 20% are non-empty, and which ones changes per batch). You want
FlashAttention-style fusion — never materialize the full score matrix — over only the live
blocks. Nothing off the shelf implements this pattern. The kernel will be rewritten
repeatedly over the next three months as the algorithm changes.

## Scenario D — A research prototype that must run on AMD *and* NVIDIA

You are publishing a paper. The artifact must reproduce on a reviewer's A100 **and** on your
lab's MI300X cluster. The kernel is a fused normalization + GEMM + custom reduction. You have
six weeks, you are the only implementer, and "within 20% of peak" is acceptable — the paper's
claim is about the algorithm, not about kernel engineering.

---

## Part 2 — Build the decision table

Consolidate your four answers into one table a reviewer could read in 30 seconds:

| Scenario | Choice | Decisive factor | Biggest risk accepted |
|---|---|---|---|
| A | | | |
| B | | | |
| C | | | |
| D | | | |

## Part 3 — The meta-question

Write three sentences on this: **cuBLAS is the right default, and yet CUTLASS exists and is
heavily used. What does that tell you about where hand-written kernels create value?**

Then answer the follow-up an interviewer will ask: *"You said you'd use CUTLASS for scenario
B. How would you prove the fusion was worth it, in numbers, before you spent two weeks on
it?"*

---

## What "done" looks like

You can defend each choice against the obvious objection, you named a *measurement* rather
than an intuition for every scenario, and you did not pick the most sophisticated tool in any
scenario where the simple one wins. Reviewers notice the last one most.
