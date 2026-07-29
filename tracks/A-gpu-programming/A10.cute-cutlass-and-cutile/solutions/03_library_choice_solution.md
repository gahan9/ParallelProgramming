# Solution 03 — The engineering decision

Reference answers. Exercise: [`../exercises/03_library_choice.md`](../exercises/03_library_choice.md).

These are **defensible positions, not the only positions.** What a reviewer actually grades is
whether you (a) named a measurement instead of an intuition, (b) declined the sophisticated tool
where the simple one wins, and (c) noticed the portability constraint when it existed. Each
answer below therefore includes the objection you should expect.

---

## Scenario A — Standard fp16 GEMM in a training loop

**Primary: cuBLAS (i.e. leave `torch.nn.Linear` alone).**

1. **Why.** Large dense standard-shape fp16 GEMM is precisely the case cuBLAS is tuned for on
   every architecture NVIDIA ships. There is no layout freedom to exploit, no fusion opportunity
   in the statement of the problem, and no unusual shape. The expected gain from a hand-written
   CUTLASS kernel is near zero, and the expected *loss* — weeks of work, template complexity,
   compile time, and a kernel someone must re-tune for the next GPU — is large and certain.
2. **Runner-up: cuBLASLt.** Switch if you need control over algorithm selection, workspace, or
   epilogue variants from its fixed menu (bias, ReLU, GELU), or heuristic caching for a fixed
   shape set. Still zero kernel code.
3. **Tradeoffs accepted.** (i) No visibility into or control over the kernel — you cannot fix a
   pathological shape yourself. (ii) Closed source, so you debug performance through a profiler
   rather than through code.
4. **Measure first.** `ncu`/`nsys` on the actual training step: what fraction of step time is
   these GEMMs, and what fraction of your GPU's datasheet tensor-core peak do they reach? If they
   are at a high fraction of peak, the ceiling on *any* GEMM work is that remaining gap — usually
   a few percent of a few percent of step time. Also check whether the step is GEMM-bound at all;
   data loading, communication, and unfused elementwise ops are more common culprits.
5. **AMD question.** Survives trivially — the equivalent is rocBLAS/hipBLASLt, and PyTorch
   already dispatches to it. This is the *only* scenario where portability is free.

> **Expected objection:** *"But NVIDIA's own charts show CUTLASS matching or beating cuBLAS on
> some shapes."* Correct, and irrelevant to the decision: "matching" is not a reason to take on
> a maintenance burden, and "beating on some shapes" means you must first show *your* shape is
> one of them. That is a measurement, and it costs an afternoon rather than three weeks.

## Scenario B — GEMM fused with a custom activation epilogue

**Primary: CUTLASS/CuTe** (with **Triton** as a serious contender — see the objection).

1. **Why.** The win here is not a faster GEMM; it is **deleting HBM round trips**. Three
   elementwise kernels each re-read and re-write the full activation tensor. Fusing them into the
   GEMM epilogue means the output tile is consumed while it is still in registers. cuBLASLt's
   epilogue menu is fixed and will not contain your gated activation with a per-channel scale, so
   the fixed menu is exactly what forces you off the default path — and CUTLASS's epilogue is
   designed as an extension point (NVIDIA ships worked examples, e.g. Hopper GEMM with activation
   fusion, example 113).
2. **Runner-up: Triton.** Switch if the team is Python-first, iteration speed matters more than
   the last few percent, or you need the AMD path. Triton fuses epilogues naturally and in far
   less code.
3. **Tradeoffs accepted.** (i) You now own a GEMM kernel: template complexity, compile time, and
   a re-tune each new architecture. (ii) Correctness risk moves into your code — you must verify
   the fused result against the unfused pipeline numerically, not just eyeball it.
4. **Measure first — and this is the answer to the interviewer's follow-up.** Bound the win
   *before* writing the kernel, with arithmetic and a profiler, not a prototype:
   - From `ncu`, get the measured time of the GEMM and of each of the three elementwise kernels.
   - The three elementwise kernels are memory-bound, so their combined time ≈ (bytes they move) /
     (your measured achievable bandwidth). Fusing removes essentially all of that traffic.
   - **Upper bound on the win = sum of the three elementwise kernel times**, as a fraction of
     end-to-end step time. If that is 2%, stop — you have your answer for the cost of an
     afternoon.
   - Cheap intermediate experiment: fuse the three elementwise kernels *with each other* first
     (one kernel, one read, one write). That is an hour of work and captures a large share of the
     traffic saving. If the remaining GEMM-epilogue fusion is then worth <1%, you are done and you
     never wrote a CUTLASS kernel.
5. **AMD question.** CUTLASS does **not** survive. Options: maintain a Composable Kernel/CK-Tile
   path in parallel (real duplication, and CK's epilogue story is the analogue you would use), or
   choose Triton up front and accept a smaller peak in exchange for one source tree.

> **Expected objection:** *"Why not just Triton?"* Often you should, and saying so scores better
> than reflexively reaching for CUTLASS. Pick CUTLASS when you are shipping this at scale on
> NVIDIA and the last 10–20% is worth real money; pick Triton when iteration speed or portability
> is worth more than that margin.

## Scenario C — A novel sparse-ish attention variant

**Primary: Triton** for the next three months, with **CUTLASS/CuTe** as the productionization
path if the algorithm survives.

1. **Why.** The dominant constraint is stated in the problem: *"rewritten repeatedly over the
   next three months."* The cost function is **iteration speed**, not peak. Triton gives you
   fused tile-level kernels in far less code, with autotuning over block sizes for free, which is
   what you want while the block mask semantics are still changing. A data-dependent block mask
   with 20% density is also a natural fit for a tile-level model: you loop over the live blocks
   and the compiler handles the thread mapping. CUTLASS would let you go faster *eventually* and
   would slow every one of the next twenty iterations.
2. **Runner-up: CUTLASS/CuTe.** Switch once the algorithm is frozen and the kernel is on the
   critical path of something that matters — or immediately, if you find Triton cannot express
   the mask indirection or the pipelining you need.
3. **Tradeoffs accepted.** (i) You concede some peak — how much is workload-specific and I am not
   going to invent a number. (ii) You are exposed to Triton compiler behaviour (autotuning time,
   register pressure, occasional codegen surprises) that you cannot fix at the source level.
4. **Measure first.** Two numbers. (a) **Density-adjusted roofline:** at 20% block density, what
   is the achievable FLOP count and therefore the fastest this kernel could possibly be? Compare
   against the dense FlashAttention baseline you are trying to beat — if dense-and-fast beats
   sparse-and-slow, the whole premise is wrong and that is worth knowing in week one. (b) The
   dense Triton FlashAttention tutorial kernel's fraction of peak on your GPU, as a realistic
   ceiling for what your Triton version will achieve.
5. **AMD question.** Triton survives — that is a genuine bonus here, since research artifacts get
   run on other people's hardware. CUTLASS would not.

> **Expected objection:** *"FlashAttention-class kernels are the canonical CUTLASS use case."*
> True for the shipped, frozen implementation. It is not true for the *research* phase, and
> conflating the two is the most common way teams burn a quarter.

## Scenario D — A research prototype that must run on AMD *and* NVIDIA

**Primary: Triton.** This one is not close.

1. **Why.** "Must reproduce on an A100 **and** an MI300X" eliminates CUTLASS/CuTe and CuTile
   outright — neither has an AMD backend. Combined with "one implementer, six weeks, within 20% of
   peak is acceptable", Triton is the only option that satisfies the hard constraint while fitting
   the budget. The paper's claim is about the algorithm; kernel engineering is overhead, not
   contribution.
2. **Runner-up: two backends** — CUTLASS on NVIDIA plus Composable Kernel/CK-Tile on AMD. Choose
   this only if a reviewer will reject the paper on absolute performance grounds. With one
   implementer and six weeks, it is not realistic, and you should say so out loud rather than
   promise it.
3. **Tradeoffs accepted.** (i) You concede peak performance, explicitly and in writing — put the
   measured fraction of peak in the paper so nobody has to guess. (ii) You inherit Triton's AMD
   support surface, which is real but historically less mature than its NVIDIA path; budget time
   to validate on MI300X **early**, not in week five.
4. **Measure first.** In week one, run a *trivial* Triton kernel (vector add, then a small GEMM)
   on both the A100 and the MI300X and record fraction of peak on each. You are measuring the
   toolchain, not your algorithm. If the AMD path is broken or far off peak, you need to know
   before you have written the real kernel — that discovery in week five is what sinks artifacts.
5. **AMD question.** This *is* the AMD question, and Triton is the answer. Corollary worth
   stating: the fact that A10 is the only NVIDIA-only module in this curriculum is the same
   observation from the other direction.

> **Expected objection:** *"Just ship the NVIDIA version and note AMD as future work."* Sometimes
> acceptable — but the scenario states the artifact *must* reproduce on both, so this answer fails
> the requirement. Notice when a constraint is hard.

---

## Part 2 — The decision table

| Scenario | Choice | Decisive factor | Biggest risk accepted |
|---|---|---|---|
| A — standard fp16 GEMM | **cuBLAS** | No fusion or layout freedom to exploit; cuBLAS is tuned for exactly this | No control if a shape turns out pathological |
| B — custom fused epilogue | **CUTLASS/CuTe** (Triton if Python-first or AMD needed) | cuBLASLt's epilogue menu is fixed; the win is deleted HBM traffic | Owning a GEMM kernel: complexity, compile time, per-arch re-tuning |
| C — novel block-sparse attention | **Triton** now, CUTLASS later if it ships | Iteration speed dominates while the algorithm is still moving | Conceding some peak; dependent on compiler behaviour |
| D — dual-vendor research artifact | **Triton** | Hard portability constraint eliminates every NVIDIA-only option | Lower peak, stated explicitly; AMD toolchain maturity risk |

Where does **CuTile** land? Nowhere as a primary choice in these four, and the reason is
scenario-independent: it is NVIDIA-only (so D is out) and its surface is newer than the
alternatives (see [`../README.md` §3.5](../README.md#35-cutile--what-it-is-and-how-new-it-is) —
Python 1.0.0 in Dec 2025, Hopper and C++ support only in CUDA 13.3 in May 2026, plus an
explicitly experimental API package). For A and B the incumbents are simply better-understood
choices, and for C, Triton buys you AMD for free. That is a **maturity and portability**
judgement, not a performance one — I have no verified cuTile-vs-Triton or cuTile-vs-CUTLASS
benchmark and will not invent one. If you are NVIDIA-only, greenfield, and willing to track
releases, cuTile is a reasonable thing to evaluate; "evaluate" is the honest verb.

## Part 3 — The meta-question

**Three sentences.** cuBLAS is the right default because a vendor library concentrates enormous
tuning effort on the small set of shapes and dtypes that dominate real workloads, and you cannot
out-tune that on the shapes it targets. CUTLASS exists and thrives because the *interesting* work
is increasingly **not** a standalone GEMM: it is a GEMM fused with something, on a shape or dtype
the library does not cover, or with a data-dependent structure a fixed API cannot express. So
hand-written kernels create value at the **boundaries** — fusion, unusual shapes, new numeric
formats, novel algorithms — and destroy value in the middle, where they merely reimplement what a
library already does well.

**The follow-up — proving the fusion is worth it before spending two weeks:** see A-B-4 above.
The short version: use `ncu` to get the measured times of the GEMM and the three elementwise
kernels; the elementwise kernels are memory-bound, so their combined time is your **upper bound**
on the fusion win; express that as a fraction of end-to-end step time. Then buy information
cheaply before buying the kernel — fuse the three elementwise kernels with each other first (an
hour), re-measure, and only then decide whether the residual justifies a CUTLASS epilogue. If you
cannot state the upper bound as a number, you are not ready to start.

---

## The pattern to take away

Across all four scenarios the deciding factor was never "which tool is most powerful". It was, in
order: **is the constraint hard** (portability in D), **where is the actual time going**
(A and B), and **what is the cost function** (iteration speed in C). Reach for the most powerful
tool only after those three questions are answered, and be able to name the measurement that
would change your mind.
