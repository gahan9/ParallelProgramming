# Module Template — the 9-section gold standard

Every module in this curriculum follows the **same nine sections, in the same order**. This is
not bureaucracy: it is a quality contract. It guarantees that each topic is explained simply,
derived from first principles, taken to expert depth, proven with runnable code and real numbers,
honest about its tradeoffs, grounded in the literature, and testable against interview-grade drills.

Copy this file's structure verbatim when authoring a new module `README.md`. If a section does not
apply, keep the heading and write one sentence explaining why — never silently drop it.

---

## Folder layout for a module

```
<Track><NN>.<slug>/
├── README.md            <- the 9 sections below
├── cuda/                <- NVIDIA-native code (.cu), built with nvcc
├── hip/                 <- AMD/portable code (.cpp), built with hipcc
├── triton/              <- Triton kernels (.py), run on either vendor
├── Makefile             <- dual hipcc/nvcc targets + profile + clean
├── exercises/           <- starter files with TODOs (learner writes the code)
└── solutions/           <- reference solutions with explanation
```

Not every module needs all three code dirs (a Track C design module may have none). Include what
serves the topic; state what you omitted and why in Section 4.

---

## The nine sections

### 1. TL;DR + Layman analogy
- A 3–5 sentence summary a busy engineer can read in 20 seconds.
- Then a **plain-language analogy** that gives a non-expert the correct intuition. The analogy
  must be *load-bearing* — it should predict the right behavior, not just decorate.
- End with a one-line "by the end of this module you can…" outcome.

### 2. First Principles
- Derive *why the technique exists* from scratch. Start from a problem, not from an API.
- Show the naive approach and where it breaks. Motivate the real approach as the fix.
- Prefer a small worked example / back-of-envelope calculation over prose.

### 3. Deep Dive
- Expert-level, architecture-aware content. Map concepts onto real hardware: AMD **CDNA**
  (CU, wavefront=64 lanes, LDS, HBM) and NVIDIA **Hopper/Ada** (SM, warp=32 lanes, shared memory,
  tensor cores). Call out where the two diverge.
- This is where a Principal Engineer expects rigor: memory models, ISA-level behavior, occupancy
  math, algorithmic complexity.

### 4. Hands-On Labs
- Progressive labs, each with a clear goal and a "run it" command.
- Dual-track: show `cuda/`, `hip/`, and (where relevant) `triton/` versions. Highlight the diffs.
- Each lab states what you should *observe*, not just what to type.

### 5. Performance Analysis
- Numbers, not adjectives. Give the profiling command and a representative result.
- Tie results to the **roofline**: is this kernel compute-bound or memory-bound, and why?
- Show at least one before/after optimization with the measured delta.

### 6. Challenges, Drawbacks & Tradeoffs
- Where this breaks: race conditions, synchronization traps, numerical issues, occupancy cliffs.
- When *not* to use the technique. What it costs (complexity, portability, memory).
- The failure modes a reviewer should look for.

### 7. Real-World Use Cases
- Where this shows up in production ML (training, inference, data pipelines).
- Concrete systems/libraries that use it (name them, link them).

### 8. Cited References
- Papers, official docs, and reputable blogs — each with a link, attributed correctly.
- Every non-obvious claim in the module should trace to something here or in
  [REFERENCES.md](REFERENCES.md).

### 9. Self-Assessment & Interview Drills
- **Conceptual** questions (with answers) that check understanding.
- **Coding & Algorithms** drills — clean, bug-free, edge-case-aware code, no pseudo-code unless
  the drill asks for it. Include at least one GPU parallel-algorithm drill where relevant.
- Point to `exercises/` (do-it-yourself) and `solutions/` (reference).

---

## Style rules for module authors

1. **Layman-first, expert-deep.** If a smart non-specialist can't follow Section 1, rewrite it.
   If a Principal Engineer would find Section 3 shallow, deepen it.
2. **Show the naive version, then fix it.** Learners must see *why* the fast version is fast.
3. **Every perf claim is reproducible.** Give the command and the hardware it ran on.
4. **Errors are always checked** in example code (`HIP_CHECK` / `CUDA_CHECK`). No silent failure.
5. **Cite as you go.** No orphan claims.
6. **Call out AMD vs NVIDIA differences explicitly** wherever they matter.
7. **No dead abstractions.** Don't add a helper used once, or a config knob nobody sets.

---

## Visual standards

Every module README should include **at least two** Mermaid diagrams or ASCII infographics that
*teach* — not decorate. Follow the **Parallel Spectrum** palette in [BRAND.md](BRAND.md) and load
the project skill at `.cursor/skills/parallel-programming-brand/` when authoring visuals.

### Nine-section learning pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryBorderColor': '#0891B2', 'lineColor': '#64748B', 'fontFamily': 'Inter, sans-serif'}}}%%
flowchart LR
  S1["1 TL;DR"] --> S2["2 First principles"]
  S2 --> S3["3 Deep dive"]
  S3 --> S4["4 Labs"]
  S4 --> S5["5 Performance"]
  S5 --> S6["6 Tradeoffs"]
  S6 --> S7["7 Real world"]
  S7 --> S8["8 References"]
  S8 --> S9["9 Drills"]

  classDef trackA fill:#0891B2,stroke:#0F172A,color:#fff
  class S1,S2,S3,S4,S5,S6,S7,S8,S9 trackA
```

### Minimum diagram set per module

| Section | Suggested visual |
|---------|------------------|
| 2–3 | Architecture or data-flow diagram (host ↔ device, hierarchy, algorithm) |
| 5 | Roofline placement, pipeline timeline, or before/after bar chart (Mermaid or table) |
| 6 | Pitfall decision tree or failure-mode flowchart |
