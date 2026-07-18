# GPU + ML Expert Tutor

> A hands-on, module-by-module curriculum that takes you from "what is a GPU thread"
> to "I can design, write, profile, and ship production ML systems on AMD and NVIDIA GPUs."

This repository is run in **expert tutor mode**. Every module is built to a single promise:

> **Explain it so a layman gets the intuition, then go deep enough that a Principal Engineer
> would nod along** — with runnable dual-vendor code, measured performance, cited research,
> and an honest account of the tradeoffs.

It is dual-track by design: examples run on **AMD (ROCm / HIP, `hipcc`, `gfx942`/MI300)** and
**NVIDIA (CUDA, `nvcc`)**, with a portable **Triton** track that runs on both. NVIDIA-only
advanced material (CuTe / CUTLASS / CuTile) is clearly marked optional.

---

## Who this is for

- Engineers preparing for **GPU kernel / ML performance / ML system-design** interviews.
- Practitioners who want first-principles depth, not copy-paste recipes.
- Anyone who wants to understand *why* a kernel is slow and *how* to make it fast — and prove it.

You need: comfort with C/C++ and Python, basic linear algebra, and curiosity. No prior GPU
experience is assumed — Module A01 starts from zero.

---

## How this repo is organized

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryBorderColor': '#0891B2', 'lineColor': '#64748B'}}}%%
flowchart TB
  ROOT["ParallelProgramming/"]

  ROOT --> README["README.md<br/>course catalog"]
  ROOT --> CURR["CURRICULUM.md<br/>index + paths"]
  ROOT --> DOCS["docs/"]
  ROOT --> TRACKS["tracks/"]

  DOCS --> MT["MODULE_TEMPLATE"]
  DOCS --> SETUP["SETUP"]
  DOCS --> REF["REFERENCES"]
  DOCS --> GLOSS["GLOSSARY"]
  DOCS --> BRAND["BRAND<br/>theme + diagrams"]

  TRACKS --> TA["A GPU Programming<br/>#0891B2"]
  TRACKS --> TB2["B ML Performance<br/>#6366F1"]
  TRACKS --> TC["C ML System Design<br/>#F59E0B"]

  classDef trackA fill:#0891B2,stroke:#0F172A,color:#fff
  classDef trackB fill:#6366F1,stroke:#0F172A,color:#fff
  classDef trackC fill:#F59E0B,stroke:#0F172A,color:#0F172A
  classDef neutral fill:#F8FAFC,stroke:#0891B2,color:#0F172A
  class TA trackA
  class TB2 trackB
  class TC trackC
  class ROOT,README,CURR,DOCS,TRACKS,MT,SETUP,REF,GLOSS,BRAND neutral
```

Visual identity (colors, Mermaid styling, slide tokens): [docs/BRAND.md](docs/BRAND.md).

Each **module** is a self-contained folder with a `README.md` (9 fixed sections), `cuda/`,
`hip/`, and `triton/` code, a `Makefile`, and `exercises/` + `solutions/`. See
[docs/MODULE_TEMPLATE.md](docs/MODULE_TEMPLATE.md) for the exact structure.

---

## The three tracks

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryBorderColor': '#0891B2', 'lineColor': '#64748B'}}}%%
flowchart LR
  subgraph A["Track A · Write kernels · #0891B2"]
    A1["CUDA + HIP"]
    A2["Triton"]
    A3["CuTe optional"]
  end

  subgraph B["Track B · Measure perf · #6366F1"]
    B1["Architecture"]
    B2["Roofline + profile"]
    B3["Serving + sharding"]
  end

  subgraph C["Track C · Ship systems · #F59E0B"]
    C1["Problem framing"]
    C2["Train + deploy"]
    C3["Monitor + capstones"]
  end

  A --> B
  B --> C
  A -.-> C

  classDef trackA fill:#0891B2,stroke:#0F172A,color:#fff
  classDef trackB fill:#6366F1,stroke:#0F172A,color:#fff
  classDef trackC fill:#F59E0B,stroke:#0F172A,color:#0F172A
  class A1,A2,A3 trackA
  class B1,B2,B3 trackB
  class C1,C2,C3 trackC
```

### Track A — GPU Programming Languages
Learn to *write* fast kernels. CUDA and HIP side by side, then Triton, then (optional) CuTe/CuTile.
Covers the programming model, memory hierarchy, the execution model, and the canonical parallel
patterns: **reduction, scan, tiled matmul, softmax, and fused attention**.

### Track B — GPU Understanding & ML Performance
Learn to *reason about* performance. Accelerator architecture (CDNA vs Hopper), the **roofline
model**, profiling with `rocprofv3` and Nsight, numeric precision, transformer performance,
**inference-serving optimizations** (continuous batching, paged attention, speculative decoding),
and **model sharding**.

### Track C — ML System Design
Learn to *architect* real systems. Problem framing and metrics, data pipelines, distributed
training, production model optimization (quantization, distillation, caching), deployment and
A/B testing, monitoring and drift, plus **end-to-end capstone case studies**.

A **Coding & Algorithms** interview thread (data structures, edge cases, clean code, plus GPU
parallel-algorithm drills) is woven into every module's Section 9 rather than living in a
separate track.

---

## Start here

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryBorderColor': '#0891B2', 'lineColor': '#64748B'}}}%%
flowchart TD
  START([New learner]) --> SETUP["1 SETUP.md<br/>pick a backend"]
  SETUP --> PATH["2 CURRICULUM.md<br/>pick a learning path"]
  PATH --> A01["3 Module A01<br/>foundations"]
  A01 --> NEXT{"Goal?"}
  NEXT -->|kernels| AK["A02 → A07 → A08"]
  NEXT -->|ML perf| BP["B01 → B06"]
  NEXT -->|systems| CS["C01 → C07"]

  classDef success fill:#10B981,stroke:#0F172A,color:#fff
  classDef neutral fill:#F8FAFC,stroke:#0891B2,color:#0F172A
  class START neutral
  class A01 success
  class SETUP,PATH,AK,BP,CS neutral
```

1. Read [docs/SETUP.md](docs/SETUP.md) and get at least one backend working (`hipcc`, `nvcc`,
   or a Triton-capable Python environment).
2. Open [CURRICULUM.md](CURRICULUM.md), pick a learning path, and check the progress tracker.
3. Begin with
   [tracks/A-gpu-programming/A01.foundations-and-programming-model/](tracks/A-gpu-programming/A01.foundations-and-programming-model/).

If you only do one module first, do **A01** — it is the fully-built gold-standard reference for
every module that follows.

---

## Parallel programming models (background primer)

The GPU is one point in a larger landscape of parallelism. Keep these in your mental model:

| Model | What it is | Where it fits here |
|---|---|---|
| **OpenMP** | Compiler-directive shared-memory multithreading (CPU, and GPU offload). | Contrast with SIMT; used in `hipcc`/`nvcc` builds via `-fopenmp`. |
| **MPI** | Message passing across distributed-memory nodes. | Foundation for multi-node training (Track B07, C03). |
| **CUDA** | NVIDIA's GPGPU platform and API. | Track A, native NVIDIA path. |
| **HIP** | Portable C++ GPU API that compiles for AMD *and* NVIDIA. | Track A, AMD path (and portability story). |
| **Triton** | Python DSL that JIT-compiles fast GPU kernels for both vendors. | Track A08+, and every module's `triton/`. |

For more, see the official homes: [OpenMP](https://www.openmp.org/),
[MPI Forum](https://www.mpi-forum.org/), [CUDA Zone](https://developer.nvidia.com/cuda-zone),
[HIP](https://github.com/ROCm/HIP), [Triton](https://triton-lang.org/).

Video companion (system-design concepts for parallel programming):
[YouTube playlist](https://youtube.com/playlist?list=PLWyBQeJgIuzD2o9ZVw5oI-P2fX4uyNKZA&si=947INnGI2lcnDbJE).

---

## Conventions

- **Dual-vendor first.** If a concept differs between AMD and NVIDIA, both are shown and the
  difference is called out (e.g. 64-lane wavefront vs 32-lane warp).
- **Evidence over assertion.** Performance claims come with a command you can run and a number
  you can reproduce — never "this is faster, trust me."
- **Errors are always checked.** Every runtime call is wrapped (`HIP_CHECK` / `CUDA_CHECK`).
  Silent failure is treated as a bug, and Module A01 explains why.
- **Cite your sources.** Claims trace to a paper, a vendor doc, or a reputable blog. The master
  list lives in [docs/REFERENCES.md](docs/REFERENCES.md).

---

## License

See [LICENSE](LICENSE).
