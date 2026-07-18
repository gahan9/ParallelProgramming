# Track A — GPU Programming Languages

Learn to **write** fast GPU kernels, CUDA and HIP side by side, then Triton, then (optionally)
CuTe/CuTile. You will build the canonical parallel patterns from scratch and understand why the
fast version is fast.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryBorderColor': '#0891B2', 'lineColor': '#64748B'}}}%%
flowchart LR
  A01["A01 Foundations<br/>DONE"] --> A02["A02 Memory"]
  A02 --> A03["A03 Execution"]
  A03 --> A04["A04 Reduction"]
  A04 --> A06["A06 Matmul"]
  A06 --> A07["A07 Softmax"]
  A03 --> A08["A08 Triton"]

  classDef trackA fill:#0891B2,stroke:#0F172A,color:#fff
  classDef success fill:#10B981,stroke:#0F172A,color:#fff
  classDef neutral fill:#F8FAFC,stroke:#64748B,color:#0F172A
  class A01 success
  class A02,A03,A04,A06,A07,A08 trackA
```

## Modules

| ID | Module | Status |
|----|--------|--------|
| [A01](A01.foundations-and-programming-model/) | Foundations & programming model | **DONE** |
| A02 | Memory hierarchy & coalescing | planned |
| A03 | Execution model: warps/wavefronts, occupancy, divergence | planned |
| A04 | Parallel reduction (7-stage optimization) | planned |
| A05 | Parallel scan / prefix sum | planned |
| A06 | Tiled matmul | planned |
| A07 | Softmax & fused kernels | planned |
| A08 | Triton foundations | planned |
| A09 | Advanced Triton (FlashAttention-style) | planned |
| A10 | CuTe / CUTLASS / CuTile (NVIDIA-only, optional) | planned |

See the top-level [CURRICULUM.md](../../CURRICULUM.md) for the dependency graph and learning paths.

## Legacy raw examples

The folders `01.hello_world/`, `02.vector_add/`, and `03.hip_directives/` are the original
hand-written examples this repository started from. They are kept as raw reference material and are
being progressively refactored into the structured module format. **Start with `A01`**, which
absorbs and upgrades the hello-world and vector-add material into the full 9-section template
(with error checking, a CUDA path, a Triton path, profiling, and drills).
