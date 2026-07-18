# Track B — GPU Understanding & ML Performance

Learn to **reason about** performance: what makes a kernel or a model fast or slow, and how to
prove it with measurements. This track turns you from someone who writes kernels into someone who
can diagnose and defend performance decisions.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryBorderColor': '#6366F1', 'lineColor': '#64748B'}}}%%
flowchart TD
  B01["B01 Architecture"] --> B02["B02 Roofline"]
  B02 --> B03["B03 Profiling"]
  B03 --> B04["B04 Numerics"]
  B04 --> B05["B05 Transformers"]
  B05 --> B06["B06 Serving"]
  B01 --> B07["B07 Sharding"]

  classDef trackB fill:#6366F1,stroke:#0F172A,color:#fff
  class B01,B02,B03,B04,B05,B06,B07 trackB
```

## Modules

| ID | Module | Status |
|----|--------|--------|
| B01 | Accelerator architecture (CDNA vs Hopper) | planned |
| B02 | Roofline model & arithmetic intensity | planned |
| B03 | Profiling (`rocprofv3` / Nsight) | planned |
| B04 | Precision & numerics (fp32→fp8, quantization) | planned |
| B05 | Transformer architecture from a perf lens | planned |
| B06 | Inference serving optimizations | planned |
| B07 | Model sharding & distributed | planned |

Prerequisite: Track A01 for GPU literacy. See [CURRICULUM.md](../../CURRICULUM.md) for details.

Modules are built just-in-time; folders appear as they are authored.
