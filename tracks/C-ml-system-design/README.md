# Track C — ML System Design

Learn to **architect** real, end-to-end ML systems that survive production: framing the problem,
building data pipelines, training at scale, optimizing for serving, deploying safely, and
monitoring for drift and fairness. Emphasis on tradeoffs, scalability, and clear communication.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#F8FAFC', 'primaryBorderColor': '#F59E0B', 'lineColor': '#64748B'}}}%%
flowchart LR
  C01["C01 Framing"] --> C02["C02 Data"]
  C02 --> C03["C03 Training"]
  C03 --> C04["C04 Optimization"]
  C04 --> C05["C05 Deploy"]
  C05 --> C06["C06 Monitor"]
  C06 --> C07["C07 Capstones"]

  classDef trackC fill:#F59E0B,stroke:#0F172A,color:#0F172A
  class C01,C02,C03,C04,C05,C06,C07 trackC
```

## Modules

| ID | Module | Status |
|----|--------|--------|
| C01 | Problem framing, metrics & decomposition | planned |
| C02 | Data pipelines & feature engineering | planned |
| C03 | Training systems (distributed, fault tolerance) | planned |
| C04 | Production model optimization | planned |
| C05 | Deployment & serving architecture | planned |
| C06 | Monitoring, drift, feedback loops, ethics | planned |
| C07 | Capstone case studies (LLM serving; recommender) | planned |

These modules are design-doc heavy (some have no GPU code). See
[CURRICULUM.md](../../CURRICULUM.md) for the recommended path and interview-skill mapping.

Modules are built just-in-time; folders appear as they are authored.
