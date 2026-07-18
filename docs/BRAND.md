# Brand & Visual Identity — Parallel Spectrum

This document is the single source of truth for colors, diagrams, and slide styling across the
**GPU + ML Expert Tutor** curriculum. AI agents should load the
[`.cursor/skills/parallel-programming-brand/`](../.cursor/skills/parallel-programming-brand/SKILL.md)
skill when authoring visual material.

---

## Design intent

| Principle | Meaning |
|-----------|---------|
| **Throughput cyan** (`#0891B2`) | Parallel lanes, data movement, Track A |
| **Compute indigo** (`#6366F1`) | Deep analysis, profiling, Track B |
| **Systems amber** (`#F59E0B`) | Tradeoffs, production design, Track C |
| **Slate foundation** (`#0F172A` / `#F8FAFC`) | Readable docs and slides for long study sessions |

Vendor colors (AMD red, NVIDIA green) appear **only** as small labels — never as the primary brand.

---

## Color tokens

| Token | Hex | RGB | Usage |
|-------|-----|-----|-------|
| `brand-primary` | `#0891B2` | 8, 145, 178 | Links, Track A, primary diagrams |
| `brand-deep` | `#0F172A` | 15, 23, 42 | Dark mode, code slide backgrounds |
| `brand-accent` | `#6366F1` | 99, 102, 241 | Track B, CTAs, emphasis |
| `brand-highlight` | `#F59E0B` | 245, 158, 11 | Track C, warnings, interview drills |
| `brand-success` | `#10B981` | 16, 185, 129 | DONE, PASSED, verified |
| `brand-surface` | `#F8FAFC` | 248, 250, 252 | Light backgrounds |
| `brand-muted` | `#64748B` | 100, 116, 139 | Captions, footers |
| `vendor-amd` | `#ED1C24` | 237, 28, 36 | AMD callout pills |
| `vendor-nvidia` | `#76B900` | 118, 185, 0 | NVIDIA callout pills |

### Semantic status colors

| Status | Hex | Example |
|--------|-----|---------|
| DONE | `#10B981` | Module A01 |
| DRAFT | `#F59E0B` | Partial modules |
| PLANNED | `#64748B` | Future modules |

---

## Mermaid theme

Paste this init block at the top of **every** Mermaid diagram in this repo:

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'primaryColor': '#F8FAFC',
    'primaryTextColor': '#0F172A',
    'primaryBorderColor': '#0891B2',
    'lineColor': '#64748B',
    'secondaryColor': '#E0F2FE',
    'tertiaryColor': '#EEF2FF',
    'fontFamily': 'Inter, Segoe UI, sans-serif'
  }
}}%%
```

### Reusable class definitions

Append after your diagram nodes:

```
classDef trackA fill:#0891B2,stroke:#0F172A,color:#fff
classDef trackB fill:#6366F1,stroke:#0F172A,color:#fff
classDef trackC fill:#F59E0B,stroke:#0F172A,color:#0F172A
classDef neutral fill:#F8FAFC,stroke:#0891B2,color:#0F172A
classDef success fill:#10B981,stroke:#0F172A,color:#fff
classDef warn fill:#FEF3C7,stroke:#F59E0B,color:#0F172A
classDef amd fill:#FEE2E2,stroke:#ED1C24,color:#0F172A
classDef nvidia fill:#ECFCCB,stroke:#76B900,color:#0F172A
```

---

## Slide deck template

For PowerPoint, Canvas, or AI-generated slide images:

```
┌─────────────────────────────────────────────────────────────┐
│ ████  Module A01 — Foundations                    [Track A] │  ← 4px #0891B2 bar
│                                                             │
│  Title (32pt, #0F172A)                                      │
│  Subtitle (18pt, #64748B)                                   │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐ │
│  │  Body / bullets     │  │  Diagram or code snippet    │ │
│  │  (#334155)          │  │  (monospace on #0F172A)     │ │
│  └─────────────────────┘  └─────────────────────────────┘ │
│                                                             │
│  Parallel Programming · A01 · Foundations          10pt    │
└─────────────────────────────────────────────────────────────┘
```

**Dark deck variant:** background `#0F172A`, title `#F8FAFC`, accent bar `#0891B2`.

---

## Typography

| Role | Font stack |
|------|------------|
| Headings | Inter, Segoe UI, system-ui |
| Body | Inter, Segoe UI, system-ui |
| Code | JetBrains Mono, Cascadia Code, Consolas |

---

## Where visuals live

| Doc | Minimum visuals |
|-----|-----------------|
| `README.md` | Repo map + learning-path flow |
| `CURRICULUM.md` | Dependency graph + path selector |
| Module README | Host/device flow + one concept diagram |
| `docs/SETUP.md` | Backend decision tree |
| `docs/MODULE_TEMPLATE.md` | Nine-section pipeline |

See [MODULE_TEMPLATE.md](MODULE_TEMPLATE.md) Section "Visual standards" for author rules.
