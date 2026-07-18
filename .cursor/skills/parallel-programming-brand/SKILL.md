---
name: parallel-programming-brand
description: Applies the Parallel Programming curriculum visual identity — theme colors, Mermaid styling, ASCII infographics, and slide design tokens. Use when creating or editing docs, diagrams, slides, canvases, or any visual material for this repository.
---

# Parallel Programming — Brand & Visual Identity

## When to apply

Use this skill whenever you:
- Add or edit Mermaid diagrams in markdown
- Create slides, canvases, or infographics for this curriculum
- Choose colors for tables, callouts, or architecture diagrams
- Author a new module README or shared doc

**Always read** [docs/BRAND.md](../../docs/BRAND.md) for the full token table and copy-paste Mermaid init blocks.

---

## Brand name

**Parallel Spectrum** — throughput (cyan) meets compute depth (indigo) on a slate foundation.

---

## Core palette (use these hex values verbatim)

| Token | Hex | Use |
|-------|-----|-----|
| `brand-primary` | `#0891B2` | Headings, primary links, Track A accent |
| `brand-deep` | `#0F172A` | Dark backgrounds, slide titles on dark decks |
| `brand-accent` | `#6366F1` | CTAs, key callouts, Track B accent |
| `brand-highlight` | `#F59E0B` | Warnings, interview tips, Track C accent |
| `brand-success` | `#10B981` | DONE status, passed tests, verification |
| `brand-surface` | `#F8FAFC` | Light slide/doc background |
| `brand-muted` | `#64748B` | Secondary text, captions |
| `vendor-amd` | `#ED1C24` | AMD-specific labels only (sparingly) |
| `vendor-nvidia` | `#76B900` | NVIDIA-specific labels only (sparingly) |

### Track colors

| Track | Color | Hex |
|-------|-------|-----|
| A — GPU Programming | Cyan | `#0891B2` |
| B — ML Performance | Indigo | `#6366F1` |
| C — ML System Design | Amber | `#F59E0B` |

---

## Mermaid rules

1. **Every new diagram** starts with the standard init block from `docs/BRAND.md` (section "Mermaid theme").
2. Prefer **flowchart**, **sequenceDiagram**, and **block-beta** for architecture; use **mindmap** for concept maps.
3. Color nodes by track when the content is track-specific; use `brand-primary` for neutral GPU concepts.
4. Keep diagrams **load-bearing** — they should teach, not decorate. Pair each diagram with one sentence of interpretation.
5. GitHub renders Mermaid natively; avoid custom CSS or HTML in markdown.

### Node styling shorthand

In Mermaid classDef blocks, map to brand tokens:

```
classDef trackA fill:#0891B2,stroke:#0F172A,color:#fff
classDef trackB fill:#6366F1,stroke:#0F172A,color:#fff
classDef trackC fill:#F59E0B,stroke:#0F172A,color:#0F172A
classDef neutral fill:#F8FAFC,stroke:#0891B2,color:#0F172A
classDef success fill:#10B981,stroke:#0F172A,color:#fff
classDef warn fill:#FEF3C7,stroke:#F59E0B,color:#0F172A
```

---

## Slide rules (PowerPoint, Canvas, or image slides)

When generating slides for this project:

| Element | Spec |
|---------|------|
| Background (light deck) | `#F8FAFC` |
| Background (dark deck) | `#0F172A` gradient to `#1E293B` |
| Title text | `#0F172A` (light) or `#F8FAFC` (dark) |
| Body text | `#334155` (light) or `#CBD5E1` (dark) |
| Accent bar / underline | `#0891B2` 4px |
| Code blocks on slides | `#0F172A` background, `#0891B2` keywords |
| Footer | `Parallel Programming · Track X · Module Y` in `#64748B` 10pt |

**Typography:** Prefer **Inter**, **Segoe UI**, or **Source Sans 3**. Monospace: **JetBrains Mono** or **Cascadia Code**.

**Layout:** One idea per slide. Left-aligned title + accent bar. Diagrams right or full-bleed below title.

**Vendor callouts:** Use small colored pills — AMD `#ED1C24`, NVIDIA `#76B900` — never as slide backgrounds.

---

## ASCII infographic patterns

Use for quick inline visuals in README sections where Mermaid is heavy:

```
 Host (CPU)          PCIe           Device (GPU)
┌──────────┐    ┌──────────┐    ┌──────────────┐
│  malloc  │───▶│  H2D copy│───▶│  hipMalloc   │
│  launch  │    │  ~50 GB/s│    │  kernel <<<>>>│
└──────────┘    └──────────┘    └──────────────┘
     ▲                                  │
     └──────── D2H copy ◀───────────────┘
```

Keep box widths consistent; use `▶`/`◀` for data flow; label bandwidth or latency when teaching performance.

---

## Module README checklist

Each module should include **at least two** of:
- One architecture / flow Mermaid diagram (Sections 2–3)
- One performance or pipeline diagram (Section 5)
- One pitfall / tradeoff flowchart (Section 6)

Reference module A01 as the gold standard.

---

## Do not

- Invent new brand colors outside the token table
- Use rainbow gradients or vendor colors as primary branding
- Add diagrams without explanatory prose
- Create slides with more than ~40 words of body text per slide
