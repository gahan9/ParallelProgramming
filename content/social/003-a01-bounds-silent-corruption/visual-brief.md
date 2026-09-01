# Visual brief — 003-a01-bounds-silent-corruption

| Field | Value |
|-------|-------|
| Track | A — GPU Programming |
| Accent | `#0891B2` |
| Format | 4-panel square infographic (1024×1024) or LinkedIn carousel |
| Background | Dark `#0F172A` → `#1E293B` |
| Footer | `Parallel Spectrum · Track A · A01` in `#64748B` |

## Panels

| # | Purpose | Exact on-art text | Notes |
|---|---------|-------------------|-------|
| 1 · Setup | Trap | Title: `Looks right` · Big: `PASSED` in `#10B981` · Sub: `CPU checked N = 1000` | Caption: `vector_add · block = 256` |
| 2 · Change | The launch | Title: `The extra 24` · `4 blocks × 256 = 1024 threads` · Amber `#F59E0B`: `24 out of bounds` | Four block rectangles; last block shows 24 amber cells + 232 cyan |
| 3 · Result | One line | Title: `One line` · Code: `if (idx < n)` in `#0891B2` · `24 threads skip` in `#10B981` | Before: OOB writes; after: idle/skip |
| 4 · Assess | Human check | `Would you ship the unguarded kernel?` · Sub: `Tests never see index N+1` | Muted supporting line in `#64748B` |

## GIF / carousel note

Same slate art direction across frames. If splitting into a carousel, one panel per image, identical footer. If a GIF: only the last-block cells flip amber → idle. No emoji stickers. Vendor pills optional (small AMD `#ED1C24` / NVIDIA `#76B900`) — not required; this lesson is vendor-identical.

## ASCII fallback

```
  unguarded                 guarded
 ┌──────────┐              ┌──────────┐
 │ PASSED   │     ──▶      │ PASSED   │
 │ 24 OOB   │              │ 24 skip  │
 └──────────┘              └──────────┘
  N=1000  block=256  4 blocks  1024 threads
  assess: would you ship the unguarded kernel?
```

## Generation prompt

Dark slate multi-panel tech infographic for LinkedIn, four stacked numbered sections, background gradient #0F172A to #1E293B, 4px cyan accent bar #0891B2 at top, Inter-style sans-serif, JetBrains Mono for numbers and code. Panel 1: "Looks right" with large green #10B981 text "PASSED" and subtitle "CPU checked N = 1000". Panel 2: "The extra 24" showing 4 blocks times 256 equals 1024 threads, with amber #F59E0B callout "24 out of bounds". Panel 3: "One line" with monospace "if (idx < n)" in cyan #0891B2 and "24 threads skip" in green. Panel 4: assessment question "Would you ship the unguarded kernel?" with muted caption "Tests never see index N+1". Footer text exactly "Parallel Spectrum · Track A · A01" in #64748B. Clean, educational, no purple glow, no rainbow gradients, no emoji, no stock photos.

## Source of truth

Hand-authored SVG: `visual.svg` (pure ASCII, brand tokens). Raster `visual.png` is derived from that file — edit the SVG, not the PNG.
