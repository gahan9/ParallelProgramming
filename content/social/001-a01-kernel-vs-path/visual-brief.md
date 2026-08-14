# Visual brief — 001-a01-kernel-vs-path

| Field | Value |
|-------|-------|
| Track | A — GPU Programming |
| Accent | `#0891B2` |
| Format | 4-panel carousel (or 3-frame GIF) |
| Background | Dark `#0F172A` → `#1E293B` |
| Footer | `Parallel Spectrum · Track A · A01` in `#64748B` |

## Panels

| # | Purpose | Exact on-art text | Notes |
|---|---------|-------------------|-------|
| 1 · Setup | Problem | Title: `Kernel looks fine` · Big number: `44 µs` in `#10B981` | Light caption: `vector_add · n ≈ 16.7M` |
| 2 · Change | Insight | Title: `Where time went` · Horizontal bars: Alloc ~68ms · H2D ~2.4ms · Kernel ~0.044ms · D2H ~1.2ms | Kernel bar is a sliver; accent cyan `#0891B2` |
| 3 · Result | Multi-result | Left: `Kernel-only` `44 µs` · Right: `End-to-end` `~72 ms` · Badge: `0.06%` in `#F59E0B` | Before/after methodology, not vanity speedup |
| 4 · Assess | Human check | `Is PCIe in your story?` · Sub: `Measure the path — not just the kernel` | Muted `#64748B` supporting line |

## GIF / carousel note

Same slate art direction across frames. Only the result region changes: panel 1 number → panel 2 bars fill → panel 3 split compare → panel 4 question. No emoji stickers. Vendor pills only if needed (optional small AMD `#ED1C24` for "HIP / rocprofv3").

## ASCII fallback

```
  kernel-only              end-to-end path
 ┌──────────────┐         ┌──────────────┐
 │    44 µs     │  ──▶    │   ~72 ms     │
 │  vector_add  │         │ alloc+PCIe+k │
 └──────────────┘         └──────────────┘
  share of wall: ~0.06%
  assess: is PCIe in your story?
```

## Generation prompt

Dark slate multi-panel tech infographic for LinkedIn, four horizontal sections, background gradient #0F172A to #1E293B, 4px cyan accent bar #0891B2 at top, Inter-style sans-serif, monospace timings. Panel 1: "Kernel looks fine" with large green #10B981 text "44 µs". Panel 2: labeled horizontal time bars showing allocation dominating, tiny kernel sliver. Panel 3: split "Kernel-only 44 µs" vs "End-to-end ~72 ms" with amber #F59E0B badge "0.06%". Panel 4: assessment question "Is PCIe in your story?" Footer text exactly "Parallel Spectrum · Track A · A01" in #64748B. Clean, educational, no purple glow, no rainbow gradients, no emoji, no stock photos of people.
