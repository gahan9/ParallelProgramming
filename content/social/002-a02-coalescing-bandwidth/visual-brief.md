# Visual brief — 002-a02-coalescing-bandwidth

| Field | Value |
|-------|-------|
| Track | A — GPU Programming |
| Accent | `#0891B2` |
| Format | 4-panel carousel (LinkedIn multi-image) or stacked PNG |
| Background | Dark `#0F172A` → `#1E293B` |
| Footer | `Parallel Spectrum · Track A · A02` in `#64748B` |

## Panels

| # | Purpose | Exact on-art text | Notes |
|---|---------|-------------------|-------|
| 1 · Setup | Trap | Title: `PASSED — still slow` · Big: `~700 GB/s` in `#F59E0B` · Sub: `~15% of copy ceiling` | Caption: `transpose_naive · n=4096` |
| 2 · Change | Insight | Title: `Addresses, not math` · Left: `strided writes` (scattered dots) · Right: `coalesced` (one full bar) | Truck / shelf metaphor OK as simple icons; no emoji stickers |
| 3 · Result | Ladder | Three bars: `~700` amber → `~2600` cyan → `~3400` green `#10B981` · Badge: `~5×` | Labels: `naive` / `tiled` / `tiled+pad` |
| 4 · Assess | Human check | `Would you ship the naive kernel?` · Sub: `Correctness never catches this` | Muted supporting line in `#64748B` |

## GIF / carousel note

Same slate art direction across frames. Animate only the GB/s bars filling left→right on panel 3 if making a GIF. Vendor pills optional (small AMD `#ED1C24` / NVIDIA `#76B900`) — not required.

## ASCII fallback

```
  naive                 tiled+padded
 ┌──────────┐          ┌──────────┐
 │ ~700 GB/s│  ──▶     │~3400 GB/s│
 │ 15% peak │          │ ~70% peak│
 └──────────┘          └──────────┘
  assess: would you ship the naive kernel?
```

## Generation prompt

Dark slate multi-panel tech infographic for LinkedIn, four horizontal sections, background gradient #0F172A to #1E293B, 4px cyan accent bar #0891B2 at top, Inter-style sans-serif, monospace for GB/s numbers. Panel 1: "PASSED — still slow" with large amber #F59E0B text "~700 GB/s" and subtitle "~15% of copy ceiling". Panel 2: "Addresses, not math" comparing strided vs coalesced access as simple geometric dots vs a solid cyan bar. Panel 3: three horizontal bandwidth bars labeled naive ~700, tiled ~2600, tiled+pad ~3400 with green #10B981 on the best bar and amber badge "~5x". Panel 4: assessment question "Would you ship the naive kernel?" with muted caption "Correctness never catches this". Footer text exactly "Parallel Spectrum · Track A · A02" in #64748B. Clean, educational, no purple glow, no rainbow gradients, no emoji, no stock photos.
