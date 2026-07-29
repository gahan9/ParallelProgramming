# Brand overlay spec — on-screen styling for video

Every video in this series must look like the same series. These are the exact values to set once in
your editor (Premiere / Resolve / CapCut) and save as presets. Tokens come from
[`../docs/BRAND.md`](../docs/BRAND.md) — do not invent new colors.

---

## Palette on screen

| Element | Color | Hex |
|---|---|---|
| Background / terminal | Slate deep | `#0F172A` |
| Primary accent (titles, underlines, highlights) | Throughput cyan | `#0891B2` |
| Secondary accent (callouts) | Compute indigo | `#6366F1` |
| Warning / "gotcha" moments | Systems amber | `#F59E0B` |
| Success / PASSED / speedup | Green | `#10B981` |
| Body text on dark | Light slate | `#F8FAFC` |
| Captions / muted metadata | Muted slate | `#64748B` |
| AMD label pill | AMD red | `#ED1C24` |
| NVIDIA label pill | NVIDIA green | `#76B900` |

Vendor colors appear **only** as small pills when a shot is vendor-specific. Never as a background.

---

## Title card (long-form, 0:00–0:03 max)

```
┌──────────────────────────────────────────────┐
│ ████                                         │  4px #0891B2 bar, full width
│                                              │
│   PARALLEL PROGRAMMING              (12pt)   │  #64748B, letter-spaced
│   Why your kernel is 8x slow        (48pt)   │  #F8FAFC, Inter Bold
│   Module A02 · Memory Coalescing    (16pt)   │  #0891B2
│                                              │
└──────────────────────────────────────────────┘
```

Background `#0F172A`. No animation beyond a 200 ms fade — animated logo stings destroy retention.
**Shorts get no title card at all.**

## Lower third (speaker / topic label)

- Position: bottom-left, 8% margin.
- Bar: 3px `#0891B2`, 180px wide, above the text.
- Line 1: topic, 20pt `#F8FAFC`.
- Line 2: module ID, 14pt `#64748B`.
- Duration: 4 s, fade 150 ms.

## Code on screen

| Setting | Value |
|---|---|
| Font | JetBrains Mono / Cascadia Code |
| Size (16:9) | ≥ 16pt at 1080p |
| Size (9:16 Shorts) | ≥ 28pt — assume a phone held at arm's length |
| Background | `#0F172A` |
| Keyword accent | `#0891B2` |
| Highlight box for the line that matters | 2px `#F59E0B` border, no fill |
| Max lines visible at once | 12 (16:9) · 6 (Shorts) |

Zoom to the lines under discussion. A full file on screen is unreadable and signals to the viewer
that nothing specific is happening.

## Numbers and metrics

When a measured result appears:

- Value in `#10B981` if it is an improvement, `#F59E0B` if it is a problem.
- Always adjacent: **the unit** (GB/s, ms, GFLOP/s) and **the GPU name**.
- Format: `1,842 GB/s — MI300X (gfx942)`. Unattributed numbers are the fastest way to lose credibility.

## Captions

- Shorts: **burned in**, always. 28pt, `#F8FAFC`, 60% `#0F172A` backing, bottom third, 2 lines max.
- Long-form: uploaded SRT. Do not burn in — it blocks code.

---

## Shots that work on camera

| Shot | Why it works |
|---|---|
| Terminal running the lab, output scrolling | Proof it is real |
| Side-by-side code diff (naive vs optimized) | The idea in one frame |
| Mermaid diagram from the module README | Already brand-styled, zero extra work |
| Curriculum dashboard canvas (ticking a module off) | Shows structure and progress |
| `rocm-smi` / `nvidia-smi` once per video | Establishes real hardware |
| Profiler timeline (`rocprofv3` / Nsight) | Makes "memory-bound" visible instead of abstract |

## Shots to avoid

- Reading a README aloud on screen — write a script instead.
- Full-file scrolling with no zoom.
- A claimed speedup with no terminal output behind it.
- Talking head for more than ~20 s without something changing on screen.
