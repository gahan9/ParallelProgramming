# Example pack — Host↔Device data path (Foundations tone)

Illustrative pack aligned to Track A / module foundations style. Swap numbers for your real profile run.

---

## 1. LinkedIn

Most "GPU hello world" posts stop at `vectorAdd` succeeding.

I timed the boring part: host alloc → H2D → kernel → D2H.

On a small N, the kernel looked "fast." The copies ate the story.

Honest check before I'd demo this in an interview:
→ Is PCIe time in the trace, or am I only admiring the kernel timer?

Pursue the lab: `tracks/A-gpu-programming/A01.foundations-and-programming-model/`
Run the HIP path, then answer: where did the milliseconds actually go?

#GPUProgramming #HIP #CUDA #ParallelSpectrum #Performance

---

## 2. Shorts script (~55s)

| t | On-screen | Voiceover |
|---|-----------|-----------|
| 0–3s | `Kernel 0.2 ms` | "Green kernel. Still slow." |
| 3–15s | `Where did time go?` | "Beginners time the kernel. Pros time the path." |
| 15–40s | `H2D · kernel · D2H` | "Watch the copy bars. On small N, they dominate." |
| 40–55s | `Measure the full path` | "I'd verify end-to-end before calling it a win." |
| 55–60s | `Lab in description` | "Full write-up and snippet linked below." |

---

## 3. Blog stub (Medium / GitHub)

# Don't trust the kernel timer alone

**TL;DR:** A correct `vectorAdd` can still be a bad demo if you only measure device time. This note walks a minimal HIP/CUDA path, shows where to look in the timeline, and what I'd assess before teaching the result.

## What we measured

End-to-end host→device→kernel→device→host on a foundations lab. Kernel alone looked excellent; the full path told a different story at small problem sizes.

## The snippet

```cpp
// Teaching core only — see module for full error checks
hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);
vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);
hipDeviceSynchronize();
```

## Reproduce

```bash
cd tracks/A-gpu-programming/A01.foundations-and-programming-model
# follow module Makefile / README for hip or cuda target
```

## Honest assessment

- **Watch for:** reporting kernel-only ms as "GPU performance."
- **When not to:** microbenchmarking launch overhead with huge N (different lesson).
- **I'd verify:** a timeline that includes H2D/D2H, not just `hipEvent` around the kernel.

## Links

- Module README under A01
- Brand visuals: Parallel Spectrum cyan `#0891B2` for Track A diagrams

---

## 4. Code pursue

- Path: `tracks/A-gpu-programming/A01.foundations-and-programming-model/`
- Stretch: overlap a copy with a kernel using streams; re-assess the timeline.

---

## 5. Visual brief — multi-result

**Format:** 4-panel carousel (or 3-frame GIF: naive timer → full path → assess)

| Frame | Content |
|-------|---------|
| 1 | Title "Kernel looks fine" · big `0.2 ms` on `#0F172A` |
| 2 | Bars: H2D / Kernel / D2H — kernel tiny, copies large · accent `#0891B2` |
| 3 | Before/after framing: "Kernel-only" vs "End-to-end" · success check on methodology not vanity ms |
| 4 | Assess: "Is PCIe in your story?" · footer `Parallel Spectrum · Track A · A01` |

**Generation prompt sketch:** Dark slate infographic, four horizontal panels, cyan `#0891B2` accent bar, Inter-style sans, monospace for timings, no logos except footer text "Parallel Spectrum · Track A · A01", no purple glow, clean bars comparing H2D vs kernel vs D2H.
