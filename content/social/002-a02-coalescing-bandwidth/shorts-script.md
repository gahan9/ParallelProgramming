# Shorts script — Post 002 (A02 coalescing)

Aligned with `content/A02-memory-coalescing/shorts.md` Short 2 — the LinkedIn story cut.
~55 s · 9:16 · burned-in captions · no title card.

| t | On-screen (≤6 words) | Voiceover |
|---|----------------------|-----------|
| 0–3s | `PASSED` · `~700 GB/s` | "This kernel passes every test — and wastes most of your GPU." |
| 3–12s | `Transpose · strided writes` | "It's a matrix transpose. Reading rows is fine. Writing columns scatters every thread across memory." |
| 12–28s | `1 truck` vs `32 trucks` | "The GPU fetches whole cache lines — full trucks. Adjacent addresses: one truck. Column writes: almost empty trucks, over and over." |
| 28–42s | `~700 → ~3400 GB/s` | "Stage the tile in shared memory, pad by one column. Seven hundred to thirty-four hundred gigabytes a second. Same answers. No math changed." |
| 42–52s | `Correctness won't catch this` | "I'd verify bandwidth against a coalesced copy ceiling — not just PASSED." |
| 52–58s | `Write-up in description` | "Full ladder and lab linked below. Parallel Spectrum, post two." |

**On-screen sequence:** `PASSED ✓` → `15% of peak` → `strided writes` → `~5× after tiling+pad`

**Title (YouTube):** `GPU kernel PASSED but only 15% bandwidth — coalescing fix`

**Pinned comment:** Medium URL + `tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/`
