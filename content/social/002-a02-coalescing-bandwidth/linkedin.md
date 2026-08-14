# LinkedIn — Post 002

## ContentBrief: a02-coalescing-bandwidth
- Goal: reach + saves (comments secondary via judgment question)
- Platforms: LinkedIn (primary) · Medium · YouTube Shorts
- Audience: 22–40 / mixed — GPU learners + PyTorch practitioners
- Core promise: A kernel can PASS and still waste most of your bandwidth; access pattern, not math, moved ~700 → ~3,400 GB/s.
- Mode: offline-first hybrid
- Constraints: assessment voice (not hype); no external link in body (first comment); 3–5 hashtags; Track A cyan visual; callback to 001

## Engagement autopsy → fixes (from Post 001)

| What hurt 001 (~1.5k impressions / 9 reactions) | Fix in 002 |
|---|---|
| External Medium link in the body (algorithm demotes outbound) | Body has **no URL**; Medium + lab path go in **first comment** |
| Dense jargon (`rocprofv3`, `hipMalloc`) without a plain gloss | Open with "PASSED but slow"; gloss *coalescing* in five words |
| Text-heavy, low comment invite | End with one judgment question (not "comment YES") |
| Likely weak / missing native visual | Attach **4-panel carousel** (`visual.png`) — native docs beat link posts |
| Narrow "profiler" framing | Bridge to PyTorch `.contiguous()` so ML engineers recognize the pain |
| Hook buried after two timing lines | Pain-point hook in line 1 that stands alone in the preview |

---

## Copy-paste (primary post)

This kernel printed PASSED.
It still used only ~15% of the GPU's memory bandwidth.

Last post: the kernel was ~0.06% of the run — the *path* was the story.
This time the path is fine. The *reads and writes* decide the speed.

Same matrix transpose. Same answers. Zero change to the math.

On a reference run (MI300X, n=4096):
→ naive (strided writes): ~700 GB/s
→ tiled through on-chip shared memory: ~2,600 GB/s
→ pad the tile by one column: ~3,400 GB/s

~5×. Because neighboring threads grabbed neighboring addresses — coalescing — so the memory system sent full trucks instead of nearly empty ones.

If you write PyTorch: this is why `.contiguous()` exists. A view can look transposed while the bytes still sit in the old order. Adjacent threads then land far apart. You pay once to reorder, or you pay on every access.

Honest check before I'd ship the naive version:
It passes every correctness test. Correctness will never catch this. Would you?

Save this if you profile kernels that "look fine."
First comment: full write-up + the three kernels + how to reproduce.

Series: Parallel Spectrum · Post 002 · Track A
Prev: measure the path, not just the kernel (A01)

#GPUProgramming #CUDA #ROCm #PerformanceEngineering #ParallelComputing

---

## First comment (post immediately)

Full write-up (table, snippet, reproduce steps) — paste Medium URL here after publish:
`TBD-verify — Medium URL for 002`

Lab (HIP + CUDA + Triton):
`tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/`
`make hip && make run-hip`  → watch naive << tiled < tiled+padded

Numbers are from a documented MI300X reference run. Re-run on your GPU and quote *your* GB/s — the ratios are the lesson.

Prev post (kernel 44 µs / program ~72 ms):
https://gahan9.medium.com/your-gpu-kernel-took-44-microseconds-the-program-took-72-milliseconds-09c5e3b920c1

---

## Hook variants (kept for A/B)

| ID | Type | Line |
|----|------|------|
| A | Curiosity | "Same math. Zero FLOPs changed. ~5× more bandwidth." |
| B | Pain-point | "This kernel printed PASSED. It still used only ~15% of GPU bandwidth." ← **ship this** |
| C | Promise | "One shared-memory tile (and a +1 pad) took transpose from ~700 to ~3,400 GB/s." |

Recommended: **B** — names the trap every engineer has shipped; preview-safe without jargon.

---

## Posting notes

- Attach `visual.png` as a **carousel / multi-image** (panels 1→4), not a single cropped thumbnail.
- Do **not** put the Medium URL in the primary post if reach is the goal this week.
- Pin the first comment.
- Best window: Tue–Thu, 9–11 AM audience timezone.
- After publish: fill Publish log in `../HISTORY.md` with the LinkedIn URL.
