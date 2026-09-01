# LinkedIn — Post 003

## ContentBrief: a01-bounds-silent-corruption
- Goal: reach + saves (comments secondary via judgment question)
- Platforms: LinkedIn (primary) · Medium · YouTube Shorts
- Audience: 22–40 / mixed — GPU learners + people writing custom CUDA/HIP/Triton kernels
- Core promise: A kernel can print the right answer while extra threads write past the end of the array. N=1000, block=256 → 4 blocks, 1,024 threads, 24 out of bounds. One line fixes it: `if (idx < n)`.
- Mode: offline-first hybrid
- Constraints: assessment voice; no external link in body (first comment); 3–5 hashtags; Track A cyan visual; callback to 002 (or 001 if 002 is still unpublished); numbers are ceil-division arithmetic from A01 §9, not a GPU profile

## Engagement autopsy → carry-forward (from Posts 001–002)

| What 001 taught | What 002 applied | Keep in 003 |
|---|---|---|
| Outbound URL in body hurt reach | URL in first comment | Same |
| Jargon without a five-word gloss | Opened in plain words | Gloss *bounds guard*: "skip threads past the array" |
| Weak comment invite | Judgment question | Same |
| Native visual beats a link post | 4-panel carousel | Attach `visual.png` |
| ML-engineer bridge | `.contiguous()` | Triton `mask = offsets < n` |
| Hook must stand alone in the preview | Pain-point line 1 | Ship hook B |

**Queue note:** 002 is `ready` but not yet in the Publish log. If 002 has not gone live, use the **001-callback variant** below so "last post" is not a lie.

---

## Copy-paste (primary post — callback to 002)

This kernel can print the right answer.
24 threads still wrote past the end of the array.

Last post: a PASSED kernel wasted most of the bandwidth.
This time the tests are the trap. They only check the first N elements.

N = 1,000. Block size = 256.
You round up and launch 4 blocks.
That is 1,024 threads. 24 of them have no element to touch.

Without one line — if (idx < n) — those 24 read and write memory that is not yours.
Sometimes it crashes. Often it does not. Next week's bug lives in a buffer you never looked at.

If you write Triton: this is why mask = offsets < n exists. The compiler does not add a bounds guard for you.

Honest check before I'd ship the unguarded kernel:
Your CPU reference compared N floats. The damage is past index N. Would you?

Save this if you review GPU kernels.
First comment: the drill, the one-line fix, how to reproduce.

Series: Parallel Spectrum · Post 003 · Track A
Prev: coalescing — PASSED at ~15% bandwidth (A02)

#GPUProgramming #CUDA #ROCm #PerformanceEngineering #ParallelComputing

---

## Copy-paste (001-callback variant — use if 002 is still unpublished)

This kernel can print the right answer.
24 threads still wrote past the end of the array.

Last time the kernel was ~0.06% of the run — the *path* was the story.
This time the answer looks right because the tests never see the damage.

N = 1,000. Block size = 256.
You round up and launch 4 blocks.
That is 1,024 threads. 24 of them have no element to touch.

Without one line — if (idx < n) — those 24 read and write memory that is not yours.
Sometimes it crashes. Often it does not. Next week's bug lives in a buffer you never looked at.

If you write Triton: this is why mask = offsets < n exists. The compiler does not add a bounds guard for you.

Honest check before I'd ship the unguarded kernel:
Your CPU reference compared N floats. The damage is past index N. Would you?

Save this if you review GPU kernels.
First comment: the drill, the one-line fix, how to reproduce.

Series: Parallel Spectrum · Post 003 · Track A
Prev: measure the path, not just the kernel (A01)

#GPUProgramming #CUDA #ROCm #PerformanceEngineering #ParallelComputing

---

## First comment (post immediately)

Full write-up (the arithmetic, snippet, reproduce steps) — paste Medium URL here after publish:
`TBD-verify — Medium URL for 003`

The drill is whiteboard-able. No GPU required for the 24-thread count:

N = 1000, block = 256
blocks = ceil(1000 / 256) = 4
threads = 4 × 256 = 1024
out of bounds = 1024 − 1000 = 24

Lab (fill in the guard yourself):
`tracks/A-gpu-programming/A01.foundations-and-programming-model/exercises/01_saxpy.cpp`
`cd …/exercises && make 01_saxpy.exe GPU_ARCH=<your-arch> && ./01_saxpy.exe`
Expect: SAXPY PASSED (0 errors) once the kernel has `if (idx < n)`.

HIP / CUDA twins of the same guard: `…/A01.foundations-and-programming-model/hip/vector_add.cpp`
Triton spelling: `mask = offsets < n` in `triton/vector_add.py`

Prev post (coalescing — PASSED at ~15% BW):
`TBD-verify — Medium URL for 002`  (or 001 if 002 is still unpublished:
https://gahan9.medium.com/your-gpu-kernel-took-44-microseconds-the-program-took-72-milliseconds-09c5e3b920c1)

---

## Hook variants (kept for A/B)

| ID | Type | Line |
|----|------|------|
| A | Curiosity | "N=1000, block=256. How many threads write past the array?" |
| B | Pain-point | "This kernel can print the right answer. 24 threads still wrote past the end of the array." ← **ship this** |
| C | Promise | "One line — if (idx < n) — is the difference between a passing test and silent corruption." |

Recommended: **B** — names the trap; preview-safe; no jargon; continues the 002 "PASSED is not the whole story" through-line.

---

## Posting notes

- Attach `visual.png` as a **carousel / multi-image** if you split panels, or as a single square infographic (1024×1024). Native image beats a link post.
- Do **not** put the Medium URL in the primary post if reach is the goal this week.
- Pin the first comment.
- Best window: Tue–Thu, 9–11 AM audience timezone.
- This post's number (24 threads) is **arithmetic**, not a profiled GPU run. Do not dress it as a benchmark. The SAXPY `PASSED` line in the module is from a guarded kernel on `gfx950` / ROCm 7; the unguarded OOB case is the documented failure mode, not a trace from this machine.
- After publish: fill Publish log in `../HISTORY.md` with the LinkedIn URL.
