# Shorts script — Post 003 (A01 bounds guard)

Aligned with `content/A01-foundations/shorts.md` Short 2 — the LinkedIn story cut.
~55 s · 9:16 · burned-in captions · no title card.

The 24-thread count is **ceil-division arithmetic** (A01 §9 drill 1), not a profiler number.
Do not put a GPU model on screen as if this were a timed run.

| t | On-screen (≤6 words) | Voiceover |
|---|----------------------|-----------|
| 0–3s | `PASSED` · `still wrong` | "This kernel can print the right answer — and still write past the end of your array." |
| 3–12s | `N=1000` · `block=256` | "N is a thousand. Block size is two fifty-six. You round up and launch four blocks." |
| 12–28s | `1024 threads` · `24 extra` | "Four times two fifty-six is one thousand twenty-four threads. Twenty-four of them have no element. Without a guard they read and write memory that is not yours." |
| 28–42s | `if (idx < n)` | "One line. If index is less than N, do the work. Otherwise skip. Those twenty-four go idle instead of corrupting a buffer you will debug next week." |
| 42–52s | `Tests check 0..N-1` | "I'd verify the guard is there — not just that the first N answers match a CPU. Damage lives past index N." |
| 52–58s | `Write-up in description` | "The drill and the SAXPY lab are linked below. Parallel Spectrum, post three." |

**On-screen sequence:** `PASSED` → `4 blocks` → `24 out of bounds` → `if (idx < n)`

**Title (YouTube):** `GPU kernel looks right — 24 threads write off the array`

**Pinned comment:** Medium URL + `tracks/A-gpu-programming/A01.foundations-and-programming-model/exercises/01_saxpy.cpp`

**Alt-text (for upload):** Dark slate four-panel infographic — PASSED on N=1000, launch of 4×256=1024 threads with 24 extra in amber, one-line `if (idx < n)` fix in cyan, assessment question; footer Parallel Spectrum Track A A01.
