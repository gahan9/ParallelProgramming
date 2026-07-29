# A02 Memory Coalescing — long-form script (16:9, ~10 min)

Source module: [`../../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/`](../../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/)

**Working title:** "Same math, 4.8× faster: how you *read* memory decides your GPU speed"

---

## Pre-flight

| Field | Value |
|---|---|
| Module | A02 — Memory hierarchy & coalescing |
| The ONE idea | Threads in a wavefront must touch *adjacent* addresses, or you throw bandwidth away. |
| The measured number | Naive transpose ~700 GB/s → padded tiled ~3,400 GB/s (~4.8×), against a ~4,800 GB/s copy ceiling |
| Source of number | Module A02 Section 5 reference run (`make run-hip`: `bandwidth`, `transpose`) |
| GPU used | AMD MI-series — **state your own GPU and re-run before recording** |
| Prerequisite | A01: knows grid/block/thread and the global index line |
| Will NOT cover | Occupancy tuning (A03), matmul tiling (A06) |

Ladder used throughout, all four printing `PASSED` — the story is bandwidth, not correctness:

| Kernel | Access pattern | GB/s | % of ceiling |
|---|---|---:|---:|
| `bandwidth` (copy) | coalesced read + write | ~4,800 | ceiling |
| `transpose_naive` | coalesced read, **strided write** | ~700 | ~15% |
| `transpose_tiled` | coalesced both, bank conflicts | ~2,600 | ~54% |
| `transpose_tiled_padded` | coalesced both, conflict-free | ~3,400 | ~70% |

Re-measure on your hardware; quote your own numbers on camera.

---

## Script

| # | Time | VISUAL | SAY |
|---|------|--------|-----|
| 1 | 0:00 | Terminal: all four GB/s results on screen, naive vs padded boxed | "Four kernels. All four produce the identical, verified-correct answer. The slowest runs at seven hundred gigabytes a second, the fastest at three thousand four hundred. Same arithmetic, same data, nearly five times the throughput. The only thing that changed is *how the threads touch memory* — and that's the most valuable performance lesson on a GPU." |
| 2 | 0:40 | Illustration: 32 delivery trucks each carrying one parcel, vs one truck carrying 32 | "The intuition first. When a group of GPU threads reads memory, the hardware doesn't fetch one value per thread — it fetches whole *cache lines*, fixed-size blocks. So if thirty-two threads ask for thirty-two adjacent values, that's one truck delivering thirty-two parcels. If they ask for thirty-two values scattered far apart, that's thirty-two separate trucks, each delivering one parcel and throwing the rest of its load away." |
| 3 | 1:20 | Same illustration, bytes-wasted counter ticking up | "You paid for the whole truck either way. That's the key: bandwidth isn't about how many bytes you *need*, it's about how many bytes the hardware had to *move*. Where the analogy breaks: caches do help you if a neighbouring thread later wants the same line — but you can't count on it at this scale." |
| 4 | 2:00 | Diagram: memory hierarchy — registers → LDS/shared → L2 → HBM, with latency/bandwidth labels | "Quick map of where data can live. Registers, private per thread, fastest. Then shared memory — AMD calls it LDS — small, on-chip, and *you* control it. Then L2. Then HBM, your main memory: tens of gigabytes, terabytes per second, but hundreds of cycles away. Every optimization in this module is about moving work up this pyramid." |
| 5 | 2:45 | Terminal: run `bandwidth` | "First, establish the ceiling. This kernel does the simplest possible thing — a fully coalesced copy — so whatever it reports is roughly the best your hardware will ever do. About forty-eight hundred gigabytes a second here. Now every other kernel gets judged as a percentage of *that*, not against some spec-sheet number." |
| 6 | 3:20 | Editor: naive transpose kernel, read index and write index highlighted separately | "Now a matrix transpose, which is the perfect teaching case because it's *forced* to be awkward. Reading along rows is coalesced — adjacent threads, adjacent addresses, one truck. But writing the transpose means writing down a *column* — and adjacent threads are now a whole row-width apart. Every lane lands in a different cache line." |
| 7 | 4:00 | Terminal: naive result, ~15% of ceiling | "Seven hundred gigabytes a second. Fifteen percent of what the hardware can do. And notice: it's *correct*. This is why correctness testing alone never finds performance bugs." |
| 8 | 4:30 | Diagram: tile staged through shared memory, coalesced in and out | "Here's the fix, and it's the single most reusable trick in GPU programming. You can't make both the read and the write coalesced in global memory — the transpose fundamentally scatters. So you stage a tile through shared memory: read a tile coalesced, write it into fast on-chip memory, then read it back out *transposed* and write to global memory coalesced. The awkward strided access still happens — but now it happens in shared memory, where a scatter costs almost nothing." |
| 9 | 5:20 | Editor: tiled kernel, `__syncthreads()` highlighted | "Note the barrier. Every thread must finish writing its piece of the tile before anyone reads a different piece. Skip it, and you read data that hasn't been written yet — a race that produces wrong answers intermittently, which is the worst kind of bug." |
| 10 | 5:50 | Terminal: tiled result, ~54% | "Twenty-six hundred. From fifteen percent to fifty-four percent of peak, and we haven't touched the math. But we're leaving something on the table." |
| 11 | 6:20 | Diagram: 32 lanes → same bank (conflict) vs padded stride 33 → 32 distinct banks | "Shared memory is split into banks — thirty-two of them. Parallel access is only fast if the lanes hit *different* banks. Our tile is thirty-two words wide, so when we read a column, every lane's address is a multiple of thirty-two apart, and they all map to the *same* bank. Thirty-two-way conflict, fully serialized." |
| 12 | 7:00 | Editor: change `tile[32][32]` to `tile[32][33]` | "The fix is almost insulting. Make the row thirty-three words instead of thirty-two. Now consecutive column elements are thirty-three apart, thirty-three mod thirty-two is one, so the lanes spread across all thirty-two banks. One wasted word per row buys you conflict-free access." |
| 13 | 7:30 | Terminal: padded result, ~70% | "Thirty-four hundred. Seventy percent of the copy ceiling — for a one-character change." |
| 14 | 8:00 | Slide: the diagnostic checklist | "So the workflow you should actually internalize. One: measure a plain copy to get your real ceiling. Two: if a kernel is far below it, look at whether adjacent threads touch adjacent addresses — that's coalescing. Three: if you're using shared memory and it's still slow, check whether a column index has stride thirty-two — that's a bank conflict, and padding fixes it." |
| 15 | 8:40 | Slide: tradeoffs | "Honest tradeoffs. Tiling costs shared memory, and shared memory per block limits how many blocks fit on an engine — so you can tile yourself into low occupancy. Padding wastes a little memory and makes indexing less obvious, so comment it. And this is a *transpose* — a memory-bound operation. If your kernel is compute-bound, none of this is your bottleneck, and A06 will show you the other side of that coin." |
| 16 | 9:10 | Slide: real ML systems | "Where this shows up: this is why frameworks keep tensors contiguous, why `.contiguous()` exists in PyTorch, why a badly-chosen memory format silently halves your training throughput, and why attention kernels obsess over tile shapes. Layout *is* performance." |
| 17 | 9:40 | Repo, exercises folder | "Three drills: fix a strided kernel and measure the delta, kill a bank conflict, and a written memory-hierarchy exercise. Next up, A03 — the execution model, where we find out what a wavefront actually is and why branching costs you. Links below." |

---

## Chapters

```
0:00 Four kernels, same answer, 4.8x apart
0:40 Cache lines: trucks and parcels
2:00 The memory hierarchy
2:45 Measuring your real ceiling
3:20 Why transpose is awkward
4:30 Staging through shared memory
5:50 From 15% to 54% of peak
6:20 Bank conflicts
7:00 The one-character fix
8:00 The diagnostic checklist
8:40 Tradeoffs
9:10 Why PyTorch has .contiguous()
9:40 Your drills + what's next
```

## Description block

```
Four transpose kernels. Identical correct output. 700 GB/s to 3,400 GB/s.
The difference is memory access patterns: coalescing, shared-memory tiling, and bank conflicts.

Runnable code (HIP + CUDA + Triton):
https://github.com/gahan9/ParallelProgramming/tree/main/tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing

Full curriculum: https://github.com/gahan9/ParallelProgramming

Measured on <YOUR GPU>, <ROCm/CUDA version>.
```
