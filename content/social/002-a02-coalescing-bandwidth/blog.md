<!--
PRE-PUBLISH CHECKLIST (delete before pasting to Medium)
- [ ] Re-run: `make run-hip` in A02 on YOUR GPU (transpose + bandwidth)
- [ ] Update these slots with your numbers:
      1. Title + TL;DR (~700 / ~2600 / ~3400 GB/s and ~5×)
      2. "The measurement" table — all four rows
      3. "GPU, driver, and N" line (reference: MI300X gfx942, ROCm 6.2, n=4096)
      4. Hero image badge (~5×) in visual.png
- [ ] Name your GPU + ROCm/CUDA version where marked
- [ ] Hero image: attach visual.png, set alt-text (provided below)
- [ ] Tags (5): GPU Programming · CUDA · ROCm · Performance Engineering · Parallel Computing
- [ ] SEO title/description in story settings
- [ ] Link Prev: Medium 001 URL already known
- [ ] After publish: paste URL into linkedin.md first-comment + HISTORY.md
-->

# Your GPU kernel passed every test. It still wasted ~85% of the bandwidth.

**Same matrix. Same answers. ~700 → ~3,400 GB/s — because neighboring threads finally grabbed neighboring addresses.**

*Parallel Spectrum · Post 002 · Track A — GPU Programming*

> **Hero image:** `visual.png` (this folder).
> **Alt-text:** Dark slate four-panel infographic — naive transpose ~700 GB/s amber, tiled ~2600 GB/s cyan, tiled+padded ~3400 GB/s green; truck metaphor for coalesced vs strided access; footer "Parallel Spectrum · Track A · A02".

## TL;DR

Last time ([Post 001](https://gahan9.medium.com/your-gpu-kernel-took-44-microseconds-the-program-took-72-milliseconds-09c5e3b920c1)) the lesson was: measure the *path*, not just the kernel. This time the path is fine — data already lives on the device — and the kernel still leaves most of the GPU on the table. A matrix transpose that prints `PASSED` at ~700 GB/s climbs to ~3,400 GB/s with shared-memory tiling and a one-column pad. The arithmetic never changes. Only the addresses do.

## The measurement

Transpose does *zero* floating-point math worth counting. It is a pure test of how threads touch memory. I ran three kernels against a coalesced copy ceiling on the same problem size:

| Kernel | Access pattern | Effective BW | vs copy ceiling |
|--------|----------------|-------------:|----------------:|
| `bandwidth` (copy) | coalesced read + write | ~4,800 GB/s | 100% (reference) |
| `transpose_naive` | coalesced read, **strided write** | ~700 GB/s | ~15% |
| `transpose_tiled` | coalesced R+W, shared-memory **bank conflicts** | ~2,600 GB/s | ~54% |
| `transpose_tiled_padded` | coalesced R+W, conflict-free | ~3,400 GB/s | ~70% |

Reference run: **AMD MI300X (`gfx942`), ROCm 6.2, n = 4096**. Documented in module A02 §5. **Your absolute GB/s will differ — re-run and quote your own ladder; the ratios are the lesson.**

~5× from naive to padded. Correctness tests pass on every rung. That is the trap.

## Why addresses beat arithmetic here

**Layman picture.** Global memory is a warehouse across town. A memory transaction is a truck that leaves whether you filled it or not. *Coalescing* — neighboring threads touching neighboring addresses — means the 32 (NVIDIA warp) or 64 (AMD wavefront) workers who ride together grab from **one shelf**, so one full truck serves them all. Strided access is each worker sending their own truck for a single item. You paid for every empty seat.

Hardware detail that makes the picture load-bearing: HBM is fetched in **cache-line bursts** (typically 64 or 128 bytes), not one float at a time. When lane `i` reads `base + i*4`, the whole warp fits in one line. When lane `i` writes `base + i*n*4` (a column of an `n×n` matrix), each lane lands in a different line — up to 32 transactions for 32 useful floats. Most of the bytes the controller fetched get thrown away. You still paid for them.

That is why the naive transpose is slow even though the *reads* are fine: it writes down columns. Same FLOPs (none). Different addresses. Different bill.

**Shared memory** is the workbench inside the building. Haul a tile over once with coalesced loads, transpose on-chip, write the output tile with coalesced stores. The strided step now happens in fast local memory instead of across town.

One more footgun after tiling: a `tile[32][32]` in shared memory makes column reads hit the **same bank** — a *bank conflict*, where the hardware serializes what should have been parallel. Change the declaration to `tile[32][33]` and the column stride is no longer a multiple of 32 banks. On the reference run that alone moved ~2,600 → ~3,400 GB/s. One character. Same answers.

## If you live in PyTorch

A transposed tensor view often changes **strides**, not bytes. Logically transposed; physically still in the old order. The next kernel then walks memory the hard way — the same strided pattern this lab measures. `.contiguous()` pays one copy so every later access can be coalesced. This post is the hardware reason that API exists.

## The snippet

Teaching core only (the lab adds timing, verification, and a bandwidth ceiling):

```cpp
#define TILE 32

__global__ void transpose_naive(const float* in, float* out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n)
        out[x * n + y] = in[y * n + x];  // write strided by n
}

__global__ void transpose_tiled_padded(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];    // +1 kills the bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n)
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];  // coalesced load
    __syncthreads();
    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < n && ty < n)
        out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];  // coalesced store
}
```

Full HIP / CUDA / Triton twins: `tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/`.

## Reproduce it

```bash
git clone <repo-url> && cd ParallelProgramming
cd tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing
make hip GPU_ARCH=<your-arch>     # e.g. gfx942
make run-hip                      # bandwidth ceiling + three transpose rungs
# Expect: naive << tiled < tiled+padded, all PASSED
```

NVIDIA: `make cuda SM_ARCH=<sm_XX> && make run-cuda`. Optional: `make profile-hip` / `make profile-cuda` and look for global store efficiency on naive, and shared-memory bank conflicts on tiled vs padded.

## Honest assessment

- **Watch for:** celebrating `PASSED` on a memory-bound kernel without quoting GB/s against a coalesced copy ceiling. Correctness will never catch a bad access pattern.
- **When not to obsess over padding:** if the kernel is already compute-bound, or the tile never does column-major shared reads, the `+1` buys nothing. Measure before you sprinkle pads everywhere.
- **What I'd verify before presenting this:** (1) the copy ceiling on *this* GPU, (2) that all three transpose kernels match a CPU reference, (3) that the GB/s ladder matches the story (naive << tiled < padded), (4) optionally bank-conflict counters on tiled vs padded so the last rung isn't a timing myth.
- **What held:** the arithmetic was never the lever. Layout and address patterns were. That is the durable lesson for every memory-bound kernel in this curriculum — GEMM tiling (A06), attention traffic (A09), and framework layout choices included.

## What's next

Next in the series: a bounds-guard kernel that corrupts memory and still looks right ([Post 003 · A01](../003-a01-bounds-silent-corruption/blog.md)). After that: the reduction ladder's stage spread (A04) once re-run on your GPU.

---

## SEO package

- **Primary keyword:** GPU memory coalescing
- **Secondary:** shared memory bank conflict, matrix transpose CUDA/HIP, PyTorch contiguous
- **Title variants:**
  1. Your GPU kernel passed. It still wasted ~85% of bandwidth
  2. Same math, ~5× bandwidth: GPU coalescing measured
  3. Why `.contiguous()` exists — measured on a transpose
- **Meta description (≤155):** A transpose that prints PASSED at ~700 GB/s hits ~3,400 GB/s after tiling and a one-column pad. Same answers — access pattern, not math.
- **Tags:** GPU Programming · CUDA · ROCm · Performance Engineering · Parallel Computing

---

*Parallel Spectrum · Post 002 · Track A*
*Prev: [001 — kernel vs path](https://gahan9.medium.com/your-gpu-kernel-took-44-microseconds-the-program-took-72-milliseconds-09c5e3b920c1) · Next: [003 — bounds / silent corruption](../003-a01-bounds-silent-corruption/blog.md) · Code: [`A02.memory-hierarchy-and-coalescing`](https://github.com/gahan9/ParallelProgramming/tree/main/tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing)*

*Numbers above are from a documented reference run; yours will differ. Run the lab — that's the point.*
