<!--
PRE-PUBLISH CHECKLIST (delete before pasting to Medium)
- [ ] Re-run: `make run-hip` + `make profile-hip` on YOUR GPU
- [ ] Update these slots with your numbers:
      1. Title + TL;DR (44 µs / 72 ms / 0.06%)
      2. "The measurement" table — all five rows
      3. "GPU, driver, and N" line
      4. Hero image badge (0.06%) in visual.png
- [ ] Name your GPU + ROCm/CUDA version where marked
- [ ] Hero image: attach visual.png, set alt-text (provided below)
- [ ] Tags (5): GPU Programming · CUDA · ROCm · Performance Engineering · Parallel Computing
- [ ] SEO title/description in story settings
-->

# Your GPU kernel took 44 microseconds. The program took 72 milliseconds.

**Where the other 99.9% went — and the one question to ask before trusting any kernel benchmark.**

*Parallel Spectrum · Post 001 · Track A — GPU Programming*

> **Hero image:** `visual.png` (this folder).
> **Alt-text:** Dark slate infographic — kernel 44 µs vs ~72 ms end-to-end; cyan bars for allocation and PCIe copies dwarfing a sliver of a kernel bar; amber badge "0.06%"; footer "Parallel Spectrum · Track A · A01".

## TL;DR

I profiled the simplest possible GPU program — vector add, ~16.7M elements — end to end instead of timing just the kernel. The kernel took 44 µs of a ~72 ms wall-clock run: **0.06%**. The rest was memory allocation, two PCIe copies, and cleanup. If you only ever look at kernel time, you will optimize the wrong thing for years — so this post is about the path, not the kernel.

## The measurement

The program is the five-step host↔device dance every GPU program performs: allocate device memory, copy inputs host→device, launch the kernel (asynchronously), copy results device→host, free. I timed the kernel with GPU events — the correct tool for an async launch — and then profiled the *whole program* with `rocprofv3`.

| Phase | Time | Share of wall clock |
|-------|-----:|--------------------:|
| `hipMalloc` ×3 (incl. first-touch GPU setup) | ~68.1 ms | ~94% |
| Host→Device copies (A, B) | ~2.4 ms | ~3% |
| **`vector_add` kernel** | **~0.044 ms** | **~0.06%** |
| Device→Host copy (C) | ~1.2 ms | ~2% |
| `hipFree` ×3 | ~0.7 ms | ~1% |

Reference run: AMD MI-series (`gfx950`), ROCm 7.x, n = 16,777,216. **Your numbers will differ — re-run and quote your own trace; that's the exercise, not a caveat.**

One honest footnote before the comments section does it for me: most of that ~68 ms is a *one-time* first-touch cost the first time the process touches the GPU. If you allocate once and run a thousand frames, you won't pay it a thousand times. But that's exactly why the durable lesson isn't "allocation is slow." The durable lesson is the next section.

## Why the path dominates

A CPU is a few geniuses; a GPU is a stadium of fifty thousand diligent students. Ask the geniuses to add two huge lists of numbers and they'll do it fast — one pair at a time. Hand every student in the stadium one pair and shout "add, now," and the whole job finishes in a single moment.

But notice what you had to do: hand out fifty thousand slips of paper, then collect fifty thousand answers. That handout is the host↔device path — and it's the part everybody forgets to measure.

Three reasons it dominates here:

**1. The arithmetic is trivially cheap.** Per element, vector add reads 8 bytes, does one addition, writes 4 bytes — one floating-point operation per 12 bytes moved, an arithmetic intensity of ~0.08 FLOP/byte. A data-center GPU offers hundreds of TFLOP/s of compute against a few TB/s of memory bandwidth. This kernel is *hopelessly* memory-bound: no amount of faster math changes anything. Its ceiling is bandwidth, full stop.

**2. The data starts on the wrong side of the bus.** The GPU has its own memory. Everything it needs must cross PCIe — tens of GB/s against the GPU's own TB/s. For a one-shot task at this size, the copies alone outweigh the kernel by ~80×.

**3. The launch is asynchronous, so the obvious timer lies.** `vector_add<<<...>>>` returns immediately; the work happens later. Time it with a CPU clock and you've measured a function call, not a kernel. This is why the lab times with GPU *events* — and why a whole-program profiler tells a truer story than any single timer.

## The snippet

The teaching core (the full lab adds error checks, event timing, CPU verification, and a bandwidth report — see the repo):

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {                  // bounds guard — not optional
        C[idx] = A[idx] + B[idx];
    }
}

// Host path: allocate → copy in → launch → copy out
hipMalloc(&d_A, bytes);
hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
vector_add<<<blocks, 256>>>(d_A, d_B, d_C, n);
hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);  // also synchronizes
```

Full production-shaped version: `tracks/A-gpu-programming/A01.foundations-and-programming-model/hip/vector_add.cpp` (HIP), with a CUDA twin in the same module.

## Reproduce it

```bash
git clone <repo-url> && cd ParallelProgramming
cd tracks/A-gpu-programming/A01.foundations-and-programming-model
make hip GPU_ARCH=<your-arch>     # e.g. gfx942
make run-hip                      # prints PASSED + effective GB/s
make profile-hip                  # rocprofv3 summary + traces
```

NVIDIA: `make cuda SM_ARCH=<sm_XX> && make run-cuda`, profile with `nsys` per the Makefile. The program prints `PASSED` only after checking every element against a CPU reference — a fast wrong answer is worthless.

## Honest assessment

- **Watch for:** anyone (including you) quoting kernel-only milliseconds as "GPU performance." Ask: *is PCIe in your story?*
- **When not to use a GPU at all:** tiny, one-shot workloads where setup and transfer cost more than the work. The handout costs more than the math. GPUs earn their keep when data stays resident and kernels run many times.
- **What I'd verify before presenting this:** a timeline that includes allocation and both copies — not just events around the kernel; that first-touch cost isn't mistaken for steady-state; and that the result was actually checked against a reference.
- **What held:** the kernel itself was already near-optimal — coalesced, verified, a healthy fraction of peak HBM bandwidth. There was nothing left to optimize *in the kernel*. The wins live in the path: keep data on device, fuse work, overlap copies with compute. Later modules do exactly that.

## What's next

Next in the series: same math, ~5× more bandwidth — how *the way* threads read memory decides your speed, before you touch a single arithmetic instruction ([Post 002 · A02 coalescing](../002-a02-coalescing-bandwidth/blog.md)).

---

*Parallel Spectrum · Post 001 · Track A*
*Prev: — · Next: [002 — coalescing bandwidth](../002-a02-coalescing-bandwidth/blog.md) · Code: [`A01.foundations-and-programming-model`](https://github.com/gahan9/ParallelProgramming/tree/main/tracks/A-gpu-programming/A01.foundations-and-programming-model) · Video script: `content/A01-foundations/`*

*Numbers above are from a documented reference run; yours will differ. Run the lab — that's the point.*
