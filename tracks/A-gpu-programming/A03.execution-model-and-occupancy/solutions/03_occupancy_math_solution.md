# Exercise 3 solution - Occupancy math when *shared memory* is the limiter

Limits: 2048 threads/SM, 32 blocks/SM, 65,536 registers/SM, 65,536 B shared per SM, 32-lane
warps. Kernel: 128 threads/block, 40 registers/thread, `40 * 128 * 4 = 20,480 B` shared per
block.

## Worked answers

**Q1 — thread limit.** `2048 / 128 = 16` blocks.

**Q2 — register limit.** Per block: `128 threads × 40 regs = 5,120 regs`.
`floor(65,536 / 5,120) = floor(12.8) = 12` blocks.

**Q3 — shared-memory limit.** `floor(65,536 / 20,480) = floor(3.2) = 3` blocks.

**Q4 — block-count limit.** `32` blocks.

**Q5 — resident blocks + limiter.** `min(16, 12, 3, 32) = 3` blocks.
**Shared memory is the limiter**, and it is not close: it costs you 4× versus the register
limit and 5× versus the thread limit.

**Q6 — occupancy.**
Resident threads `= 3 × 128 = 384`. Resident warps `= 384 / 32 = 12`.
Max warps `= 2048 / 32 = 64`. Occupancy `= 12 / 64 =` **18.75%**.

**Q7 — sizing for 8 resident blocks.**
`65,536 / 8 = 8,192 B = 8 KB per block` → `8,192 / 4 = 2,048` floats → a **16 × 128** tile
instead of 40 × 128. Occupancy jumps to `8 × 128 / 32 = 32` warps `= 50%`.
**The cost:** the tile is 2.5× shallower, so you need 2.5× as many blocks to cover the same
domain, and every one of them re-reads its halo rows from global memory. You bought latency
hiding with extra global traffic. Whether that is a win depends on which one you are
actually bound by — measure both, do not argue about it.

**Q8 — `-maxrregcount=24`.**
It achieves **nothing** for occupancy here. Registers allow 12 blocks while shared memory
allows 3; lowering the register limiter from 12 to some larger number of blocks cannot move
a `min()` that is pinned at 3. What it *does* introduce is **register spilling**: values that
no longer fit in registers move to "local" memory, which is backed by global memory. You get
the same 18.75% occupancy plus slower threads. This is the classic occupancy trap — tune the
**binding** constraint, never the one that happens to be easiest to change.

**Q9 — same 20 KB, 256 threads per block.**

| Limit | 256 threads/block |
|---|---|
| threads | `2048 / 256 = 8` |
| registers | `256 × 40 = 10,240` → `floor(65,536/10,240) = 6` |
| shared | `floor(65,536/20,480) = 3` ← still the limiter |
| blocks | 32 |

Resident blocks stay at **3**, but each block now carries twice the threads:
`3 × 256 = 768` threads `= 24` warps `= ` **37.5%** — occupancy doubled with *zero* change to
the shared-memory budget.

**The rule:** when shared memory is the limiter, a block's allocation is paid **per block**,
so occupancy scales with threads *per* block. Make blocks bigger (equivalently: give each
thread more work) to amortize the same tile over more warps.

It does not scale forever — push further and the register limit takes over:

| threads/block | thread lim | reg lim | shared lim | resident blocks | resident warps | occupancy |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 12 | **3** | 3 | 12 | 18.75% |
| 256 | 8 | 6 | **3** | 3 | 24 | 37.5% |
| 512 | 4 | **3** | **3** | 3 | 48 | 75% |
| 1024 | 2 | **1** | 3 | 1 | 32 | 50% |

Occupancy peaks at 512 threads/block and then *falls*, because at 1024 threads a single
block needs 40,960 registers and only one fits. This non-monotonic curve is exactly why you
sweep block size empirically (Lab 2) instead of reasoning once and hardcoding it.

**Q10 — the number you cannot assume.**
**Shared memory per SM** (and to a lesser extent max warps per SM). On NVIDIA, shared memory
is a *configurable carveout* of the unified L1/shared block and differs per compute
capability; a single block may also exceed 48 KB only after opting in via
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`. On AMD CDNA
the LDS is a fixed 64 KB per CU. Get the real numbers from:

- `hipGetDeviceProperties` / `cudaGetDeviceProperties` at runtime — the module's
  `occupancy` lab prints exactly these fields for your device;
- the **CUDA C++ Programming Guide**, *Compute Capabilities* table
  (<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities>);
- the **AMD GPU architecture specs**
  (<https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>);
- and confirm the *achieved* number with a profiler (Nsight Compute *Occupancy* section,
  `rocprofv3` / ROCm Compute Profiler), because theoretical occupancy ignores tail effects
  and launch-tail underutilization.

## Bonus — AMD CDNA (64-lane wavefront, 64 KB LDS per CU)

- **Limiting resource: unchanged.** `floor(65,536 / 20,480) = 3` workgroups per CU. LDS is
  still the binding constraint.
- **Occupancy percentage: unchanged at 18.75%.** Resident threads `= 3 × 128 = 384`;
  wavefronts `= 384 / 64 = 6`; max wavefronts `= 2048 / 64 = 32`; `6 / 32 = 18.75%`.
- **The reported count of resident "warps" *does* change: 6 wavefronts, not 12 warps.** The
  percentage is identical because the wavefront is twice as wide, so the factor of 2 cancels
  in numerator and denominator. This is why "we run 12 warps" is a meaningless statement
  across vendors while "we are at 18.75% occupancy" is portable.
- **One AMD-specific wrinkle:** LDS is allocated in granules, so a request is rounded up to
  the next granule before the division is done. 20,480 B is already a clean multiple; a
  20,481-byte request would consume a whole extra granule and could silently cost you a
  resident workgroup. Check the granularity for your target in the AMD arch specs.

## The interview point

Anyone can recite "occupancy = resident warps / max warps." The signal the examiner is
looking for is whether you (a) compute occupancy as a **`min()` over four independent
limiters**, (b) name the **binding** one before proposing a fix, (c) know that the fix for a
shared-memory-bound kernel is *bigger blocks or a smaller tile*, not fewer registers, and
(d) treat occupancy as a **means to latency hiding**, not a score to maximize. Ending with
"and then I would measure it, because the curve is not monotonic" is the answer that closes
the loop.
