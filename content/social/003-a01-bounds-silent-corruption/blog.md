<!--
PRE-PUBLISH CHECKLIST (delete before pasting to Medium)
- [ ] This post's headline number (24 threads) is arithmetic, not a GPU profile.
      Do not add a GPU model as if you timed it. Optional: run the SAXPY exercise
      on YOUR GPU and quote the PASSED line from that run.
- [ ] If you run SAXPY, update the "Reproduce" GPU_ARCH and any PASSED quote.
- [ ] Hero image: attach visual.png, set alt-text (provided below)
- [ ] Tags (5): GPU Programming · CUDA · ROCm · Performance Engineering · Parallel Computing
- [ ] SEO title/description in story settings
- [ ] Link Prev: 002 Medium URL if live; else 001
      001: https://gahan9.medium.com/your-gpu-kernel-took-44-microseconds-the-program-took-72-milliseconds-09c5e3b920c1
- [ ] After publish: paste URL into linkedin.md first-comment + HISTORY.md
-->

# This GPU kernel can print the right answer. 24 threads still wrote off the array.

**N = 1,000. Block size = 256. Four blocks. 1,024 threads. The tests only check the first thousand.**

*Parallel Spectrum · Post 003 · Track A — GPU Programming*

> **Hero image:** `visual.png` (this folder).
> **Alt-text:** Dark slate four-panel infographic — PASSED on a 1,000-element check; launch of 4 blocks × 256 threads = 1,024 with 24 extra in amber; one-line `if (idx < n)` fix in cyan; assessment question "Would you ship the unguarded kernel?"; footer "Parallel Spectrum · Track A · A01".

## TL;DR

Last time ([Post 002](../002-a02-coalescing-bandwidth/blog.md)) a kernel printed `PASSED` and still wasted most of the GPU's bandwidth. This time the answer can look right for a worse reason: the extra threads never touch the elements you check. A launch of `N = 1000` with 256 threads per block creates **24 threads whose index is past the end of the array**. Without `if (idx < n)` they read and write memory you did not allocate. One line is the fix. The arithmetic does not need a GPU — which is why this is the interview drill.

## The arithmetic

You do not pick the grid size to be pretty. You pick it so every element gets a thread, which means **rounding up**:

```
blocks = ceil(N / blockDim.x)
       = ceil(1000 / 256)
       = 4
```

Four blocks × 256 threads = **1,024 threads**. `N` is 1,000. **24 threads have `idx >= N`.**

That 24 is not a profiled millisecond. It is the A01 §9 drill, and it is exact on any vendor:

| Quantity | Value | Why |
|----------|------:|-----|
| `N` | 1,000 | elements in the array you allocated |
| `blockDim.x` | 256 | a default that is a multiple of both warp 32 and wavefront 64 |
| blocks launched | 4 | `ceil(1000/256)` — integer form: `(n + 255) / 256` |
| threads launched | 1,024 | 4 × 256 |
| threads with `idx >= N` | **24** | 1,024 − 1,000 |

Two sibling bugs hide in the same launch math. They fail differently, which is why people ship the silent one.

| Mistake | What happens | What your test shows |
|---------|--------------|----------------------|
| Integer divide (`n / 256` → 3 blocks) | last 232 elements never computed | `FAILED` — visible |
| Ceil divide, **no** `if (idx < n)` | 24 threads go past the allocation | `PASSED` on `C[0..n)` — the damage is at index 1,000…1,023 |
| Ceil divide **and** the guard | extra threads skip; every element is covered | `PASSED` for the right reason |

The under-launch is the bug beginners notice. The over-launch without a guard is the one that reviews as "tests are green."

## Why the extra threads exist — and why they are dangerous

**Layman picture.** You hired a stadium of workers and told them to fill 1,000 boxes. The stadium only seats people in rows of 256, so you book four rows: 1,024 seats. Twenty-four people walk in, find no box with their number, and — if you forgot to tell them to sit down — start writing on the wall behind the last box.

That wall is adjacent device memory. Another buffer. Allocator metadata. Sometimes an unmapped page, and the kernel dies. Often a mapped page that belongs to something else, and the kernel returns. The module's own wording: the corruption *may look like it works* because it is often silent ([A01 §3](https://github.com/gahan9/ParallelProgramming/blob/main/tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#3-deep-dive)).

Hardware detail that makes the picture load-bearing: a GPU thread is not an OS thread you `join()`. The launch creates a **grid** of **blocks**; every thread in that grid runs the same kernel. The index

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

is unique, sequential, and **not automatically clipped to `n`**. The programming model does not insert the clip. NVIDIA's CUDA C++ Programming Guide and AMD's HIP programming model both describe this grid; neither promises a bounds check on your pointers ([CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html); [HIP Programming Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)).

The guard is also cheap. On a GPU, neighboring threads execute in lockstep — 32 at a time on NVIDIA (a *warp*), 64 on AMD (a *wavefront*). `if (idx < n)` only splits that lockstep on the **last partial bundle**. The rest of the kernel runs as if the branch were not there. Data-dependent branching in a hot loop is a different tax ([A01 §3](https://github.com/gahan9/ParallelProgramming/blob/main/tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#3-deep-dive)); this one is nearly free.

## If you write Triton — or a custom PyTorch op

Framework kernels you call (`torch.add`, `aten::…`) already guard. Kernels you **write** do not get a gift wrap.

Triton's spelling of the same one-liner is a mask, because each program handles a whole block of offsets at once:

```python
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n                       # vectorized if (idx < n)
a = tl.load(a_ptr + offsets, mask=mask)
tl.store(c_ptr + offsets, a + b, mask=mask)
```

`triton.cdiv(n, BLOCK_SIZE)` is the ceil-division. The mask is the guard. Drop the mask and you have the same 24-thread bug with nicer syntax. The comment in the curriculum's `triton/vector_add.py` says this in one line: *the `mask` == the `if (idx < n)` bounds guard, vectorized.*

## The snippet

Teaching core only. The lab adds `HIP_CHECK` / `CUDA_CHECK` on every runtime call, a launch-error check after the `<<<>>>`, and a CPU reference:

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {                 // last block is partial — not optional
        C[idx] = A[idx] + B[idx];
    }
}

const int threads = 256;
const int blocks  = (n + threads - 1) / threads;  // ceil
vector_add<<<blocks, threads>>>(d_A, d_B, d_C, n);
```

Fill it in yourself on SAXPY (`Y = a*X + Y`): `tracks/A-gpu-programming/A01.foundations-and-programming-model/exercises/01_saxpy.cpp`. Solution: `solutions/01_saxpy_solution.cpp`. HIP / CUDA twins of vector add live next to the module README.

## Reproduce it

**Whiteboard (no GPU):** N = 1000, block = 256. Confirm 4, 1024, 24. That is the post.

**Then the lab** — the exercise kernel starts empty, so `Y` comes back unchanged and the CPU reference disagrees. After the guard, the ceil launch, and the two post-launch checks, it prints `PASSED`:

```bash
git clone https://github.com/gahan9/ParallelProgramming
cd ParallelProgramming/tracks/A-gpu-programming/A01.foundations-and-programming-model/exercises
make 01_saxpy.exe GPU_ARCH=<your-arch>     # e.g. gfx942
./01_saxpy.exe
# Expect after TODOs: SAXPY PASSED (0 errors)
```

Documented empty-kernel fail and guarded pass: [`exercises/README.md`](https://github.com/gahan9/ParallelProgramming/blob/main/tracks/A-gpu-programming/A01.foundations-and-programming-model/exercises/README.md) (reference shape captured on `gfx950`, ROCm 7). **Your wall-clock will differ. The PASS/FAIL state is the lesson.**

NVIDIA: rename to `.cu`, swap `hip*` → `cuda*`, `nvcc -O3 -arch=sm_XX`. Same guard. Same 24.

Optional, and the honest way to *see* the unguarded case: comment out `if (idx < n)`, launch N = 1000 / 256, and run under NVIDIA `compute-sanitizer`. The CPU loop over `C[0..n)` can still match. The illegal accesses are at 1000…1023. I am not quoting a sanitizer log from this machine — that run is yours.

## Honest assessment

- **Watch for:** treating `PASSED` on the first `N` elements as "the kernel is safe." The out-of-bounds writes are *past* `N`. A reference that never looks at index `N` cannot catch them.
- **When not to panic about the 24:** if you pad `N` to a multiple of the block size **and allocate the pad**, those threads have real slots. Some production libraries do this. Still write the guard. The next caller will pass 1,000.
- **What I'd verify before I'd ship:** (1) `blocks = (n + threads - 1) / threads`, not `n / threads`; (2) `if (idx < n)` on every kernel that indexes a finite array; (3) `hipGetLastError` / `cudaGetLastError` after the launch — a bad config fails differently, and silently if you skip this; (4) optionally a memcheck tool on a deliberately unguarded build so I have seen the 24, not just counted them.
- **What held:** the missing-work bug fails tests. The extra-thread bug can pass them. That is the durable lesson, and it is why the one-liner is in every kernel in this curriculum — vector add, SAXPY, coalescing probes, tiled GEMM included.

## What's next

Next in the series is blocked on a real GPU run: the reduction ladder's stage-1-to-stage-9 spread (A04). Until that number exists, do not invent one.

---

## SEO package

- **Primary keyword:** GPU kernel bounds check
- **Secondary:** CUDA `if (idx < n)`, HIP bounds guard, Triton mask, silent memory corruption
- **Title variants:**
  1. This GPU kernel can print the right answer. 24 threads still wrote off the array
  2. N=1000, block=256: 24 GPU threads write past your array
  3. Why Triton `mask = offsets < n` exists — the bounds-guard drill
- **Meta description (≤155):** N=1000 and block=256 launch 1024 threads. 24 write past the array. Tests that check N elements can still pass. One line fixes it.
- **Tags:** GPU Programming · CUDA · ROCm · Performance Engineering · Parallel Computing

---

## References

1. Parallel Spectrum, module A01 §3 — "Why `if (idx < n)` is mandatory, not optional." [GitHub](https://github.com/gahan9/ParallelProgramming/blob/main/tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#why-if-idx--n-is-mandatory-not-optional)
2. Parallel Spectrum, module A01 §9 drill 1 — `ceil(1000/256) = 4` blocks, 24 threads with `idx >= N`. [GitHub](https://github.com/gahan9/ParallelProgramming/blob/main/tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#9-self-assessment--interview-drills)
3. NVIDIA. *CUDA C++ Programming Guide* — grid, block, thread index. <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
4. AMD. *HIP Programming Model* — workgroups and the equivalent index. <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html>
5. Parallel Spectrum, `triton/vector_add.py` — `mask = offsets < n` as the vectorized guard.

---

*Parallel Spectrum · Post 003 · Track A*
*Prev: [002 — coalescing bandwidth](../002-a02-coalescing-bandwidth/blog.md) · Next: reduction ladder (A04, planned — needs a measured run) · Code: [`A01.foundations-and-programming-model`](https://github.com/gahan9/ParallelProgramming/tree/main/tracks/A-gpu-programming/A01.foundations-and-programming-model)*

*The 24 is arithmetic. Run the drill — that's the point.*
