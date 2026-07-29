# A01 Foundations — long-form script (16:9, ~11 min)

Source module: [`../../tracks/A-gpu-programming/A01.foundations-and-programming-model/`](../../tracks/A-gpu-programming/A01.foundations-and-programming-model/)

**Working title:** "Your GPU kernel took 44 microseconds. The program took 72 milliseconds."

---

## Pre-flight

| Field | Value |
|---|---|
| Module | A01 — Foundations & the GPU programming model |
| The ONE idea | On a GPU, moving the data usually costs more than doing the math. |
| The measured number | Kernel 44.2 µs out of ~72.4 ms wall clock — the kernel is ~0.06% of the run |
| Source of number | `make profile-hip` (`rocprofv3 --summary --sys-trace`) on `hip/vector_add.exe`, n = 16,777,216 |
| GPU used | AMD MI-series, `gfx950`, ROCm 7.x — **state your own GPU on camera** |
| Prerequisite | Can read C; has never written GPU code |
| Will NOT cover | Shared memory, coalescing (that's A02), streams/overlap (later) |

Phase breakdown from that run, used throughout the script:

| Phase | Time | Share of wall clock |
|---|---:|---:|
| `hipMalloc` ×3 | 68.112 ms | 94.1% |
| Host→Device copy (A, B) | 2.357 ms | 3.3% |
| **`vector_add` kernel** | **0.044 ms** | **0.06%** |
| Device→Host copy (C) | 1.205 ms | 1.7% |
| `hipFree` ×3 | 0.655 ms | 0.9% |

The allocation cost is dominated by a one-time context / lazy-init cost on the first call — say so
on camera, because a viewer who allocates once per frame in a loop will not see 68 ms every time.

---

## Script

| # | Time | VISUAL | SAY |
|---|------|--------|-----|
| 1 | 0:00 | Terminal: the profiler summary already on screen, kernel row highlighted amber | "This GPU kernel ran in forty-four microseconds. The program that ran it took seventy-two milliseconds. So where did the other ninety-nine point nine percent go? That question is the whole reason GPU programming is hard — and by the end of this video you'll be able to answer it for your own code." |
| 2 | 0:35 | Illustration: a few people at desks vs a packed stadium | "Here's the mental model I want you to keep. A CPU is a few geniuses. A GPU is a stadium of fifty thousand diligent students. Ask the geniuses to add two huge lists of numbers and they'll do it fast — but one pair at a time. Hand every student in the stadium one pair and shout 'add your pair, now' — the whole job finishes in a single moment." |
| 3 | 1:10 | Same illustration, arrows for handing out paper | "But notice what you had to do. You had to hand out fifty thousand slips of paper, and collect fifty thousand answers. That handout is the part everybody forgets, and it's exactly where our seventy-two milliseconds went. Where the analogy breaks: the students aren't independent — they work in locked groups, all doing the same step at the same time. Hold onto that, it matters in a minute." |
| 4 | 1:45 | Editor: a CPU `for` loop adding two arrays | "Let's derive the programming model instead of memorizing it. On a CPU, you write a loop. The loop *index* walks through the data, one element per iteration." |
| 5 | 2:10 | Editor: the loop transforms into a kernel; `blockIdx.x * blockDim.x + threadIdx.x` appears | "On a GPU you turn that inside out. You launch as many threads as you have elements, and each thread figures out *which* element is its own. The loop didn't disappear — the loop *became* the grid of threads. And this one line is how every thread learns who it is. Block index, times block size, plus thread index. You'll write this line for the rest of your career." |
| 6 | 2:45 | Diagram: grid → blocks → threads, mapped to CU/SM | "Why two levels — blocks *and* threads? Because hardware is physical. Threads that need to cooperate have to sit on the same engine, so they're grouped into a block. Blocks are independent of each other, and that independence is exactly why the same code runs on a laptop GPU with twenty engines and a data-center GPU with three hundred. You don't rewrite anything. You just get more blocks running at once." |
| 7 | 3:30 | Editor: writing `vector_add`, including `if (idx < n)` | "Let me write the kernel. Two inputs, one output, and this guard. The guard is not optional. I'm launching a whole number of blocks, so if my array size isn't an exact multiple of the block size, the last block has extra threads with an index past the end of the array. Without the guard they write out of bounds — and the ugly part is it often *looks* like it worked." |
| 8 | 4:15 | Editor: adding `HIP_CHECK` around every call | "Now the part that separates code that works from code you can debug. Every one of these runtime calls returns an error code, and the single most common beginner mistake is assigning it to a variable and never looking at it. Worse: a kernel launch is *asynchronous*, so its error doesn't show up on the launch line — it shows up later, at the next synchronizing call. So I check the launch error explicitly, right here." |
| 9 | 5:00 | Split screen: `hip/vector_add.cpp` and `cuda/vector_add.cu` | "Quick note on portability, because this series is dual-vendor. This is HIP, for AMD. This is CUDA, for NVIDIA. Look how similar they are: `hipMalloc`, `cudaMalloc`. `hipMemcpy`, `cudaMemcpy`. Learning one gets you most of the other. The one difference that will actually bite you: the lockstep group of threads is thirty-two lanes on NVIDIA — a warp — and sixty-four on AMD — a wavefront. Pick a block size that's a multiple of sixty-four and you're safe on both." |
| 10 | 5:45 | Terminal: `make hip && make run-hip`, output shows PASSED + GB/s | "Let's run it. It verifies against a CPU reference — always do this, a fast wrong answer is worthless — and it reports effective bandwidth. Note we're timing with GPU *events*, not a CPU clock, because that launch is asynchronous and a CPU timer would measure nothing." |
| 11 | 6:30 | Slide: the arithmetic-intensity calculation | "Now, is this kernel good? Vector add reads eight bytes and writes four per element, and does exactly one addition. That's one floating-point operation per twelve bytes. A data-center GPU can do hundreds of trillions of operations per second but only move a few terabytes per second. So this kernel is *hopelessly* memory-bound. It will never be limited by math. Faster arithmetic would change nothing." |
| 12 | 7:15 | Terminal: `make profile-hip`, then the summary table | "Which brings us back to the opening. Let's profile the whole program, not just the kernel." |
| 13 | 7:45 | Zoom the phase table; bars animate in | "Allocation: sixty-eight milliseconds — and most of that is a one-time setup cost the first time you touch the GPU. Copy in: two point four. The kernel: forty-four *microseconds* — you can barely see it. Copy out: one point two. The lesson isn't 'allocation is slow.' The lesson is that on a GPU, *moving data usually costs more than the math*, and if you only ever look at kernel time you will optimize the wrong thing for years." |
| 14 | 8:45 | Slide: tradeoffs | "So the honest tradeoffs. Don't use a GPU for tiny data — the handout costs more than the work. Don't use it for branchy, pointer-chasing, inherently serial logic. And block size is a real knob: too small starves the engine, too large can hurt occupancy through register pressure. A hundred twenty-eight to two fifty-six is a sane default, and we'll tune it properly in module A03." |
| 15 | 9:30 | Slide: PyTorch / real systems | "Where does this show up in real work? Every elementwise operation in PyTorch — add, scale, activation, dropout — is this kernel with more types. And all those tricks you've seen in training scripts, `pin_memory`, non-blocking transfers, keeping activations on the device — every one of them exists to fight the data movement we just measured." |
| 16 | 10:15 | Repo on screen, exercises folder | "Your turn. There are three drills in the module: write SAXPY yourself, fix a histogram kernel that has a real data race, and do the occupancy arithmetic by hand. Solutions are there too, but do them first. Next module: memory coalescing — where *how* you read memory changes your bandwidth by several times, without touching the math. Links below." |

**Word count target:** ~1,500 spoken words. Trim shot 6 or 15 first if you run long.

---

## Chapters

```
0:00 A kernel that's 0.06% of the runtime
0:35 CPU vs GPU: the stadium
1:45 Deriving the programming model
3:30 Writing the kernel (and the guard you can't skip)
4:15 Checking errors properly
5:00 HIP vs CUDA: warps vs wavefronts
5:45 Running it
6:30 Why this kernel is memory-bound
7:15 Profiling the whole program
8:45 Honest tradeoffs
9:30 Where this shows up in PyTorch
10:15 Your drills + what's next
```

## Description block

```
A GPU kernel that runs in 44 microseconds inside a program that takes 72 milliseconds.
We profile where the time actually goes, and build the mental model that explains it.

Runnable code (HIP for AMD, CUDA for NVIDIA, plus Triton):
https://github.com/gahan9/ParallelProgramming/tree/main/tracks/A-gpu-programming/A01.foundations-and-programming-model

Full curriculum: https://github.com/gahan9/ParallelProgramming

Measured on <YOUR GPU>, <ROCm/CUDA version>, n = 16,777,216.
```
