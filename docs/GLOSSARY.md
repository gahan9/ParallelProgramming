# Glossary — plain-language definitions

Every term the curriculum uses, defined so a newcomer understands it and an expert doesn't wince.
AMD and NVIDIA use different words for the same idea; both are given, cross-referenced.

---

## Hardware & execution

**GPU (Graphics Processing Unit)** — a processor with thousands of small cores optimized for doing
the *same* operation on *lots* of data at once. Contrast with a CPU: a few very fast, very smart
cores optimized for doing *different* things quickly.

**SIMT (Single Instruction, Multiple Threads)** — the GPU execution style: many threads run the
same instruction in lockstep, each on its own data. It's how one instruction fetch drives dozens
of lanes at once.

**Thread** — the smallest unit of work; one lane of execution with its own registers and indices.

**Warp (NVIDIA) / Wavefront (AMD)** — a group of threads that execute together in lockstep. A warp
is **32** threads; a wavefront is **64** threads. This size difference matters for occupancy and
divergence math — the curriculum flags it whenever it does.

**Block (NVIDIA) / Workgroup (AMD)** — a group of threads that can cooperate: they share fast
on-chip memory and can synchronize with a barrier. Scheduled onto one SM/CU.

**Grid** — all the blocks launched by one kernel call. The whole "army" for one job.

**SM (Streaming Multiprocessor, NVIDIA) / CU (Compute Unit, AMD)** — the physical engine that runs
blocks/workgroups. A GPU has many. Each has registers, schedulers, and fast on-chip memory.

**Tensor Core (NVIDIA) / Matrix Core (AMD)** — specialized hardware that does small matrix
multiply-accumulate operations extremely fast, in reduced precision. The workhorse of modern
deep learning throughput.

**Occupancy** — how many warps/wavefronts are resident on an SM/CU relative to the max. Higher
occupancy helps hide memory latency — up to a point. Limited by registers and shared memory per
thread/block.

**Warp/wavefront divergence** — when threads in the same warp take different branches of an `if`,
the hardware runs both paths serially with the inactive lanes masked off. It wastes throughput.

---

## Memory

**Global memory (VRAM / HBM / GDDR)** — the GPU's main memory. Big (tens of GB) but relatively
slow and high-latency. **HBM** (High Bandwidth Memory) is the fast, stacked variant on data-center
GPUs (MI300, H100).

**Shared memory (NVIDIA) / LDS (Local Data Share, AMD)** — small, fast, on-chip memory that a block
controls explicitly. The key to fast tiled algorithms (matmul, reduction).

**Registers** — the fastest storage, private per thread. Scarce; using too many per thread lowers
occupancy ("register pressure").

**L1 / L2 cache** — automatic on-chip caches between registers and global memory.

**Coalescing** — when the threads of a warp access consecutive memory addresses, the hardware
combines them into a few wide transactions. Coalesced access is fast; strided/scattered access
wastes bandwidth. See Module A02.

**Bandwidth** — how many bytes/second you can move to/from memory (e.g. ~3 TB/s HBM3 on MI300/H100).

**Arithmetic intensity** — FLOPs performed per byte moved from memory. Determines whether a kernel
is **memory-bound** (low intensity) or **compute-bound** (high intensity). Central to the roofline.

**PCIe / host-device transfer** — the (comparatively slow) link between CPU (host) memory and GPU
(device) memory. A classic hidden bottleneck; see Module A01 §6.

---

## Programming models & tools

**Host / Device** — CPU side / GPU side. Host code orchestrates; device code (the kernel) runs on
the GPU.

**Kernel** — a function that runs on the GPU, executed by many threads in parallel. Marked
`__global__`.

**CUDA** — NVIDIA's GPU compute platform + C++ API. Compiled with `nvcc`.

**HIP** — AMD's portable C++ GPU API. Compiled with `hipcc`; runs on AMD and (via a CUDA backend)
NVIDIA. HIP mirrors CUDA closely — `cudaMalloc` ↔ `hipMalloc`.

**Triton** — a Python DSL for writing GPU kernels; JIT-compiles to fast code for both vendors.
Great for fused ML kernels without hand-writing low-level code.

**CuTe / CUTLASS** — NVIDIA C++ template libraries for high-performance GEMM and tensor layouts.
Advanced, NVIDIA-only (Track A10).

**Stream** — an ordered queue of GPU operations. Different streams can overlap (compute with copy).

**Barrier / `__syncthreads()`** — a synchronization point where all threads in a block wait for
each other. Misusing it (e.g. inside divergent branches) causes hangs or wrong results.

**Roofline model** — a plot that bounds achievable performance by memory bandwidth (the slanted
"roof") and peak compute (the flat "roof"), telling you which one limits your kernel.

**Profiler** — a tool that measures where time and bandwidth go. `rocprofv3`/Omniperf (AMD),
`nsys`/`ncu` (NVIDIA).

---

## ML performance & systems

**Reduction** — combining an array into one value (sum, max) in parallel. Foundational pattern
(Module A04).

**Scan / prefix sum** — computing running aggregates in parallel (Module A05).

**GEMM (General Matrix-Matrix Multiply)** — the dominant compute in deep learning. Tiled matmul is
Module A06.

**Softmax** — turns a vector of scores into probabilities; needs a numerically stable, often fused,
implementation on GPUs (Module A07).

**Attention / FlashAttention** — the core transformer operation; FlashAttention makes it IO-aware
and memory-efficient by fusing and tiling (Modules A09, B05).

**KV cache** — stored keys/values from previous tokens so autoregressive generation doesn't recompute
them. Big memory consumer; managed by PagedAttention (Module B06).

**Continuous batching** — dynamically adding/removing sequences from an in-flight inference batch to
keep the GPU busy (Module B06).

**Speculative decoding** — using a small fast model to draft tokens a big model then verifies, to
speed up generation (Module B06).

**Quantization** — representing weights/activations in fewer bits (int8, fp8) to save memory and
speed up compute, trading some accuracy (Module B04).

**Mixed precision** — training/inferring in lower precision (bf16/fp16) with a few operations kept
in fp32 for stability (Module B04).

**Data / Tensor / Pipeline / Expert parallelism** — the four ways to split a model+data across GPUs
(Module B07).

**Roofline-bound vs latency-bound vs throughput-bound** — different limits a serving system hits;
diagnosing which is the first step to optimizing it.

**Drift** — when live data diverges from training data over time, degrading a deployed model
(Module C06).

**A/B testing / canary** — releasing a model change to a fraction of traffic to measure impact
before full rollout (Module C05).
