# Glossary — plain-language definitions

Every term the curriculum uses, defined so a newcomer understands it and an expert doesn't wince.
AMD and NVIDIA use different words for the same idea; both are given, cross-referenced.

## How to link to a term

Each term is a `###` heading, so it has a stable anchor on both GitHub and the published site.
When a module uses a defined term for the first time, it links here:

```markdown
[coalescing](../../../docs/GLOSSARY.md#coalescing)
```

Two conventions keep this useful rather than noisy:

- **Link the first use in a module, not every use.** A page where every technical word is blue is
  harder to read than one where none of them are.
- **Where a term is *taught*, link the module; where it is merely *used*, link the glossary.** The
  `Taught in` line on each entry below is the return path.

`check_links.py` resolves every one of these anchors, so a reworded term fails CI rather than
leaving a dead link behind.

---

## Hardware and execution

### GPU

A processor with thousands of small cores optimized for doing the *same* operation on *lots* of
data at once. Contrast with a CPU: a few very fast, very smart cores optimized for doing
*different* things quickly.

**Taught in** [A01 §2](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#2-first-principles)

### SIMT

Single Instruction, Multiple Threads — the GPU execution style: many threads run the same
instruction in lockstep, each on its own data. It's how one instruction fetch drives dozens of
lanes at once.

**Taught in** [A01 §3](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#3-deep-dive) ·
**Source** [CUDA C++ Programming Guide](REFERENCES.md#cuda-c-programming-guide-nvidia)

### Thread

The smallest unit of work; one lane of execution with its own registers and indices.

### Warp (NVIDIA) and wavefront (AMD)

A group of threads that execute together in lockstep. A warp is **32** threads; a wavefront is
**64**. This size difference matters for occupancy and divergence math — the curriculum flags it
whenever it does.

**Taught in** [A03 §3](../tracks/A-gpu-programming/A03.execution-model-and-occupancy/README.md#3-deep-dive) ·
**Source** [HIP Programming Model](REFERENCES.md#hip-programming-model-amd)

### Block (NVIDIA) and workgroup (AMD)

A group of threads that can cooperate: they share fast on-chip memory and can synchronize with a
barrier. Scheduled onto one SM/CU.

**Taught in** [A01 §3](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#3-deep-dive)

### Grid

All the blocks launched by one kernel call. The whole "army" for one job.

**Taught in** [A01 §2](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#2-first-principles)

### SM (NVIDIA) and CU (AMD)

Streaming Multiprocessor / Compute Unit — the physical engine that runs blocks and workgroups. A
GPU has many. Each has registers, schedulers, and fast on-chip memory.

**Taught in** [B01 §3](../tracks/B-gpu-ml-performance/B01.accelerator-architecture/README.md#3-deep-dive) ·
**Source** [AMD GPU architecture specifications](REFERENCES.md#amd-gpu-architecture-specifications-amd)

### Tensor Core (NVIDIA) and Matrix Core (AMD)

Specialized hardware that does small matrix multiply-accumulate operations extremely fast, in
reduced precision. The workhorse of modern deep learning throughput.

**Taught in** [B01 §3](../tracks/B-gpu-ml-performance/B01.accelerator-architecture/README.md#3-deep-dive) ·
**Source** [AMD matrix instruction calculator](REFERENCES.md#amd-matrix-instruction-calculator-amd)

### Occupancy

How many warps or wavefronts are resident on an SM/CU relative to the maximum. Higher occupancy
helps hide memory latency — up to a point. Limited by registers and shared memory per thread and
per block.

**Taught in** [A03 §3](../tracks/A-gpu-programming/A03.execution-model-and-occupancy/README.md#3-deep-dive) ·
**Source** [Little's Law](REFERENCES.md#littles-law-little-1961)

### Divergence

When threads in the same warp or wavefront take different branches of an `if`, the hardware runs
both paths serially with the inactive lanes masked off. It wastes throughput.

**Taught in** [A03 §3](../tracks/A-gpu-programming/A03.execution-model-and-occupancy/README.md#3-deep-dive)

---

## Memory

### Global memory

The GPU's main memory (VRAM). Big — tens of GB — but relatively slow and high-latency. **HBM**
(High Bandwidth Memory) is the fast, stacked variant on data-center GPUs; **GDDR** is the
consumer variant.

**Taught in** [A02 §2](../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#2-first-principles)

### Shared memory (NVIDIA) and LDS (AMD)

Small, fast, on-chip memory that a block controls explicitly — Local Data Share on AMD. The key
to fast tiled algorithms such as matmul and reduction.

**Taught in** [A02 §3](../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#3-deep-dive)

### Registers

The fastest storage, private per thread. Scarce; using too many per thread lowers occupancy
("register pressure").

**Taught in** [A03 §3](../tracks/A-gpu-programming/A03.execution-model-and-occupancy/README.md#3-deep-dive)

### L1 and L2 cache

Automatic on-chip caches between registers and global memory. AMD data-center parts add a large
**Infinity Cache** behind L2.

**Taught in** [B01 §3](../tracks/B-gpu-ml-performance/B01.accelerator-architecture/README.md#3-deep-dive)

### Coalescing

When the threads of a warp access consecutive memory addresses, the hardware combines them into a
few wide transactions. Coalesced access is fast; strided or scattered access wastes bandwidth.

**Taught in** [A02 §2](../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#2-first-principles) ·
**Source** [How to Access Global Memory Efficiently](REFERENCES.md#how-to-access-global-memory-efficiently-in-cuda-cc-kernels-harris)

### Bank conflict

Shared memory is split into banks. A warp's accesses are served in one cycle only if they hit
distinct banks (or all read one address, which broadcasts). If `k` lanes hit the same bank at
different addresses, the access serializes into `k` cycles. The classic fix is padding the minor
dimension of a shared tile by one word.

**Taught in** [A02 §3](../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#3-deep-dive) ·
**Source** [CUDA C++ Best Practices Guide](REFERENCES.md#cuda-c-best-practices-guide-nvidia)

### Bandwidth

How many bytes per second you can move to and from memory. The practical ceiling for every
memory-bound kernel; measure yours rather than quoting the datasheet.

**Taught in** [A02 §5](../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#5-performance-analysis)

### Arithmetic intensity

FLOPs performed per byte moved from memory. Determines whether a kernel is **memory-bound** (low
intensity) or **compute-bound** (high intensity). Central to the roofline.

**Taught in** [B02 §2](../tracks/B-gpu-ml-performance/B02.roofline-and-arithmetic-intensity/README.md#2-first-principles) ·
**Source** [Roofline](REFERENCES.md#roofline-williams-et-al-2009)

### Tiling

Splitting a problem into blocks small enough to stage in on-chip memory, so each byte fetched
from global memory is reused many times. The single most important structural optimization in
Track A.

**Taught in** [A06 §2](../tracks/A-gpu-programming/A06.tiled-matrix-multiply/README.md#2-first-principles)

### PCIe and host-device transfer

The comparatively slow link between CPU (host) memory and GPU (device) memory. A classic hidden
bottleneck.

**Taught in** [A01 §6](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#6-challenges-drawbacks--tradeoffs)

---

## Programming models and tools

### Host and device

CPU side and GPU side. Host code orchestrates; device code — the kernel — runs on the GPU.

**Taught in** [A01 §2](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#2-first-principles)

### Kernel

A function that runs on the GPU, executed by many threads in parallel. Marked `__global__`.

**Taught in** [A01 §2](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#2-first-principles)

### CUDA

NVIDIA's GPU compute platform and C++ API. Compiled with `nvcc`.

**Source** [CUDA C++ Programming Guide](REFERENCES.md#cuda-c-programming-guide-nvidia)

### HIP

AMD's portable C++ GPU API. Compiled with `hipcc`; runs on AMD and, through a CUDA backend, on
NVIDIA. HIP mirrors CUDA closely — `cudaMalloc` maps to `hipMalloc`.

**Source** [HIP Programming Guide](REFERENCES.md#hip-programming-guide-amd)

### Triton

A Python DSL for writing GPU kernels; JIT-compiles to fast code for both vendors. Good for fused
ML kernels without hand-writing low-level code.

**Taught in** [A08](../tracks/A-gpu-programming/A08.triton-foundations/README.md) ·
**Source** [Triton documentation](REFERENCES.md#triton-documentation-and-tutorials-triton-project)

### CuTe and CUTLASS

NVIDIA C++ template libraries for high-performance GEMM and tensor layouts. Advanced, and
NVIDIA-only.

**Taught in** [A10](../tracks/A-gpu-programming/A10.cute-cutlass-and-cutile/README.md) ·
**Source** [CUTLASS and CuTe](REFERENCES.md#cutlass-and-cute-nvidia)

### Stream

An ordered queue of GPU operations. Different streams can overlap — compute with copy, for
instance.

**Taught in** [A01 §3](../tracks/A-gpu-programming/A01.foundations-and-programming-model/README.md#3-deep-dive)

### Barrier

A synchronization point where all threads in a block wait for each other, written
`__syncthreads()`. Misusing it — for example inside a divergent branch, which not all threads
reach — causes hangs or wrong results.

**Taught in** [A02 §6](../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/README.md#6-challenges-drawbacks--tradeoffs)

### Fusion

Combining several operations into one kernel so intermediate values stay in registers or shared
memory instead of making a round trip to global memory. The main lever for memory-bound
elementwise chains.

**Taught in** [A07 §2](../tracks/A-gpu-programming/A07.softmax-and-fused-kernels/README.md#2-first-principles)

### Roofline model

A plot that bounds achievable performance by memory bandwidth (the slanted roof) and peak compute
(the flat roof), telling you which one limits your kernel.

**Taught in** [B02](../tracks/B-gpu-ml-performance/B02.roofline-and-arithmetic-intensity/README.md) ·
**Source** [Roofline](REFERENCES.md#roofline-williams-et-al-2009)

### Profiler

A tool that measures where time and bandwidth go: `rocprofv3` and `rocprof-compute` on AMD,
`nsys` and `ncu` on NVIDIA.

**Taught in** [B03](../tracks/B-gpu-ml-performance/B03.profiling-and-bottleneck-analysis/README.md) ·
**Source** [rocprofiler-sdk](REFERENCES.md#rocprofiler-sdk-and-rocprofv3-amd) ·
[Nsight Compute](REFERENCES.md#nsight-compute-nvidia)

---

## ML performance and systems

### Reduction

Combining an array into one value — a sum or a max — in parallel. Foundational pattern.

**Taught in** [A04](../tracks/A-gpu-programming/A04.parallel-reduction/README.md) ·
**Source** [Optimizing Parallel Reduction in CUDA](REFERENCES.md#optimizing-parallel-reduction-in-cuda-harris)

### Scan (prefix sum)

Computing running aggregates in parallel. The building block behind compaction, sorting and
sparse layouts.

**Taught in** [A05](../tracks/A-gpu-programming/A05.parallel-scan-prefix-sum/README.md) ·
**Source** [Parallel Prefix Sum with CUDA](REFERENCES.md#parallel-prefix-sum-with-cuda-harris-sengupta-and-owens)

### GEMM

General Matrix-Matrix Multiply — the dominant compute in deep learning.

**Taught in** [A06](../tracks/A-gpu-programming/A06.tiled-matrix-multiply/README.md)

### Softmax

Turns a vector of scores into probabilities. Needs a numerically stable, and usually fused,
implementation on GPUs.

**Taught in** [A07](../tracks/A-gpu-programming/A07.softmax-and-fused-kernels/README.md) ·
**Source** [Online normalizer calculation for softmax](REFERENCES.md#online-normalizer-calculation-for-softmax-milakov-and-gimelshein-2018)

### Attention and FlashAttention

The core transformer operation. FlashAttention makes it IO-aware and memory-efficient by fusing
and tiling, so the quadratic intermediate never reaches HBM.

**Taught in** [A09](../tracks/A-gpu-programming/A09.advanced-triton-flash-attention/README.md) ·
[B05 §3](../tracks/B-gpu-ml-performance/B05.transformer-performance/README.md#3-deep-dive) ·
**Source** [Attention Is All You Need](REFERENCES.md#attention-is-all-you-need-vaswani-et-al-2017) ·
[FlashAttention](REFERENCES.md#flashattention-dao-et-al-2022)

### Prefill and decode

The two phases of autoregressive generation. **Prefill** processes the whole prompt at once and
is compute-bound; **decode** emits one token at a time and is memory-bound. Almost every serving
tradeoff follows from the fact that they have opposite bottlenecks.

**Taught in** [B05 §3](../tracks/B-gpu-ml-performance/B05.transformer-performance/README.md#3-deep-dive) ·
**Source** [Efficiently Scaling Transformer Inference](REFERENCES.md#efficiently-scaling-transformer-inference-pope-et-al-2022)

### KV cache

Stored keys and values from previous tokens, so autoregressive generation doesn't recompute them.
A large memory consumer; managed by PagedAttention in modern servers.

**Taught in** [B06 §3](../tracks/B-gpu-ml-performance/B06.inference-serving-optimizations/README.md#3-deep-dive) ·
**Source** [PagedAttention and vLLM](REFERENCES.md#pagedattention-and-vllm-kwon-et-al-2023)

### Continuous batching

Dynamically adding and removing sequences from an in-flight inference batch to keep the GPU busy,
rather than waiting for every sequence in a batch to finish.

**Taught in** [B06 §3](../tracks/B-gpu-ml-performance/B06.inference-serving-optimizations/README.md#3-deep-dive) ·
**Source** [Orca](REFERENCES.md#orca-continuous-batching-yu-et-al-2022)

### Speculative decoding

Using a small fast model to draft tokens that a big model then verifies in one pass, to speed up
generation without changing the output distribution.

**Taught in** [B06 §3](../tracks/B-gpu-ml-performance/B06.inference-serving-optimizations/README.md#3-deep-dive) ·
**Source** [Speculative decoding](REFERENCES.md#speculative-decoding-leviathan-et-al-2023)

### Quantization

Representing weights and activations in fewer bits — int8, fp8 — to save memory and speed up
compute, trading some accuracy.

**Taught in** [B04](../tracks/B-gpu-ml-performance/B04.precision-and-quantization/README.md) ·
**Source** [LLM.int8()](REFERENCES.md#llmint8-dettmers-et-al-2022) ·
[GPTQ](REFERENCES.md#gptq-frantar-et-al-2022) ·
[AWQ](REFERENCES.md#awq-lin-et-al-2023)

### Mixed precision

Training or inferring in lower precision (bf16, fp16) with a few operations kept in fp32 for
stability.

**Taught in** [B04 §3](../tracks/B-gpu-ml-performance/B04.precision-and-quantization/README.md#3-deep-dive) ·
**Source** [Mixed Precision Training](REFERENCES.md#mixed-precision-training-micikevicius-et-al-2017)

### FP8 and the FNUZ/OCP split

Two 8-bit float interpretations that are **not** bit-compatible. AMD CDNA 3 implements **FNUZ**
(exponent bias 8, max 240); Hopper implements **OCP** (bias 7, max 448). The same byte decodes
about 2x apart, and `0x80` is NaN under FNUZ but negative zero under OCP.

**Taught in** [B04 §3](../tracks/B-gpu-ml-performance/B04.precision-and-quantization/README.md#3-deep-dive) ·
**Source** [FP8 Formats for Deep Learning](REFERENCES.md#fp8-formats-for-deep-learning-micikevicius-et-al-2022) ·
[OCP OFP8 specification](REFERENCES.md#ocp-8-bit-floating-point-specification-ofp8)

### Data, tensor, pipeline and expert parallelism

The four ways to split a model and its data across GPUs. They differ in what they replicate, what
they shard, and how much they communicate.

**Taught in** [B07](../tracks/B-gpu-ml-performance/B07.distributed-training-and-sharding/README.md) ·
**Source** [Megatron-LM](REFERENCES.md#megatron-lm-shoeybi-et-al-2019) ·
[ZeRO](REFERENCES.md#zero-rajbhandari-et-al-2020) ·
[GPipe](REFERENCES.md#gpipe-huang-et-al-2019)

### Memory-bound, compute-bound and latency-bound

The three limits a kernel or a serving system can hit. Diagnosing which one you are against is
the first step of any optimization; optimizing for the wrong one changes nothing.

**Taught in** [B02 §3](../tracks/B-gpu-ml-performance/B02.roofline-and-arithmetic-intensity/README.md#3-deep-dive)

### Drift

When live data diverges from training data over time, degrading a deployed model.

**Taught in** [C06](../tracks/C-ml-system-design/C06.monitoring-drift-and-ethics/README.md) ·
**Source** [Dataset Shift in Machine Learning](REFERENCES.md#dataset-shift-in-machine-learning-2009)

### A/B testing and canary

Releasing a model change to a fraction of traffic to measure its impact before full rollout.

**Taught in** [C05](../tracks/C-ml-system-design/C05.deployment-and-serving-architecture/README.md)

### Training-serving skew

When the features a model sees in production are computed differently from the ones it trained
on. A leading cause of models that look good offline and fail live.

**Taught in** [C02](../tracks/C-ml-system-design/C02.data-pipelines-and-features/README.md) ·
**Source** [Hidden Technical Debt in Machine Learning Systems](REFERENCES.md#hidden-technical-debt-in-machine-learning-systems-sculley-et-al-2015)
