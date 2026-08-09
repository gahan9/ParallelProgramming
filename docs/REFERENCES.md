# Master References

The curated bibliography for the whole curriculum. Every non-obvious claim in a module should
trace to an entry here or to a module-local citation in that module's section 8.

## How to cite into this page

Every source below is a `###` heading, which gives it a stable anchor on **both** GitHub and the
published site. Cite a specific source rather than the bibliography as a whole:

```markdown
[Roofline (Williams et al., 2009)](../../../docs/REFERENCES.md#roofline-williams-et-al-2009)
```

`check_links.py` resolves every one of those anchors, so a reworded heading fails CI rather than
rotting silently. Depth of `../` depends on where you are: three levels up from a module README,
four from an `exercises/` or `solutions/` page.

## What gets a date, and what does not

Following the repository's *cite or hedge* rule, the two kinds of source are dated differently:

- **Papers** are identified by **DOI or arXiv ID**, which are permanent. A retrieval date on
  `arxiv.org/abs/1706.03762` would be decoration — the identifier already pins the artifact.
- **Vendor documentation, spec pages and tool guides** are **version-sensitive and move**, so each
  carries a **retrieval date**. If a claim depends on one of these, the date tells you how stale
  your evidence is. Links were verified on the dates shown; search the title if one 404s.

Some publishers (ACM, MIT Press, several corporate blogs) return `403` to automated requests. Those
links are correct in a browser; the link checker's `--external` mode reports them and that is
expected, not a defect.

---

## Official programming guides

Authoritative for API behaviour and GPU correctness. All retrieved **8 Aug 2026**.

### CUDA C++ Programming Guide (NVIDIA)

The normative description of the CUDA execution and memory model.
<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>

### CUDA C++ Best Practices Guide (NVIDIA)

Coalescing, shared memory, bank conflicts, alignment, occupancy guidance.
<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

### HIP Programming Guide (AMD)

AMD's portable C++ GPU API; the CUDA-to-HIP mapping lives here.
<https://rocm.docs.amd.com/projects/HIP/en/latest/>

### HIP Programming Model (AMD)

The execution model in isolation — wavefronts, workgroups, the memory hierarchy.
<https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html>

### ROCm documentation portal (AMD)

Entry point for every ROCm component's documentation.
<https://rocm.docs.amd.com/>

### AMD GPU architecture specifications (AMD)

The authoritative per-SKU table: compute units, wavefront size, LDS, cache sizes, VRAM.
Cited whenever the curriculum states an AMD hardware number.
<https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>

### Triton documentation and tutorials (Triton project)

Language reference and the tutorial series the Triton labs parallel.
<https://triton-lang.org/>

---

## Architecture and performance analysis

### Roofline (Williams et al., 2009)

Williams, S., Waterman, A., Patterson, D. *Roofline: An Insightful Visual Performance Model for
Multicore Architectures.* Communications of the ACM 52(4), 65–76.
DOI [10.1145/1498765.1498785](https://dl.acm.org/doi/10.1145/1498765.1498785).

### Little's Law (Little, 1961)

Little, J. D. C. *A Proof for the Queuing Formula L = λW.* Operations Research 9(3), 383–387.
The concurrency-latency-throughput identity behind every occupancy and batching argument.
DOI [10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383).

### NVIDIA Hopper Architecture Whitepaper (NVIDIA)

Retrieved **8 Aug 2026**. <https://resources.nvidia.com/en-us-tensor-core>

### AMD CDNA architecture (AMD)

Landing page for the MI200/MI300 whitepapers. Retrieved **8 Aug 2026**.
<https://www.amd.com/en/technologies/cdna.html>

### AMD CDNA 3 whitepaper (AMD)

Per-CU matrix throughput, XCD layout, Infinity Cache. **Note:** this document reports Infinity
Cache bandwidth as 17.2 TB/s while AMD's own Hot Chips 2024 deck reports 14.7 TB/s. The
curriculum reports both rather than picking one; see B01 §3.3. Retrieved **8 Aug 2026**.
<https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf>

### AMD Instinct cluster design reference (AMD)

Infinity Fabric per-pair and aggregate bandwidth, node and rack topology. Retrieved
**8 Aug 2026** — AMD retired the standalone `instinct-mi300-series-cluster-reference-guide.pdf`
this page replaces.
<https://instinct.docs.amd.com/projects/cluster-documentation/latest/reference/cluster-design.html>

### Optimizing Parallel Reduction in CUDA (Harris)

The seven-step reduction ladder that A04 rebuilds. Retrieved **8 Aug 2026**.
<https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>

### How to Access Global Memory Efficiently in CUDA C/C++ Kernels (Harris)

The coalescing model used throughout Track A. Retrieved **8 Aug 2026**.
<https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/>

### Parallel Prefix Sum with CUDA (Harris, Sengupta and Owens)

GPU Gems 3, chapter 39 — work-efficient scan. Retrieved **8 Aug 2026**.
<https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>

### Optimizing Matrix Transpose in CUDA (Ruetsch and Micikevicius, 2009)

The naive/tiled/padded progression that A02's Lab 2 reproduces. Retrieved **8 Aug 2026**.
<https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf>

### CUTLASS and CuTe (NVIDIA)

Template library for high-performance GEMM and tensor layouts; the subject of A10. Retrieved
**8 Aug 2026**. <https://github.com/NVIDIA/cutlass>

### AMD matrix instruction calculator (AMD)

Per-instruction MFMA shapes, cycle counts and register layouts for CDNA. Retrieved
**8 Aug 2026**. <https://github.com/ROCm/amd_matrix_instruction_calculator>

---

## Transformers, attention and serving

### Attention Is All You Need (Vaswani et al., 2017)

The transformer architecture. arXiv [1706.03762](https://arxiv.org/abs/1706.03762).

### FlashAttention (Dao et al., 2022)

*Fast and Memory-Efficient Exact Attention with IO-Awareness* — tiling and recomputation that
make attention memory-bound rather than quadratic in HBM traffic.
arXiv [2205.14135](https://arxiv.org/abs/2205.14135).

### FlashAttention-2 (Dao, 2023)

Better work partitioning across warps and thread blocks.
arXiv [2307.08691](https://arxiv.org/abs/2307.08691).

### Online normalizer calculation for softmax (Milakov and Gimelshein, 2018)

The single-pass streaming softmax that A07 and A09 depend on.
arXiv [1805.02867](https://arxiv.org/abs/1805.02867).

### PagedAttention and vLLM (Kwon et al., 2023)

*Efficient Memory Management for LLM Serving with PagedAttention* — KV cache paging.
arXiv [2309.06180](https://arxiv.org/abs/2309.06180).

### Speculative decoding (Leviathan et al., 2023)

*Fast Inference from Transformers via Speculative Decoding.*
arXiv [2211.17192](https://arxiv.org/abs/2211.17192).

### Efficiently Scaling Transformer Inference (Pope et al., 2022)

The partitioning and cost model behind prefill/decode reasoning.
arXiv [2211.05102](https://arxiv.org/abs/2211.05102).

### Multi-query and grouped-query attention (Ainslie et al., 2023)

*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — the
KV-cache reduction B05 and B06 assume.
arXiv [2305.13245](https://arxiv.org/abs/2305.13245).

### Orca: continuous batching (Yu et al., 2022)

*Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI '22 —
iteration-level scheduling. <https://www.usenix.org/conference/osdi22/presentation/yu>

### Scaling laws for neural language models (Kaplan et al., 2020)

arXiv [2001.08361](https://arxiv.org/abs/2001.08361).

### Training compute-optimal large language models (Hoffmann et al., 2022)

The Chinchilla result. arXiv [2203.15556](https://arxiv.org/abs/2203.15556).

### The Llama 3 herd of models (Grattafiori et al., 2024)

Cited for concrete production training and serving configurations.
arXiv [2407.21783](https://arxiv.org/abs/2407.21783).

### vLLM (project)

Reference implementation for paged KV cache and continuous batching. Retrieved **8 Aug 2026**.
<https://github.com/vllm-project/vllm>

### SGLang (project)

Serving runtime with RadixAttention prefix caching. Retrieved **8 Aug 2026**.
<https://github.com/sgl-project/sglang>

---

## Distributed training and sharding

### Megatron-LM (Shoeybi et al., 2019)

*Training Multi-Billion Parameter Language Models Using Model Parallelism* — tensor parallelism.
arXiv [1909.08053](https://arxiv.org/abs/1909.08053).

### ZeRO (Rajbhandari et al., 2020)

*Memory Optimizations Toward Training Trillion Parameter Models* — optimizer, gradient and
parameter sharding. arXiv [1910.02054](https://arxiv.org/abs/1910.02054).

### GPipe (Huang et al., 2019)

*Efficient Training of Giant Neural Networks using Pipeline Parallelism.*
arXiv [1811.06965](https://arxiv.org/abs/1811.06965).

### PyTorch FSDP (PyTorch)

The production implementation of parameter sharding. Retrieved **8 Aug 2026**.
<https://docs.pytorch.org/docs/stable/fsdp.html>

### RCCL (AMD)

ROCm Collective Communication Library. Retrieved **8 Aug 2026**.
<https://github.com/ROCm/rccl>

### NCCL (NVIDIA)

NVIDIA Collective Communication Library. Retrieved **8 Aug 2026**.
<https://developer.nvidia.com/nccl>

---

## Precision, quantization and efficiency

### Mixed Precision Training (Micikevicius et al., 2017)

Loss scaling and the fp32 master-weight pattern.
arXiv [1710.03740](https://arxiv.org/abs/1710.03740).

### FP8 Formats for Deep Learning (Micikevicius et al., 2022)

E4M3 and E5M2. Note that CDNA 3 implements the **FNUZ** interpretation and Hopper the **OCP**
one, so the same bits decode differently; see B04.
arXiv [2209.05433](https://arxiv.org/abs/2209.05433).

### OCP 8-bit Floating Point Specification (OFP8)

The vendor-neutral fp8 specification Hopper follows. Retrieved **8 Aug 2026** (the Open Compute
site returns `403` to automated requests; the link is correct in a browser).
<https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1>

### LLM.int8() (Dettmers et al., 2022)

*8-bit Matrix Multiplication for Transformers at Scale* — outlier-aware int8.
arXiv [2208.07339](https://arxiv.org/abs/2208.07339).

### GPTQ (Frantar et al., 2022)

*Accurate Post-Training Quantization for GPTs.*
arXiv [2210.17323](https://arxiv.org/abs/2210.17323).

### AWQ (Lin et al., 2023)

*Activation-aware Weight Quantization for LLM Compression and Acceleration.*
arXiv [2306.00978](https://arxiv.org/abs/2306.00978).

### Distilling the Knowledge in a Neural Network (Hinton et al., 2015)

arXiv [1503.02531](https://arxiv.org/abs/1503.02531).

### What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg, 1991)

The source for every numerical-stability claim in the curriculum.
DOI [10.1145/103162.103163](https://dl.acm.org/doi/10.1145/103162.103163) ·
<https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>

---

## ML systems design

### Hidden Technical Debt in Machine Learning Systems (Sculley et al., 2015)

NeurIPS. The reference for why the model is the small part of the system.
<https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>

### Designing Machine Learning Systems (Huyen, 2022)

O'Reilly. Retrieved **8 Aug 2026**.
<https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/>

### Rules of Machine Learning (Zinkevich, Google)

Retrieved **8 Aug 2026**.
<https://developers.google.com/machine-learning/guides/rules-of-ml>

### The ML Test Score (Breck et al., 2017)

*A Rubric for ML Production Readiness and Technical Debt Reduction.*
<https://research.google/pubs/pub46555/>

### Software Engineering for Machine Learning (Amershi et al., 2019)

ICSE. The nine-stage ML workflow Track C is organized around.
<https://www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/>

### Dataset Shift in Machine Learning (2009)

Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., Lawrence, N. (eds). MIT Press. The
taxonomy behind C06's drift vocabulary. Retrieved **8 Aug 2026** (MIT Press
returns `403` to automated requests; the link is correct in a browser).
<https://mitpress.mit.edu/9780262170055/dataset-shift-in-machine-learning/>

---

## Tooling and profiling

All retrieved **8 Aug 2026**.

### rocprofiler-sdk and rocprofv3 (AMD)

**Note:** `rocprofv3` defaults to `rocpd` SQLite output, not CSV; the curriculum's `profile-hip`
targets pass `--output-format csv` explicitly.
<https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/>

### Nsight Systems (NVIDIA)

Timeline profiler — where the time goes across the whole program.
<https://docs.nvidia.com/nsight-systems/>

### Nsight Compute (NVIDIA)

Kernel profiler — memory charts, occupancy, bank conflicts.
<https://docs.nvidia.com/nsight-compute/>

### Nsight Compute Profiling Guide (NVIDIA)

The metric definitions B03 relies on.
<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

### ROCm compute and systems profilers (AMD)

Formerly Omniperf and Omnitrace; renamed to `rocprof-compute` and `rocprof-sys-*` in ROCm 6.3.
<https://rocm.docs.amd.com/projects/omniperf/en/latest/>

### ROCm workload optimization guide (AMD)

CDNA 3 vs CDNA 4 comparison, peak theoretical tables, partitioning modes. Replaces the retired
`rocmdocs.amd.com` page of the same name.
<https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html>

---

## Style and engineering craft

All retrieved **8 Aug 2026**.

### Google C++ Style Guide

<https://google.github.io/styleguide/cppguide.html>

### Conventional Comments

The severity vocabulary used in this repository's review notes.
<https://conventionalcomments.org/>

### Clean Code (Martin, 2008)

*A Handbook of Agile Software Craftsmanship.*
<https://www.oreilly.com/library/view/clean-code-a/9780136083238/>

---

## Where each source is cited

The table below is **generated** by `build_reference_index.py` from the links that modules
actually contain, so it cannot drift out of step with the prose. Regenerate it after adding a
citation; CI runs `--check` and fails if it is stale.

<!-- BEGIN GENERATED: reference-index -->
| Source | Cited by |
|---|---|
| [CUDA C++ Programming Guide (NVIDIA)](#cuda-c-programming-guide-nvidia) | A01, A03, A04, A06, A07, A10, B04 |
| [CUDA C++ Best Practices Guide (NVIDIA)](#cuda-c-best-practices-guide-nvidia) | A01, A02, A03, A04, A05, A06, A07, A10, B02, B03 |
| [HIP Programming Guide (AMD)](#hip-programming-guide-amd) | A02, A03, A05, A06, A07 |
| [HIP Programming Model (AMD)](#hip-programming-model-amd) | A01, A03, A04, A08 |
| [AMD GPU architecture specifications (AMD)](#amd-gpu-architecture-specifications-amd) | A02, A03, B01 |
| [Triton documentation and tutorials (Triton project)](#triton-documentation-and-tutorials-triton-project) | A01, A05, A06, A07, A08, A09 |
| [Roofline (Williams et al., 2009)](#roofline-williams-et-al-2009) | A01, A02, A04, A05, A06, A08, A10, B01, B02, B03, B05 |
| [Little's Law (Little, 1961)](#littles-law-little-1961) | A03, B01, C05 |
| [NVIDIA Hopper Architecture Whitepaper (NVIDIA)](#nvidia-hopper-architecture-whitepaper-nvidia) | A10, B01 |
| [AMD CDNA architecture (AMD)](#amd-cdna-architecture-amd) | A01, A02 |
| [AMD CDNA 3 whitepaper (AMD)](#amd-cdna-3-whitepaper-amd) | A03, A06, B01, B04 |
| [AMD Instinct cluster design reference (AMD)](#amd-instinct-cluster-design-reference-amd) | B01, B07 |
| [Optimizing Parallel Reduction in CUDA (Harris)](#optimizing-parallel-reduction-in-cuda-harris) | A04 |
| [How to Access Global Memory Efficiently in CUDA C/C++ Kernels (Harris)](#how-to-access-global-memory-efficiently-in-cuda-cc-kernels-harris) | A02 |
| [Parallel Prefix Sum with CUDA (Harris, Sengupta and Owens)](#parallel-prefix-sum-with-cuda-harris-sengupta-and-owens) | A05 |
| [Optimizing Matrix Transpose in CUDA (Ruetsch and Micikevicius, 2009)](#optimizing-matrix-transpose-in-cuda-ruetsch-and-micikevicius-2009) | A02, B03 |
| [CUTLASS and CuTe (NVIDIA)](#cutlass-and-cute-nvidia) | A02, A03, A06, A10, B01 |
| [AMD matrix instruction calculator (AMD)](#amd-matrix-instruction-calculator-amd) | A06, A10, B01 |
| [Attention Is All You Need (Vaswani et al., 2017)](#attention-is-all-you-need-vaswani-et-al-2017) | A09, B05, B06 |
| [FlashAttention (Dao et al., 2022)](#flashattention-dao-et-al-2022) | A02, A03, A04, A05, A06, A07, A08, A09, B01, B02, B05, B06, C07 |
| [FlashAttention-2 (Dao, 2023)](#flashattention-2-dao-2023) | A07, A09 |
| [Online normalizer calculation for softmax (Milakov and Gimelshein, 2018)](#online-normalizer-calculation-for-softmax-milakov-and-gimelshein-2018) | A07, A09, B02 |
| [PagedAttention and vLLM (Kwon et al., 2023)](#pagedattention-and-vllm-kwon-et-al-2023) | A09, B05, B06, B07, C07 |
| [Speculative decoding (Leviathan et al., 2023)](#speculative-decoding-leviathan-et-al-2023) | B06, C04, C07 |
| [Efficiently Scaling Transformer Inference (Pope et al., 2022)](#efficiently-scaling-transformer-inference-pope-et-al-2022) | B05, B06, B07, C07 |
| [Multi-query and grouped-query attention (Ainslie et al., 2023)](#multi-query-and-grouped-query-attention-ainslie-et-al-2023) | B05, C07 |
| [Orca: continuous batching (Yu et al., 2022)](#orca-continuous-batching-yu-et-al-2022) | B06, C07 |
| [Scaling laws for neural language models (Kaplan et al., 2020)](#scaling-laws-for-neural-language-models-kaplan-et-al-2020) | B05, C03 |
| [Training compute-optimal large language models (Hoffmann et al., 2022)](#training-compute-optimal-large-language-models-hoffmann-et-al-2022) | B05, C03 |
| [The Llama 3 herd of models (Grattafiori et al., 2024)](#the-llama-3-herd-of-models-grattafiori-et-al-2024) | B05, B07, C03 |
| [vLLM (project)](#vllm-project) | B06, C05, C07 |
| [SGLang (project)](#sglang-project) | B06, C05, C07 |
| [Megatron-LM (Shoeybi et al., 2019)](#megatron-lm-shoeybi-et-al-2019) | B07, C03 |
| [ZeRO (Rajbhandari et al., 2020)](#zero-rajbhandari-et-al-2020) | B07, C03 |
| [GPipe (Huang et al., 2019)](#gpipe-huang-et-al-2019) | B07 |
| [PyTorch FSDP (PyTorch)](#pytorch-fsdp-pytorch) | B07, C03 |
| [RCCL (AMD)](#rccl-amd) | A04, B07 |
| [NCCL (NVIDIA)](#nccl-nvidia) | A04, B07 |
| [Mixed Precision Training (Micikevicius et al., 2017)](#mixed-precision-training-micikevicius-et-al-2017) | B04, B07 |
| [FP8 Formats for Deep Learning (Micikevicius et al., 2022)](#fp8-formats-for-deep-learning-micikevicius-et-al-2022) | B04, C07 |
| [OCP 8-bit Floating Point Specification (OFP8)](#ocp-8-bit-floating-point-specification-ofp8) | B04 |
| [LLM.int8() (Dettmers et al., 2022)](#llmint8-dettmers-et-al-2022) | B04, C04 |
| [GPTQ (Frantar et al., 2022)](#gptq-frantar-et-al-2022) | B04, C04 |
| [AWQ (Lin et al., 2023)](#awq-lin-et-al-2023) | B04, C04 |
| [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](#distilling-the-knowledge-in-a-neural-network-hinton-et-al-2015) | C04 |
| [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg, 1991)](#what-every-computer-scientist-should-know-about-floating-point-arithmetic-goldberg-1991) | A04, A07, B04, C03 |
| [Hidden Technical Debt in Machine Learning Systems (Sculley et al., 2015)](#hidden-technical-debt-in-machine-learning-systems-sculley-et-al-2015) | C01, C02, C03, C04, C05, C06, C07 |
| [Designing Machine Learning Systems (Huyen, 2022)](#designing-machine-learning-systems-huyen-2022) | C01, C02 |
| [Rules of Machine Learning (Zinkevich, Google)](#rules-of-machine-learning-zinkevich-google) | C01, C02, C05, C06, C07 |
| [The ML Test Score (Breck et al., 2017)](#the-ml-test-score-breck-et-al-2017) | C01, C02, C03, C04, C05, C06, C07 |
| [Software Engineering for Machine Learning (Amershi et al., 2019)](#software-engineering-for-machine-learning-amershi-et-al-2019) | C01, C02, C05, C06 |
| [Dataset Shift in Machine Learning (2009)](#dataset-shift-in-machine-learning-2009) | C06 |
| [rocprofiler-sdk and rocprofv3 (AMD)](#rocprofiler-sdk-and-rocprofv3-amd) | A03, B03 |
| [Nsight Systems (NVIDIA)](#nsight-systems-nvidia) | B03 |
| [Nsight Compute (NVIDIA)](#nsight-compute-nvidia) | A10, B02, B03 |
| [Nsight Compute Profiling Guide (NVIDIA)](#nsight-compute-profiling-guide-nvidia) | A03, B02, B03 |
| [ROCm compute and systems profilers (AMD)](#rocm-compute-and-systems-profilers-amd) | A03, B02, B03 |
| [ROCm workload optimization guide (AMD)](#rocm-workload-optimization-guide-amd) | B01 |

**Not yet cited from any module (4):** ROCm documentation portal (AMD); Google C++ Style Guide; Conventional Comments; Clean Code (Martin, 2008). These are here as background reading, not as support for a claim.
<!-- END GENERATED: reference-index -->
