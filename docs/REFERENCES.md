# Master References

The curated bibliography for the whole curriculum. Modules cite into this list. Every non-obvious
claim in a module should trace to an entry here (or a module-local citation). Grouped by theme.

Links were valid at authoring time; vendor docs move — search the title if a link 404s.

---

## Official programming guides (authoritative for API/GPU correctness)

- **CUDA C++ Programming Guide** — NVIDIA.
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- **CUDA C++ Best Practices Guide** — NVIDIA.
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- **HIP Programming Guide** — AMD ROCm.
  <https://rocm.docs.amd.com/projects/HIP/en/latest/>
- **HIP Programming Model** — AMD ROCm.
  <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html>
- **ROCm documentation portal** — AMD.
  <https://rocm.docs.amd.com/>
- **AMD GPU architecture specs (gfx targets)** — AMD.
  <https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>
- **Triton documentation & tutorials** — OpenAI/Triton.
  <https://triton-lang.org/>

---

## Architecture & performance analysis

- Williams, Waterman, Patterson (2009). *Roofline: An Insightful Visual Performance Model for
  Multicore Architectures.* CACM. <https://dl.acm.org/doi/10.1145/1498765.1498785>
- **NVIDIA Hopper Architecture Whitepaper.**
  <https://resources.nvidia.com/en-us-tensor-core>
- **AMD CDNA Architecture** (MI200/MI300 whitepapers) — AMD.
  <https://www.amd.com/en/technologies/cdna.html>
- Harris, M. *Optimizing Parallel Reduction in CUDA* (classic slides).
  <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>
- Harris, Sengupta, Owens. *Parallel Prefix Sum (Scan) with CUDA* (GPU Gems 3, Ch. 39).
  <https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda>
- **CUTLASS / CuTe documentation** — NVIDIA.
  <https://github.com/NVIDIA/cutlass>

---

## Transformers, attention & serving

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
  <https://arxiv.org/abs/1706.03762>
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*
  <https://arxiv.org/abs/2205.14135>
- Dao (2023). *FlashAttention-2.* <https://arxiv.org/abs/2307.08691>
- Milakov & Gimelshein (2018). *Online normalizer calculation for softmax.*
  <https://arxiv.org/abs/1805.02867>
- Kwon et al. (2023). *Efficient Memory Management for LLM Serving with PagedAttention (vLLM).*
  <https://arxiv.org/abs/2309.06180>
- Leviathan et al. (2023). *Fast Inference from Transformers via Speculative Decoding.*
  <https://arxiv.org/abs/2211.17192>
- Pope et al. (2022). *Efficiently Scaling Transformer Inference.*
  <https://arxiv.org/abs/2211.05102>

---

## Distributed training & sharding

- Shoeybi et al. (2019). *Megatron-LM: Training Multi-Billion Parameter Language Models Using
  Model Parallelism.* <https://arxiv.org/abs/1909.08053>
- Rajbhandari et al. (2020). *ZeRO: Memory Optimizations Toward Training Trillion Parameter
  Models.* <https://arxiv.org/abs/1910.02054>
- Huang et al. (2019). *GPipe: Efficient Training of Giant Neural Networks using Pipeline
  Parallelism.* <https://arxiv.org/abs/1811.06965>
- **RCCL** (ROCm Collective Communication Library). <https://github.com/ROCm/rccl>
- **NCCL** (NVIDIA Collective Communication Library). <https://developer.nvidia.com/nccl>

---

## Precision, quantization & efficiency

- Micikevicius et al. (2017). *Mixed Precision Training.* <https://arxiv.org/abs/1710.03740>
- Dettmers et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.*
  <https://arxiv.org/abs/2208.07339>
- Frantar et al. (2022). *GPTQ: Accurate Post-Training Quantization for GPTs.*
  <https://arxiv.org/abs/2210.17323>
- Hinton et al. (2015). *Distilling the Knowledge in a Neural Network.*
  <https://arxiv.org/abs/1503.02531>
- Micikevicius et al. (2022). *FP8 Formats for Deep Learning.* <https://arxiv.org/abs/2209.05433>

---

## ML systems design (production ML)

- Sculley et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
  <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
- Huyen, C. *Designing Machine Learning Systems* (O'Reilly, 2022). Book.
- **Google Rules of Machine Learning** — Martin Zinkevich.
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
- Breck et al. (2017). *The ML Test Score: A Rubric for ML Production Readiness.*
  <https://research.google/pubs/pub46555/>
- Amershi et al. (2019). *Software Engineering for Machine Learning: A Case Study.* ICSE.
  <https://www.microsoft.com/en-us/research/publication/software-engineering-for-machine-learning-a-case-study/>

---

## Tooling & profiling

- **rocprofiler-sdk / rocprofv3** — AMD. <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/>
- **Nsight Systems** — NVIDIA. <https://docs.nvidia.com/nsight-systems/>
- **Nsight Compute** — NVIDIA. <https://docs.nvidia.com/nsight-compute/>
- **Omniperf / Omnitrace** — AMD. <https://rocm.docs.amd.com/projects/omniperf/en/latest/>

---

## Style & engineering craft

- **Google C++ Style Guide.** <https://google.github.io/styleguide/cppguide.html>
- **Conventional Comments.** <https://conventionalcomments.org/>
- Martin, R. C. *Clean Code* (2008). Book.
