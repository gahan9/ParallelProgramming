# Exercise 2 — Spec-sheet audit (written drill)

**Goal.** Learn to read a performance claim the way a reviewer does. Every number
below is **real and correctly quoted from a vendor document** — none of them are
lies. They are still capable of misleading you, because a peak figure carries
assumptions that the headline omits. Your job is to name the assumptions.

No code. Answer in writing, then check
[`../solutions/02_spec_sheet_audit_solution.md`](../solutions/02_spec_sheet_audit_solution.md).

---

## The audit checklist

For every performance claim, ask these six questions in order. If a claim cannot
answer all six, it is not yet an engineering number.

1. **Which precision?** fp64, fp32, tf32, bf16, fp16, fp8, fp6, fp4, int8?
2. **Which engine?** The vector ALUs, or the matrix/tensor cores? (Often ~8–16×
   apart on the same chip.)
3. **Sparsity assumed?** A "with structured sparsity" number is exactly 2× the dense
   number and requires your weights to actually have the required zero pattern.
4. **Per what?** Per GPU, per package, per node, per rack? An "8 GPU platform"
   number is 8× a single-device number.
5. **Peak or measured?** Peak is `units × lanes × clock × FLOP/lane`, an arithmetic
   ceiling nobody reaches. Measured comes with a workload name.
6. **Which clock?** Peak/boost or sustained? And is the clock sustainable at the
   quoted TDP across all dies? (On MI300X, XCD clocks vary 3–10% between dies.)

---

## The claims to audit

### Claim 1

> "AMD Instinct MI355X: MXFP4 matrix performance **80.5 PFLOPS**."
> "AMD Instinct MI355X: FP16 matrix performance **5 PFLOPS**."

Source: [AMD Instinct MI350 Series blog](https://www.amd.com/en/blogs/2025/amd-instinct-mi350-series-and-beyond-accelerating-the-future-of-ai-and-hpc.html)
and [MI355X GPU brochure](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi355x-gpu-brochure.pdf) (as of 29 Jul 2026).

**Audit it.** AMD's own tables in those same documents also give MI355X MXFP4 matrix
performance as **10.1 PFLOPS** and FP16 matrix performance as **2.5 PFLOPS**. All
four numbers are correct and none contradict each other. Reconcile each pair — the
two discrepancies have *different* causes — and say which two numbers belong in a
per-GPU roofline.

### Claim 2

> "NVIDIA H100 SXM: **3,958 teraFLOPS** of FP8 Tensor Core performance."

Source: [NVIDIA H100 product page](https://www.nvidia.com/en-us/data-center/h100/)
(as of 29 Jul 2026). Note the asterisk NVIDIA prints next to it.

**Audit it.** What does the asterisk mean, what is the dense number, and what must
be true of your *model weights* before the headline figure is reachable? Then
contrast this figure with the same page's **67 teraFLOPS** FP32 entry: what is the
ratio, and what does that ratio tell you about where a transformer's time goes?

### Claim 3

> "AMD Instinct MI300X: memory bandwidth **5.3 TB/s**."
> "NVIDIA H100 SXM: GPU memory bandwidth **3.35 TB/s**."

Sources: [MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) ·
[H100 product page](https://www.nvidia.com/en-us/data-center/h100/) (as of 29 Jul 2026).

**Audit it.** Both are honest peak theoretical figures. Yet a well-written coalesced
copy kernel will not hit either. Name three physical reasons, and state what
fraction of peak you would accept as "healthy" for a memory-bound kernel — and how
you would *establish* that fraction on a machine you had never seen before.

### Claim 4

> "AMD Infinity Cache on MI300X: **256 MB** at **17.2 TB/s** peak bandwidth."

Source: [AMD CDNA 3 architecture white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) (as of 29 Jul 2026).

**Audit it.** AMD's own Hot Chips 2024 MI300X presentation states "256 MB at
**14.7 TB/s** peak BW"
([Hot Chips 2024 slides](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith\(MI300X\).v1.Final.20240817.pdf)).
Two official AMD documents, two different numbers for the same cache. What do you do
about that as an engineer who needs *a* number today? (Hint: the answer is not
"pick the bigger one", and it is not "the vendor is lying".)

### Claim 5

> "8 × AMD Instinct MI300X: **~10.4 PF** BF16/FP16."

Source: [AMD Hot Chips 2024 MI300X presentation](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith\(MI300X\).v1.Final.20240817.pdf) (as of 29 Jul 2026).

**Audit it.** Divide by 8. Which of the six checklist questions does the platform
framing hide, and why is a platform-level FLOP number nearly useless for deciding
whether *your* kernel is fast?

### Claim 6

> "MI300X Infinity Fabric: **128 GB/s** bidirectional bandwidth between each GPU,
> **896 GB/s** peak aggregate."

Source: [MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) ·
[MI300 Series cluster design reference](https://instinct.docs.amd.com/projects/cluster-documentation/latest/reference/cluster-design.html) (retrieved 8 Aug 2026).

**Audit it.** You are sizing a tensor-parallel shard that does an all-reduce of a
`[8192, 8192]` bf16 activation on every one of 80 layers. Which of these two numbers
do you use, and why is the other one the wrong one? What does "bidirectional" do to
your arithmetic if the collective is effectively unidirectional per step?

---

## Deliverable

Write a short table: one row per claim, columns
`precision | engine | sparsity | scope (per GPU/node) | peak or measured | the number I would actually use`.

Then write the **one paragraph** you would put in a design document to state your
hardware assumptions in a way that a reviewer six months later cannot
misinterpret. That paragraph is the real output of this drill.
