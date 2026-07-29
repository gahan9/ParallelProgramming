# Solution 2 — Spec-sheet audit

Every figure below is quoted from the vendor document linked in
[`../exercises/02_spec_sheet_audit.md`](../exercises/02_spec_sheet_audit.md), verified
as of **29 Jul 2026**. Nothing here accuses a vendor of lying. The point is that a
correct number can still be the wrong number for your calculation.

---

## Claim 1 — MI355X: 80.5 PFLOPS MXFP4 and 5 PFLOPS FP16

**The two reconciliations have different causes, and that is the lesson.**

*80.5 vs 10.1 PFLOPS MXFP4* — this is a **scope** difference.
`8 × 10.0663 = 80.53`. The 80.5 figure is the **MI355X Platform**: eight OAM modules
on one UBB baseboard. AMD's table labels the columns "AMD INSTINCT MI355X GPU" and
"AMD INSTINCT MI355X PLATFORM"; the headline in a press summary usually drops the
label.

*5 vs 2.5 PFLOPS FP16* — this is a **sparsity** difference. `2 × 2.5 = 5`. The 2.5
PFLOPS figure is dense; 5 PFLOPS assumes structured sparsity, which the CDNA
architecture supports for fp16, bf16, fp8 and int8 when at least two of every four
input values are zero.

**For a per-GPU roofline: 10.1 PFLOPS MXFP4 and 2.5 PFLOPS FP16 dense.** Both
per-device, both dense.

The generalization: *the same claim can be inflated by 8× (scope) or 2× (sparsity)
or 16× (both), and the inflation is always documented — in the column header, the
asterisk, or the endnote you did not read.*

---

## Claim 2 — H100 SXM: 3,958 teraFLOPS FP8 Tensor Core

**The asterisk means "with sparsity".** NVIDIA's product page footnotes the starred
rows explicitly, and the Hopper whitepaper states that "the Sparsity feature exploits
fine-grained structured sparsity in deep learning networks, **doubling** the
performance of standard Tensor Core operations."

- **Dense FP8 Tensor Core: 1,978.9 TFLOP/s.**
- With 2:4 structured sparsity: 3,957.8 TFLOP/s.

**What must be true of your weights.** Hopper's sparsity is *structured*, not
arbitrary: within every contiguous group of four values, at least two must be zero,
and the non-zeros are stored compressed with metadata recording their positions. A
model that is 90% sparse in an *unstructured* pattern gets **zero** benefit. In
practice you reach the sparse number only by pruning to the 2:4 pattern and
fine-tuning back to accuracy — a training-time commitment, not a compiler flag. If
your model is dense (and almost all deployed LLMs are), 1,978.9 is your ceiling and
the headline number is unreachable in principle.

**The FP8-to-FP32 ratio.** `1978.9 / 66.9 = 29.6×`. On one piece of silicon, at one
moment, the answer to "how fast is this GPU?" varies by nearly **30×** depending only
on the precision and the engine. So:

- The fp32 vector ALUs are, for transformer workloads, close to irrelevant as a
  performance ceiling.
- Almost all the arithmetic capability sits behind the Tensor Cores, reachable only
  through fixed-shape MMA instructions, which in turn are reachable only through
  cuBLAS/hipBLASLt, CUTLASS/Composable Kernel, or handwritten
  `wmma`/`mma`/MFMA code.
- A transformer's time therefore does **not** go where the FLOPs are. GEMMs use the
  matrix path and finish fast; softmax, layernorm, activations, residual adds, KV
  cache reads and the all-reduces are all memory- or interconnect-bound and are where
  the wall clock actually goes. That asymmetry is the entire reason kernel fusion
  (A07) and attention rewrites like FlashAttention exist.

---

## Claim 3 — 5.3 TB/s (MI300X) and 3.35 TB/s (H100 SXM)

Both are correct **peak theoretical** figures, and both are the product of a bus
width and a transfer rate (verified in solution 1). No coalesced kernel reaches
either. Three physical reasons:

1. **DRAM refresh.** DRAM cells leak; every row must be periodically read and
   rewritten. Refresh cycles occupy the memory controller and are unavailable to your
   kernel. This cost is unavoidable and grows with capacity and temperature.
2. **Bus turnaround and bank conflicts.** A DRAM bus cannot switch between read and
   write in zero time (`tWTR`/`tRTW` penalties), and a bank that was just activated
   for one row cannot immediately serve a different row (`tRC`, `tRP`). A copy kernel
   is *both* reading and writing, so it pays turnaround continuously. Peak assumes an
   idealized uninterrupted stream.
3. **ECC and command overhead.** Data-centre HBM is ECC-protected; on parts using
   inline ECC, some fraction of the bus carries check bits rather than payload. Even
   with sideband ECC there is address/command traffic that consumes cycles. Add
   request queueing and page-table walks and the controller is never 100% efficient.

**A healthy fraction.** For a fully coalesced streaming kernel, **70–90% of peak** is
the normal band, and 80%+ is good. Below ~60% on a pure copy, suspect your code
(non-coalesced access, too few resident waves, buffers small enough to fit in
Infinity Cache and distort the result) before you suspect the hardware.

**How to establish it on an unknown machine** — and this is the part that matters:
you do not look it up, you measure it. Run the most bandwidth-friendly kernel that
exists (a fully coalesced grid-stride copy on buffers several times larger than the
last-level cache) and treat *its* number as your ceiling. That is exactly what
[`03_measure_vs_spec.cpp`](03_measure_vs_spec_solution.cpp) and A02's `bandwidth`
lab do. Build the B02 roofline on the measured number; if you build it on 5.3 TB/s,
every real kernel will appear to be failing when in fact it is at 95% of achievable.

---

## Claim 4 — Infinity Cache: 256 MB at 17.2 TB/s vs 14.7 TB/s

Two AMD documents, two numbers for the same cache on the same product:

- The [CDNA 3 architecture white
  paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
  states: "The peak bandwidth from the Infinity Cache is an astounding **17.2 TB/s**."
- AMD's [Hot Chips 2024 MI300X
  presentation](https://hc2024.hotchips.org/assets/program/conference/day1/23_HC2024.AMD.MI300X.ASmith\(MI300X\).v1.Final.20240817.pdf)
  states: "256 MB at **14.7 TB/s** peak BW."

**What is going on.** Almost certainly a different measurement boundary, not an
error. A cache has more than one bandwidth: the raw rate of the data arrays, the rate
across the Infinity Fabric interface toward the XCDs, and the rate achievable at a
given sustainable clock. The whitepaper's own Figure 6 annotations list *three*
different aggregate figures for different edges of the hierarchy (51.6 TB/s, 17.2
TB/s, 5.3 TB/s), which is strong evidence that "the" Infinity Cache bandwidth depends
on which arrow you measure.

**What you do about it as an engineer today.** Not "pick the bigger one". Not "the
vendor is lying".

1. **Cite both, with their sources and dates**, in whatever document you are writing.
   A reviewer who sees both numbers and the boundary ambiguity trusts you more than
   one who sees a single confident number.
2. **Use the smaller one for planning.** If a design only works at 17.2 TB/s and
   fails at 14.7, the design has no margin and is not a design.
3. **Measure the thing you actually care about.** You almost never care about "cache
   bandwidth" in the abstract; you care about whether *your* working set fits in
   256 MB and what *your* kernel achieves when it does. Size a benchmark buffer just
   under and just over 256 MB and look at the bandwidth cliff. That measurement is
   worth more than either published number.
4. **If it is load-bearing, ask.** This is the point where you go to the vendor's
   architecture team or your AMD field application engineer and ask which boundary
   each figure describes. Vendors answer this question; the numbers exist because
   someone measured something specific.

This claim is in the drill because it is the most realistic one. Real spec sheets
disagree with each other, and "which document, measured at which boundary, as of
what date" is a professional habit, not pedantry.

---

## Claim 5 — 8 × MI300X: ~10.4 PF BF16/FP16

`8 × 1.3074 PF = 10.46 PF`. It is a **platform** number: eight GPUs, dense, peak
theoretical, bf16/fp16 matrix.

**Which checklist questions the framing hides:** primarily **#4 (per what?)** — the
scope is a node, not a device. It also quietly leans on **#2 (which engine)** and
**#5 (peak or measured)**, since neither is stated in the headline.

**Why a platform FLOP number is nearly useless for kernel work.** Your kernel runs on
*one* GPU at a time. A per-node aggregate is only reachable if the work partitions
perfectly across eight devices with zero communication, which is precisely the thing
that never happens: the moment you shard a model you introduce all-reduces that run
at Infinity Fabric speed (896 GB/s aggregate), not at HBM speed, and you introduce
load imbalance and synchronization. Node-level FLOPs are a **procurement** number —
useful for comparing the cost of one rack against another. They tell you nothing
about whether the GEMM you just wrote is at 30% or 80% of its achievable rate, which
is the only question a performance engineer is ever asked.

---

## Claim 6 — Infinity Fabric: 128 GB/s per pair, 896 GB/s aggregate

`7 links × 128 GB/s = 896 GB/s` — seven links because each MI300X connects directly to
each of the other seven GPUs in the fully meshed 8-GPU node, plus one PCIe Gen 5 x16
link (128 GB/s) to the host.

**Which number for the tensor-parallel all-reduce.** Neither, naively — and getting
this right is the whole reason B07 exists.

- **128 GB/s** is the bandwidth of *one* pair's link. If your collective is
  implemented as a ring or tree, each GPU is simultaneously driving several of its
  links, so a single-link figure understates the available bandwidth.
- **896 GB/s** is the aggregate across all seven of one GPU's links. You only
  approach it if the collective actually engages all seven links at once — which is
  why ROCm's guidance is that collectives over 2 or 4 GPUs "can only use a fraction of
  the potential bandwidth" of the node.

The number you use is the **effective algorithm bandwidth of the specific collective
at your message size**, which you get from `rccl-tests`/`nccl-tests`, not from a
datasheet. Then you check it against 896 GB/s to see how much of the fabric the
collective is managing to use.

**What "bidirectional" does to your arithmetic.** A "128 GB/s bidirectional" link is
64 GB/s in each direction. A ring all-reduce is *symmetric* — every GPU sends and
receives simultaneously — so it can genuinely use both directions and the 128 GB/s
figure is meaningful. But a plain unidirectional `hipMemcpyPeer`, or an
all-gather step that only moves data one way at a given instant, sees **64 GB/s**.
Halving errors here are extremely common; the `topology` lab measures a unidirectional
copy specifically so you see the 2× for yourself.

**The sizing arithmetic for the stated case.** A `[8192, 8192]` bf16 activation is
`8192 × 8192 × 2 B = 134.2 MB`. A ring all-reduce moves approximately `2(N-1)/N ≈ 1.75×`
the tensor size per GPU for `N = 8`, so ~235 MB per GPU per all-reduce. At an
optimistic 700 GB/s of achieved aggregate bandwidth that is ~0.34 ms, and across 80
layers ~27 ms of pure communication per forward pass. Compare that against the GEMM
time on the same layers and you have the tensor-parallel scaling story in one
calculation — which is exactly the calculation B07 formalizes.

---

## The summary table

| Claim | Precision | Engine | Sparsity | Scope | Peak/measured | The number I would use |
|---|---|---|---|---|---|---|
| 1a | MXFP4 | matrix | dense | **node (8 GPU)** | peak | **10.1 PFLOP/s** per GPU |
| 1b | FP16 | matrix | **sparse 2:4** | per GPU | peak | **2.5 PFLOP/s** dense |
| 2 | FP8 | tensor | **sparse 2:4** | per GPU | peak | **1,978.9 TFLOP/s** dense |
| 3 | — | HBM | — | per GPU | **peak** | **measured** coalesced copy GB/s |
| 4 | — | Infinity Cache | — | per GPU | peak, **boundary unclear** | **14.7 TB/s** (the conservative one), both cited |
| 5 | BF16/FP16 | matrix | dense | **node (8 GPU)** | peak | **1.3 PFLOP/s** per GPU |
| 6 | — | Infinity Fabric | — | per link **and** per GPU | peak | **measured** RCCL bus bandwidth at my message size |

---

## The paragraph for a design document

> **Hardware assumptions.** Target is a single AMD Instinct MI300X (CDNA 3, `gfx942`),
> 304 active CUs at up to 2100 MHz, 192 GB HBM3, in an 8-GPU OAM node. Compute budget
> assumes **bf16 Matrix Core, dense, per GPU: 1.3 PFLOP/s peak theoretical**
> ([MI300X data
> sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf),
> retrieved 29 Jul 2026); we plan against 60% of that as achievable and will revise
> from profiling. No structured sparsity is assumed. Memory budget uses **measured**
> coalesced-copy bandwidth from `hip/device_query` + `03_measure_vs_spec` on the
> target node, **not** the 5.3 TB/s peak theoretical figure. Scale-up communication is
> budgeted from measured RCCL bus bandwidth at our message sizes, sanity-checked
> against the 896 GB/s peak aggregate Infinity Fabric figure; a single peer link is
> 128 GB/s bidirectional, i.e. 64 GB/s per direction. All figures are per GPU unless
> stated.

Every clause in that paragraph exists to stop a specific misreading. That is what
makes it worth writing.
