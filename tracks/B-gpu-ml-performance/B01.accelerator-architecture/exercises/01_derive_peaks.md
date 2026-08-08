# Exercise 1 — Derive the peaks yourself (written drill)

**Goal.** Stop treating "5.3 TB/s" and "1.3 PFLOPS" as magic. Both are short
multiplications over numbers a vendor already published, and doing them by hand once
means you will never again be confused by two datasheets that disagree.

No code. Work it on paper, then check
[`../solutions/01_derive_peaks_solution.md`](../solutions/01_derive_peaks_solution.md).

---

## The two identities

**Peak memory bandwidth**

```
bytes/s = transfer_rate (transfers/s)  ×  bus_width_bits / 8
```

The trap: vendors quote *either* a memory **clock** *or* a memory **transfer rate**,
and the two differ by the DDR factor of **2** (HBM moves data on both clock edges).
`5.2 Gbps` and `2600 MHz DDR` are the same thing. Read the unit, not the number.

**Peak vector FLOP/s**

```
FLOP/s = engine_clock (Hz)  ×  compute_units  ×  lanes_per_unit  ×  FLOP_per_lane_per_clock
```

`FLOP_per_lane_per_clock` is **2** for a plain fused multiply-add (one multiply +
one add). It is **4** if the ISA has a packed instruction that does two FMAs per lane
per clock, and it becomes a much larger number for a matrix instruction, which is the
whole point of Section 3.

---

## Given data (all figures verified — see the links)

### Device A — AMD Instinct MI300X (CDNA 3, `gfx942`)

| Quantity | Value | Source |
|---|---|---|
| Memory interface | **8192 bits** | [MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) |
| Memory bus rate | **5.2 Gbps** (a transfer rate, *not* a clock) | [CDNA 3 white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf) |
| Active compute units | **304** | [ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) |
| Lanes per CU (wavefront) | **64** | [ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) |
| Max engine clock | **2100 MHz** | [ROCm MI300/MI350 workload optimization guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/optimization/workload-optimization.html) · [MI300X data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) |
| Matrix FLOP per CU per clock, fp16 | **2048** | [CDNA 3 white paper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf), Table 1 |

### Device B — NVIDIA H100 SXM5 (Hopper, compute capability 9.0)

| Quantity | Value | Source |
|---|---|---|
| Memory interface | **5120 bits** HBM3 | [H100 architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) |
| Memory data rate | **2619 MHz DDR** (a clock — mind the DDR factor) | [H100 architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) |
| SMs | **132** | [H100 architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) |
| FP32 cores per SM | **128** | [H100 architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) |
| Boost clock, non-tensor FP32/FP64 | **1980 MHz** | [H100 architecture white paper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) |
| FP64 cores per SM | **64** | derive it from the published FP64:FP32 ratio |

---

## Your tasks

1. **Bandwidth, Device A.** Compute MI300X peak HBM bandwidth in GB/s and TB/s.
   Show the units at each step. Does your answer match the number AMD prints on the
   product page? If it is off by exactly 2×, you applied the DDR factor to something
   that already included it.

2. **Bandwidth, Device B.** Compute H100 SXM5 peak HBM bandwidth. Here the datasheet
   gives you a *clock*, so the DDR factor **does** apply. Compare against the 3.35
   TB/s NVIDIA advertises.

3. **Vector FP64, Device A.** Using 2 FLOP per lane per clock, compute MI300X peak
   FP64 *vector* FLOP/s. AMD publishes **81.7 TFLOP/s** — do you land on it?

4. **The factor-of-two puzzle.** AMD publishes MI300X FP32 vector peak as
   **163.4 TFLOP/s**, exactly 2× your FP64 answer. A lane is a lane. What
   ISA-level feature explains the doubling, and what does it demand of the
   *programmer* before it materializes?

5. **Vector FP32 and FP64, Device B.** Compute H100 SXM5 peak FP32 (non-tensor) and
   FP64 (non-tensor). Published values are **66.9** and **33.5 TFLOP/s**. Then state
   how many FP64 cores per SM your FP64 answer implies.

6. **Matrix FLOP/s, Device A.** Using 2048 fp16 matrix FLOP per CU per clock, compute
   MI300X peak fp16 *matrix* throughput. Express the result as a multiple of your
   FP32 vector answer from task 4. That multiple is the reason "TFLOPs" is a
   meaningless word without a precision *and* an engine.

7. **Ridge points.** For each device, compute the arithmetic intensity
   (FLOP per byte) at which the *matrix* compute ceiling crosses the *bandwidth*
   ceiling:

   ```
   ridge_point = peak_matrix_FLOP_per_s / peak_bytes_per_s
   ```

   Use fp16 matrix peak for both (MI300X from task 6; H100 SXM5 dense fp16 tensor is
   **989.4 TFLOP/s**). A GEMM has intensity roughly `K` FLOP/byte for a `K`-deep
   reduction in fp16 — roughly how deep does `K` have to be before either chip is
   compute-bound rather than memory-bound?

8. **The honest sentence.** Write one sentence you could say in a design review that
   quotes MI300X performance without being misleading. It must name the precision,
   say whether matrix cores are used, and say whether sparsity is assumed.

---

## What good work looks like

- Units carried through every line (`bits`, `bytes`, `transfers/s`, `FLOP/clock`).
- Your derived numbers within ~1% of the published ones, with any discrepancy
  *explained* rather than fudged.
- Task 7's ridge points in the low hundreds of FLOP/byte. If you get single digits,
  you divided TFLOP/s by TB/s without converting both to base units.
