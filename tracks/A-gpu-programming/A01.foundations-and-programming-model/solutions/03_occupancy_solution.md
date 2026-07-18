# Exercise 3 solution - Occupancy whiteboard drill

Recall the limits: 2048 threads/SM, 32 blocks/SM, 65,536 registers/SM, 64 KB shared/SM.
Kernel: 256 threads/block, 40 registers/thread, 8 KB shared/block.

## Worked answers

**Q1 - thread limit.** `2048 / 256 = 8` blocks.

**Q2 - register limit.** Registers are per thread, so per block:
`256 threads x 40 regs = 10,240 regs/block`.
`floor(65,536 / 10,240) = floor(6.4) = 6` blocks.

**Q3 - shared-memory limit.** `floor(64 KB / 8 KB) = 8` blocks.

**Q4 - block-count limit.** `32` blocks.

**Q5 - actual resident blocks + limiter.**
`min(8, 6, 8, 32) = 6` blocks. **Registers are the limiter.**

**Q6 - occupancy.**
Resident threads `= 6 x 256 = 1536`. Warps `= 1536 / 32 = 48`.
Max warps `= 2048 / 32 = 64`. Occupancy `= 48 / 64 = 75%`.

**Q7 - one change + tradeoff.**
Cut register usage per thread (refactor the kernel, or cap with `-maxrregcount` / launch bounds).
At 32 regs/thread: per block `= 256 x 32 = 8192`; register limit `= floor(65,536/8192) = 8` blocks,
which now ties the thread limit -> 8 resident blocks = 2048 threads = 64 warps = **100% occupancy**.
**Tradeoff:** forcing fewer registers can cause *register spilling* to local memory (backed by slow
global memory), which may hurt more than the extra occupancy helps. Always measure; higher occupancy
is a means (latency hiding), not the goal (throughput). Increasing threads/block does *not* help
here, because the kernel is register-bound, not thread-bound.

## Bonus - AMD CDNA (64-lane wavefront)

- **Register reasoning is conceptually identical:** VGPRs are allocated per work-item for the whole
  wavefront, so "registers/thread x threads/block" still drives how many blocks fit - but the VGPR
  file size, per-SIMD allocation, and granularity differ from the NVIDIA numbers above. Use the real
  figures for your target from the
  [AMD GPU architecture specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html).
- **The occupancy *fraction* changes unit:** you count **wavefronts** (64 lanes), not 32-lane warps.
  Resident threads / 64 gives resident wavefronts; divide by the CU's max wavefronts for occupancy.
- **Practical takeaway:** a block of 256 threads is 4 wavefronts on AMD but 8 warps on NVIDIA - the
  same code, different scheduling granularity. This is why block sizes that are multiples of 64 are
  the safe portable choice (Module A01 S3).

## The interview point

The examiner is checking whether you know that occupancy is a **min over multiple resource limits**
(threads, blocks, registers, shared memory), can identify the *binding* constraint, and understand
that maximizing occupancy is not automatically the right objective. State the limiter, then reason
about whether raising occupancy would actually raise throughput.
