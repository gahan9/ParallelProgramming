# Solution 1 — Diagnosing the mystery kernels

> **On numbers in this key.** No profiler output here is a measurement taken by
> the author. Values that appear are either derived arithmetically from the
> source (clearly shown), or stated as a **predicted direction/magnitude** that
> your own run should confirm. If your measurement contradicts a prediction
> here, trust your measurement and work out why — that is a better outcome than
> agreeing with the key.

---

## The answer table, up front

| kernel | Bottleneck | Discriminating evidence (AMD) | Discriminating evidence (NVIDIA) |
|---|---|---|---|
| `kernel_alpha` | **None — at the bandwidth roof.** Memory-bound and that is correct. | Effective GB/s ≈ A02's copy ceiling; `VALUUtilization` ~100%; healthy `MeanOccupancyPerCU` | SpeedOfLight: Memory Throughput high, Compute (SM) Throughput low; Achieved Occupancy healthy |
| `kernel_beta` | **Uncoalesced global access.** | `MemUnitStalled` high and `L2CacheHit` collapses vs. alpha, while `VALUUtilization` and occupancy are *unchanged* | MemoryWorkloadAnalysis: sectors-per-request far above 1; low global load/store efficiency |
| `kernel_gamma` | **Branch divergence** (execution-bound, not memory-bound). | `VALUUtilization` very low (~5%) while `VALUBusy` is high and `MemUnitStalled` is low | Branch Efficiency low / high `smsp__sass_branch_targets_threads_divergent`; SpeedOfLight shows Compute high, Memory near zero |
| `kernel_delta` | **Not enough parallelism** (launch/latency-bound). | `MeanOccupancyPerCU` near zero; `Grid_Size_X` = 4096 in `kernel_trace.csv` | Occupancy section: Achieved Occupancy near zero; LaunchStats shows a tiny grid; both SoL percentages near zero |

The four together are a complete taxonomy: *nothing wrong*, *memory system used
badly*, *execution units used badly*, *hardware not used at all*.

---

## `kernel_alpha` — the control

```cpp
out[i] = SCALE * in[i] + BIAS;        // coalesced grid-stride, 4096 x 256 threads
```

**Arithmetic intensity.** Per element: 4 bytes read, 4 bytes written, one FMA
(2 FLOP). That is **0.25 FLOP/byte** — far to the left of any data-center GPU's
roofline ridge point (which sits in the tens of FLOP/byte). The kernel *must* be
memory-bound; there is no version of this computation that is not.

**Why "memory-bound" is the healthy verdict here, not a complaint.** Memory-bound
is a *classification*, not a defect. The defect would be being memory-bound and
*below* the bandwidth roof. Your proof that alpha is done:

- effective GB/s is within a small factor of A02 Lab 1's copy ceiling — the
  maximum any 1-read-1-write elementwise kernel can reach;
- `VALUUtilization` near 100% — every lane is doing useful work;
- `MeanOccupancyPerCU` is healthy — 4096 blocks × 256 threads keeps every CU fed.

**Answer to Step 4 question 1** — the proof that no optimization is available is
*not* "I couldn't find one". It is: the kernel moves the minimum possible bytes
(each element read once, written once) at close to the maximum possible rate. To
go faster you would have to move fewer bytes, which means **fusing this operation
into a neighbouring kernel** so the intermediate never reaches HBM. That is a
change to the *program*, not to the kernel — and it is exactly the argument for
fused kernels in A07.

## `kernel_beta` — uncoalesced

```cpp
size_t idx = (t % ROWS) * COLS + (t / ROWS);      // ROWS = COLS = 4096
out[idx] = SCALE * in[idx] + BIAS;
```

**What the index does.** Consecutive threads `t, t+1, t+2, …` map to
`idx, idx + 4096, idx + 8192, …`. Consecutive lanes are **16 KB apart** in
memory. Every lane in the wavefront lands in a different cache line, so a
64-byte line is fetched to deliver 4 useful bytes — **1/16 of the traffic is
useful** (1/32 on a 128-byte line). This is A02's coalescing lesson with the
stride hidden inside an innocuous-looking index expression.

**Why the permutation exists at all.** `ROWS * COLS == N` exactly, so the mapping
is a **bijection**: every element is still read and written exactly once, and
`kernel_beta`'s output is byte-for-byte identical to `kernel_alpha`'s. That is
what makes it a clean experiment — the *only* variable is the address pattern.

**The discriminating evidence.** Any single counter can be argued with. The
combination cannot:

| Counter | alpha | beta | What the *difference* rules out |
|---|---|---|---|
| `MeanOccupancyPerCU` | healthy | **unchanged** | not a parallelism problem |
| `VALUUtilization` | ~100% | **unchanged** | not divergence |
| `MemUnitStalled` | moderate | **high** | the memory unit is the constraint |
| `L2CacheHit` | higher | **collapses** | the traffic is scattered, not reused |

Occupancy and VALU identical, memory behaviour catastrophically different → the
address pattern is the only remaining explanation.

**Fix.** Make the innermost index stride-1 — i.e. become `kernel_alpha`. If the
column-major traversal is genuinely required by the algorithm (a real transpose,
say), you cannot delete it; you **relocate** it into on-chip memory, which is
precisely A02's tiled transpose and exercise 3.

## `kernel_gamma` — divergence

```cpp
const int reps = ((i & 31u) == 0u) ? HEAVY : LIGHT;   // 512 : 8
float acc = 1.0f;
for (int k = 0; k < reps; ++k) acc = acc * x + HORNER_C;
```

**Step 4 question 3, worked.** Take one 32-lane group. One lane has `reps = 512`;
the other 31 have `reps = 8`. SIMT executes the loop **the maximum number of
times** and masks off lanes that have finished:

```
useful lane-iterations  = 1 x 512  +  31 x 8   =  512 + 248  =    760
issued  lane-iterations = 32 lanes x 512 iters =            = 16,384
useful / issued         = 760 / 16,384         =  4.64%
```

So **~95% of issued lane slots do nothing.** On AMD the wavefront is 64 lanes and
contains *two* heavy lanes, but the maximum trip count is still 512:

```
useful  = 2 x 512 + 62 x 8 = 1024 + 496 = 1,520
issued  = 64 x 512                      = 32,768
ratio   = 4.64%     <- identical, because the heavy-lane density is the same
```

That 4.6% is what `VALUUtilization` measures: *"percentage of active vector ALU
threads in a wave"* ([MI300/MI200 performance counters](https://rocm.docs.amd.com/en/docs-7.2.1/conceptual/gpu-arch/mi300-mi200-performance-counters.html), ROCm 7.2.1 docs).
Seeing ~5% there, with `MemUnitStalled` low, is the whole diagnosis.

**The trap this kernel exists to teach.** `kernel_gamma`'s *effective GB/s* is
terrible — it moves the same 8 bytes per element as alpha but takes far longer,
so the GB/s column looks like a memory disaster. It is not. The bytes-per-second
metric assumed the kernel was memory-bound; gamma breaks that assumption. **A
derived bandwidth number is only meaningful for a kernel you have already
classified as memory-bound.** This is Step 1's "trap warning" and the answer to
that question.

**Fix.** Make the lanes in a wavefront agree:

1. **Bucket the work.** Partition indices into a heavy list and a light list
   (a stream-compaction pass — A05), then launch one kernel per bucket. Every
   wavefront is then uniform and `VALUUtilization` returns to ~100%.
2. **Hoist the rare path.** If heavy elements are rare, run the light path over
   everything and fix up the heavy ones in a tiny second kernel.

Both trade one launch for ~20× less issued work. Note what does *not* help:
raising occupancy, changing block size, or improving coalescing. Divergence is a
lane-utilization problem, and only lane reorganization fixes it.

## `kernel_delta` — not enough parallelism

```cpp
{DELTA, "kernel_delta", 64, 64, 8.0},     // grid = 64 blocks, block = 64 threads
```

**`kernel_delta`'s body is character-for-character identical to
`kernel_alpha`'s.** Same instructions, same access pattern, same data. The only
difference is the launch geometry: **64 × 64 = 4,096 threads**, which is 64
wavefronts on AMD (128 warps on NVIDIA) spread over a GPU with hundreds of CUs
and tens of thousands of thread slots. Most of the machine is idle, and the few
resident wavefronts cannot keep enough memory requests in flight to hide HBM
latency.

This answers **Step 4 question 2**: `alpha`, `beta`, and `delta` all produce an
identical output array; `alpha` and `delta` have identical *bodies*. When two
identical kernel bodies differ in runtime by a large factor, the cause is
necessarily **outside the kernel body** — launch configuration, or the data it is
pointed at. That is a diagnosis you can reach before running any profiler.

**Evidence, cheapest first.** You do not need counters for this one. `Grid_Size_X`
and `Workgroup_Size_X` are columns in `*_kernel_trace.csv` from the level-1
`--sys-trace` run. Multiply them, compare to your CU count. Counters
(`MeanOccupancyPerCU` near zero) merely confirm what the timeline already told
you — which is the module's whole thesis about ordering.

**Fix.** Launch enough blocks: `alpha`'s 4096 × 256. In real code this bug
usually arrives as a hard-coded grid, a grid sized from a *previous* problem
size, or `hipOccupancyMaxPotentialBlockSize` never being consulted.

---

## Step 5 — proving a fix

The highest-leverage single change is `kernel_delta`'s launch geometry, because
the fix is one line and the predicted result is exact: it should become
`kernel_alpha`. That precision is what makes it a good first fix — a prediction
that *cannot* be fudged after the fact.

After rebuilding, check that **the counter you blamed moved**:
`MeanOccupancyPerCU` should rise to match alpha's. If runtime improved but
occupancy did not, your diagnosis was wrong and something else changed.

## Common wrong answers, and why they are wrong

| Wrong answer | Why it fails |
|---|---|
| "`kernel_gamma` is memory-bound — look at the GB/s." | GB/s is derived from an assumed byte count. `MemUnitStalled` is low; the memory system is idle while the VALU grinds through masked lanes. |
| "`kernel_beta` needs higher occupancy." | Occupancy is *identical* to alpha's. Raising it adds more threads issuing the same wasteful transactions. |
| "`kernel_delta` needs a better access pattern." | Its access pattern is already perfect — identical to alpha's. |
| "`kernel_alpha` is memory-bound, so it needs optimizing." | Memory-bound at the roof is the finish line for this computation. The only remaining lever is fusion, which is a program-level change. |
| "I profiled with `--set full` and read every counter." | You collected a wall of numbers without a hypothesis, and paid replay overhead for sections you never read. Start with SpeedOfLight / the four counters above. |
