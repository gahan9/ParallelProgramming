# Solution 3 — Reduction roofline drill

Setup recap: `N = 2^28 = 268,435,456` float32 elements, MI300X, datasheet peak **5.3 TB/s**,
measured copy ceiling **4.8 TB/s**, measured reduction **0.265 ms**, ~5 µs per kernel launch.
All bandwidth arithmetic uses decimal units (10⁹, 10¹²) to match vendor datasheets.

---

## 1. Bytes moved

```
read  = N × 4 B = 268,435,456 × 4 = 1,073,741,824 B ≈ 1.074 GB
write = 4 B (one float)
total ≈ 1.074 GB
```

The write term is **one scalar** — nine orders of magnitude below the read term. That asymmetry is
what makes reduction the purest bandwidth benchmark in the curriculum: unlike a copy (read + write)
or a transpose (read + write), there is essentially nothing to measure except how fast the machine
can *stream data in*. Multi-pass partials add a little back; see Q6.

## 2. Theoretical minimum time

```
datasheet floor  = 1.0737e9 B / 5.3e12 B/s = 2.026e-4 s = 202.6 µs
copy-ceiling floor = 1.0737e9 B / 4.8e12 B/s = 2.237e-4 s = 223.7 µs
```

The **copy ceiling is the honest target**. The datasheet number is a theoretical pin-level figure
that no kernel — not even a pure `memcpy` — reaches on real silicon, because of DRAM page
activation, refresh, ECC, and the fact that a real access stream is never perfectly scheduled.
Measuring your own ceiling (A02 Lab 1) and judging against *that* is the difference between "we are
leaving 24% on the table" and "we are at 84% of what this machine can do".

## 3. Fraction of peak

```
vs datasheet    : 202.6 / 265 = 76.5%
vs copy ceiling : 223.7 / 265 = 84.4%
```

84% of achievable. In practice this is the point where you stop tuning the reduction and go profile
something else — the remaining 16% is worth ~42 µs, and any of the causes in Q5 costs more
engineering time than it returns.

## 4. Arithmetic intensity and roofline placement

```
FLOPs  = N − 1 ≈ 2.684e8 additions
bytes  ≈ 1.0737e9
AI     = 2.684e8 / 1.0737e9 = 0.25 FLOP/byte
```

The **ridge point** — where a machine stops being memory-bound and starts being compute-bound — is
`peak_FLOPs / peak_bytes_per_s`. For an MI300X at order 1.6e14 FLOP/s (fp32 vector) and 5.3e12 B/s:

```
ridge ≈ 1.6e14 / 5.3e12 ≈ 30 FLOP/byte
```

At 0.25 FLOP/byte the reduction sits **~120× to the left of the ridge**: hopelessly memory-bound.
A 2× faster FPU buys exactly **nothing**. Neither does fusing more math into the kernel — in fact
you could do a hundred extra FLOPs per element for free, which is precisely why frameworks fuse
`x*x` into the same pass when computing an L2 norm.

## 5. Where the missing 16% goes

Ranked by expected cost at this `N`:

1. **Kernel launch and drain latency** — two launches × ~5 µs ≈ 10 µs ≈ **3.8%** of 265 µs. Plus
   the tail of each launch: the last wave of blocks finishes while the machine is already emptying.
2. **Insufficient memory-level parallelism** — a thread that issues one 4-byte load per loop
   iteration cannot keep enough requests in flight to cover HBM latency. Unrolling the grid-stride
   loop 2–4× or switching to `float4` loads is usually worth several percent, and is the single
   highest-yield remaining change.
3. **Pure-read traffic** — the copy ceiling was measured with a balanced read+write stream. A
   read-only stream exercises the memory controller differently and typically lands slightly below
   the copy figure.
4. **Wave quantization / tail effect** — the grid rarely divides evenly into resident blocks per
   CU, so the final partial wave underfills the machine.
5. **Second-pass traffic** — real but tiny at this size (Q6), and the second launch's cost is
   already counted in item 1.

Note what is *not* on the list: divergence, bank conflicts, and instruction count. Ladder stages 1–6
removed those, and once you are at 84% of bandwidth they are no longer the constraint.

## 6. Multi-pass overhead

**Stage-6 style** (one block per `2 × 256 = 512` elements):

```
partials  = 2^28 / 512 = 524,288
pass 1 writes = 524,288 × 4 B = 2.10 MB
pass 2 reads  = 2.10 MB, writes 1024 × 4 B = 4 KB
pass 3 reads  = 4 KB, writes 4 B
extra traffic ≈ 4.2 MB  →  4.2e6 / 1.0737e9 ≈ 0.39% of total
```

**Cascade with a fixed grid of 2048 blocks:**

```
partials  = 2048
extra traffic = 2048 × 4 B (write) + 2048 × 4 B (read) ≈ 16 KB
             →  16e3 / 1.0737e9 ≈ 0.0015%
```

Two conclusions. First, the *bytes* of the second pass never mattered much — even the naive scheme
costs under half a percent. Second, the reason the cascade wins is not traffic at all: it is that
524,288 blocks means 524,288 block-trees' worth of barriers and log-depth overhead, versus 2048.
The cascade replaces `log(N)` levels of parallel tree with a long, free, coalesced serial fold in
registers, keeping only the last `log(2048)` levels as an actual tree.

## 7. When fixed overhead wins

Set the transfer time equal to the ~10 µs launch cost:

```
bytes = 5.3e12 B/s × 10e-6 s = 5.3e7 B = 53 MB
N     = 5.3e7 / 4 ≈ 13.25 million elements
```

Below roughly **13 M elements**, two kernel launches cost at least as much as moving the data. Below
~1 M elements the reduction is essentially *pure launch latency* and the bandwidth number is
meaningless.

That is the everyday case in a training loop: a scalar loss, a per-layer gradient norm, a batch mean
over a few hundred thousand activations. The fixes are not kernel-level:

- **Fuse** the reduction into the kernel that produced the data (compute the sum of squares in the
  same pass that writes the gradient).
- **Batch** many small reductions into one launch (one block per tensor — this is what a fused
  optimizer / `torch._foreach_norm` does).
- **Capture** the sequence in a CUDA/HIP **graph** so the per-launch CPU cost is paid once.
- **Do not synchronize** on the result. `loss.item()` in the inner loop turns a 5 µs kernel into a
  full pipeline stall.

## 8. bfloat16 input, float32 accumulator

**Time.** Bytes halve, so the floor halves: `0.537 GB / 5.3e12 ≈ 101 µs`. A memory-bound kernel
scales directly with element width — this is the single biggest lever available, and it is a *data
type* decision, not a kernel one.

**Accuracy.** bf16 keeps 8 explicit mantissa bits, so every input is already quantized with relative
error up to ~2⁻⁹ ≈ 2e-3 *before* the reduction starts. The fp32 accumulator protects the
*summation* (without it, a bf16 running total stops growing after a few hundred additions, because
`x + δ == x` once `δ` falls below the accumulator's ulp — the classic "my mean is wrong for large
batches" bug). It does nothing about the input quantization, and if the values share a sign those
errors accumulate as a bias rather than cancelling.

**Which to mention first in a design review: the accuracy.** The 2× speedup is uncontroversial and
easy to verify. The accuracy change is the one that can silently degrade a model — and the reviewer
needs to know which consumer is downstream, because a gradient-norm used for clipping tolerates
2e-3 while a loss value used to compare two training runs may not.

---

## Bonus — the atomicAdd code review

> **Recommendation.** The 6 µs win is 2.3% of a 265 µs kernel, and it costs bit-reproducibility:
> `atomicAdd` makes the summation order depend on block scheduling, and float addition is not
> associative, so this value will differ in its low bits from run to run — which means training
> runs are no longer byte-identical across restarts and "did my change break anything?" becomes
> unanswerable by diffing. Please keep the second launch on any value that feeds training math
> (loss, gradient norms, normalization statistics), and re-open this if we ever measure the launch
> as a real bottleneck.

**When I would accept it:** when the consumer provably tolerates nondeterminism — a metric that is
only logged for monitoring, an inference-time statistic, or a debug histogram — or when the
reduction is genuinely launch-bound (small `N`, per Q7) *and* determinism is recovered another way:
accumulate into fixed-point/integer atomics, use a deterministic library reduction (rocPRIM/CUB with
a fixed grid), or keep a pre-allocated partials buffer and a deterministic second pass. "It's only
the last two bits" is not an argument; the question is always *who consumes this number*.
