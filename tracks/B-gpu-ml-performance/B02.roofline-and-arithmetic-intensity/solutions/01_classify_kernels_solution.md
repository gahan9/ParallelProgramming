# Solution 1 — Classify these kernels

Reference answers for [`../exercises/01_classify_kernels.md`](../exercises/01_classify_kernels.md).
The arithmetic matters less than the *habit*: FLOPs, bytes, divide, compare to the ridge,
pick the lever.

Device R: `P_v = 60 TFLOP/s` (fp32 vector), `P_m = 1200 TFLOP/s` (bf16 matrix),
`B = 4.0 TB/s`, `N = 2^26 = 67,108,864`.

---

## Part A — Ridge points

**A1.**

```
I_ridge(fp32 vector) = 60e12 FLOP/s  ÷ 4.0e12 byte/s =   15 FLOP/byte
I_ridge(bf16 matrix) = 1200e12 FLOP/s ÷ 4.0e12 byte/s =  300 FLOP/byte
```

**A2.** Matrix cores raised `P` by 20× while `B` barely moved, so the ridge point moved 20×
to the right. Every kernel keeps the intensity its *algorithm* gives it, so kernels that used
to sit near the ridge are now far to the left of it. Faster compute makes *more* things
memory-bound, not fewer — this is the whole reason fusion, quantization, and FlashAttention
became load-bearing rather than nice-to-have.

---

## Part B — Classifications

`I` is exact; attainable rate is `min(P, I × B)` with `B = 4000 GB/s`.

| # | Kernel | FLOPs | bytes | `I` | attainable | verdict | first lever |
|---|--------|-------|-------|-----|-----------|---------|-------------|
| 1 | `saxpy` fp32 | `2N` | `12N` (read x, read y, write y) | **0.167** | 667 GFLOP/s | memory-bound | fuse it into whatever produced `x`; nothing else moves the needle |
| 2 | `relu` fp32 out-of-place | `N` | `8N` | **0.125** | 500 GFLOP/s | memory-bound | do it in-place (`4N`→ `I = 0.25`) or fuse into the preceding GEMM epilogue |
| 3 | Sum reduction fp32 | `N` | `4N` (+1 scalar) | **0.25** | 1000 GFLOP/s | memory-bound | already near-optimal; target % of copy ceiling, not FLOP/s |
| 4 | 5 unfused elementwise fp32 | `5N` | `5 × 8N = 40N` | **0.125** | 500 GFLOP/s | memory-bound | fuse (row 5) |
| 5 | The same 5, fused | `5N` | `8N` | **0.625** | 2500 GFLOP/s | memory-bound | now bandwidth-optimal; next lever is dtype |
| 6 | Softmax fp32, 3-pass | `5N` | `4N + 8N + 8N = 20N` | **0.25** | 1000 GFLOP/s | memory-bound | single-pass (online) softmax → `8N` bytes → `I = 0.625` |
| 7 | fp32 GEMM 8192³ | `2 × 8192³ = 1.100e12` | `3 × 8192² × 4 = 8.053e8` | **1365** | 60 TFLOP/s (= `P_v`) | compute-bound | instruction mix, matrix cores, tile/register blocking |
| 8 | bf16 GEMM 8192×8192×128 | `2 × 8192 × 8192 × 128 = 1.718e10` | `2 × (8192·128 + 128·8192 + 8192²) = 1.384e8` | **124** | 496 TFLOP/s | **memory-bound** vs `P_m` | this is an epilogue problem: fuse the projection with what follows so `C` never round-trips |
| 9 | bf16 GEMM 8192³ | `1.100e12` | `3 × 8192² × 2 = 4.027e8` | **2731** | 1200 TFLOP/s (= `P_m`) | compute-bound | matrix-core utilization, software pipelining |

Worth noticing in row 8: the `C` matrix is `8192² × 2 B = 134 MB` while `A` and `B` together
are only `4 MB`. A thin-K GEMM is dominated by *writing its own output*. That is why fused
epilogues (bias + activation + cast, all before the store) exist.

**B10.** Rows **4 and 5** — same verdict (memory-bound), same FLOPs, but `40N` bytes versus
`8N`, hence 500 vs 2500 GFLOP/s attainable and a ~5× wall-clock win. **Fusion does not usually
change your classification; it moves you up the memory-bound slope by deleting bytes.** People
misremember fusion as "making a kernel compute-bound." It rarely does. It makes a memory-bound
kernel *finish sooner*, which is what you were actually paid for.

**B11.** Row 8 has `I = 124 FLOP/byte`.

- Against the **fp32 vector** ridge (15): `124 > 15` → looks compute-bound.
- Against the **bf16 matrix-core** ridge (300): `124 < 300` → memory-bound.

The kernel executes on matrix cores, so the matrix-core ceiling is the only relevant one:
**memory-bound**, attainable `124 × 4000 = 496 TFLOP/s`, i.e. at most ~41% of `P_m` no matter
how good the inner loop is. Comparing a bf16 matrix-core kernel to the fp32 ridge is the most
common way engineers talk themselves into optimizing the wrong thing.

---

## Part C — Where the model stops helping

**C1.** `N = 4096` fp32 vector add:

```
bytes = 3 × 4096 × 4 = 49,152 B
time at 4.0 TB/s = 49,152 / 4.0e12 = 12.3 ns
launch overhead                    = 4,000 ns
```

The launch costs **~325×** more than the memory traffic. `I` is unchanged at 0.083, so the
roofline gives the same verdict as the huge case — and that verdict is **useless**, because the
kernel never reaches steady state. It is **launch/latency-bound**. Fixes: batch the work into
one larger launch, fuse it into a neighbouring kernel, use CUDA/HIP graphs to amortize launch
cost, or keep it on the CPU. The roofline assumes an infinitely long steady-state kernel; a
12 ns kernel is not one.

**C2.** `I = 40` FLOP/byte, measured 8,000 GFLOP/s.

```
attainable = min(60,000, 40 × 4000 = 160,000) = 60,000 GFLOP/s   (compute roof binds)
achieved / attainable = 8,000 / 60,000 = 13%
```

**Neither.** A kernel at 13% of its own roof is not limited by that roof — it is limited by
something the roofline does not model. Three likeliest causes:

| Cause | What to check |
|---|---|
| **Latency-bound / too few resident warps** — not enough parallelism to hide instruction and memory latency | occupancy achieved vs theoretical; warp-stall reasons (Nsight Compute *Warp State Statistics*, `rocprof-compute` wavefront-occupancy panel) |
| **Dependency chains / poor ILP** — the FLOPs are issued serially | instructions-per-cycle vs pipe utilization; SASS/ISA inspection for back-to-back dependent FMAs |
| **Wrong instruction mix** — the "FLOPs" are being executed as transcendentals, fp64, or scalar ops rather than packed FMAs | instruction-mix / pipe-utilization breakdown (VALU vs MFMA vs transcendental) |

Also worth ruling out: launch-bound tail effects (grid not a multiple of the CU count, so the
last wave runs alone) and register spilling turning "register-resident" work into local-memory
traffic.

**C3.** 3-pass softmax on a 4096-element fp32 row (16 KB), scored against **L2** at 12 TB/s.
The row is read/written 5 times, but all 5 hit L2, so *L2* traffic is still `20N` bytes while
*HBM* traffic is only `4N + 4N = 8N` (one read in, one write out).

```
I_HBM = 5N / 8N   = 0.625 FLOP/byte  → vs HBM ridge  15  → memory-bound
I_L2  = 5N / 20N  = 0.25  FLOP/byte  → vs L2 ridge  60e12/12e12 = 5 → memory-bound
```

The verdict does not change, but the *diagnosis* does: against HBM the kernel looks 2.5× better
than the naive count suggested, which means the remaining cost is on-chip traffic and pass
overhead, not HBM. A model carrying one ceiling per memory level is a **hierarchical roofline**
(Yang, Kurth & Williams). Both Nsight Compute and ROCm Compute Profiler produce them, and this
is precisely the case they exist for.

**C4.** Two ways "95% of roofline" hides a bad kernel:

1. **The bytes are wrong.** The roofline is drawn from the traffic the kernel *actually*
   generated. An uncoalesced kernel moving 8× more bytes than it needs to has an 8× lower
   intensity, so its roof is 8× lower — and hitting 95% of that lowered roof means it is
   efficiently wasting bandwidth. Always compare against the intensity of the *ideal*
   implementation, not the one you wrote.
2. **The algorithm is wrong.** A perfectly executed `O(n²)` kernel at 95% of roofline loses to
   a sloppy `O(n log n)` kernel at 40%. The roofline scores the implementation, never the
   algorithm — and it says nothing about whether the kernel needed to run at all (an
   unnecessary layout conversion at 99% of roofline is still pure waste).
