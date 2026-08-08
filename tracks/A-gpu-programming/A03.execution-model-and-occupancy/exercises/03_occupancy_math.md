# Exercise 3 - Occupancy math when *shared memory* is the limiter

No code. Reason it out on paper, then check
[`../solutions/03_occupancy_math_solution.md`](../solutions/03_occupancy_math_solution.md).

[A01](../../A01.foundations-and-programming-model/README.md)'s drill was register-bound. This
one is **shared-memory-bound**, which behaves very
differently under tuning — and it is the case you actually meet in tiled GEMM, stencils,
and attention kernels.

## Setup

An SM/CU with these (simplified, representative) limits:

| Resource | Limit |
|---|---|
| Max resident threads per SM | **2048** |
| Max resident blocks per SM | **32** |
| Registers per SM | **65,536** |
| Shared memory / LDS available to blocks per SM | **64 KB (65,536 B)** |
| Lanes per warp/wavefront | **32** (see the bonus for 64) |

Your kernel is a 2D stencil:

- **128 threads per block**
- **40 registers per thread**
- each block stages a **40 × 128 float tile** (halo included) in shared memory
  → `40 * 128 * 4 = 20,480 B = 20 KB per block`

## Questions

1. From the **thread** limit alone, how many blocks could be resident per SM?
2. From the **register** limit, how many blocks fit?
3. From the **shared-memory** limit, how many blocks fit?
4. From the **block-count** limit, how many blocks fit?
5. Resident blocks = the **minimum** of the above. What is it, and which resource is the
   **limiter**?
6. What occupancy does that give (resident warps / max warps per SM)?
7. You want **8** resident blocks. What is the largest shared-memory allocation per block
   that allows it, and what does that do to your tile (and therefore to the amount of
   global traffic the kernel must issue)?
8. A colleague suggests compiling with `-maxrregcount=24` to "raise occupancy." Given your
   answer to Q5, what will that actually achieve, and what new risk does it introduce?
9. Now keep the tile — and therefore the 20 KB per block — exactly as it is, but launch
   **256 threads per block** instead of 128 (each thread doing half as much work). What
   happens to resident blocks, resident *threads*, and occupancy? State the general rule
   you just derived about block size when shared memory is the limiter.
10. Which single number in this drill can you *not* safely assume, and where would you go
    to get the real one for your GPU?

## Bonus — AMD CDNA (64-lane wavefront, 64 KB LDS per CU)

Redo Q5 and Q6 for a CDNA-class CU with 64 KB of LDS and 2048 max threads per CU. Does the
**limiting resource** change? Does the **occupancy percentage** change? Does the number you
report as "resident warps" change? Explain why the last answer differs from the other two.
