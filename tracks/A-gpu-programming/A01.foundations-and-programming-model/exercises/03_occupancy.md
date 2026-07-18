# Exercise 3 - Occupancy whiteboard drill

No code. Reason it out on paper, then check [`../solutions/03_occupancy_solution.md`](../solutions/03_occupancy_solution.md).

## Setup

You launch a kernel on an NVIDIA SM with these (simplified, representative) limits:

- Max resident threads per SM: **2048**
- Max resident blocks per SM: **32**
- Registers per SM: **65,536**
- Shared memory per SM: **64 KB**

Your kernel is launched with:

- **256 threads per block**
- Each thread uses **40 registers**
- Each block uses **8 KB of shared memory**

## Questions

1. From the **thread** limit alone, how many blocks could be resident per SM?
2. From the **register** limit, how many blocks fit? (Registers are allocated per thread, for all
   threads in the block.)
3. From the **shared-memory** limit, how many blocks fit?
4. From the **block-count** limit, how many blocks fit?
5. The actual resident blocks = the **minimum** of the above. What is it, and which resource is the
   **limiter**?
6. Occupancy = (resident warps) / (max warps per SM). With 32 threads/warp and max 2048
   threads/SM (= 64 max warps), what occupancy do you achieve?
7. Name **one** concrete change that would raise occupancy, and state its likely tradeoff.

## Bonus

Redo Q2 for an **AMD** CU where the wavefront is 64 lanes. Does the register-per-thread reasoning
change? Does the *warp/wavefront count* used for the occupancy fraction change?
