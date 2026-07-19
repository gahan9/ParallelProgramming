# Exercise 3 - Memory-hierarchy sizing drill

No code. Reason on paper, then check [`../solutions/03_hierarchy_solution.md`](../solutions/03_hierarchy_solution.md).

## Setup

An NVIDIA SM (simplified, representative) offers:

- Shared memory per SM: **100 KB** (configurable partition)
- Registers per SM: **65,536** (32-bit)
- Max resident threads per SM: **2048**

Your tiled kernel uses:

- **256 threads per block** (a 16×16 tile of threads, or 32×8 — your choice)
- **48 registers per thread**
- A shared-memory tile of `float tile[TILE][TILE + 1]`

## Questions

1. For `TILE = 32`, how many **bytes** of shared memory does one block use? (Remember the `+1` pad.)
2. From the **shared-memory** budget alone, how many blocks can be resident per SM at `TILE = 32`?
3. From the **register** budget, how many blocks fit? (registers/thread × threads/block).
4. From the **thread** limit, how many blocks fit?
5. The resident blocks = min of the above. What is it, and which resource is the **limiter**?
6. You want **≥ 2 resident blocks** to hide latency. If shared memory is the limiter, what is the
   largest square `TILE` (padded) that still allows 2 blocks? Show the inequality.
7. State the fundamental **tradeoff** a larger tile buys you and what it costs.

## Bonus

Redo Q1–Q2 for an **AMD CDNA CU** with **64 KB of LDS**. Which platform is more shared-memory
constrained for this kernel, and how would you adapt the tile size to stay portable?
