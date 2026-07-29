# Solution 02 — Layout algebra drill

Reference answers with the reasoning. Exercise:
[`../exercises/02_layout_algebra.md`](../exercises/02_layout_algebra.md).

**The one rule, restated.** A `Layout` is `(Shape, Stride)`, and the index is the inner
product of the natural coordinate with the stride. Everything below is that rule applied
mechanically — there is no second trick.

---

## Part 1 — Column-major offsets

`L1 = (4,2):(1,4)` → **`index(m,n) = m*1 + n*4`**

| coordinate | offset |
|---|---|
| `(0,0)` | **0** |
| `(1,0)` | **1** |
| `(3,0)` | **3** |
| `(0,1)` | **4** |
| `(2,1)` | **6** |
| `(3,1)` | **7** |

Full table (rows = `m`, columns = `n`):

```
  0   4
  1   5
  2   6
  3   7
```

Consecutive `m` differ by 1 — the **columns are contiguous**.

## Part 2 — Row-major offsets

`L2 = (4,2):(2,1)` → **`index(m,n) = m*2 + n*1`**

| coordinate | offset |
|---|---|
| `(0,0)` | **0** |
| `(1,0)` | **2** |
| `(3,0)` | **6** |
| `(0,1)` | **1** |
| `(2,1)` | **5** |
| `(3,1)` | **7** |

```
  0   1
  2   3
  4   5
  6   7
```

**What distinguishes them:** nothing but **which mode carries stride 1**. Majorness is a
property of the `Stride` *value*, not of the code that consumes the layout — which is exactly
why a CuTe kernel written once handles both, and why the CuTe docs say "majorness is simply
which mode has stride-1".

## Part 3 — A hierarchical layout

`L3 = (2,(2,2)):(4,(2,1))`

**1.** Natural coordinate `(i,(j,k))` → **`index = i*4 + j*2 + k*1`**.

**2.**

| natural coordinate | offset |
|---|---|
| `(0,(0,0))` | **0** |
| `(1,(0,0))` | **4** |
| `(0,(1,0))` | **2** |
| `(0,(0,1))` | **1** |
| `(1,(1,1))` | **7**  (4 + 2 + 1) |

**3.** Colexicographical order over the `(2,2)` multi-mode — read right-to-left, so the
**left-most sub-mode varies fastest**:

| 1-D `n` | natural `(j,k)` |
|---|---|
| 0 | `(0,0)` |
| 1 | `(1,0)` |
| 2 | `(0,1)` |
| 3 | `(1,1)` |

Full 2×4 offset table:

```
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
```

(This matches `print_layout` in the CuTe layout tutorial and in Lab 2's output — check it.)

**4.** **Neither.** For row-major you would need the *last* mode to have stride 1 and the
strides to be a clean prefix product of the shape; for column-major, the first. Here the
column stride is 4 while the row multi-mode carries `(2,1)` — the 1-stride sits in the
*inner* sub-mode of mode 1, so the offsets interleave (`0 2 1 3`). This is the whole point of
hierarchical layouts: the interesting layouts in a GEMM are neither.

## Part 4 — One element, three coordinates

`L4 = (3,(2,3)):(3,(12,1))`, size = 3 × 2 × 3 = 18.
Index formula: **`i*3 + j*12 + k*1`**.

**1-D coordinate `16` → natural coordinate.** Colex means the left-most mode varies fastest,
so peel modes left to right, dividing by each mode's size:

```
i    = 16 mod 3 = 1        rest = 16 / 3 = 5      (mode 0 has size 3)
j    =  5 mod 2 = 1        rest =  5 / 2 = 2      (sub-mode size 2)
k    =  2
-> natural coordinate (1,(1,2))
```

**Index** = `1*3 + 1*12 + 2*1` = **17**.

**2-D coordinate `(1,5)`.** The `5` is a 1-D coordinate into the `(2,3)` multi-mode:
`j = 5 mod 2 = 1`, `k = 5/2 = 2` → natural `(1,(1,2))` → **17**.

**Natural coordinate `(1,(1,2))`** → **17** directly.

**Why they agree:** every `Layout` provides two mappings — input coordinate → *natural*
coordinate (via the `Shape`, by colex order) and natural coordinate → index (inner product
with the `Stride`). All three inputs above resolve to the same natural coordinate, so they must
produce the same index. That is why one layout can be indexed as an 18-element vector, a 3×6
matrix, or a 3×(2×3) tensor without any change to the code holding it. (Corroborated directly
by the CuTe layout tutorial, which prints `crd2idx(16, shape, stride) == 17`.)

## Part 5 — A transposed view

Original: `(4,8):(8,1)` → `index(m,n) = m*8 + n`.

**1. Transposed view: `Shape (8,4)`, `Stride (1,8)`.**

Same pointer. You have swapped the two modes: the new mode 0 is the old mode 1 (stride 1) and
the new mode 1 is the old mode 0 (stride 8). In CuTe:
`make_layout(layout<1>(L), layout<0>(L))`.

**2. Proof.**

```
original(1,5)   = 1*8 + 5*1 = 13
transposed(5,1) = 5*1 + 1*8 = 13      same offset -> same element
```

**3. What it tells you:** a transpose is **free** in this formalism — it is a compile-time
permutation of the stride tuple, not a data movement. A real GEMM still cares because the
*hardware* cares: the MMA instruction wants a specific majorness for A and B (`.row.col.` in
the PTX name), and coalescing and bank conflicts depend on which mode is contiguous *in
memory*. So the transpose is free but its consequences are not: you may still have to
physically restage the tile through shared memory, and *that* is where the swizzle comes in
(part 7).

## Part 6 — A tiled view

Original: `(8,8):(8,1)`. Tiles of 4×4, arranged 2×2.

**1. Derivation.** With `row = 4*ti + i` and `col = 4*tj + j`:

```
offset = row*8 + col
       = (4*ti + i)*8 + (4*tj + j)
       = i*8  +  j*1  +  ti*32  +  tj*4
         ^^^^     ^^^^     ^^^^^^     ^^^^
         intra    intra    tile       tile
         row      col      row        col
```

**2. Read the four strides straight off:**

```
Shape  ((4,4),(2,2))
Stride ((8,1),(32,4))
```

Mode 0 = intra-tile `(i,j)`, mode 1 = tile `(ti,tj)`. **This is exactly the mode profile CuTe's
`zipped_divide` produces**: the docs state `zipped_divide : ((TileM,TileN), (RestM,RestN,...))`
— mode 0 is the tile itself, mode 1 is the layout of the tiles.

**3. Spot checks.**

```
tile (1,1), element (0,0):  0*8 + 0*1 + 1*32 + 1*4 = 36
                            direct: row 4, col 4 -> 4*8 + 4 = 36    OK

tile (0,1), element (3,3):  3*8 + 3*1 + 0*32 + 1*4 = 31
                            direct: row 3, col 7 -> 3*8 + 7 = 31    OK
```

**4. Why the compiler doing it matters more for a tensor-core GEMM:** a copy needs *one* tiling
of *one* tensor. A tensor-core GEMM needs a distinct tiling/partitioning at every hop
(global→shared, shared→register, register→MMA operand, accumulator→epilogue) for A, B, and C,
at three levels (CTA, warp, atom), and each one changes when you change a tile size, a data
type, or the target architecture. Hand-deriving four strides is a five-minute exercise;
hand-deriving and *maintaining* several dozen of them across architectures is where kernels go
to die. See the CuTe layout-algebra doc for `composition`, `logical_divide`, and
`zipped_divide`: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html>

## Part 7 — Banks, padding, and swizzling

`float` = 4 bytes, 32 banks, `bank = word_offset mod 32`.

**1. Unpadded `(32,32):(32,1)`, reading column 5.**

```
row r -> offset = 32*r + 5
bank        = (32*r + 5) mod 32 = 5      for every r, because 32 mod 32 = 0
```

| row | 0 | 1 | 2 | 3 | … | 31 |
|---|---|---|---|---|---|---|
| offset | 5 | 37 | 69 | 101 | … | 997 |
| bank | 5 | 5 | 5 | 5 | … | 5 |

**1 distinct bank → a 32-way conflict**, serialized into 32 accesses. This is A02's transpose
bug exactly.

**2. Padded `(32,32):(33,1)`, reading column 5.**

```
offset = 33*r + 5
bank   = (33*r + 5) mod 32 = (32*r + r + 5) mod 32 = (r + 5) mod 32
```

| row | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| bank | 5 | 6 | 7 | 8 |

`(r + 5) mod 32` is a bijection over `r = 0..31`, so all **32 banks distinct → conflict-free**.
Cost: one dead word per row (32 words = 128 bytes total here) **and** a row stride of 33, which
is no longer a power of two.

**3. Swizzle for row stride 32 words.**

Choose the parameters from the two logs:

```
BBits  = log2(banks)      = log2(32)  = 5      how many bits must vary
SShift = log2(row stride) = log2(32)  = 5      which bits drive the permutation (the row index)
MBase  = 0                                     nothing to protect for single-word access
-> Swizzle<5, 0, 5>
```

(The header requires `abs(SShift) >= BBits`; 5 ≥ 5 holds.)

Trace it. `offset = 32*r + c`, so the low 5 bits are `c` and bits 5–9 are `r`. The mask selects
bits 5–9, shifts them down by 5, and XORs them into bits 0–4:

```
swizzled_offset = offset ^ ((offset & (31 << 5)) >> 5) = (32*r + c) ^ r
bank            = (c XOR r) mod 32
```

Column 5:

| row `r` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| bank = `5 XOR r` | 5 | 4 | 7 | 6 |

XOR with a fixed `c` is a bijection on `0..31`, so all 32 lanes land in **32 distinct banks →
conflict-free**, and the row stride is still 32.

**4. Swizzle for row stride 128 words.** Only `SShift` changes:

```
BBits = 5, MBase = 0, SShift = log2(128) = 7   ->  Swizzle<5, 0, 7>
swizzled_offset = (128*r + c) ^ r
bank            = ((c mod 32) XOR r) mod 32
```

Column 5 gives banks `5, 4, 7, 6` for rows 0–3 again — conflict-free. This is the case Lab 2
prints; run it and check the last column of the table.

> **A real-kernel caveat.** `MBase = 0` is correct for the single-word accesses in this drill.
> When each lane loads a *vector* (e.g. 128 bits = 8 fp16 elements), those contiguous elements
> must not be permuted apart, so `MBase` is set to protect them. CuTe's stock shared-memory
> swizzle atoms therefore use non-zero `MBase`; take the exact triples from
> [`include/cute/swizzle.hpp`](https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp)
> and the CUTLASS layout atoms for your data type rather than deriving them from this drill.

**5. Why a tensor-core kernel must swizzle rather than pad.**

Padding makes the row stride 33 words = 132 bytes. That is **not** a multiple of 16 bytes, so
successive rows are no longer 128-bit aligned. Vectorized `ld.shared.v4` / `st.shared.v4`,
`ldmatrix` (which exists precisely to deliver a tile into MMA fragment order), and TMA/`wgmma`
shared-memory descriptors all require power-of-two, aligned layouts. Padding therefore fixes
the bank conflict by breaking the thing that makes the tensor-core data path fast — you trade a
32-way conflict for scalarized, unaligned loads.

Swizzling permutes *within* the same power-of-two footprint: the stride stays 128, alignment is
preserved, no memory is wasted, and — the part that matters at scale — because a swizzle is
itself a layout-like function, it **composes** with any rank and any tile shape through the same
algebra instead of being re-derived per geometry.

---

## The one-paragraph summary

Every question above was the same operation: inner product of coordinate and stride. Transposing
permutes the stride tuple. Tiling splits a mode and produces a new stride per sub-mode.
Swizzling XORs the offset. Because all four are functions with the same type, CuTe can compose
them at compile time — and that composability, not any single clever index formula, is why
CUTLASS can retarget from Volta quadpairs to Hopper warpgroups without rewriting its index math.
