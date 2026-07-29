# Exercise 02 — Layout algebra drill (written, no compiler)

**Goal.** Be able to go from a `(Shape, Stride)` pair to a linear offset, and back, on a
whiteboard. This is the arithmetic CuTe automates; you cannot debug a CuTe kernel — or read
`sgemm_2.cu` — without being fluent in it.

**Do it on paper first.** You can check parts 1–4 by running Lab 2
(`make run-cute CUTLASS_DIR=...`), which prints the same tables via `cute::print_layout`.
Reference answers with reasoning: [`../solutions/02_layout_algebra_solution.md`](../solutions/02_layout_algebra_solution.md).

**The one rule you need.** A `Layout` is a `(Shape, Stride)` pair. The index is the **inner
product of the (natural) coordinate with the stride**:

```
Shape  (s0, s1, ...)
Stride (d0, d1, ...)
index(c0, c1, ...) = c0*d0 + c1*d1 + ...
```

For a hierarchical shape, the rule applies recursively: `Shape (3,(2,3))` with
`Stride (3,(12,1))` and natural coordinate `(i,(j,k))` gives `i*3 + j*12 + k*1`.

---

## Part 1 — Column-major offsets

Layout `L1 = (4,2):(1,4)`.

Write the index formula, then compute the offsets:

| coordinate | offset |
|---|---|
| `(0,0)` | |
| `(1,0)` | |
| `(3,0)` | |
| `(0,1)` | |
| `(2,1)` | |
| `(3,1)` | |

Then draw the full 4×2 table of offsets.

## Part 2 — Row-major offsets

Layout `L2 = (4,2):(2,1)` — **same shape as `L1`**, different stride.

| coordinate | offset |
|---|---|
| `(0,0)` | |
| `(1,0)` | |
| `(3,0)` | |
| `(0,1)` | |
| `(2,1)` | |
| `(3,1)` | |

Then answer in one sentence: *what, exactly, distinguishes "row-major" from "column-major"
in this formalism?*

## Part 3 — A hierarchical layout

Layout `L3 = (2,(2,2)):(4,(2,1))`.

1. Write the index formula for a natural coordinate `(i,(j,k))`.
2. Compute the offsets for `(0,(0,0))`, `(1,(0,0))`, `(0,(1,0))`, `(0,(0,1))`, `(1,(1,1))`.
3. `L3` is logically a 2×4 matrix, so it also accepts a 2-D coordinate `(m,n)` where `n` is a
   1-D coordinate into the `(2,2)` multi-mode. CuTe enumerates those in **colexicographical**
   order (right-to-left, i.e. the *left-most* sub-mode varies fastest). Fill in the mapping
   and then the full 2×4 offset table:

   | 1-D `n` | natural `(j,k)` |
   |---|---|
   | 0 | |
   | 1 | |
   | 2 | |
   | 3 | |

4. Is `L3` row-major, column-major, or neither? Justify from the strides.

## Part 4 — One element, three coordinates

Layout `L4 = (3,(2,3)):(3,(12,1))` (size 18).

Compute the index for each of these, and explain why they agree:

1. the 1-D coordinate `16`
2. the 2-D coordinate `(1,5)`
3. the natural coordinate `(1,(1,2))`

(Show the 1-D → natural conversion explicitly; that step is where people go wrong.)

## Part 5 — Design a transposed view

You have a **4×8 row-major** `float` matrix in memory, i.e. layout `(4,8):(8,1)`.

1. Write the `(Shape, Stride)` of an **8×4 transposed view** of the same memory — no data
   moves, no copy.
2. Prove it: show that element `(1,5)` of the original and element `(5,1)` of your view
   compute the same offset.
3. What does this tell you about the cost of "transposing" a tensor in this formalism, and
   why does a real GEMM still care about which operand is transposed?

## Part 6 — Design a tiled view

You have an **8×8 row-major** matrix, layout `(8,8):(8,1)`. You want to view it as a
**2×2 grid of 4×4 tiles**, so that a kernel can index it as `(intra_tile_coord, tile_coord)`.

1. Derive the offset formula for tile `(ti,tj)`, intra-tile element `(i,j)`. Start from
   `row = 4*ti + i`, `col = 4*tj + j` and substitute into the row-major formula.
2. Read the four strides straight off your formula and write the rank-2 hierarchical layout
   in the form `((4,4),(2,2)) : ((?,?),(?,?))`, where mode 0 is the intra-tile coordinate
   and mode 1 is the tile coordinate.
3. Verify with two spot checks: tile `(1,1)` element `(0,0)`, and tile `(0,1)` element `(3,3)`.
4. This is what CuTe's `logical_divide` / `zipped_divide` compute for you. Say in one sentence
   why having the compiler do it matters more for a tensor-core GEMM than for a simple copy.
   (Reference: [CuTe layout algebra docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html).)

## Part 7 — Banks, padding, and swizzling

A shared-memory tile of `float` (4 bytes), 32 banks, `bank = word_offset mod 32`.

1. **Unpadded.** Layout `(32,32):(32,1)`. A warp reads **column 5** (one element per lane,
   rows 0..31). Write the offsets and the banks. How many distinct banks? How many-way is the
   conflict?
2. **Padded (A02's fix).** Change the layout to `(32,32):(33,1)`. Recompute the bank for
   rows 0,1,2,3 reading column 5, and state the general formula. Is it conflict-free?
3. **Swizzled.** CuTe's `Swizzle<BBits, MBase, SShift>` computes
   `offset ^ ((offset & yyy_msk) >> SShift)`, where the mask selects `BBits` bits starting
   `MBase + SShift` bits up, and the XOR lands on the `BBits` bits starting at `MBase`.
   For a **row-major tile with row stride 32 words**, choose `BBits`, `MBase`, `SShift` so
   that a column read hits 32 distinct banks. Show the resulting bank for rows 0..3, column 5.
4. Repeat part 3 for a row stride of **128 words**.
5. Padding and swizzling both remove the conflict. Give the concrete reason a **tensor-core**
   kernel must use the swizzle. (Hint: what is special about a power-of-two stride when you
   issue 128-bit loads or `ldmatrix`?)

---

## What "done" looks like

You can, without notes:
- turn any `(Shape, Stride)` into an index formula and evaluate it;
- convert a 1-D coordinate to a natural coordinate through a hierarchical shape;
- write the layout of a transposed or tiled view of existing memory;
- compute shared-memory banks and say whether an access pattern conflicts.

If part 6 or part 7 took you more than ten minutes, redo them — this is the arithmetic you
will be asked to do live in a kernel-engineering interview.
