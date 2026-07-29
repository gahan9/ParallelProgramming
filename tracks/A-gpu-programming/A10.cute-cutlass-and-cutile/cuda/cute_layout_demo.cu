// cute_layout_demo.cu - A10 Lab 2: CuTe layout algebra, on the host.
//
// REQUIRES CUTLASS. This file is compiled only when CUTLASS_DIR is set; the Makefile hard-
// fails with an explanatory message otherwise. Nothing here is vendored or shimmed.
//
// Deliberately host-only and GPU-free. A CuTe `Layout` is a compile-time object that maps a
// coordinate to a linear index, so you can learn the algebra by printing tables - no kernel
// and no device needed. That also keeps this file small enough to audit line by line.
//
// Every construct below comes from official NVIDIA sources:
//   CuTe Layouts (Shape/Stride/Layout/Tensor, print_layout, crd2idx, flatten, layout<I>):
//     https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html
//   Swizzle<BBits, MBase, SShift>:
//     https://github.com/NVIDIA/cutlass/blob/main/include/cute/swizzle.hpp
//
// The runnable, NVIDIA-maintained CuTe examples (partitioning, TiledCopy, TiledMMA, a
// pipelined GEMM) live in the CUTLASS repo and you should work through them there:
//   https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial
//
// Build/run:  make run-cute CUTLASS_DIR=/path/to/cutlass SM_ARCH=sm_90

#include <cstdio>

#include <cute/tensor.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

// ---------------------------------------------------------------------------
// 1. Majorness is data, not code.
// ---------------------------------------------------------------------------
static void demo_majorness()
{
    std::printf("=== 1. Same Shape, different Stride ===\n\n");

    // index(m,n) = m*1 + n*4  -> stride-1 down the columns ("column-major")
    auto col_major = make_layout(make_shape(Int<4>{}, Int<2>{}),
                                 make_stride(Int<1>{}, Int<4>{}));
    // index(m,n) = m*2 + n*1  -> stride-1 across the rows ("row-major")
    auto row_major = make_layout(make_shape(Int<4>{}, Int<2>{}),
                                 make_stride(Int<2>{}, Int<1>{}));

    std::printf("column-major (4,2):(1,4)\n");
    print_layout(col_major);
    std::printf("\nrow-major    (4,2):(2,1)\n");
    print_layout(row_major);

    std::printf("\nSame logical 4x2 matrix. Majorness is just which mode has stride 1,\n"
                "so it lives in the Stride *value* - not in a branch in your kernel.\n\n");
}

// ---------------------------------------------------------------------------
// 2. Hierarchical layouts: one object, neither row- nor column-major.
// ---------------------------------------------------------------------------
static void demo_hierarchical()
{
    std::printf("=== 2. A hierarchical layout ===\n\n");

    // Natural coordinate (i,(j,k)) -> index = i*4 + j*2 + k*1.
    auto h = make_layout(make_shape (Int<2>{}, make_shape (Int<2>{}, Int<2>{})),
                         make_stride(Int<4>{}, make_stride(Int<2>{}, Int<1>{})));

    std::printf("(2,(2,2)):(4,(2,1))\n");
    print_layout(h);

    std::printf("\nStill addressable with a plain 2-D coordinate even though the second mode\n"
                "is itself a pair. size = %d, size<0> = %d, size<1> = %d\n",
                static_cast<int>(size(h)),
                static_cast<int>(size<0>(h)),
                static_cast<int>(size<1>(h)));

    std::printf("flatten -> ");
    print(flatten(h));
    std::printf("\n\n");
}

// ---------------------------------------------------------------------------
// 3. A transposed view is a permutation of modes - no data moves.
// ---------------------------------------------------------------------------
static void demo_transpose_view()
{
    std::printf("=== 3. Transposed view, zero data movement ===\n\n");

    // A 4x8 row-major matrix: index(m,n) = m*8 + n*1.
    auto rowm = make_layout(make_shape (Int<4>{}, Int<8>{}),
                            make_stride(Int<8>{}, Int<1>{}));

    // Swap the two sub-layouts. The result is an 8x4 layout over the *same* memory:
    // index(n,m) = n*1 + m*8, i.e. Shape (8,4), Stride (1,8).
    auto transposed = make_layout(layout<1>(rowm), layout<0>(rowm));

    std::printf("original   (4,8):(8,1)\n");
    print_layout(rowm);
    std::printf("\ntransposed (8,4):(1,8)  <- same pointer, modes swapped\n");
    print_layout(transposed);

    // Spot-check: element (1,5) of the original must be element (5,1) of the transpose.
    std::printf("\noriginal(1,5)   = %d\n", static_cast<int>(rowm(1, 5)));
    std::printf("transposed(5,1) = %d   (equal -> it really is the same element)\n\n",
                static_cast<int>(transposed(5, 1)));
}

// ---------------------------------------------------------------------------
// 4. crd2idx: the index mapping, spelled out.
// ---------------------------------------------------------------------------
static void demo_crd2idx()
{
    std::printf("=== 4. Coordinate -> index, three equivalent coordinates ===\n\n");

    // From the CuTe Layouts tutorial: layout (3,(2,3)):(3,(12,1)).
    // A natural coordinate (i,(j,k)) maps to i*3 + j*12 + k*1.
    auto shape  = make_shape (Int<3>{}, make_shape (Int<2>{},  Int<3>{}));
    auto stride = make_stride(Int<3>{}, make_stride(Int<12>{}, Int<1>{}));

    std::printf("layout (3,(2,3)):(3,(12,1))\n");
    std::printf("  1-D coord 16          -> index %d\n",
                static_cast<int>(crd2idx(16, shape, stride)));
    std::printf("  2-D coord (1,5)       -> index %d\n",
                static_cast<int>(crd2idx(make_coord(1, 5), shape, stride)));
    std::printf("  natural coord (1,(1,2)) -> index %d\n",
                static_cast<int>(crd2idx(make_coord(1, make_coord(1, 2)), shape, stride)));
    std::printf("\nAll three name the same element, so all three give the same index.\n"
                "That is why one Layout can be indexed as a vector, a matrix, or a tensor.\n\n");
}

// ---------------------------------------------------------------------------
// 5. Swizzle: A02's padding trick, generalized.
// ---------------------------------------------------------------------------
static void demo_swizzle()
{
    std::printf("=== 5. Swizzle vs. a 32-way bank conflict ===\n\n");

    // A 32x128 row-major fp32 shared tile. Reading column c touches offsets
    // c, 128+c, 256+c, ... - and bank = offset % 32 is the same for every lane.
    constexpr int kRows      = 32;
    constexpr int kRowStride = 128;         // 2^7 words
    constexpr int kBanks     = 32;          // 2^5 banks of 4-byte words
    constexpr int kColumn    = 0;           // the column being read

    // BBits = log2(banks) = 5, MBase = 0 (nothing to protect for single-word access),
    // SShift = log2(row stride) = 7, so the row index drives the XOR permutation.
    using Swz = Swizzle<5, 0, 7>;

    std::printf("row-major tile, row stride %d words, reading column %d\n\n",
                kRowStride, kColumn);
    std::printf("%5s %10s %6s %12s %6s\n", "row", "offset", "bank", "swizzled", "bank");
    std::printf("%5s %10s %6s %12s %6s\n", "-----", "----------", "------",
                "------------", "------");

    bool plain_seen[kBanks]    = {};
    bool swizzled_seen[kBanks] = {};

    for (int r = 0; r < kRows; ++r) {
        const int offset       = r * kRowStride + kColumn;
        const int swizzled     = Swz{}(offset);
        const int plain_bank   = offset % kBanks;
        const int swizzle_bank = swizzled % kBanks;

        plain_seen[plain_bank]       = true;
        swizzled_seen[swizzle_bank]  = true;

        if (r < 6 || r == kRows - 1) {       // first few rows plus the last, for brevity
            std::printf("%5d %10d %6d %12d %6d\n", r, offset, plain_bank,
                        swizzled, swizzle_bank);
        } else if (r == 6) {
            std::printf("%5s %10s %6s %12s %6s\n", "...", "...", "...", "...", "...");
        }
    }

    int plain_distinct = 0, swizzled_distinct = 0;
    for (int b = 0; b < kBanks; ++b) {
        plain_distinct    += plain_seen[b]    ? 1 : 0;
        swizzled_distinct += swizzled_seen[b] ? 1 : 0;
    }

    std::printf("\ndistinct banks touched by the %d lanes:\n", kRows);
    std::printf("  unswizzled : %2d  -> %d-way conflict (serialized)\n",
                plain_distinct, kRows / plain_distinct);
    std::printf("  swizzled   : %2d  -> conflict-free\n", swizzled_distinct);
    std::printf("\nNote what did NOT change: the row stride is still %d, a power of two, so\n"
                "128-bit vector loads and ldmatrix stay aligned. A02's +1 padding also removes\n"
                "the conflict, but a stride of %d breaks that alignment - which is why\n"
                "tensor-core kernels swizzle instead of padding.\n\n",
                kRowStride, kRowStride + 1);
}

int main()
{
    std::printf("CuTe layout algebra - host-only demo (no GPU used)\n");
    std::printf("==================================================\n\n");

    demo_majorness();
    demo_hierarchical();
    demo_transpose_view();
    demo_crd2idx();
    demo_swizzle();

    std::printf("Next: the real thing. Work through examples/cute/tutorial/sgemm_2.cu in the\n"
                "CUTLASS repo, where these same layouts drive TiledCopy and TiledMMA in a\n"
                "pipelined tensor-core GEMM.\n");
    return 0;
}
