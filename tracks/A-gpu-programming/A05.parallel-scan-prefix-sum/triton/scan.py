"""Lab (Triton) - Prefix sum with tl.cumsum: the block scan you don't have to write.

The C++ labs build the scan tree by hand (Hillis-Steele's ping-pong buffers, Blelloch's
up-sweep/down-sweep, the bank-conflict padding). Triton hands you the intra-block scan as
a primitive - `tl.cumsum` - and leaves you responsible for the part that is genuinely a
design decision: how to compose per-block scans into a scan of the whole array.

So this file is the same lesson from the other end:

  * rowwise_cumsum - one program per row, whole row inside one block. This is the shape of
    most real ML scans (per-sequence, per-row, per-segment offsets), and it needs no
    cross-block communication at all.
  * scan_1d        - the 3-phase decomposition from README S3, in ~20 lines: per-tile
    `tl.cumsum` + tile totals, scan the totals, add the offsets back.

`tl.cumsum` is the sum-specific case of `tl.associative_scan`, which takes any associative
combine function - the API is shaped by exactly the algebra that makes scan parallel.

Run: python scan.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rowwise_cumsum_kernel(in_ptr, out_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * row_stride + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * row_stride + offs, tl.cumsum(x, axis=0), mask=mask)


@triton.jit
def tile_cumsum_kernel(in_ptr, out_ptr, tile_sums_ptr, n, BLOCK: tl.constexpr):
    """Phase 1: inclusive scan inside a tile + that tile's total (phase 2's input)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.cumsum(x, axis=0), mask=mask)
    tl.store(tile_sums_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def add_tile_offsets_kernel(data_ptr, offsets_ptr, n, BLOCK: tl.constexpr):
    """Phase 3: shift every element of tile `pid` by that tile's exclusive offset."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    off = tl.load(offsets_ptr + pid)
    x = tl.load(data_ptr + offs, mask=mask, other=0.0)
    tl.store(data_ptr + offs, x + off, mask=mask)


def rowwise_cumsum(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "2-D GPU tensor"
    assert x.is_contiguous(), "row_stride below assumes a contiguous layout"
    rows, cols = x.shape
    assert cols <= block, f"one row must fit one block ({cols} > {block})"
    out = torch.empty_like(x)
    rowwise_cumsum_kernel[(rows,)](x, out, cols, cols, BLOCK=block)
    return out


def scan_1d(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """Inclusive scan of a 1-D array via the 3-phase decomposition."""
    assert x.is_cuda and x.dim() == 1 and x.is_contiguous(), "contiguous 1-D GPU tensor"
    n = x.numel()
    out = torch.empty_like(x)
    num_tiles = triton.cdiv(n, block)

    tile_sums = torch.empty(num_tiles, device=x.device, dtype=x.dtype)
    tile_cumsum_kernel[(num_tiles,)](x, out, tile_sums, n, BLOCK=block)
    if num_tiles == 1:
        return out

    # Phase 2 reuses phase 1's kernel on the (much smaller) tile-sum array. One tile is
    # enough as long as num_tiles <= block; a production scan recurses for deeper inputs
    # or skips this hierarchy entirely with decoupled look-back (README S3).
    assert num_tiles <= block, f"{num_tiles} tiles needs a recursive phase 2"
    tile_incl = torch.empty_like(tile_sums)
    scratch = torch.empty(1, device=x.device, dtype=x.dtype)
    tile_cumsum_kernel[(1,)](tile_sums, tile_incl, scratch, num_tiles, BLOCK=block)

    # Inclusive -> exclusive is one shift: tile t starts after all tiles before it, and
    # tile 0 starts at the identity. This is a copy, not another scan.
    offsets = torch.zeros_like(tile_sums)
    offsets[1:] = tile_incl[:-1]

    add_tile_offsets_kernel[(num_tiles,)](out, offsets, n, BLOCK=block)
    return out


def _bench(fn, iters: int = 50) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../docs/SETUP.md")

    torch.manual_seed(0)

    # Small integer values keep every partial sum exactly representable in float32, so the
    # tree-reassociated Triton result must match torch's element for element. On real data
    # a re-associated scan differs in the last bits - README S6.
    rows, cols = 4096, 1024
    x2d = torch.randint(0, 4, (rows, cols), device="cuda", dtype=torch.int32).float()
    out2d = rowwise_cumsum(x2d)
    ref2d = torch.cumsum(x2d, dim=1)
    err2d = (out2d - ref2d).abs().max().item()
    ms2d = _bench(lambda: rowwise_cumsum(x2d))
    gbps2d = (2 * x2d.element_size() * rows * cols) / (ms2d / 1e3) / 1e9

    n = 1 << 20
    x1d = torch.randint(0, 4, (n,), device="cuda", dtype=torch.int32).float()
    out1d = scan_1d(x1d)
    ref1d = torch.cumsum(x1d, dim=0)
    err1d = (out1d - ref1d).abs().max().item()
    ms1d = _bench(lambda: scan_1d(x1d))
    gbps1d = (2 * x1d.element_size() * n) / (ms1d / 1e3) / 1e9

    print(f"rowwise cumsum : {rows} rows x {cols} cols, BLOCK = 1024")
    print(f"  time / iter  : {ms2d:.3f} ms")
    print(f"  effective BW : {gbps2d:.1f} GB/s  (read + write)")
    print(f"  max abs err  : {err2d:g} vs torch.cumsum")
    print(f"  result       : {'PASSED' if err2d == 0.0 else 'FAILED'}")
    print()
    print(f"1-D scan (3-phase): n = {n}, BLOCK = 1024, tiles = {triton.cdiv(n, 1024)}")
    print(f"  time / iter  : {ms1d:.3f} ms")
    print(f"  effective BW : {gbps1d:.1f} GB/s  (useful bytes only: read n + write n)")
    print(f"  max abs err  : {err1d:g} vs torch.cumsum")
    print(f"  result       : {'PASSED' if err1d == 0.0 else 'FAILED'}")


if __name__ == "__main__":
    main()
