"""Lab 2 (Triton) - Tiled matrix transpose, the portable way.

The C++ labs make the memory hierarchy explicit (you hand-manage the shared-memory
tile and the padding). Triton raises the abstraction: you describe a *block* of work
and Triton's compiler handles the staging and vectorized, masked loads/stores. The
transpose is expressed as "load a BLOCK x BLOCK tile, store it with axes swapped".

Mental-model mapping (see README S3):
  * one Triton program  == one CUDA block / HIP workgroup handling a tile
  * tl.arange + 2D offset == the per-thread (x, y) indexing done by hand in C++
  * the `mask`           == the `if (x < n && y < n)` bounds guards, vectorized

Run: python transpose.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid_x = tl.program_id(0)      # tile column
    pid_y = tl.program_id(1)      # tile row

    offs_x = pid_x * BLOCK + tl.arange(0, BLOCK)
    offs_y = pid_y * BLOCK + tl.arange(0, BLOCK)

    # Load a BLOCK x BLOCK tile from the source (row-major: row*n + col).
    in_ptrs = in_ptr + offs_y[:, None] * n + offs_x[None, :]
    mask = (offs_y[:, None] < n) & (offs_x[None, :] < n)
    tile = tl.load(in_ptrs, mask=mask, other=0.0)

    # Store transposed: swap the roles of x and y in the destination index.
    out_ptrs = out_ptr + offs_x[:, None] * n + offs_y[None, :]
    out_mask = (offs_x[:, None] < n) & (offs_y[None, :] < n)
    tl.store(out_ptrs, tl.trans(tile), mask=out_mask)


def transpose(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2 and x.shape[0] == x.shape[1], "square GPU matrix"
    n = x.shape[0]
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, block), triton.cdiv(n, block))
    transpose_kernel[grid](x, out, n, BLOCK=block)
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../docs/SETUP.md")

    torch.manual_seed(0)
    n = 4096
    x = torch.rand((n, n), device="cuda", dtype=torch.float32)

    out = transpose(x)
    passed = torch.equal(out, x.t().contiguous())

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        transpose(x)
    torch.cuda.synchronize()
    start.record()
    iters = 50
    for _ in range(iters):
        transpose(x)
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / iters

    gbps = (2 * x.element_size() * n * n) / (ms / 1e3) / 1e9  # read + write
    print(f"triton transpose: n = {n}, BLOCK = 32")
    print(f"time / iter      : {ms:.3f} ms")
    print(f"effective BW     : {gbps:.1f} GB/s")
    print(f"result           : {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
