"""A08 Exercise 2 - masked mean over each row.

Given `x` of shape (rows, cols) and an integer `keep` mask of the same shape, compute
for every row the mean of the entries where `keep != 0`. This is the reduction under
every attention-padding mask and every variable-length-sequence pooling layer.

Two masks meet in this kernel and they are NOT the same thing:
  * the BOUNDS mask `cols < n_cols` - a memory-safety guard, exactly A01's
    `if (idx < n)`. Its job is to stop out-of-range addresses being touched.
  * the DATA mask `keep != 0` - a semantic decision about which valid elements
    participate. Its job is to stop unwanted values entering the reduction.

The edge case that decides whether this code is production-grade: a row where
*nothing* is kept. Count is 0, the mean is 0/0, and a nan leaves your kernel and
poisons every downstream layer. Convention here: a fully masked row returns 0.0.

TASK
  1. Load the row of x and the row of keep, both guarded by the bounds mask.
  2. Sum the kept values and count them.
  3. Divide, guarding the empty-row case so the result is 0.0 and never nan.

HINT
  * `other=0.0` on the x load makes padded lanes neutral for a SUM (unlike softmax,
    where the neutral value for a MAX is -inf).
  * `tl.where(cond, a, b)` evaluates both sides. Guarding a divide means clamping
    the DENOMINATOR, not just selecting the result afterwards.
  * The output is one scalar per row, so the store needs no mask.

Run: python 02_masked_mean.py     (FAILED is expected until the TODOs are done)
Requires: torch + triton (see ../../../docs/SETUP.md).
"""

import torch
import triton
import triton.language as tl

ROWS = 4096
COLS = 1024


@triton.jit
def masked_mean_kernel(
    x_ptr,
    keep_ptr,
    out_ptr,
    x_row_stride,
    keep_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < n_cols

    # TODO 1: load this row of x (other=0.0) and this row of keep (other=0),
    #         both guarded by `in_bounds`.
    # TODO 2: build `valid = in_bounds & (k != 0)`, then reduce with tl.sum to get
    #         the total of the kept values and the count of kept lanes.
    # TODO 3: divide with the empty-row guard, then store one scalar to out_ptr + row.
    #
    # Placeholder so the file runs before you solve it: every row reports 0.0, which
    # is right only for the fully masked row, so the check below reports FAILED.
    tl.store(out_ptr + row, 0.0)


def masked_mean(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and keep.is_cuda, "both tensors must live on the GPU"
    assert x.dim() == 2 and keep.shape == x.shape, "keep must match x"

    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(cols)
    num_warps = 4 if block < 2048 else 8

    masked_mean_kernel[(rows,)](
        x, keep, out,
        x.stride(0), keep.stride(0),
        cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def reference_masked_mean(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Torch reference with the same empty-row convention (0.0, never nan)."""
    kept = keep != 0
    total = (x * kept).sum(dim=-1)
    count = kept.sum(dim=-1)
    return torch.where(count > 0, total / count.clamp(min=1), torch.zeros_like(total))


def benchmark(fn, iters: int = 50, warmup: int = 10) -> float:
    """Milliseconds per call, measured with GPU events after a warmup."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU visible to torch - torch.cuda.is_available() is False.\n"
            "On AMD you need a ROCm torch build; on NVIDIA a CUDA build.\n"
            "See ../../../docs/SETUP.md"
        )

    torch.manual_seed(0)
    x = torch.randn((ROWS, COLS), device="cuda", dtype=torch.float32)
    keep = (torch.rand((ROWS, COLS), device="cuda") > 0.3).to(torch.int32)

    # The two rows that break a naive implementation.
    keep[0] = 0                       # fully masked  -> must be 0.0, not nan
    keep[1] = 0
    keep[1, 5] = 1                     # exactly one kept element -> mean == x[1, 5]

    out = masked_mean(x, keep)
    reference = reference_masked_mean(x, keep)

    finite = bool(torch.isfinite(out).all())
    passed = torch.allclose(out, reference, atol=1e-5, rtol=1e-5)
    empty_ok = out[0].item() == 0.0
    single_ok = abs(out[1].item() - x[1, 5].item()) < 1e-5

    ms = benchmark(lambda: masked_mean(x, keep))
    # x (4 B) + keep (4 B) read per element; the per-row output is negligible.
    bytes_moved = ROWS * COLS * (x.element_size() + keep.element_size())

    print(f"rows x cols      : {ROWS} x {COLS}")
    print(f"time / iter      : {ms:.3f} ms")
    print(f"effective BW     : {bytes_moved / (ms / 1e3) / 1e9:.1f} GB/s")
    print(f"all finite       : {finite}")
    print(f"empty row -> 0.0 : {empty_ok}")
    print(f"single kept elem : {single_ok}")
    print(f"vs torch ref     : {'PASSED' if passed and finite else 'FAILED'}")

    raise SystemExit(0 if passed and finite else 1)


if __name__ == "__main__":
    main()
