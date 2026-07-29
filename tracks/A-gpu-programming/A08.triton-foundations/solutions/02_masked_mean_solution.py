"""A08 Exercise 2 - SOLUTION: masked row mean, including the empty-row case.

Three things this drill exists to teach:

1. **The two masks are different.** `in_bounds` is memory safety; `k != 0` is
   semantics. They are combined into `valid` for the reduction, but only `in_bounds`
   may guard a load - masking a load with `valid` would still be correct here, yet it
   couples a data decision to an addressing decision and breaks the moment the keep
   mask is broadcast, strided, or absent.

2. **`other=` must be the identity of the reduction that follows.** For a SUM that
   is 0.0. For the MAX in softmax it is -inf. Using the wrong identity is a silent
   correctness bug that only shows up on the tail block, where masked lanes exist.

3. **The empty row.** `total / count` with `count == 0` yields nan (0/0) or inf, and
   a nan does not stay local - it propagates through every subsequent layer and
   usually surfaces as "the loss went to nan on step 4000", far from this kernel.

The guard is written as a clamped denominator, not just a selected result:

    safe_count = tl.where(count > 0, count, 1.0)
    mean       = tl.where(count > 0, total / safe_count, 0.0)

`tl.where` is a select, so both sides are evaluated - the divide happens whether or
not it is chosen. Clamping the denominator means no nan is ever produced. Relying on
the outer `where` alone happens to work for a single select, but it stops working the
moment the value is used in further arithmetic, and it produces floating-point
exceptions that some tools will flag. Clamp the denominator; it is free.

Convention: a fully masked row returns 0.0. That is a *choice* - other reasonable
conventions are nan-on-purpose (fail loudly) or a sentinel. Whatever you pick, write
it down where the caller can see it, because the caller cannot infer it.

Run: python 02_masked_mean_solution.py
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

    # other=0.0 / other=0: the identity for both reductions below, so a padded lane
    # can neither add to the total nor be counted.
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=in_bounds, other=0.0)
    k = tl.load(keep_ptr + row * keep_row_stride + cols, mask=in_bounds, other=0)

    valid = in_bounds & (k != 0)
    total = tl.sum(tl.where(valid, x, 0.0), axis=0)
    count = tl.sum(tl.where(valid, 1.0, 0.0), axis=0)

    # Clamp the denominator, do not merely select the result: tl.where evaluates both
    # branches, so an unclamped 0 denominator still produces a nan.
    alive = count > 0.0
    mean = tl.where(alive, total / tl.where(alive, count, 1.0), 0.0)

    tl.store(out_ptr + row, mean)


def masked_mean(x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Per-row mean over entries where keep != 0. A fully masked row returns 0.0."""
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

    keep[0] = 0                       # fully masked  -> must be 0.0, not nan
    keep[1] = 0
    keep[1, 5] = 1                     # exactly one kept element -> mean == x[1, 5]

    out = masked_mean(x, keep)
    reference = reference_masked_mean(x, keep)

    finite = bool(torch.isfinite(out).all())
    passed = torch.allclose(out, reference, atol=1e-5, rtol=1e-5)
    empty_ok = out[0].item() == 0.0
    single_ok = abs(out[1].item() - x[1, 5].item()) < 1e-5

    # A non-power-of-two, non-block-aligned width exercises the padded lanes that the
    # `other=` identity exists for. All-kept and all-dropped rows sit side by side.
    ragged_cols = 771
    rx = torch.randn((8, ragged_cols), device="cuda", dtype=torch.float32)
    rkeep = torch.ones((8, ragged_cols), device="cuda", dtype=torch.int32)
    rkeep[3] = 0
    ragged_ok = torch.allclose(
        masked_mean(rx, rkeep), reference_masked_mean(rx, rkeep), atol=1e-5, rtol=1e-5
    )

    ms = benchmark(lambda: masked_mean(x, keep))
    bytes_moved = ROWS * COLS * (x.element_size() + keep.element_size())

    print(f"rows x cols       : {ROWS} x {COLS}")
    print(f"time / iter       : {ms:.3f} ms")
    print(f"effective BW      : {bytes_moved / (ms / 1e3) / 1e9:.1f} GB/s"
          "  (x + keep read once each)")
    print(f"all finite        : {finite}")
    print(f"empty row -> 0.0  : {empty_ok}")
    print(f"single kept elem  : {single_ok}")
    print(f"ragged width {ragged_cols} : {'PASSED' if ragged_ok else 'FAILED'}")
    print(f"vs torch ref      : "
          f"{'PASSED' if passed and finite and ragged_ok else 'FAILED'}")
    print()
    print("This kernel is memory-bound: it reads 8 bytes per element and writes one")
    print("float per row, so judge it against the copy ceiling from A02 Lab 1, not")
    print("against a FLOP number.")

    raise SystemExit(0 if passed and finite and ragged_ok else 1)


if __name__ == "__main__":
    main()
