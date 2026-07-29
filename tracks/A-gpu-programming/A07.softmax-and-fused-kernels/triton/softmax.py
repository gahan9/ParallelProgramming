"""Lab 3 (Triton) - Fused row softmax, verified against torch.softmax.

The C++ labs make every pass explicit: you can count the reads of the row and watch
the pass count turn into runtime. Triton removes that bookkeeping. One program owns
one row, `tl.load` brings the whole row into registers, and the max/sum reductions and
the normalisation all happen on-chip. The result is the fused, single-HBM-read kernel
from hip/softmax.cpp variant 4 - in about fifteen lines.

Mental-model mapping (see README S3 and S4):
  * one Triton program        == one HIP workgroup / CUDA block handling one row
  * `tl.arange(0, BLOCK)`     == the grid-stride loop over the row, vectorised
  * `other=-float("inf")`     == the identity element of the online-softmax monoid,
                                 so padded lanes cannot raise the row max
  * `tl.max` / `tl.sum`       == the wavefront/warp shuffle reductions written by hand
                                 in the C++ labs

Two things Triton does NOT do for you:
  1. The max subtraction. `tl.exp(x)` overflows exactly like `expf(x)`; the `x - m`
     below is the same numerical-stability fix, written by you.
  2. The fully masked row. If every entry of a row is -inf, `m` is -inf and
     `x - m` is nan. The masked variant below shows the guard.

Run: python softmax.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    # -inf for the padding lanes: the identity of (max, sum-of-exp), so they can
    # neither raise the row max nor add to the normalizer.
    x = tl.load(in_ptr + row * in_row_stride + cols,
                mask=cols < n_cols,
                other=-float("inf"))

    m = tl.max(x, axis=0)                       # the max trick, still your job
    e = tl.exp(x - m)                           # every exponent is now <= 0
    y = e / tl.sum(e, axis=0)

    tl.store(out_ptr + row * out_row_stride + cols, y, mask=cols < n_cols)


@triton.jit
def masked_softmax_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    """Causal softmax with the fully-masked-row guard.

    Row 0 keeps one valid column, so a causal mask never empties a row - but a padding
    mask over an all-padding sequence does, and that is the NaN factory this guard
    exists for. Convention here: a fully masked row emits all zeros.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    keep = (cols < n_cols) & (cols <= row)      # additive causal mask
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=cols < n_cols, other=0.0)
    x = tl.where(keep, x, -float("inf"))

    m = tl.max(x, axis=0)
    # If m is -inf the row is empty: exp(x - m) would be exp(nan). Force the whole row
    # to 0 instead of letting a NaN escape into the next layer.
    alive = m > -float("inf")
    e = tl.where(keep & alive, tl.exp(x - tl.where(alive, m, 0.0)), 0.0)
    denom = tl.sum(e, axis=0)
    y = tl.where(denom > 0.0, e / tl.where(denom > 0.0, denom, 1.0), 0.0)

    tl.store(out_ptr + row * out_row_stride + cols, y, mask=cols < n_cols)


def _launch(kernel, x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "2-D GPU tensor expected"
    rows, cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    # More lanes per row once the row stops fitting comfortably in one warp's registers.
    num_warps = 4 if block < 2048 else (8 if block < 4096 else 16)
    kernel[(rows,)](
        x, out, x.stride(0), out.stride(0), cols, BLOCK=block, num_warps=num_warps
    )
    return out


def softmax(x: torch.Tensor) -> torch.Tensor:
    return _launch(softmax_kernel, x)


def causal_softmax(x: torch.Tensor) -> torch.Tensor:
    return _launch(masked_softmax_kernel, x)


def _bench(fn, x: torch.Tensor, iters: int = 50) -> float:
    for _ in range(5):
        fn(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(x)
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../docs/SETUP.md")

    torch.manual_seed(0)
    rows, cols = 8192, 1024
    x = torch.randn((rows, cols), device="cuda", dtype=torch.float32)

    # The same poison as the C++ lab: one score that overflows fp32 under exp().
    x[0, 7] = 100.0

    y = softmax(x)
    ref = torch.softmax(x, dim=-1)
    ok = torch.allclose(y, ref, atol=1e-6, rtol=1e-5)
    finite = bool(torch.isfinite(y).all())

    ms = _bench(softmax, x)
    # A fused row softmax touches HBM exactly twice: one read, one write.
    gbps = (2 * x.element_size() * rows * cols) / (ms / 1e3) / 1e9

    print(f"triton fused softmax : rows = {rows}, cols = {cols}, BLOCK = "
          f"{triton.next_power_of_2(cols)}")
    print(f"time / iter          : {ms:.3f} ms")
    print(f"effective BW         : {gbps:.1f} GB/s  (modelled traffic = 1 read + 1 write)")
    print(f"all finite           : {finite}")
    print(f"vs torch.softmax     : {'PASSED' if ok and finite else 'FAILED'}")

    # Causal variant, checked against a torch reference built the explicit way.
    s = 2048
    scores = torch.randn((s, s), device="cuda", dtype=torch.float32) / (128 ** 0.5)
    causal = torch.ones((s, s), device="cuda").tril().bool()
    ref_causal = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    y_causal = causal_softmax(scores)
    ok_causal = torch.allclose(y_causal, ref_causal, atol=1e-6, rtol=1e-5)

    ms_causal = _bench(causal_softmax, scores)
    gbps_causal = (2 * scores.element_size() * s * s) / (ms_causal / 1e3) / 1e9
    print()
    print(f"triton causal softmax: {s} x {s}")
    print(f"time / iter          : {ms_causal:.3f} ms")
    print(f"effective BW         : {gbps_causal:.1f} GB/s")
    print(f"vs torch reference   : {'PASSED' if ok_causal else 'FAILED'}")

    # A fully masked row: every entry -inf. The guard must produce zeros, not NaN.
    empty = torch.full((1, 64), float("-inf"), device="cuda", dtype=torch.float32)
    unguarded = _launch(softmax_kernel, empty)
    print()
    print("fully masked row (all -inf):")
    print(f"  unguarded softmax_kernel  -> finite? {bool(torch.isfinite(unguarded).all())}"
          "   (False is the lesson)")
    print("  guarded masked kernel     -> see exercises/02_masked_softmax.cpp for the "
          "same guard in HIP")


if __name__ == "__main__":
    main()
