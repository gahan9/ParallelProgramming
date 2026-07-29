"""A08 Lab 2 - row softmax: block reductions and `other=` in one kernel.

Lab 1 used only load/store. This lab adds the two pieces you need for almost every
real kernel: **block reductions** (`tl.max`, `tl.sum`) and a **fill value for masked
lanes** (`other=`). One program owns one row; the row is loaded once into registers
and never touched in HBM again, so the whole max -> exp -> sum -> divide chain is
fused into a single read and a single write.

A07 derives the numerics (the max-subtraction trick, online softmax, the fully
masked row). This lab is about the Triton API surface, so it stays with the simple
two-pass-on-chip formulation and points at A07 for the rest.

Mental-model mapping (see README section 3):
  * tl.max(x, axis=0)   == the warp/wavefront shuffle reduction hand-written in A04
  * other=-float("inf") == the identity of max, so padded lanes cannot win the max
  * num_warps           == how many warps/wavefronts cooperate on ONE program;
                           the compiler splits the block-shaped tensor across them

The constraint to notice: BLOCK must be a power of two and must cover the whole
row, so this shape of kernel only works while a row fits in registers. Wider rows
need the streaming/online formulation from A07.

Run: python fused_softmax.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl

ROWS = 8192
COLS = 1024


@triton.jit
def softmax_kernel(
    in_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # -inf, not 0.0: the padding lanes must be neutral for BOTH reductions below.
    # With other=0.0 a row of negative values would report a max of 0 and the whole
    # normalization would be wrong.
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)              # every exponent is now <= 0, so exp cannot overflow
    y = e / tl.sum(e, axis=0)      # exp(-inf) == 0, so padded lanes add nothing

    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 2, "2-D GPU tensor expected"
    rows, cols = x.shape
    out = torch.empty_like(x)

    block = triton.next_power_of_2(cols)
    # Wider rows need more lanes to hold them without spilling. This hand-written
    # heuristic is exactly what @triton.autotune replaces in Lab 3.
    num_warps = 4 if block < 2048 else (8 if block < 4096 else 16)

    softmax_kernel[(rows,)](
        x, out,
        x.stride(0), out.stride(0),
        cols,
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


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

    # One score large enough that expf(x) overflows fp32. It proves the `x - m`
    # subtraction is load-bearing: drop it and this element alone produces inf/nan.
    x[0, 7] = 100.0

    out = softmax(x)
    reference = torch.softmax(x, dim=-1)
    passed = torch.allclose(out, reference, atol=1e-6, rtol=1e-5)
    finite = bool(torch.isfinite(out).all())

    ms = benchmark(lambda: softmax(x))
    ms_torch = benchmark(lambda: torch.softmax(x, dim=-1))
    # A fused row softmax touches HBM exactly twice: one read, one write.
    bytes_moved = 2 * x.element_size() * ROWS * COLS
    block = triton.next_power_of_2(COLS)

    print(f"rows x cols      : {ROWS} x {COLS}")
    print(f"BLOCK            : {block}   programs = {ROWS}  (one row each)")
    print(f"time / iter      : {ms:.3f} ms")
    print(f"effective BW     : {bytes_moved / (ms / 1e3) / 1e9:.1f} GB/s"
          "  (modelled traffic = 1 read + 1 write)")
    print(f"all finite       : {finite}")
    print(f"vs torch.softmax : {'PASSED' if passed and finite else 'FAILED'}")
    print()
    print(f"torch.softmax    : {ms_torch:.3f} ms   "
          f"{bytes_moved / (ms_torch / 1e3) / 1e9:.1f} GB/s")
    print("torch.softmax is itself a fused kernel, so expect the same order of")
    print("magnitude. The win of writing it yourself is fusing the ops AROUND it.")

    # Exit code mirrors the C++ labs so `make run` and CI can detect a failure.
    raise SystemExit(0 if passed and finite else 1)


if __name__ == "__main__":
    main()
