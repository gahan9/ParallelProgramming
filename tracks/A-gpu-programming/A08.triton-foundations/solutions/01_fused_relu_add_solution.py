"""A08 Exercise 1 - SOLUTION: relu(x + y) fused into one Triton kernel.

The kernel is the vector-add kernel with one extra operation inserted *between* the
loads and the store. That placement is the entire lesson: the intermediate `x + y`
lives in registers and never becomes an HBM round trip, so fusing costs nothing and
removes 2 of the 5 traffic units the two-kernel version pays.

Why `tl.maximum` and not an `if`:
  There is no `tl.relu`, and you must not reach for control flow here. `v` is a
  block-shaped tensor, not a scalar - a Python `if v > 0` has no meaning on it.
  `tl.maximum(v, 0.0)` is the elementwise, branch-free form, which is also what a
  hand-written CUDA/HIP kernel should emit (`fmaxf`) to avoid warp divergence (A03).

Why the modelled 40% saving is a ceiling, not a promise:
  * At this size the intermediate does not fit in cache, so the two-kernel version
    really does pay HBM for it. At small N it would hit L2 and the gap would shrink.
  * The two-kernel version also pays a second kernel launch (a few microseconds),
    which matters at small N and is invisible at large N.
  * `torch.relu(x + y)` under `torch.compile` would be fused by TorchInductor into a
    generated Triton kernel very much like this one - the comparison here is against
    *eager* torch on purpose.

Run: python 01_fused_relu_add_solution.py
Requires: torch + triton (see ../../../docs/SETUP.md).
"""

import torch
import triton
import triton.language as tl

N = 1 << 24
BLOCK_SIZE = 1024


@triton.jit
def fused_relu_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # The fusion: the sum stays in registers and is clamped in place. No `other=` is
    # needed on the loads - the masked lanes are never stored, so their value cannot
    # reach the output.
    tl.store(out_ptr + offsets, tl.maximum(x + y, 0.0), mask=mask)


def fused_relu_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "shape mismatch"
    assert x.is_cuda and y.is_cuda, "both operands must live on the GPU"

    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    fused_relu_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
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
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)

    out = fused_relu_add(x, y)
    reference = torch.relu(x + y)
    passed = torch.allclose(out, reference, atol=1e-6, rtol=1e-5)

    # The boundary itself: sums that are exactly 0 must survive as 0, and a tail that
    # is not a multiple of BLOCK_SIZE must not be dropped or over-written.
    tail_n = (7 * BLOCK_SIZE) + 3
    tx = torch.randn(tail_n, device="cuda", dtype=torch.float32)
    ty = -tx                                     # every sum is exactly 0.0
    tail_ok = torch.allclose(fused_relu_add(tx, ty), torch.zeros_like(tx))

    unit = x.element_size() * N
    ms_fused = benchmark(lambda: fused_relu_add(x, y))
    ms_eager = benchmark(lambda: torch.relu(x + y))

    print(f"n                 : {N}   ({unit / 1e6:.1f} MB per array)")
    print(f"torch (2 kernels) : {ms_eager:.3f} ms   "
          f"{5 * unit / (ms_eager / 1e3) / 1e9:.1f} GB/s   modelled traffic 5 units")
    print(f"triton (1 kernel) : {ms_fused:.3f} ms   "
          f"{3 * unit / (ms_fused / 1e3) / 1e9:.1f} GB/s   modelled traffic 3 units")
    print(f"HBM bytes saved   : {2 * unit / 1e6:.1f} MB per call  (2 of 5 units, 40%)")
    print(f"speedup           : {ms_eager / ms_fused:.2f}x")
    print(f"tail + zero sums  : {'PASSED' if tail_ok else 'FAILED'}  "
          f"(n = {tail_n}, not a multiple of BLOCK_SIZE)")
    print(f"vs torch.relu(x+y): {'PASSED' if passed and tail_ok else 'FAILED'}")
    print()
    print("Modelled traffic drops 40%; measured speedup is usually somewhat less.")
    print("If yours is far less, check whether the arrays are small enough to be")
    print("cache-resident - then the intermediate was never going to HBM anyway.")

    raise SystemExit(0 if passed and tail_ok else 1)


if __name__ == "__main__":
    main()
