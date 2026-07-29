"""A08 Exercise 1 - fuse relu(x + y) into ONE Triton kernel.

PyTorch eager runs `torch.relu(x + y)` as two kernels. Count the HBM traffic in
units of one N-element fp32 array:

    kernel 1  tmp = x + y     read x, read y, write tmp   -> 3 units
    kernel 2  out = relu(tmp) read tmp, write out         -> 2 units
                                                    total -> 5 units

A single fused kernel reads x and y and writes out: 3 units. The intermediate
never reaches HBM, so you delete 2 of 5 units - 40% of the traffic - on a kernel
that is entirely memory-bound. That is the whole reason kernel fusion exists.

TASK
  1. Fill in the kernel so it computes relu(x + y) in one pass.
  2. Run it: the script verifies against torch.relu(x + y) and prints the measured
     bytes-saved and the speedup.

HINT
  * relu is `tl.maximum(v, 0.0)`. There is no `tl.relu`.
  * You need exactly one mask, used by both loads and the store.
  * Nothing about the launch has to change.

Run: python 01_fused_relu_add.py     (FAILED is expected until the TODOs are done)
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

    # TODO 1: load x and y at `offsets`, guarded by `mask`.
    # TODO 2: compute relu(x + y) and store it to out_ptr + offsets under `mask`.
    #
    # Placeholder so the file runs before you solve it - it writes zeros, which is
    # wrong for any positive input, so the check below reports FAILED.
    tl.store(out_ptr + offsets, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=mask)


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
    # Centred on zero so roughly half the sums are clamped: a relu that is never
    # exercised is not tested.
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)

    out = fused_relu_add(x, y)
    reference = torch.relu(x + y)
    passed = torch.allclose(out, reference, atol=1e-6, rtol=1e-5)

    unit = x.element_size() * N
    ms_fused = benchmark(lambda: fused_relu_add(x, y))
    ms_eager = benchmark(lambda: torch.relu(x + y))

    print(f"n                : {N}   ({unit / 1e6:.1f} MB per array)")
    print(f"torch (2 kernels): {ms_eager:.3f} ms   "
          f"{5 * unit / (ms_eager / 1e3) / 1e9:.1f} GB/s   modelled traffic 5 units")
    print(f"triton (1 kernel): {ms_fused:.3f} ms   "
          f"{3 * unit / (ms_fused / 1e3) / 1e9:.1f} GB/s   modelled traffic 3 units")
    print(f"HBM bytes saved  : {2 * unit / 1e6:.1f} MB per call  (2 of 5 units, 40%)")
    print(f"speedup          : {ms_eager / ms_fused:.2f}x")
    print(f"vs torch.relu(x+y): {'PASSED' if passed else 'FAILED'}")
    print()
    print("A 40% traffic cut is the ceiling, not the promise: launch overhead, cache")
    print("hits on the intermediate, and any fusion torch already does all eat into")
    print("it. Report what you measured, not what the model predicted.")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
