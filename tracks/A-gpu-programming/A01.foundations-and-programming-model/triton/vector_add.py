"""Lab 2 (Triton) - Vector addition, the portable way.

Triton is a Python DSL that JIT-compiles GPU kernels for BOTH NVIDIA and AMD.
Compare this to hip/vector_add.cpp and cuda/vector_add.cu: the same idea in a
fraction of the code, with the block/bounds logic expressed at the "program"
(block) level instead of per-thread.

Mental-model mapping (see README S3):
  * A Triton "program" == one CUDA block / HIP workgroup.
  * tl.program_id(0)   == blockIdx.x
  * BLOCK_SIZE         == blockDim.x (chosen at launch; a compile-time constant)
  * the `mask`         == the `if (idx < n)` bounds guard, vectorized.

Run: python vector_add.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each program handles one contiguous BLOCK_SIZE-wide chunk (this is coalesced).
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n                       # vectorized bounds guard
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "shape mismatch"
    assert a.is_cuda and b.is_cuda, "inputs must be on the GPU"
    c = torch.empty_like(a)
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)      # ceil(n / BLOCK_SIZE) programs
    vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
    return c


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../docs/SETUP.md")

    torch.manual_seed(0)
    n = 1 << 24
    a = torch.rand(n, device="cuda", dtype=torch.float32)
    b = torch.rand(n, device="cuda", dtype=torch.float32)

    c = vector_add(a, b)
    reference = a + b
    passed = torch.allclose(c, reference, atol=1e-5)

    # Time with CUDA events for a clean kernel measurement.
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(5):                        # warmup
        vector_add(a, b)
    torch.cuda.synchronize()
    start.record()
    iters = 50
    for _ in range(iters):
        vector_add(a, b)
    stop.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(stop) / iters

    gbps = (3 * a.element_size() * n) / (ms / 1e3) / 1e9  # 2 reads + 1 write
    print(f"n = {n}, BLOCK_SIZE = 1024, programs = {triton.cdiv(n, 1024)}")
    print(f"kernel time      : {ms:.3f} ms")
    print(f"effective BW     : {gbps:.1f} GB/s")
    print(f"result           : {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
