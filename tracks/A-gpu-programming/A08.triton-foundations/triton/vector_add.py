"""A08 Lab 1 - vector add: the smallest complete Triton kernel.

This is the same computation as A01's hip/vector_add.cpp and cuda/vector_add.cu,
which are ~90 lines each of allocation, copies, error checks, and index math. The
whole kernel here is five lines, and it compiles for AMD and NVIDIA from one source.

Mental-model mapping (see README section 3):
  * one Triton program   == one HIP workgroup / CUDA block
  * tl.program_id(0)     == blockIdx.x
  * BLOCK_SIZE           == blockDim.x, but a COMPILE-TIME constant (tl.constexpr)
  * `mask=offsets < n`   == the `if (idx < n)` guard, applied to a whole block at once
  * grid = (cdiv(n, B),) == the <<<grid, block>>> launch configuration

What you do NOT write, and the compiler decides for you: how the BLOCK_SIZE
elements are split across lanes, the vector width of each load, whether anything
is staged through shared memory, and the instruction schedule.

Run: python vector_add.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl

N = 1 << 24
BLOCK_SIZE = 1024


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # One contiguous BLOCK_SIZE-wide slice per program. Contiguous is deliberate:
    # it is what lets the compiler emit coalesced, vectorized accesses (A02).
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape, "shape mismatch"
    assert a.is_cuda and b.is_cuda, "both operands must live on the GPU"

    c = torch.empty_like(a)
    n = a.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)
    return c


def compiler_report(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Best-effort peek at what the JIT produced for this kernel.

    The launch returns a compiled-kernel handle whose attribute names are not a
    stable public API - they have changed across Triton releases - so every field
    is read defensively and simply omitted when this version does not expose it.
    """
    c = torch.empty_like(a)
    n = a.numel()
    handle = vector_add_kernel[(triton.cdiv(n, BLOCK_SIZE),)](
        a, b, c, n, BLOCK_SIZE=BLOCK_SIZE
    )
    fields = {name: getattr(handle, name, None)
              for name in ("n_regs", "n_spills", "shared")}
    return {name: value for name, value in fields.items() if value is not None}


def benchmark(fn, iters: int = 50, warmup: int = 10) -> float:
    """Milliseconds per call, measured with GPU events after a warmup.

    The warmup matters more in Triton than in C++: the first call to a @triton.jit
    function pays JIT compilation, so timing it would measure the compiler.
    """
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

    props = torch.cuda.get_device_properties(0)
    # torch reports the wavefront as `warp_size` on ROCm too (64 on CDNA), but the
    # attribute is not present on every torch build, so treat it as optional.
    warp = getattr(props, "warp_size", None)
    backend = "ROCm/HIP" if torch.version.hip else "CUDA"

    torch.manual_seed(0)
    a = torch.rand(N, device="cuda", dtype=torch.float32)
    b = torch.rand(N, device="cuda", dtype=torch.float32)

    out = vector_add(a, b)
    reference = a + b
    passed = torch.allclose(out, reference, atol=1e-5, rtol=1e-5)

    ms = benchmark(lambda: vector_add(a, b))
    # 2 reads + 1 write per element: the only traffic this kernel can possibly do.
    gbps = (3 * a.element_size() * N) / (ms / 1e3) / 1e9

    print(f"backend          : {backend}  ({props.name})")
    print(f"warp/wavefront   : {warp if warp else 'not reported by this torch build'}")
    print(f"n                : {N}")
    print(f"BLOCK_SIZE       : {BLOCK_SIZE}   programs = {triton.cdiv(N, BLOCK_SIZE)}")
    print(f"time / iter      : {ms:.3f} ms")
    print(f"effective BW     : {gbps:.1f} GB/s")
    print(f"vs torch a + b   : {'PASSED' if passed else 'FAILED'}")

    report = compiler_report(a, b)
    if report:
        print(f"compiler report  : {report}")
        print("                   (n_spills > 0 means registers spilled to memory - "
              "a real slowdown)")

    ms_torch = benchmark(lambda: a + b)
    print()
    print(f"torch a + b      : {ms_torch:.3f} ms   "
          f"{(3 * a.element_size() * N) / (ms_torch / 1e3) / 1e9:.1f} GB/s")
    print("Both are memory-bound and both should land near your HBM copy ceiling")
    print("(A02 Lab 1). A hand-written kernel has nothing left to win here.")

    # Exit code mirrors the C++ labs so `make run` and CI can detect a failure.
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
