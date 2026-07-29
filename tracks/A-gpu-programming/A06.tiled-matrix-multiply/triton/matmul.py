"""A06 Lab (Triton) - Block matrix multiply, the portable way.

The C++ labs make every level of the tiling explicit: you hand-write the shared
tile, the barriers, the register micro-tile, and the boundary guards. Triton
raises the abstraction one notch - you describe a BLOCK_M x BLOCK_N tile of C
and a BLOCK_K slice of the reduction, and the compiler decides how to stage the
operands in shared memory, how to schedule the barriers, and (on hardware that
supports it) whether to emit MFMA / Tensor Core instructions for `tl.dot`.

Mental-model mapping (see README section 4):
  * one Triton program        == one HIP workgroup / CUDA block owning a C tile
  * `acc` (BLOCK_M x BLOCK_N) == the register micro-tile array `acc[TM][TN]`
  * `mask=` on load/store     == the `if (row < M && col < N)` boundary guards
  * `other=0.0`               == the mandatory zero-fill on out-of-range lanes
  * `tl.dot`                  == the whole `for kk` inner product, possibly
                                 lowered onto matrix cores

Run: python matmul.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl

M_DIM = 1024
N_DIM = 1024
K_DIM = 1024

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulate in fp32 even when the operands are lower precision: the whole
    # numerical stability of GEMM rests on a wide accumulator.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "both operands must live on the GPU"
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], "shape mismatch"

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    matmul_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


def benchmark(fn, iters: int = 50, warmup: int = 10) -> float:
    """Milliseconds per call, measured with CUDA/HIP events after a warmup."""
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
        raise SystemExit("No GPU visible to torch. See ../../../docs/SETUP.md")

    # TF32 would silently truncate the fp32 mantissa on NVIDIA and make the
    # reference itself approximate; turn it off so the comparison is meaningful.
    torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(0)
    a = torch.rand((M_DIM, K_DIM), device="cuda", dtype=torch.float32)
    b = torch.rand((K_DIM, N_DIM), device="cuda", dtype=torch.float32)

    out = matmul(a, b)
    ref = torch.matmul(a, b)

    # Same tolerance argument as the C++ labs: summation ORDER differs, so
    # compare relatively. The bound is looser here because `tl.dot` may lower to
    # matrix cores, whose fp32 path can carry reduced-precision inputs.
    max_rel = ((out - ref).abs() / ref.abs().clamp_min(1e-6)).max().item()
    passed = max_rel <= 1e-2

    flops = 2.0 * M_DIM * N_DIM * K_DIM
    ms_triton = benchmark(lambda: matmul(a, b))
    ms_torch = benchmark(lambda: torch.matmul(a, b))

    print(f"problem        : C[{M_DIM} x {N_DIM}] = A[{M_DIM} x {K_DIM}] "
          f"* B[{K_DIM} x {N_DIM}]")
    print(f"block          : BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K}")
    print(f"triton         : {ms_triton:.3f} ms   "
          f"{flops / (ms_triton / 1e3) / 1e12:.2f} TFLOP/s")
    print(f"torch.matmul   : {ms_torch:.3f} ms   "
          f"{flops / (ms_torch / 1e3) / 1e12:.2f} TFLOP/s   "
          f"(rocBLAS / cuBLAS under the hood)")
    print(f"max rel error  : {max_rel:.2e}")
    print(f"result         : {'PASSED' if passed else 'FAILED'}")
    print()
    print("A single un-autotuned block shape will not match the vendor library.")
    print("Add @triton.autotune over BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages")
    print("to close much of the gap - see README section 6.")


if __name__ == "__main__":
    main()
