"""A08 Lab 3 - block matmul with `tl.dot` and `@triton.autotune`.

Lab 1 and Lab 2 were memory-bound, where there is little to tune. Matmul is where
tuning decides everything, and where the two remaining pieces of the Triton API
appear: `tl.dot` (a whole block-by-block matrix product as one operation, lowered
onto MFMA on AMD / Tensor Cores on NVIDIA where the shapes and dtypes allow it) and
`@triton.autotune` (search the tile shape and warp/stage counts instead of guessing).

A06 hand-writes this kernel in HIP and CUDA: the shared-memory tiles, the barriers,
the register micro-tile, the boundary guards. All of that is absent below. What
replaces it is a *search*, and the first call pays for that search.

The launch order is deliberately simple - a 2-D grid of (row tile, column tile). The
official Triton tutorial adds a grouped/swizzled ordering to improve L2 reuse; it is
a real further win and is left as a stretch drill in the README.

Run: python matmul.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import time

import torch
import triton
import triton.language as tl

M_DIM = 2048
N_DIM = 2048
K_DIM = 2048


# What the autotuner searches, and why each knob matters:
#   BLOCK_M/N/K : the tile shape. Bigger tiles reuse more operand data per byte
#                 loaded, but cost registers and shared memory (occupancy, A03).
#   num_warps   : how many warps/wavefronts cooperate on one program. NOT portable
#                 in effect - a warp is 32 lanes on NVIDIA and a wavefront is 64 on
#                 AMD, so num_warps=4 is 128 threads on one vendor and 256 on the
#                 other. This is the main reason a config tuned on one vendor is not
#                 automatically right on the other.
#   num_stages  : software-pipelining depth for the K loop (prefetch depth). The
#                 useful value is hardware-dependent; Triton's own docs describe it
#                 primarily in NVIDIA SM80+ terms.
CONFIGS = [
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32},
                  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32},
                  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32},
                  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
                  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},
                  num_warps=8, num_stages=2),
]


# `key` lists the runtime arguments that invalidate the tuning result: change M, N,
# or K and the whole search runs again. Keep the key minimal - every distinct key
# tuple costs one full search.
@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 2-D pointer blocks: broadcasting a column vector against a row vector is how
    # Triton expresses "the tile of addresses this program owns".
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulate in fp32 regardless of operand dtype: the numerical stability of a
    # GEMM rests entirely on a wide accumulator.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        # other=0.0 is mandatory, not cosmetic: a padded lane must contribute a
        # zero product to the dot, and an uninitialised lane would poison the tile.
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
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

    # With @triton.autotune the grid must be a callable: the tile shape is not known
    # until the autotuner has picked a config, and it is handed back in META.
    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))

    matmul_kernel[grid](
        a, b, c,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


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

    # TF32 would silently truncate the fp32 mantissa on NVIDIA and make the
    # reference itself approximate, so the comparison below would be meaningless.
    torch.backends.cuda.matmul.allow_tf32 = False

    torch.manual_seed(0)
    a = torch.rand((M_DIM, K_DIM), device="cuda", dtype=torch.float32)
    b = torch.rand((K_DIM, N_DIM), device="cuda", dtype=torch.float32)

    # The first call compiles AND benchmarks every config in CONFIGS. Measuring it
    # separately is the point: this cost is real and it is paid per process, per key.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = matmul(a, b)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - t0

    reference = torch.matmul(a, b)
    # Same tolerance argument as A06: the summation ORDER differs from torch's, and
    # tl.dot may lower onto matrix cores whose fp32 path can carry reduced-precision
    # inputs. Compare relatively, not bit-exactly.
    max_rel = ((out - reference).abs() / reference.abs().clamp_min(1e-6)).max().item()
    passed = max_rel <= 1e-2

    flops = 2.0 * M_DIM * N_DIM * K_DIM
    ms_triton = benchmark(lambda: matmul(a, b))
    ms_torch = benchmark(lambda: torch.matmul(a, b))

    best = getattr(matmul_kernel, "best_config", None)

    print(f"problem          : C[{M_DIM} x {N_DIM}] = A[{M_DIM} x {K_DIM}] "
          f"* B[{K_DIM} x {N_DIM}]  (fp32)")
    print(f"configs searched : {len(CONFIGS)}")
    print(f"first call       : {first_call_s * 1e3:.0f} ms  "
          "(JIT compile + autotune, paid once per process per key)")
    print(f"best config      : {best if best is not None else 'not exposed by this Triton version'}")
    print(f"triton           : {ms_triton:.3f} ms   "
          f"{flops / (ms_triton / 1e3) / 1e12:.2f} TFLOP/s")
    print(f"torch.matmul     : {ms_torch:.3f} ms   "
          f"{flops / (ms_torch / 1e3) / 1e12:.2f} TFLOP/s   "
          "(rocBLAS / cuBLAS under the hood)")
    print(f"max rel error    : {max_rel:.2e}")
    print(f"result           : {'PASSED' if passed else 'FAILED'}")
    print()
    print("Expect the vendor library to stay ahead on a plain fp32 square GEMM - it")
    print("is hand-tuned per architecture and per shape. Triton's win is not beating")
    print("it; it is getting a respectable fraction of it for a kernel the vendor")
    print("library does not have (a fused, unusual, or shape-specialised one).")
    print("Ratios are hardware- and version-dependent. Measure on your own GPU.")

    # Exit code mirrors the C++ labs so `make run` and CI can detect a failure.
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
