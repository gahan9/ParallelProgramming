"""A09 Lab 1 (Triton) - `triton.autotune` in depth.

The kernel here is step 1 of FlashAttention on its own: compute the softmax
statistics of S = Q K^T (the row max `m` and the row exp-sum `l`) WITHOUT ever
storing S. It walks the K blocks and carries the running (m, l) pair, exactly like
flash_attention.py does, but writes only two (batch*heads, seq) vectors.

It is autotuned so you can watch the search happen:
  * the space is BLOCK_M x BLOCK_N x num_warps x num_stages
  * `key=["n_ctx", "HEAD_DIM"]` decides WHEN the search is redone
  * the first call pays compile + benchmark for every surviving config

Set TRITON_PRINT_AUTOTUNING=1 to have Triton itself report the winner and the
time it spent tuning:
    TRITON_PRINT_AUTOTUNING=1 python autotune_demo.py

Run: python autotune_demo.py
Requires: torch + triton (see ../../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import math
import time

import torch
import triton
import triton.language as tl


def is_hip() -> bool:
    return triton.runtime.driver.active.get_current_target().backend == "hip"


# num_stages sets the software-pipelining depth of the `for` loop: how many future
# iterations' loads the compiler keeps in flight (multi-buffered in shared memory)
# while the current iteration computes. More stages hide more memory latency but
# cost shared memory and registers, so past some depth occupancy falls and the
# kernel gets slower.
#
# The AMD (HIP) backend schedules the loop differently and the upstream Triton
# fused-attention tutorial restricts itself to num_stages=1 there; we mirror that
# so the search space contains only configs worth benchmarking on your hardware.
# See https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
NUM_STAGES_OPTIONS = [1] if is_hip() else [2, 3, 4]

CONFIGS = [
    triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n},
                  num_warps=num_warps, num_stages=num_stages)
    for block_m in (32, 64, 128)
    for block_n in (32, 64, 128)
    for num_warps in (4, 8)
    for num_stages in NUM_STAGES_OPTIONS
]

MASKED_SCORE = -1.0e9


@triton.autotune(configs=CONFIGS, key=["n_ctx", "HEAD_DIM"])
@triton.jit
def _softmax_stats_kernel(
    q_ptr, k_ptr, m_ptr, l_ptr,
    stride_qz, stride_qm, stride_qd,
    stride_kz, stride_kn, stride_kd,
    stride_mz, stride_mm,
    n_ctx, softmax_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)

    q_base = q_ptr + pid_z * stride_qz
    k_base = k_ptr + pid_z * stride_kz

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # This loop is the reason num_stages matters: it runs n_ctx/BLOCK_N times and
    # every iteration starts with a global load that the next iteration's compute
    # could be overlapping with.
    for start_n in range(0, n_ctx, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < n_ctx, other=0.0)

        qk = tl.dot(q, k) * softmax_scale
        qk = tl.where(offs_n[None, :] < n_ctx, qk, MASKED_SCORE)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(qk - m_new[:, None]), axis=1)
        m_i = m_new

    m_ptrs = m_ptr + pid_z * stride_mz + offs_m * stride_mm
    l_ptrs = l_ptr + pid_z * stride_mz + offs_m * stride_mm
    tl.store(m_ptrs, m_i, mask=offs_m < n_ctx)
    tl.store(l_ptrs, l_i, mask=offs_m < n_ctx)


def softmax_stats(q: torch.Tensor, k: torch.Tensor):
    """Row max and row exp-sum of Q K^T / sqrt(d). Inputs are (batch*heads, seq, dim)."""
    assert q.is_cuda and k.is_cuda, "inputs must live on the GPU"
    assert q.shape == k.shape and q.dim() == 3, "expected matching (bh, seq, dim)"

    n_bh, n_ctx, head_dim = q.shape
    stats_shape = (n_bh, n_ctx)
    row_max = torch.empty(stats_shape, device=q.device, dtype=torch.float32)
    row_sum = torch.empty(stats_shape, device=q.device, dtype=torch.float32)

    # BLOCK_M comes from the winning config, so the grid cannot be a plain tuple -
    # it must be a callable that the autotuner re-evaluates per config.
    grid = lambda meta: (triton.cdiv(n_ctx, meta["BLOCK_M"]), n_bh)

    _softmax_stats_kernel[grid](
        q, k, row_max, row_sum,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        row_max.stride(0), row_max.stride(1),
        n_ctx, 1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
    )
    return row_max, row_sum


def reference_stats(q: torch.Tensor, k: torch.Tensor):
    """Same statistics the slow way: materialize the whole score matrix in fp32."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    row_max = scores.max(dim=-1).values
    row_sum = torch.exp(scores - row_max[:, :, None]).sum(dim=-1)
    return row_max, row_sum


def benchmark(fn, iters: int = 50, warmup: int = 10) -> float:
    """Milliseconds per call, timed with CUDA/HIP events after a warmup."""
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


def timed_first_call(q: torch.Tensor, k: torch.Tensor) -> float:
    """Wall-clock seconds for the first call at a new key: compile + tune + run."""
    start = time.perf_counter()
    softmax_stats(q, k)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def describe_best() -> str:
    # `best_config` is set by the autotuner after it has benchmarked a key. Guarded
    # because the attribute is an implementation detail of the Autotuner object and
    # has moved between Triton versions.
    best = getattr(_softmax_stats_kernel, "best_config", None)
    return str(best) if best is not None else "<not exposed by this Triton version>"


def check(q: torch.Tensor, k: torch.Tensor) -> bool:
    row_max, row_sum = softmax_stats(q, k)
    row_max_ref, row_sum_ref = reference_stats(q, k)

    # Compare log-sum-exp rather than (m, l) separately: the split between max and
    # sum is not unique (any m' >= m gives a valid pair), but m + log(l) is. Both
    # paths consume the same fp16 inputs and accumulate in fp32, so the only source
    # of disagreement is summation order plus the vendor GEMM's own fp16 rounding -
    # hence a small relative tolerance with an absolute floor.
    lse = row_max + torch.log(row_sum)
    lse_ref = row_max_ref + torch.log(row_sum_ref)
    passed = torch.allclose(lse, lse_ref, rtol=1e-3, atol=2e-2)
    print(f"max abs err (logsumexp) : {(lse - lse_ref).abs().max().item():.3e}")
    print(f"result                  : {'PASSED' if passed else 'FAILED'}")
    return passed


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../../docs/SETUP.md")

    n_bh, head_dim = 32, 64
    dtype = torch.float16
    torch.manual_seed(0)

    n_ctx = 2048
    q = torch.randn((n_bh, n_ctx, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((n_bh, n_ctx, head_dim), device="cuda", dtype=dtype)

    print(f"backend        : {'HIP / ROCm' if is_hip() else 'CUDA'}")
    print(f"search space   : {len(CONFIGS)} configs "
          f"(BLOCK_M x BLOCK_N x num_warps x num_stages={NUM_STAGES_OPTIONS})")
    print(f"key            : ['n_ctx', 'HEAD_DIM'] - a change in either re-tunes")
    print(f"problem        : bh={n_bh} seq={n_ctx} head_dim={head_dim} dtype={dtype}")
    print()

    tune_s = timed_first_call(q, k)
    steady_ms = benchmark(lambda: softmax_stats(q, k))

    print(f"first call     : {tune_s:.2f} s   <- compiles and benchmarks every config")
    print(f"steady state   : {steady_ms:.3f} ms")
    print(f"tuning cost    : ~{tune_s * 1e3 / max(steady_ms, 1e-9):.0f}x one steady call")
    print(f"winning config : {describe_best()}")
    print()

    passed = check(q, k)
    print()

    # Same kernel, new value of a `key` argument -> the cached choice is invalid and
    # the whole search runs again. This is the cost model to internalize: tuning is
    # per distinct key tuple, not per process.
    n_ctx2 = 4096
    q2 = torch.randn((n_bh, n_ctx2, head_dim), device="cuda", dtype=dtype)
    k2 = torch.randn((n_bh, n_ctx2, head_dim), device="cuda", dtype=dtype)

    retune_s = timed_first_call(q2, k2)
    steady2_ms = benchmark(lambda: softmax_stats(q2, k2))
    print(f"seq={n_ctx2} (new key value)")
    print(f"first call     : {retune_s:.2f} s   <- re-tuned, because n_ctx is in `key`")
    print(f"steady state   : {steady2_ms:.3f} ms")
    print(f"winning config : {describe_best()}")
    print()

    # And a shape change that is NOT in `key`: the cached winner is reused, so the
    # first call costs a JIT compile at most, not a search.
    q3 = q[: n_bh // 2].contiguous()
    k3 = k[: n_bh // 2].contiguous()
    reuse_s = timed_first_call(q3, k3)
    print(f"bh={n_bh // 2}, seq={n_ctx} (key unchanged)")
    print(f"first call     : {reuse_s:.2f} s   <- no search: bh is not part of `key`")
    print()
    print("Takeaway: put in `key` the arguments that genuinely change which tile")
    print("shape wins (sequence length, head dim, dtype flags) and nothing else -")
    print("every extra key multiplies your worst-case tuning latency.")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
