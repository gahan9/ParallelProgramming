"""A09 Exercise 1 - SOLUTION. Causal masking done for both correctness and speed.

Two distinct fixes, and confusing them is the classic mistake:

  1. Block skipping (performance). Query block `pid_m` covers rows
     [pid_m*BLOCK_M, pid_m*BLOCK_M + BLOCK_M). Its largest reachable key index is
     the last row of the block, so the walk stops at

         hi = min((pid_m + 1) * BLOCK_M, n_ctx)

     The `min` with n_ctx is not cosmetic: without it the last, partial block walks
     past the end of the sequence, doing masked-out work for nothing. Every K/V
     block beyond `hi` is never loaded and never multiplied - that is the ~2x.

  2. Diagonal masking (correctness). The block containing the diagonal holds both
     allowed (j <= i) and disallowed (j > i) positions, so it needs an element-wise
     mask:

         keep &= offs_m[:, None] >= offs_n[None, :]

     Masking sets those scores to a large finite negative so exp() underflows to 0.
     Note that this is the only block that needs it; earlier blocks are entirely in
     the past. A production kernel therefore specializes the loop into two stages -
     an unmasked body plus one masked diagonal block - so the mask instructions are
     not executed for every block. See the Triton fused-attention tutorial's `STAGE`
     parameter: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

Why a finite sentinel rather than -inf: a row whose entire seen range is masked leaves
the running max at -inf, and then exp(m_old - m_new) = exp(-inf + inf) = NaN. Plain
causal masking hides this (every real row sees key 0 on the first iteration), which is
why the bug survives review and then surfaces on padding rows, sliding-window masks,
packed variable-length batches, or the first time someone stores the log-sum-exp.

Run: python 01_causal_mask_solution.py
"""

import math

import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64

MASKED_SCORE = -1.0e9


@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    n_ctx, n_heads, softmax_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_b = pid_bh // n_heads
    off_h = pid_bh % n_heads

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # FIX 1 - stop the walk at the last key this query block can reach.
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, n_ctx)
    else:
        hi = n_ctx

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < n_ctx, other=0.0)

        qk = tl.dot(q, k) * softmax_scale

        keep = offs_n[None, :] < n_ctx
        if CAUSAL:
            # FIX 2 - inside the diagonal block, a query may not see future keys.
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(keep, qk, MASKED_SCORE)

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < n_ctx, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < n_ctx)


def attention(q, k, v, causal=False):
    batch, heads, n_ctx, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (triton.cdiv(n_ctx, BLOCK_M), batch * heads)
    attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n_ctx, heads, 1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
    )
    return out


def reference_attention(q, k, v, causal=False):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        n_ctx = q.shape[-2]
        future = torch.triu(
            torch.ones(n_ctx, n_ctx, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def benchmark(fn, iters=50, warmup=10):
    """Milliseconds per call via CUDA/HIP events, after a warmup that pays the JIT."""
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


def visited_block_fraction(n_ctx: int) -> float:
    """Fraction of (query block, key block) pairs a causal kernel actually visits."""
    n_m = triton.cdiv(n_ctx, BLOCK_M)
    n_n = triton.cdiv(n_ctx, BLOCK_N)
    visited = sum(triton.cdiv(min((pid + 1) * BLOCK_M, n_ctx), BLOCK_N)
                  for pid in range(n_m))
    return visited / (n_m * n_n)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../../docs/SETUP.md")

    batch, heads, head_dim = 2, 8, 64
    dtype = torch.float16
    n_ctx = 1000

    torch.manual_seed(0)
    shape = (batch, heads, n_ctx, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    # fp16 rounding of P plus a different summation order costs a few 1e-3.
    atol = 2e-2

    all_passed = True
    for causal in (False, True):
        out = attention(q, k, v, causal=causal)
        ref = reference_attention(q, k, v, causal=causal)
        max_abs = (out.float() - ref).abs().max().item()
        passed = torch.allclose(out.float(), ref, atol=atol, rtol=0.0)
        all_passed = all_passed and passed
        label = "causal" if causal else "full"
        print(f"[{label:6s}] max abs err : {max_abs:.3e}   "
              f"{'PASSED' if passed else 'FAILED'}")

    ms_full = benchmark(lambda: attention(q, k, v, causal=False))
    ms_causal = benchmark(lambda: attention(q, k, v, causal=True))
    frac = visited_block_fraction(n_ctx)

    print()
    print(f"shape             : batch={batch} heads={heads} seq={n_ctx} "
          f"head_dim={head_dim}")
    print(f"full   (ms)       : {ms_full:.3f}")
    print(f"causal (ms)       : {ms_causal:.3f}   ({ms_full / ms_causal:.2f}x vs full)")
    print(f"blocks visited    : {frac * 100:.1f}% of the full grid "
          f"(-> ideal {1 / frac:.2f}x)")
    print()
    print("The measured speedup lands below the ideal for three reasons: the")
    print("diagonal block computes a full tile and discards half of it; programs are")
    print("now imbalanced (pid_m = 0 walks one block, the last walks them all), so")
    print("the launch finishes at the slowest program; and the Q load plus O store")
    print("are fixed costs that do not shrink.")

    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
