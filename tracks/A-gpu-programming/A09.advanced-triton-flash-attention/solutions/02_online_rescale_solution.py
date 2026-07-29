"""A09 Exercise 2 - SOLUTION. The online rescaling update, with the algebra.

The correct update, and why each line is there:

    m_new = max(m_i, rowmax(qk))      # the new reference point for exp()
    alpha = exp(m_i - m_new)          # <= 1, and exactly 1 when the max did not move
    p     = exp(qk - m_new)           # this block, referenced to m_new
    l_i   = l_i * alpha + rowsum(p)   # rebase the old sum, then add the new one
    acc   = acc * alpha + p @ V       # rebase the old numerator, then add the new one
    m_i   = m_new

Why it is exact. After processing blocks 1..t, the invariant is

    l_i = sum_{j in 1..t} exp(s_j - m_i)
    acc = sum_{j in 1..t} exp(s_j - m_i) * V_j

Both are the true sums scaled by the single common factor exp(-m_i). Changing the
reference point from m_old to m_new multiplies every term by
exp(-m_new)/exp(-m_old) = exp(m_old - m_new) = alpha - one scalar per row, applied
uniformly, so no information is lost. At the end, acc / l_i cancels the shared
factor entirely: the result is identical to a softmax computed over the whole row at
once, up to floating-point rounding. This is Milakov & Gimelshein's online softmax
(https://arxiv.org/abs/1805.02867) with an output accumulator carried alongside.

Why it is numerically safe. Every exponent is <= 0, because both `qk - m_new` and
`m_i - m_new` are non-positive by construction. exp() therefore never overflows; the
worst case is underflow to 0, which is the mathematically correct answer for a score
far below the row max. The unfused version needs exactly the same trick - it just
gets to take the max over the whole row before it starts.

Why the placeholder in the exercise is wrong but not obviously wrong. Normalizing
each block to its own max makes every block internally consistent, so the numbers
stay finite and plausible. What breaks is the stitching: terms from different blocks
end up scaled by different factors, and the final division cannot undo a
per-block-varying scale. The output is silently biased toward whichever block had
the largest scores.

Run: python 02_online_rescale_solution.py
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

    for start_n in range(0, n_ctx, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < n_ctx, other=0.0)

        qk = tl.dot(q, k) * softmax_scale
        qk = tl.where(offs_n[None, :] < n_ctx, qk, MASKED_SCORE)

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < n_ctx, other=0.0)

        # ---- the online update ----
        # On the first iteration m_i is -inf, so alpha = exp(-inf) = 0 and the
        # zero-initialized l_i / acc are wiped rather than poisoned. That is why the
        # loop needs no special-casing for its first pass.
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_new

    # One division at the very end, not once per block: the running sum is carried
    # precisely so this can be deferred.
    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < n_ctx)


def attention(q, k, v):
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
        HEAD_DIM=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def reference_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
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


def two_block_walkthrough() -> None:
    """The same algebra on one row in plain PyTorch, so you can read the numbers.

    Deliberately adversarial: the second half of the row holds much larger scores,
    which is exactly the case a missing rescale gets wrong.
    """
    torch.manual_seed(0)
    scores = torch.cat([torch.randn(8), torch.randn(8) + 12.0])

    one_shot = torch.softmax(scores, dim=0)

    m_run = torch.tensor(float("-inf"))
    l_run = torch.tensor(0.0)
    for block in scores.split(8):
        m_new = torch.maximum(m_run, block.max())
        alpha = torch.exp(m_run - m_new)
        l_run = l_run * alpha + torch.exp(block - m_new).sum()
        m_run = m_new

    # softmax denominator recovered from the streamed statistics
    streamed_denom = l_run * torch.exp(m_run)
    one_shot_denom = torch.exp(scores).sum()

    print("two-block walkthrough (one row, second block 12 larger):")
    print(f"  running (m, l)      : ({m_run.item():.4f}, {l_run.item():.6f})")
    print(f"  denominator streamed: {streamed_denom.item():.6e}")
    print(f"  denominator one-shot: {one_shot_denom.item():.6e}")
    print(f"  max prob diff       : "
          f"{(torch.exp(scores - m_run) / l_run - one_shot).abs().max().item():.3e}")
    print("  Without the alpha rescale the first block's 8 terms would stay")
    print("  referenced to their own max and be over-weighted by exp(12) ~ 1.6e5.")
    print()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../../docs/SETUP.md")

    two_block_walkthrough()

    batch, heads, n_ctx, head_dim = 2, 8, 1024, 64
    dtype = torch.float16

    torch.manual_seed(0)
    shape = (batch, heads, n_ctx, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    atol = 2e-2

    out = attention(q, k, v)
    ref = reference_attention(q, k, v)
    max_abs = (out.float() - ref).abs().max().item()
    passed = torch.allclose(out.float(), ref, atol=atol, rtol=0.0)
    ms = benchmark(lambda: attention(q, k, v))

    print(f"shape         : batch={batch} heads={heads} seq={n_ctx} head_dim={head_dim}")
    print(f"fused (ms)    : {ms:.3f}")
    print(f"max abs err   : {max_abs:.3e}   (tolerance {atol:g})")
    print(f"result        : {'PASSED' if passed else 'FAILED'}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
