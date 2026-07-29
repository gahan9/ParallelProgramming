"""A09 Exercise 2 - Implement the online rescaling update. This is the heart of
FlashAttention; everything else in the kernel is bookkeeping.

Each program owns one BLOCK_M block of queries and walks the K/V blocks. Between
iterations it carries three pieces of state:

    m_i   (BLOCK_M,)            running row max of the scores seen so far
    l_i   (BLOCK_M,)            running sum of exp(score - m_i)
    acc   (BLOCK_M, HEAD_DIM)   running sum of exp(score - m_i) * V

The invariant to preserve: `l_i` and `acc` are always expressed relative to the
CURRENT `m_i`. When a new block reveals a bigger max, both must be rescaled before
the new block's contribution is added. The update, in order:

    m = max(m_old, m_block)
    alpha = exp(m_old - m)                       correction factor, in (0, 1]
    P     = exp(qk - m)                          this block, relative to the new max
    l     = l_old * alpha + rowsum(P)
    acc   = acc_old * alpha + P @ V

Your TODO is the block marked below. The skeleton currently normalizes each block
to its OWN max and never rescales, which is a plausible-looking bug that produces
finite, wrong numbers - the failure mode you must be able to recognize in review.

Run: python 02_online_rescale.py
Requires: torch + triton (see ../../../../docs/SETUP.md).
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

        # ==================================================================
        # TODO: replace the four lines below with the correct online update.
        #   1. fold this block's row max into the running max
        #   2. compute the correction factor alpha from the old and new max
        #   3. rescale l_i AND acc by alpha before adding this block
        #   4. cast P to v.dtype before tl.dot - the matrix cores need it, and
        #      production FlashAttention makes the same compromise
        # Keep the fp32 accumulator: it is what makes the result exact enough
        # to match the unfused reference.
        # ==================================================================
        m_new = tl.max(qk, axis=1)                    # WRONG: forgets m_i
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i + tl.sum(p, axis=1)                 # WRONG: no rescale of l_i
        acc = tl.dot(p.to(v.dtype), v, acc)           # WRONG: no rescale of acc
        m_i = m_new

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
    """fp32, unfused, obviously-correct attention."""
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


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../../docs/SETUP.md")

    batch, heads, n_ctx, head_dim = 2, 8, 1024, 64
    dtype = torch.float16

    torch.manual_seed(0)
    shape = (batch, heads, n_ctx, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    # fp16 rounding of P plus a different summation order costs a few 1e-3; 2e-2
    # leaves headroom for that without hiding a genuine algebra error, which shows
    # up orders of magnitude larger.
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
    print()
    if not passed:
        print("Expected until the TODO is done. Note the failure signature: the")
        print("numbers are finite and in the right ballpark, because each block's")
        print("softmax is internally consistent - only the stitching is wrong. A")
        print("missing rescale does not crash, it quietly biases the output toward")
        print("whichever block had the largest scores.")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
