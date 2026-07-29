"""A09 Exercise 1 - Make a tiled attention kernel causal.

The kernel below is a complete, correct, NON-causal fused attention forward pass.
Your job is to make it causal, and to make it FASTER than the non-causal version
while you are at it. Two TODOs, in this order:

  TODO 1 (performance): bound the K/V loop so blocks lying strictly in the future
          of this query block are never loaded at all. A query at row i may only
          attend to keys j <= i, so for the query block owned by `pid_m` every key
          index beyond (pid_m + 1) * BLOCK_M is dead weight. Skipping those blocks
          is where the ~2x causal saving comes from - masking alone saves nothing,
          because you have already paid for the loads and the dot.

  TODO 2 (correctness): mask inside the diagonal block. The block that straddles
          the diagonal contains both allowed (j <= i) and disallowed (j > i)
          positions, so it needs element-wise masking, not block-level skipping.

Until both are done the script prints FAILED: the kernel is still computing full
bidirectional attention while the reference is causal.

Run: python 01_causal_mask.py
Requires: torch + triton (see ../../../../docs/SETUP.md).
"""

import math

import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 64

# A finite sentinel, not -inf: a row whose entire seen range is masked would leave the
# running max at -inf and produce exp(-inf + inf) = NaN in the rescale step. Keep using
# this constant.
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

    # ------------------------------------------------------------------
    # TODO 1: when CAUSAL, stop the walk at the last key this query block
    #         can possibly attend to instead of walking the whole sequence.
    #         Hint: tl.minimum(...) of two expressions, and remember the
    #         sequence may not be a whole number of blocks long.
    # ------------------------------------------------------------------
    hi = n_ctx

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < n_ctx, other=0.0)

        qk = tl.dot(q, k) * softmax_scale

        # Bounds mask: keys past the end of the sequence must not contribute.
        keep = offs_n[None, :] < n_ctx

        # ------------------------------------------------------------------
        # TODO 2: when CAUSAL, also drop positions where the key index is in
        #         the future of the query index. `offs_m` holds query indices
        #         (a column vector after [:, None]); `offs_n` holds key indices
        #         (a row vector after [None, :]).
        # ------------------------------------------------------------------

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
    """fp32, unfused, obviously-correct attention."""
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


def main():
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../../docs/SETUP.md")

    batch, heads, head_dim = 2, 8, 64
    dtype = torch.float16
    # A ragged length on purpose: n_ctx is not a multiple of BLOCK_M, so the bounds
    # masks and your loop bound must both handle the partial last block.
    n_ctx = 1000

    torch.manual_seed(0)
    shape = (batch, heads, n_ctx, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    # fp16 keeps ~10 mantissa bits and the kernel rounds P before the second dot,
    # so an absolute tolerance around 2e-2 separates rounding from real bugs.
    atol = 2e-2

    out = attention(q, k, v, causal=True)
    ref = reference_attention(q, k, v, causal=True)
    max_abs = (out.float() - ref).abs().max().item()
    passed = torch.allclose(out.float(), ref, atol=atol, rtol=0.0)

    ms_full = benchmark(lambda: attention(q, k, v, causal=False))
    ms_causal = benchmark(lambda: attention(q, k, v, causal=True))

    print(f"shape         : batch={batch} heads={heads} seq={n_ctx} head_dim={head_dim}")
    print(f"full   (ms)   : {ms_full:.3f}")
    print(f"causal (ms)   : {ms_causal:.3f}   ({ms_full / ms_causal:.2f}x vs full)")
    print(f"max abs err   : {max_abs:.3e}   (tolerance {atol:g})")
    print(f"result        : {'PASSED' if passed else 'FAILED'}")
    print()
    if not passed:
        print("Expected until TODO 2 is done: the kernel still lets every query see")
        print("every key, so it disagrees with the causal reference.")
    if ms_full / ms_causal < 1.3:
        print("Causal is not meaningfully faster than full - TODO 1 is still open.")
        print("Masking a score you already computed saves no time; skipping the")
        print("block saves the load AND the dot.")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
