"""A09 Lab 2 (Triton) - FlashAttention-style fused attention, forward pass.

The whole point of this kernel is what it does NOT do: it never materializes the
(N x N) score matrix S = Q K^T in HBM. Instead each program owns one BLOCK_M row
block of Q, walks the K/V blocks, and carries three pieces of state in registers:

    m_i   running row max          (BLOCK_M,)
    l_i   running exp-sum          (BLOCK_M,)
    acc   running output numerator (BLOCK_M, HEAD_DIM)

When a new K/V block reveals a larger max, the already-accumulated `l_i` and
`acc` are rescaled by exp(m_old - m_new) so the final result is algebraically
identical to computing softmax over the whole row at once. That rescaling is the
online-softmax trick from A07, generalized from one softmax to the
matmul -> softmax -> matmul chain (Milakov & Gimelshein 2018; Dao et al. 2022).

Reference: the algorithm follows the Triton fused-attention tutorial
(https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html);
this file is deliberately smaller and un-autotuned so the mechanism stays visible.

Run: python flash_attention.py
Requires: torch + triton (see ../../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import math

import torch
import triton
import triton.language as tl

# Tile shape. Fixed on purpose: the autotuned version lives in autotune_demo.py,
# and a fixed shape keeps this file's timings reproducible run to run.
BLOCK_M = 64
BLOCK_N = 64

# Masked scores use a large FINITE negative instead of -inf. A row whose entire seen
# range is masked (the padding rows of the tail block here; real rows under a
# windowed or packed-varlen mask) would leave m_new = -inf, and then
# exp(m_old - m_new) = exp(-inf + inf) = NaN. With -1e9 the running max stays finite
# and exp() underflows cleanly to 0.
MASKED_SCORE = -1.0e9


@triton.jit
def _flash_attn_fwd_kernel(
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

    # Batch and head are folded into one grid axis: attention is embarrassingly
    # parallel across (batch, head, query block), so there is nothing to share.
    off_b = pid_bh // n_heads
    off_h = pid_bh % n_heads

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Q block is loaded once and stays in registers for the whole K/V walk. That
    # reuse is what turns O(N^2) HBM traffic into O(N*d).
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_ctx, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal skipping: keys strictly beyond this query block's last row can never
    # contribute, so those K/V blocks are never loaded at all. For a full sequence
    # this halves both the traffic and the flops - the single biggest win after
    # fusion itself.
    if CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, n_ctx)
    else:
        hi = n_ctx

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # K is loaded already transposed, (HEAD_DIM, BLOCK_N), so tl.dot sees
        # (M x D) @ (D x N) with no explicit transpose op.
        k_ptrs = k_base + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < n_ctx, other=0.0)

        qk = tl.dot(q, k) * softmax_scale

        keep = offs_n[None, :] < n_ctx
        if CAUSAL:
            # Only the diagonal block needs element-wise masking; earlier blocks
            # are entirely in the past and later blocks were skipped above.
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(keep, qk, MASKED_SCORE)

        # ---- the online-softmax update (see README section 3) ----
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)          # correction for everything seen so far
        p = tl.exp(qk - m_new[:, None])      # this block, referenced to the new max

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < n_ctx, other=0.0)

        # P is rounded to the input dtype before the second dot so it can feed the
        # matrix/tensor cores - the same precision compromise production
        # FlashAttention makes. The accumulator stays fp32.
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_new

    # Normalization is deferred to the very end: dividing once here instead of per
    # block is why the running sum `l_i` is carried separately from `acc`.
    acc = acc / l_i[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=offs_m[:, None] < n_ctx)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = False) -> torch.Tensor:
    """Fused scaled-dot-product attention. Inputs are (batch, heads, seq, head_dim)."""
    assert q.is_cuda and k.is_cuda and v.is_cuda, "all inputs must live on the GPU"
    assert q.dim() == 4, "expected (batch, heads, seq, head_dim)"
    assert q.shape == k.shape == v.shape, "Q, K, V must have identical shapes"
    assert q.dtype == k.dtype == v.dtype, "Q, K, V must have identical dtypes"

    batch, heads, n_ctx, head_dim = q.shape
    # tl.arange requires a power-of-two length, and the kernel loads the full head
    # dimension in one tile - so head_dim is constrained, not merely preferred.
    assert head_dim in (16, 32, 64, 128), f"head_dim must be 16/32/64/128, got {head_dim}"

    out = torch.empty_like(q)
    grid = (triton.cdiv(n_ctx, BLOCK_M), batch * heads)

    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        n_ctx, heads, 1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
    )
    return out


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        causal: bool = False) -> torch.Tensor:
    """Unfused fp32 attention. This is the O(N^2)-memory version we are replacing."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        n_ctx = q.shape[-2]
        future = torch.triu(
            torch.ones(n_ctx, n_ctx, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float())


def tolerance(dtype: torch.dtype) -> float:
    """Absolute tolerance for fused-vs-unfused comparison.

    The fused and unfused paths are algebraically equal but arithmetically
    different: they sum in a different order, and the fused kernel rounds P to the
    input dtype before the second matmul. fp16 keeps 10 mantissa bits (~1e-3
    relative), bf16 only 8 (~4e-3). Outputs are convex combinations of V rows, so
    they are O(1) in magnitude - which makes an ABSOLUTE tolerance the meaningful
    one; a relative bound would blow up on the near-zero entries. The values below
    leave roughly an order of magnitude of headroom over the observed error so the
    check flags real bugs, not rounding.
    """
    return {torch.float16: 2e-2, torch.bfloat16: 5e-2, torch.float32: 1e-4}[dtype]


def benchmark(fn, iters: int = 50, warmup: int = 10) -> float:
    """Milliseconds per call, measured with CUDA/HIP events after a warmup.

    The warmup matters twice over here: it pays the Triton JIT compile on the first
    call and lets clocks settle. Timing the first call would time the compiler.
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


def attention_flops(batch: int, heads: int, n_ctx: int, head_dim: int,
                    causal: bool) -> float:
    """Two matmuls of 2*N*N*d flops each; causal masking halves the useful half."""
    total = 4.0 * batch * heads * n_ctx * n_ctx * head_dim
    return total * 0.5 if causal else total


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../../docs/SETUP.md")

    batch, heads, n_ctx, head_dim = 2, 8, 1024, 64
    dtype = torch.float16

    torch.manual_seed(0)
    shape = (batch, heads, n_ctx, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    atol = tolerance(dtype)
    print(f"shape          : batch={batch} heads={heads} seq={n_ctx} head_dim={head_dim} "
          f"dtype={dtype}")
    print(f"tile           : BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N}")
    print(f"tolerance      : atol={atol:g} (see tolerance() for the argument)")
    print()

    all_passed = True
    for causal in (False, True):
        out = flash_attention(q, k, v, causal=causal)
        ref = reference_attention(q, k, v, causal=causal)

        max_abs = (out.float() - ref).abs().max().item()
        passed = torch.allclose(out.float(), ref, atol=atol, rtol=0.0)
        all_passed = all_passed and passed

        ms = benchmark(lambda: flash_attention(q, k, v, causal=causal))
        tflops = attention_flops(batch, heads, n_ctx, head_dim, causal) / (ms / 1e3) / 1e12

        label = "causal" if causal else "full"
        print(f"[{label:6s}] fused kernel : {ms:.3f} ms   {tflops:.1f} TFLOP/s")
        print(f"[{label:6s}] max abs err  : {max_abs:.3e}")
        print(f"[{label:6s}] result       : {'PASSED' if passed else 'FAILED'}")
        print()

    print("The causal run visits roughly half the K/V blocks of the full run, so it")
    print("should be markedly faster - that saving comes from skipping whole blocks,")
    print("not from masking their scores. See README section 3.")

    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
