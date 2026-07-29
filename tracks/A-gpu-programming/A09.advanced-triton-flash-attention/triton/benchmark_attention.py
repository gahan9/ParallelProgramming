"""A09 Lab 3 (Triton) - naive PyTorch attention vs the fused kernel, swept over
sequence length.

The point of the sweep is the SHAPE of the curve, not any single number: the fused
kernel's advantage grows with sequence length, because the naive path's HBM traffic
grows as O(N^2) while the fused path's grows as O(N*d). Where the two cross depends
on your GPU, dtype, and library versions - measure, do not quote.

The byte columns are analytic estimates (see naive_hbm_bytes / fused_hbm_bytes),
not counter readings. To measure real traffic use rocprofv3 or Nsight Compute.

Run: python benchmark_attention.py [seq_len ...]
Requires: torch + triton (see ../../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import math
import sys

import torch

# Same directory as this script, so a plain import works when run as
# `python triton/benchmark_attention.py` from the module root.
from flash_attention import (
    attention_flops,
    benchmark,
    flash_attention,
    reference_attention,
    tolerance,
)

BATCH = 1
HEADS = 8
HEAD_DIM = 64
DTYPE = torch.float16
DEFAULT_SEQ_LENS = (512, 1024, 2048, 4096)


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    causal: bool = False) -> torch.Tensor:
    """Textbook attention: materialize the (N x N) score matrix, then soften it.

    This is what every from-scratch transformer implementation does, and what
    FlashAttention exists to avoid. Note the three full-size intermediates that hit
    HBM: `scores`, the mask fill, and the softmax output.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        n_ctx = q.shape[-2]
        future = torch.triu(
            torch.ones(n_ctx, n_ctx, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


def naive_hbm_bytes(batch: int, heads: int, n_ctx: int, head_dim: int,
                    elem_size: int) -> int:
    """Lower bound on HBM traffic for the unfused path.

    Q, K, V read once and O written once: 4 * B*H*N*d elements. Then the score
    matrix crosses the bus four times - written by the QK^T matmul, read by the
    softmax, written by the softmax, read by the PV matmul: 4 * B*H*N^2 elements.
    Real traffic is higher (the mask fill adds a pass, caches only partly help).
    """
    endpoints = 4 * batch * heads * n_ctx * head_dim
    score_matrix = 4 * batch * heads * n_ctx * n_ctx
    return (endpoints + score_matrix) * elem_size


def fused_hbm_bytes(batch: int, heads: int, n_ctx: int, head_dim: int,
                    elem_size: int) -> int:
    """Ideal HBM traffic for the fused path: Q, K, V in and O out. Nothing else.

    Ideal because each K/V block is re-read once per Q block; the estimate assumes
    those re-reads are served by L2 / Infinity Cache. On a cache miss the true
    figure rises toward B*H*N^2*d/BLOCK_M - still far below the naive path, but do
    not treat this column as a guarantee.
    """
    return 4 * batch * heads * n_ctx * head_dim * elem_size


def gib(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def try_bench(fn) -> float:
    """Benchmark `fn`, returning NaN when the GPU cannot hold its intermediates.

    An OOM here is not a failure of the harness - it is the headline result. The
    naive path allocates an (N x N) matrix per head, so it falls over at exactly
    the sequence lengths production models care about.
    """
    try:
        return benchmark(fn)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan")
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return float("nan")


def fmt_ms(ms: float) -> str:
    return "  OOM  " if math.isnan(ms) else f"{ms:7.3f}"


def verify(causal: bool) -> bool:
    """Correctness gate before any timing: fused must match the fp32 reference."""
    n_ctx = 512
    torch.manual_seed(0)
    shape = (BATCH, HEADS, n_ctx, HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=DTYPE)
    k = torch.randn(shape, device="cuda", dtype=DTYPE)
    v = torch.randn(shape, device="cuda", dtype=DTYPE)

    fused = flash_attention(q, k, v, causal=causal).float()
    ref = reference_attention(q, k, v, causal=causal)
    naive = naive_attention(q, k, v, causal=causal).float()

    atol = tolerance(DTYPE)
    fused_ok = torch.allclose(fused, ref, atol=atol, rtol=0.0)
    naive_ok = torch.allclose(naive, ref, atol=atol, rtol=0.0)

    label = "causal" if causal else "full"
    print(f"[{label:6s}] fused vs fp32 reference : "
          f"max abs err {(fused - ref).abs().max().item():.3e}  "
          f"{'PASSED' if fused_ok else 'FAILED'}")
    print(f"[{label:6s}] naive vs fp32 reference : "
          f"max abs err {(naive - ref).abs().max().item():.3e}  "
          f"{'PASSED' if naive_ok else 'FAILED'}")
    return fused_ok and naive_ok


def sweep(seq_lens, causal: bool) -> None:
    elem_size = torch.tensor([], dtype=DTYPE).element_size()
    label = "causal" if causal else "full"

    print()
    print(f"=== {label} attention: batch={BATCH} heads={HEADS} head_dim={HEAD_DIM} "
          f"dtype={DTYPE} ===")
    print(f"{'seq':>6} {'naive ms':>9} {'fused ms':>9} {'speedup':>8} "
          f"{'naive GiB':>10} {'fused GiB':>10} {'byte ratio':>11} {'fused TFLOP/s':>14}")

    for n_ctx in seq_lens:
        torch.manual_seed(0)
        shape = (BATCH, HEADS, n_ctx, HEAD_DIM)
        try:
            q = torch.randn(shape, device="cuda", dtype=DTYPE)
            k = torch.randn(shape, device="cuda", dtype=DTYPE)
            v = torch.randn(shape, device="cuda", dtype=DTYPE)
        except torch.cuda.OutOfMemoryError:
            print(f"{n_ctx:>6}   cannot even allocate Q/K/V - skipping")
            torch.cuda.empty_cache()
            continue

        naive_ms = try_bench(lambda: naive_attention(q, k, v, causal=causal))
        fused_ms = try_bench(lambda: flash_attention(q, k, v, causal=causal))

        naive_bytes = naive_hbm_bytes(BATCH, HEADS, n_ctx, HEAD_DIM, elem_size)
        fused_bytes = fused_hbm_bytes(BATCH, HEADS, n_ctx, HEAD_DIM, elem_size)

        speedup = ("  n/a  " if math.isnan(naive_ms) or math.isnan(fused_ms)
                   else f"{naive_ms / fused_ms:6.2f}x")
        tflops = ("   n/a" if math.isnan(fused_ms) else
                  f"{attention_flops(BATCH, HEADS, n_ctx, HEAD_DIM, causal) / (fused_ms / 1e3) / 1e12:6.1f}")

        print(f"{n_ctx:>6} {fmt_ms(naive_ms):>9} {fmt_ms(fused_ms):>9} {speedup:>8} "
              f"{gib(naive_bytes):>10.3f} {gib(fused_bytes):>10.3f} "
              f"{naive_bytes / fused_bytes:>10.1f}x {tflops:>14}")

        del q, k, v
        torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../../docs/SETUP.md")

    seq_lens = [int(a) for a in sys.argv[1:]] or list(DEFAULT_SEQ_LENS)

    print(f"device         : {torch.cuda.get_device_name(0)}")
    print(f"sequence lens  : {seq_lens}")
    print()
    # Both checks run unconditionally: a short-circuit would hide the second report.
    full_ok = verify(causal=False)
    causal_ok = verify(causal=True)
    passed = full_ok and causal_ok

    for causal in (False, True):
        sweep(seq_lens, causal)

    print()
    print("Reading this table:")
    print("  * `byte ratio` is exactly 1 + N/d for these estimates - the analytic")
    print("    reason the fused kernel pulls away as the sequence grows.")
    print("  * measured `speedup` grows with N but lags the byte ratio: the fused")
    print("    kernel becomes compute-bound, so it stops converting saved bytes")
    print("    into saved time once it is limited by matrix-core throughput.")
    print("  * an OOM row is a result, not an error: the naive path's (N x N)")
    print("    intermediates are what make long context impossible without fusion.")
    print("  * this teaching kernel will NOT match flash-attn / aotriton / cuDNN.")
    print("    See README section 6 for exactly what they do that this does not.")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
