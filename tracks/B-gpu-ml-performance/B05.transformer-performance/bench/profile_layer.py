#!/usr/bin/env python3
"""Time one decoder layer's sub-operations in both regimes, on the same weights.

The point of this benchmark is a single comparison. Run the identical layer twice:

  prefill shape  (batch, seq=2048, d)  -> big GEMMs, matrix cores busy
  decode  shape  (batch, seq=1,    d)  -> matrix-VECTOR products, HBM-starved

Then look at where the time went. In prefill, time tracks FLOPs and the achieved
TFLOP/s column is high. In decode, time tracks BYTES READ: the GEMM columns collapse
to ~1 FLOP/byte and the GB/s column is what approaches the machine ceiling instead.

The layer is a Llama-style block (RMSNorm, GQA + RoPE, SwiGLU FFN) built from plain
torch ops so every sub-op can be timed in isolation. It is deliberately NOT fused and
NOT using a flash-attention kernel: this measures the naive baseline that A07 fuses
and B06 optimizes. Two consequences worth seeing in the output:

  - the elementwise ops (norms, RoPE, SwiGLU, residuals) each pay a full HBM
    round trip, which is exactly what fusion removes;
  - attention materializes the (seq x ctx) score matrix, which is the O(N^2)
    memory that FlashAttention exists to avoid.

Timings are per sub-op. Their sum approximates an unfused layer's cost but is not
identical to running the layer end to end, since a real forward pass gets some
overlap between adjacent kernels.

Usage:
    python bench/profile_layer.py
    python bench/profile_layer.py --hidden 8192 --ffn 28672 --heads 64 --kv-heads 8
    python bench/profile_layer.py --prefill-seq 4096 --batch 4 --dtype bfloat16
"""

import argparse
import sys

try:
    import torch
except ImportError:
    sys.exit("torch is required: see ../../../docs/SETUP.md, "
             "or pip install -r requirements.txt")

GEMM, ATTN, ELEM = "gemm", "attn", "elem"


def require_gpu():
    """Refuse to run on CPU: a CPU number would teach the wrong lesson."""
    if not torch.cuda.is_available():
        sys.exit(
            "No GPU visible to torch (torch.cuda.is_available() is False).\n"
            "\n"
            "This benchmark measures HBM bandwidth and matrix-core throughput, so a\n"
            "CPU run would be misleading rather than merely slow. Refusing.\n"
            "\n"
            "  AMD    : install a ROCm build of torch, check `rocm-smi` / `amd-smi`\n"
            "  NVIDIA : install a CUDA build of torch, check `nvidia-smi`\n"
            "  Setup  : ../../../docs/SETUP.md\n"
            "\n"
            "For a no-GPU analysis of the same two regimes, run:\n"
            "  python tools/model_calculator.py --preset llama3-8b"
        )
    props = torch.cuda.get_device_properties(0)
    print(f"device : {torch.cuda.get_device_name(0)}  "
          f"({props.total_memory / 1e9:.1f} GB, torch {torch.__version__})")


class Layer:
    """A Llama-style decoder layer held as loose tensors, so each op is timable."""

    def __init__(self, d, d_ffn, heads, kv_heads, dtype, device):
        if d % heads:
            raise ValueError(f"hidden {d} not divisible by heads {heads}")
        if heads % kv_heads:
            raise ValueError(f"heads {heads} not divisible by kv-heads {kv_heads}")
        self.d, self.d_ffn = d, d_ffn
        self.heads, self.kv_heads = heads, kv_heads
        self.d_head = d // heads
        self.d_kv = kv_heads * self.d_head
        self.group = heads // kv_heads
        self.dtype = dtype
        self.elem_bytes = torch.finfo(dtype).bits // 8

        kw = dict(dtype=dtype, device=device)
        # torch.nn.functional.linear wants (out_features, in_features).
        self.wq = torch.randn(d, d, **kw) * 0.02
        self.wk = torch.randn(self.d_kv, d, **kw) * 0.02
        self.wv = torch.randn(self.d_kv, d, **kw) * 0.02
        self.wo = torch.randn(d, d, **kw) * 0.02
        self.w_gate = torch.randn(d_ffn, d, **kw) * 0.02
        self.w_up = torch.randn(d_ffn, d, **kw) * 0.02
        self.w_down = torch.randn(d, d_ffn, **kw) * 0.02
        self.g1 = torch.ones(d, **kw)
        self.g2 = torch.ones(d, **kw)

        # RoPE frequency table, precomputed as a real implementation would cache it.
        self.inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, self.d_head, 2, device=device).float() / self.d_head))

    @property
    def params(self):
        return sum(w.numel() for w in (self.wq, self.wk, self.wv, self.wo,
                                       self.w_gate, self.w_up, self.w_down))

    def rms_norm(self, x, gain):
        # The float() upcast mirrors what production implementations do for
        # numerical stability, and costs an extra fp32 temporary.
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + 1e-5)).to(x.dtype) * gain

    def rope(self, x, positions):
        # x: (B, heads, T, d_head), rotate-half formulation.
        freqs = torch.outer(positions.float(), self.inv_freq)
        cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).to(x.dtype)
        sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).to(x.dtype)
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin


def timed_ms(fn, iters, warmup):
    """Mean milliseconds per call, measured with CUDA events after warmup.

    Warmup matters more than usual here. The first call pays lazy library init,
    GEMM heuristic selection, and allocator growth - constants large enough to
    swamp the decode measurements, which are only tens of microseconds.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def build_ops(layer, batch, seq, ctx):
    """[(name, kind, fn, flops, bytes)] for one layer at this shape.

    `seq` is the number of NEW tokens in this step (2048 for prefill, 1 for decode);
    `ctx` is the total attended length, so a decode step attends over a full cache.
    FLOP counts use 2 per multiply-accumulate. Byte counts are the ideal HBM traffic
    of an unfused implementation: inputs + weights read once, output written once.
    """
    d, d_ffn = layer.d, layer.d_ffn
    h, hkv, dh, d_kv = layer.heads, layer.kv_heads, layer.d_head, layer.d_kv
    el = layer.elem_bytes
    dev, dt = layer.wq.device, layer.dtype
    T = batch * seq

    x = torch.randn(batch, seq, d, dtype=dt, device=dev)
    positions = torch.arange(ctx - seq, ctx, device=dev)

    normed = layer.rms_norm(x, layer.g1)
    q = torch.nn.functional.linear(normed, layer.wq).view(batch, seq, h, dh).transpose(1, 2)

    # The KV cache this step attends over. In prefill it is the step's own tokens.
    k_cache = torch.randn(batch, hkv, ctx, dh, dtype=dt, device=dev)
    v_cache = torch.randn(batch, hkv, ctx, dh, dtype=dt, device=dev)

    # GQA needs KV broadcast back to the query head count. Unfused, that is a real
    # materialized copy; a fused kernel does it in registers for free.
    k_rep = k_cache.repeat_interleave(layer.group, dim=1)
    v_rep = v_cache.repeat_interleave(layer.group, dim=1)
    scale = dh ** -0.5

    attn_out = torch.randn(batch, seq, d, dtype=dt, device=dev)
    scores = torch.randn(batch, h, seq, ctx, dtype=dt, device=dev)
    gate = torch.randn(batch, seq, d_ffn, dtype=dt, device=dev)
    up = torch.randn(batch, seq, d_ffn, dtype=dt, device=dev)

    lin = torch.nn.functional.linear
    return [
        ("RMSNorm (pre-attn)", ELEM,
         lambda: layer.rms_norm(x, layer.g1),
         0, 2 * T * d * el),

        ("Q projection  d x d", GEMM,
         lambda: lin(normed, layer.wq),
         2 * T * d * d, (d * d + 2 * T * d) * el),

        ("K+V projection  d x d_kv", GEMM,
         lambda: (lin(normed, layer.wk), lin(normed, layer.wv)),
         2 * (2 * T * d * d_kv), (2 * d * d_kv + 2 * T * d + 2 * T * d_kv) * el),

        ("RoPE on Q", ELEM,
         lambda: layer.rope(q, positions),
         0, 2 * T * d * el),

        ("GQA repeat K,V", ELEM,
         lambda: (k_cache.repeat_interleave(layer.group, dim=1),
                  v_cache.repeat_interleave(layer.group, dim=1)),
         0, 2 * batch * ctx * dh * (hkv + h) * el),

        ("attn: Q @ K^T", ATTN,
         lambda: torch.matmul(q, k_rep.transpose(-2, -1)),
         2 * batch * h * seq * ctx * dh,
         (batch * h * dh * (seq + ctx) + batch * h * seq * ctx) * el),

        ("attn: softmax", ATTN,
         lambda: torch.softmax(scores * scale, dim=-1),
         0, 2 * batch * h * seq * ctx * el),

        ("attn: probs @ V", ATTN,
         lambda: torch.matmul(scores, v_rep),
         2 * batch * h * seq * ctx * dh,
         (batch * h * seq * ctx + batch * h * dh * (ctx + seq)) * el),

        ("O projection  d x d", GEMM,
         lambda: lin(attn_out, layer.wo),
         2 * T * d * d, (d * d + 2 * T * d) * el),

        ("residual add", ELEM,
         lambda: x + attn_out,
         T * d, 3 * T * d * el),

        ("RMSNorm (pre-FFN)", ELEM,
         lambda: layer.rms_norm(x, layer.g2),
         0, 2 * T * d * el),

        ("FFN gate+up  d x d_ffn", GEMM,
         lambda: (lin(normed, layer.w_gate), lin(normed, layer.w_up)),
         2 * (2 * T * d * d_ffn), (2 * d * d_ffn + T * d + 2 * T * d_ffn) * el),

        ("SwiGLU activation", ELEM,
         lambda: torch.nn.functional.silu(gate) * up,
         2 * T * d_ffn, 3 * T * d_ffn * el),

        ("FFN down  d_ffn x d", GEMM,
         lambda: lin(gate, layer.w_down),
         2 * T * d_ffn * d, (d * d_ffn + T * d_ffn + T * d) * el),
    ], scores.numel() * el


def run_regime(title, layer, batch, seq, ctx, iters, warmup):
    print()
    print("=" * 78)
    print(f"  {title}")
    print(f"  batch={batch}  new tokens={seq}  attended ctx={ctx}"
          f"  ->  {batch * seq} token(s) per step")
    print("=" * 78)

    ops, score_bytes = build_ops(layer, batch, seq, ctx)
    print(f"  materialized score matrix: {score_bytes / 1e6:.1f} MB"
          f"  (batch x heads x {seq} x {ctx}, the O(N^2) term)")
    print()

    rows = []
    for name, kind, fn, flops, byts in ops:
        try:
            ms = timed_ms(fn, iters, warmup)
        except RuntimeError as exc:
            print(f"  {name:<26} SKIPPED: {str(exc).splitlines()[0][:44]}")
            continue
        rows.append((name, kind, ms, flops, byts))

    total = sum(r[2] for r in rows)
    print(f"  {'sub-operation':<26} {'kind':>5} {'ms':>9} {'share':>7}"
          f" {'TFLOP/s':>9} {'GB/s':>8} {'FLOP/B':>8}")
    print("  " + "-" * 76)
    for name, kind, ms, flops, byts in rows:
        secs = ms / 1e3
        tf = f"{flops / secs / 1e12:9.1f}" if flops else f"{'-':>9}"
        ai = f"{flops / byts:8.2f}" if flops and byts else f"{'-':>8}"
        print(f"  {name:<26} {kind:>5} {ms:9.4f} {100 * ms / total:6.1f}%"
              f" {tf} {byts / secs / 1e9:8.1f} {ai}")
    print("  " + "-" * 76)
    print(f"  {'TOTAL':<26} {'':>5} {total:9.4f} {100.0:6.1f}%")

    print()
    for kind, label in ((GEMM, "weight GEMMs"), (ATTN, "attention"),
                        (ELEM, "elementwise / memory-bound")):
        part = sum(r[2] for r in rows if r[1] == kind)
        print(f"  {label:<28} {part:8.4f} ms  ({100 * part / total:5.1f}% of the layer)")
    print(f"  {'per-token cost':<28} {1e3 * total / (batch * seq):8.2f} us/token")
    return total, batch * seq


def main():
    p = argparse.ArgumentParser(
        description="Time a decoder layer's sub-ops in the prefill and decode regimes.")
    p.add_argument("--hidden", type=int, default=4096, help="d (default: Llama-3-8B)")
    p.add_argument("--ffn", type=int, default=14336, help="FFN inner width")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8, help="< heads for GQA, 1 for MQA")
    p.add_argument("--prefill-seq", type=int, default=2048, help="prompt length")
    p.add_argument("--decode-ctx", type=int, default=2048,
                   help="cache length the one decode token attends over")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    require_gpu()
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)

    try:
        layer = Layer(args.hidden, args.ffn, args.heads, args.kv_heads,
                      dtype, torch.device("cuda"))
    except ValueError as exc:
        sys.exit(f"bad configuration: {exc}")

    print(f"layer  : d={args.hidden} d_ffn={args.ffn} heads={args.heads} "
          f"kv_heads={args.kv_heads} dtype={args.dtype}")
    print(f"params : {layer.params / 1e6:.1f} M per layer = "
          f"{layer.params * layer.elem_bytes / 1e6:.1f} MB of weights read per pass")
    print(f"timing : {args.iters} iters after {args.warmup} warmup, torch.cuda.Event")

    pre_ms, pre_tok = run_regime("PREFILL REGIME  (expect compute-bound)", layer,
                                 args.batch, args.prefill_seq, args.prefill_seq,
                                 args.iters, args.warmup)
    dec_ms, dec_tok = run_regime("DECODE REGIME  (expect bandwidth-bound)", layer,
                                 args.batch, 1, args.decode_ctx,
                                 args.iters, args.warmup)

    print()
    print("=" * 78)
    print("  THE COMPARISON")
    print("=" * 78)
    print(f"  prefill : {pre_tok:>6} tokens in {pre_ms:8.4f} ms"
          f"  ->  {1e3 * pre_ms / pre_tok:8.2f} us/token")
    print(f"  decode  : {dec_tok:>6} tokens in {dec_ms:8.4f} ms"
          f"  ->  {1e3 * dec_ms / dec_tok:8.2f} us/token")
    print()
    print(f"  Decode does {pre_tok // max(dec_tok, 1)}x less work per step, yet the step is"
          f" only {pre_ms / dec_ms:.1f}x faster.")
    print(f"  Cost per token is {(dec_ms / dec_tok) / (pre_ms / pre_tok):.0f}x higher in decode.")
    print("  That gap is the weight read: decode pays for every byte and uses almost")
    print("  none of the FLOPs. Batching amortizes it - see bench/batch_scaling.py.")


if __name__ == "__main__":
    main()
