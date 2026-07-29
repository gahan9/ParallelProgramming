#!/usr/bin/env python3
"""Measure decode throughput vs batch size, to prove decode is bandwidth-bound.

The claim under test: a decode step's cost is dominated by *reading the weights*,
not by arithmetic. If that is true, then processing 32 sequences in one step should
cost barely more than processing 1 - because the same weights are read once either
way and each sequence just adds one more column to the matrix-vector product.

What to look for in the output:

  step latency (ms)     nearly FLAT as batch grows, then bending upward
  aggregate tokens/s    nearly LINEAR in batch over that flat region
  achieved GB/s         high and roughly CONSTANT -> the HBM is the limiter
  achieved TFLOP/s      low, but rising linearly -> the math is not the limiter

The knee, where latency starts rising proportionally to batch, is where arithmetic
intensity finally reaches the machine's ridge point and the workload crosses into
compute-bound. tools/model_calculator.py predicts roughly where that should be.

Attention over the KV cache is deliberately EXCLUDED here. Its byte count scales
*with* batch (each sequence has its own cache), so including it would mask the very
effect being demonstrated. That interaction - and why it caps real serving batch
sizes - is B06's subject.

Usage:
    python bench/batch_scaling.py
    python bench/batch_scaling.py --layers 16 --batches 1,2,4,8,16,32,64,128,256
    python bench/batch_scaling.py --hidden 8192 --ffn 28672 --kv-heads 8 --bandwidth 5300
"""

import argparse
import sys

try:
    import torch
except ImportError:
    sys.exit("torch is required: see ../../../docs/SETUP.md, "
             "or pip install -r requirements.txt")


def require_gpu():
    if not torch.cuda.is_available():
        sys.exit(
            "No GPU visible to torch (torch.cuda.is_available() is False).\n"
            "\n"
            "This benchmark exists to measure HBM bandwidth, so a CPU run would be\n"
            "misleading rather than merely slow. Refusing.\n"
            "\n"
            "  AMD    : install a ROCm build of torch, check `rocm-smi` / `amd-smi`\n"
            "  NVIDIA : install a CUDA build of torch, check `nvidia-smi`\n"
            "  Setup  : ../../../docs/SETUP.md\n"
            "\n"
            "The analytical version of this experiment needs no GPU:\n"
            "  python tools/model_calculator.py --preset llama3-8b --batch 32"
        )
    props = torch.cuda.get_device_properties(0)
    print(f"device : {torch.cuda.get_device_name(0)}  "
          f"({props.total_memory / 1e9:.1f} GB, torch {torch.__version__})")
    return props.total_memory


def build_weights(layers, d, d_ffn, d_kv, dtype, device):
    """A stack of decoder layers reduced to their weight matrices.

    Only the GEMM weights are allocated: in decode they are ~all of the bytes moved
    (see the calculator's per-layer numbers), and they are what batching amortizes.
    """
    stack = []
    for _ in range(layers):
        kw = dict(dtype=dtype, device=device)
        stack.append([
            torch.randn(d, d, **kw) * 0.02,        # q
            torch.randn(d_kv, d, **kw) * 0.02,     # k
            torch.randn(d_kv, d, **kw) * 0.02,     # v
            torch.randn(d, d, **kw) * 0.02,        # o
            torch.randn(d_ffn, d, **kw) * 0.02,    # gate
            torch.randn(d_ffn, d, **kw) * 0.02,    # up
            torch.randn(d, d_ffn, **kw) * 0.02,    # down
        ])
    return stack


def decode_step(stack, x):
    """One autoregressive step for every sequence in the batch: x is (batch, d).

    Every matmul here is a matrix-VECTOR product per sequence - a (batch, d) by
    (d, out) GEMM with batch as the only "long" dimension. That is the shape that
    makes decode bandwidth-bound.
    """
    lin = torch.nn.functional.linear
    for wq, wk, wv, wo, w_gate, w_up, w_down in stack:
        h = lin(x, wq)
        lin(x, wk)
        lin(x, wv)
        h = lin(h, wo)
        x = x + h
        g = lin(x, w_gate)
        u = lin(x, w_up)
        x = x + lin(torch.nn.functional.silu(g) * u, w_down)
    return x


def timed_ms(fn, iters, warmup):
    """Mean ms per call via CUDA events, after untimed warmup passes."""
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


def main():
    p = argparse.ArgumentParser(
        description="Decode throughput vs batch size: the bandwidth-bound proof.")
    p.add_argument("--layers", type=int, default=8,
                   help="synthetic decoder layers (default 8; keeps the model small "
                        "enough to fit alongside activations)")
    p.add_argument("--hidden", type=int, default=4096, help="d")
    p.add_argument("--ffn", type=int, default=14336, help="FFN inner width")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8, help="< heads for GQA")
    p.add_argument("--batches", default="1,2,4,8,16,32,64,128,256",
                   help="comma-separated batch sizes to sweep")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--bandwidth", type=float, default=0.0,
                   help="peak HBM GB/s, to report %% of peak (e.g. 5300 for MI300X)")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()

    require_gpu()
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    el = torch.finfo(dtype).bits // 8

    if args.hidden % args.heads:
        sys.exit(f"hidden {args.hidden} not divisible by heads {args.heads}")
    d_kv = (args.hidden // args.heads) * args.kv_heads

    try:
        batches = sorted({int(b) for b in args.batches.split(",") if b.strip()})
    except ValueError:
        sys.exit(f"--batches must be comma-separated integers, got {args.batches!r}")
    if not batches:
        sys.exit("--batches produced an empty sweep")

    stack = build_weights(args.layers, args.hidden, args.ffn, d_kv, dtype, device)
    params = sum(w.numel() for layer in stack for w in layer)
    weight_bytes = params * el
    print(f"model  : {args.layers} layers, d={args.hidden}, d_ffn={args.ffn}, "
          f"kv_heads={args.kv_heads}, dtype={args.dtype}")
    print(f"weights: {params / 1e9:.3f} G params = {weight_bytes / 1e9:.3f} GB "
          f"to read per decode step, at ANY batch size")
    print(f"timing : {args.iters} iters after {args.warmup} warmup, torch.cuda.Event")
    print()

    # 2 FLOPs per weight per token, so the step's arithmetic scales with batch while
    # weight_bytes does not. That divergence is the whole experiment. Note the
    # consequence: FLOP-per-weight-byte = 2*batch/element_bytes, which for fp16 is
    # exactly the batch size - the analytical result the calculator predicts.
    flops_per_token = 2 * params

    print("  'weight GB/s' counts only the weight read (the fixed cost batching")
    print("  amortizes). It should climb toward peak, then FALL once the step becomes")
    print("  compute-bound and time starts growing with batch. Activation traffic is")
    print("  excluded, so treat the column as a lower bound on total HBM traffic.")
    print()
    header = (f"  {'batch':>6} {'step ms':>9} {'tok/s agg':>11} {'tok/s per seq':>14}"
              f" {'weight GB/s':>12} {'TFLOP/s':>9} {'FLOP/B':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for b in batches:
        x = torch.randn(b, args.hidden, dtype=dtype, device=device)
        try:
            ms = timed_ms(lambda: decode_step(stack, x), args.iters, args.warmup)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"  {b:>6}   out of memory - stopping sweep here")
            del x
            torch.cuda.empty_cache()
            break
        secs = ms / 1e3
        gbs = weight_bytes / secs / 1e9
        rows.append((b, ms, b / secs, gbs))
        pct = f"  ({100 * gbs / args.bandwidth:.0f}% of peak)" if args.bandwidth else ""
        print(f"  {b:>6} {ms:9.4f} {b / secs:11.1f} {1 / secs:14.1f}"
              f" {gbs:12.1f} {b * flops_per_token / secs / 1e12:9.1f}"
              f" {b * flops_per_token / weight_bytes:7.1f}{pct}")
        del x

    print()
    print("=" * 78)
    print("  WHAT THIS SHOWS")
    print("=" * 78)
    if len(rows) >= 2:
        b0, ms0, agg0, _ = rows[0]
        bN, msN, aggN, _ = rows[-1]
        print(f"  batch {b0} -> {bN}: step latency grew {msN / ms0:.2f}x"
              f" while throughput grew {aggN / agg0:.1f}x")
        print(f"  ({bN // max(b0, 1)}x more tokens per step for {msN / ms0:.2f}x the time)")
        print()
        # Find where latency stops being flat: the transition to compute-bound.
        knee = next((b for b, ms, _, _ in rows if ms > 1.5 * ms0), None)
        if knee:
            print(f"  Latency first exceeds 1.5x the batch-{b0} step at batch {knee}:")
            print("  that is roughly where arithmetic intensity reaches the ridge point")
            print("  and the workload stops being purely bandwidth-bound.")
        else:
            print(f"  Latency never reached 1.5x the batch-{b0} step across this sweep -")
            print("  the whole range is bandwidth-bound. Extend --batches to find the knee.")
        print()
        print("  The weight bytes are a fixed cost paid once per step. Spreading it over")
        print("  more sequences is why every serving stack batches aggressively (B06),")
        print("  and why the same trick does almost nothing for prefill, which is already")
        print("  compute-bound at a single sequence.")
    else:
        print("  Only one batch size completed - rerun with a smaller --layers.")


if __name__ == "__main__":
    main()
