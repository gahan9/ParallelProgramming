#!/usr/bin/env python3
"""Transformer performance calculator - parameters, FLOPs, KV cache, and the
memory-bound decode throughput ceiling.

Every number printed is a DERIVED ESTIMATE from closed-form formulas, never a
measurement. It is an upper bound on what the hardware can do: it assumes perfect
overlap, zero launch overhead, and 100% of vendor-published peak bandwidth. Real
systems land below it. Use bench/ for measurements.

Stdlib only, by design - this must run on a login node with no venv and no GPU.

Units: memory and bandwidth are DECIMAL (1 GB = 1e9 bytes), so
tokens/s = bandwidth_GBs / bytes_GB is an exact division. Vendor bandwidth specs
are decimal too, which keeps the whole chain consistent.

Usage:
    python tools/model_calculator.py --preset llama3-8b
    python tools/model_calculator.py --preset llama2-70b --seq 4096 --batch 32
    python tools/model_calculator.py -L 32 -d 4096 --heads 32 --kv-heads 8 \
        --ffn 14336 --glu --vocab 128256 --bandwidth 8000 --capacity 288
"""

import argparse
import sys

# Bytes per element. int8/fp8 entries are for weight-only or full quantization
# sizing; they say nothing about accuracy (that is B04's subject).
DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Published configurations only. Sources are printed with the preset so a learner
# can check them; never add a preset you cannot cite.
PRESETS = {
    "llama3-8b": dict(
        layers=32, hidden=4096, heads=32, kv_heads=8, ffn=14336, glu=True,
        vocab=128256, tie_embeddings=False, seq=8192,
        source="Meta Llama 3 8B model card / HF config.json (arXiv:2407.21783)",
    ),
    "llama2-70b": dict(
        layers=80, hidden=8192, heads=64, kv_heads=8, ffn=28672, glu=True,
        vocab=32000, tie_embeddings=False, seq=4096,
        source="Llama 2 paper Table 1 + reference impl ModelArgs (arXiv:2307.09288)",
    ),
    "gpt3-175b": dict(
        layers=96, hidden=12288, heads=96, kv_heads=96, ffn=49152, glu=False,
        vocab=50257, tie_embeddings=True, seq=2048,
        source="GPT-3 paper Table 2.1 (arXiv:2005.14165)",
    ),
}

# Vendor-published dense peaks, for the compute-vs-memory verdict. Dense (no
# sparsity doubling), bf16/fp16 matrix cores.
ACCELERATORS = {
    "mi300x": dict(bandwidth=5300.0, capacity=192.0, tflops=1307.4,
                   note="AMD Instinct MI300X, HBM3"),
    "mi355x": dict(bandwidth=8000.0, capacity=288.0, tflops=2500.0,
                   note="AMD Instinct MI355X, HBM3E"),
    "h100": dict(bandwidth=3350.0, capacity=80.0, tflops=989.4,
                 note="NVIDIA H100 SXM, HBM3"),
    "h200": dict(bandwidth=4800.0, capacity=141.0, tflops=989.4,
                 note="NVIDIA H200 SXM, HBM3E"),
}


class Model:
    """A decoder-only transformer, reduced to the shapes that decide performance."""

    def __init__(self, layers, hidden, heads, kv_heads, ffn, glu, vocab,
                 tie_embeddings):
        if hidden % heads:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        if heads % kv_heads:
            raise ValueError(f"heads ({heads}) must be divisible by kv-heads ({kv_heads})")
        self.layers = layers
        self.d = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.d_ffn = ffn
        self.glu = glu
        self.vocab = vocab
        self.tie_embeddings = tie_embeddings

        self.d_head = hidden // heads
        # d_kv is the width of ONE of K or V. MHA: d_kv == d. GQA: d/group_size.
        self.d_kv = self.kv_heads * self.d_head
        self.group_size = heads // kv_heads

    # ---- parameters ----

    @property
    def p_q(self):
        return self.d * self.d

    @property
    def p_kv(self):
        return 2 * self.d * self.d_kv

    @property
    def p_o(self):
        return self.d * self.d

    @property
    def p_attn(self):
        return self.p_q + self.p_kv + self.p_o

    @property
    def p_ffn(self):
        # GLU variants (SwiGLU) need a third matrix: gate, up, down.
        return (3 if self.glu else 2) * self.d * self.d_ffn

    @property
    def p_layer(self):
        return self.p_attn + self.p_ffn

    @property
    def p_blocks(self):
        return self.layers * self.p_layer

    @property
    def p_embed(self):
        return self.vocab * self.d * (1 if self.tie_embeddings else 2)

    @property
    def p_total(self):
        return self.p_blocks + self.p_embed

    @property
    def ffn_multiplier(self):
        """The m in the (4 + 2m)d^2 rule, back-solved from the real FFN shape."""
        return self.p_ffn / (2.0 * self.d * self.d)

    # ---- FLOPs ----

    def gemm_flops_per_token(self):
        """2 FLOPs (one multiply + one add) per weight, per token."""
        return 2 * self.p_blocks

    def attn_flops_prefill(self, seq):
        """QK^T then (softmax)V, all layers, one sequence: 4 * L * N^2 * d.

        GQA does not reduce this - K is broadcast back to all query heads, so the
        FLOP count still sees the full query width d. GQA saves memory, not math.
        """
        return 4 * self.layers * seq * seq * self.d

    def attn_flops_decode(self, ctx):
        """One new token attending over ctx cached tokens: 4 * L * ctx * d."""
        return 4 * self.layers * ctx * self.d

    def lm_head_flops_per_token(self):
        return 2 * self.vocab * self.d

    # ---- memory ----

    def weight_bytes(self, dtype_bytes):
        return self.p_total * dtype_bytes

    def kv_bytes_per_token(self, kv_dtype_bytes):
        """2 (K and V) * layers * d_kv * bytes. Per token, per sequence."""
        return 2 * self.layers * self.d_kv * kv_dtype_bytes


def human_bytes(n):
    for unit, scale in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.3f} {unit}"
    return f"{n:.0f} B"


def si(n, unit=""):
    for suffix, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.3f} {suffix}{unit}"
    return f"{n:.2f} {unit}".rstrip()


def rule(title=""):
    if title:
        print(f"\n{title}")
        print("-" * max(len(title), 60))
    else:
        print("-" * 60)


def report(m, args):
    dtype_b = DTYPE_BYTES[args.dtype]
    kv_b = DTYPE_BYTES[args.kv_dtype]
    seq, batch = args.seq, args.batch

    print("=" * 60)
    print("  Transformer performance calculator - DERIVED ESTIMATES")
    print("=" * 60)

    rule("Configuration")
    if args.preset:
        print(f"  preset            {args.preset}")
        print(f"  source            {PRESETS[args.preset]['source']}")
    print(f"  layers L          {m.layers}")
    print(f"  hidden d          {m.d}")
    print(f"  heads h           {m.heads}  (head_dim {m.d_head})")
    kind = "MHA" if m.group_size == 1 else ("MQA" if m.kv_heads == 1 else "GQA")
    print(f"  kv heads          {m.kv_heads}  -> {kind}, group size {m.group_size}, d_kv {m.d_kv}")
    print(f"  ffn d_ffn         {m.d_ffn}  ({'GLU, 3 matrices' if m.glu else '2 matrices'})"
          f"  -> effective m = {m.ffn_multiplier:.2f}")
    print(f"  vocab             {m.vocab}  ({'tied' if m.tie_embeddings else 'untied'} embeddings)")
    print(f"  weight dtype      {args.dtype} ({dtype_b:g} B)   kv dtype {args.kv_dtype} ({kv_b:g} B)")
    print(f"  seq len N         {seq}      batch B  {batch}")
    print(f"  accelerator       {args.bandwidth:.0f} GB/s, {args.capacity:.0f} GB"
          + (f", {args.peak_tflops:.1f} TFLOP/s" if args.peak_tflops else ""))

    # ---- parameters ----
    rule("Parameters")
    print(f"  Q proj      d*d           = {si(m.p_q):>12}")
    print(f"  K+V proj    2*d*d_kv      = {si(m.p_kv):>12}")
    print(f"  O proj      d*d           = {si(m.p_o):>12}")
    print(f"  attention   subtotal      = {si(m.p_attn):>12}"
          f"   ({m.p_attn / m.d ** 2:.2f} d^2)")
    print(f"  FFN         {3 if m.glu else 2}*d*d_ffn     = {si(m.p_ffn):>12}"
          f"   ({m.p_ffn / m.d ** 2:.2f} d^2)")
    print(f"  per layer                 = {si(m.p_layer):>12}"
          f"   ({m.p_layer / m.d ** 2:.2f} d^2)")
    print(f"  all {m.layers:>3} layers            = {si(m.p_blocks):>12}")
    print(f"  embeddings + lm_head      = {si(m.p_embed):>12}")
    print(f"  TOTAL                     = {si(m.p_total):>12} params")
    print()
    mha_layer = (4 + 2 * m.ffn_multiplier) * m.d ** 2
    print(f"  (4+2m)d^2 @ m={m.ffn_multiplier:.2f}, MHA = {si(mha_layer):>12}"
          f"   ({4 + 2 * m.ffn_multiplier:.2f} d^2)")
    print(f"  actual layer              = {si(m.p_layer):>12}"
          f"   ({m.p_layer / m.d ** 2:.2f} d^2)")
    if m.group_size > 1:
        print(f"    the {mha_layer / m.d ** 2 - m.p_layer / m.d ** 2:.2f} d^2 gap is GQA:"
              f" K and V are {m.d}x{m.d_kv}, not {m.d}x{m.d}")
    print(f"  12 d^2 (m=4, MHA) shortcut= {si(12 * m.d ** 2):>12}"
          f"   -> {12 * m.d ** 2 / m.p_layer:.2f}x actual")
    print("  (norm weights, biases, and positional embeddings are O(d) or O(N*d)"
          " and omitted)")

    # ---- memory ----
    rule("Memory")
    w = m.weight_bytes(dtype_b)
    kv_tok = m.kv_bytes_per_token(kv_b)
    kv_seq = kv_tok * seq
    kv_all = kv_seq * batch
    print(f"  weights, {args.dtype:<5}            = {human_bytes(w):>12}")
    print(f"  KV per token per seq      = {human_bytes(kv_tok):>12}"
          f"   (2 * {m.layers} * {m.d_kv} * {kv_b:g} B)")
    print(f"  KV per seq, N={seq:<12}= {human_bytes(kv_seq):>12}")
    print(f"  KV for batch B={batch:<11}= {human_bytes(kv_all):>12}")
    print(f"  weights + KV              = {human_bytes(w + kv_all):>12}")

    free = args.capacity * 1e9 - w
    print(f"\n  {args.capacity:.0f} GB capacity - weights = {human_bytes(free)} left for KV + activations")
    if free <= 0:
        print("  !! weights alone exceed capacity - needs tensor/pipeline sharding (B07)")
    else:
        max_seqs = int(free / kv_seq)
        print(f"  max concurrent seqs @ N={seq} = {max_seqs}"
              f"   (ignores activations, fragmentation, runtime overhead)")
        if m.group_size > 1:
            mha_kv_seq = 2 * m.layers * m.d * kv_b * seq
            print(f"  same model with MHA instead = {int(free / mha_kv_seq)} seqs"
                  f"   ({mha_kv_seq / kv_seq:.0f}x larger KV per seq)")

    # ---- FLOPs ----
    rule("FLOPs")
    gemm_tok = m.gemm_flops_per_token()
    head_tok = m.lm_head_flops_per_token()
    attn_pre = m.attn_flops_prefill(seq)
    prefill = (gemm_tok + head_tok) * seq * batch + attn_pre * batch
    print(f"  GEMM fwd per token        = {si(gemm_tok, 'FLOP'):>12}"
          f"   (2 x {si(m.p_blocks)} non-embedding params)")
    print(f"  lm_head per token         = {si(head_tok, 'FLOP'):>12}")
    print(f"  attention, 1 seq          = {si(attn_pre, 'FLOP'):>12}"
          f"   (4 * L * N^2 * d, N={seq})")
    print(f"  PREFILL total             = {si(prefill, 'FLOP'):>12}"
          f"   (B={batch}, N={seq})")
    print()
    attn_dec = m.attn_flops_decode(seq)
    decode_tok = gemm_tok + head_tok + attn_dec
    print(f"  DECODE per token per seq  = {si(decode_tok, 'FLOP'):>12}")
    print(f"    of which attn vs KV     = {si(attn_dec, 'FLOP'):>12}"
          f"   ({100 * attn_dec / decode_tok:.1f}%)")
    print(f"  training fwd+bwd / token  = {si(3 * gemm_tok, 'FLOP'):>12}"
          f"   (~6 x non-embedding params)")
    print()
    # 4*N^2*d == 2*p_layer*N  =>  N = p_layer / (2d). For an MHA model with the
    # textbook FFN this reduces to the familiar N = (2+m)d.
    crossover = int(m.p_layer / (2 * m.d))
    print(f"  Attention overtakes the GEMMs at N = p_layer/(2d) = {crossover} tokens")
    print(f"    textbook MHA form, N = (2+m)d = {int((2 + m.ffn_multiplier) * m.d)}"
          f"   (higher: MHA has more projection params to beat)")
    print(f"    at N={seq}: attention is {100 * attn_pre / (attn_pre + gemm_tok * seq):.1f}%"
          f" of a prefill layer stack's FLOPs")

    # ---- regime analysis ----
    rule("Regime analysis (arithmetic intensity)")
    ridge = args.peak_tflops * 1e12 / (args.bandwidth * 1e9) if args.peak_tflops else None
    if ridge:
        print(f"  machine ridge point          = {ridge:.0f} FLOP/byte"
              f"   ({args.peak_tflops:.1f} TFLOP/s / {args.bandwidth:.0f} GB/s)")

    # Per-GEMM intensity for a d x d projection: 2*T*d^2 FLOPs over
    # (weights 2*d^2*bytes... careful) -> weights + activations in and out.
    def gemm_intensity(tokens):
        flops = 2.0 * tokens * m.d * m.d
        byts = m.d * m.d * dtype_b + 2.0 * tokens * m.d * dtype_b
        return flops / byts

    print(f"  d x d projection, {seq:>6} tok  = {gemm_intensity(seq):>8.1f} FLOP/byte  (prefill shape)")
    if batch != 1:
        print(f"  d x d projection, {batch:>6} tok  = {gemm_intensity(batch):>8.1f} FLOP/byte  (decode shape, B={batch})")
    print(f"  d x d projection, {1:>6} tok  = {gemm_intensity(1):>8.1f} FLOP/byte  (decode shape, B=1)")
    if ridge:
        # Solve 2*B*d / (bytes * (d + 2B)) = ridge for B. Only solvable while the
        # weight read still dominates, i.e. d > ridge * bytes_per_element.
        if ridge * dtype_b < m.d:
            need = ridge * dtype_b * m.d / (2.0 * m.d - 2.0 * ridge * dtype_b)
            print(f"  batch needed for compute-bound decode ~ {need:.0f} concurrent sequences")
        else:
            print("  no batch size makes this projection compute-bound (d too small vs ridge point)")

    # ---- throughput ceiling ----
    rule("Memory-bound decode throughput (the useful prediction)")
    print("  Generating one token reads ~every weight once, plus this sequence's KV.")
    print(f"  bytes read per decode step (B={batch}) = weights + B * KV")
    step_bytes = w + batch * kv_seq
    bw = args.bandwidth * 1e9
    step_s = step_bytes / bw
    print(f"    = {human_bytes(w)} + {batch} * {human_bytes(kv_seq)}"
          f" = {human_bytes(step_bytes)}")
    print(f"  step time  >= {step_s * 1e3:.3f} ms   -> {1.0 / step_s:.1f} steps/s")
    print(f"  THROUGHPUT ceiling            = {batch / step_s:>10.1f} tokens/s aggregate")
    print(f"  per-sequence (user-visible)   = {1.0 / step_s:>10.1f} tokens/s")
    print()
    print(f"  B=1 ceiling (weights only)    = {bw / w:>10.1f} tokens/s"
          f"   ({args.bandwidth:.0f} GB/s / {human_bytes(w)})")
    if args.peak_tflops:
        # If FLOPs were the only limit, this is what the same batch would deliver.
        compute_tps = args.peak_tflops * 1e12 / decode_tok
        mem_tps = batch / step_s
        print(f"  compute-only ceiling          = {compute_tps:>10.1f} tokens/s aggregate"
              f"   (peak FLOP/s / {si(decode_tok)}FLOP per token)")
        print(f"  -> the memory ceiling is {compute_tps / mem_tps:.0f}x lower,"
              f" so decode is BANDWIDTH-BOUND")

    rule("Prefill throughput (compute-bound side)")
    if args.peak_tflops:
        pf_s = prefill / (args.peak_tflops * 1e12)
        print(f"  prefill FLOPs / peak         = {pf_s * 1e3:.2f} ms at 100% of peak")
        print(f"  at a realistic 60% of peak   = {pf_s * 1e3 / 0.6:.2f} ms"
              f"  -> {batch * seq / (pf_s / 0.6):,.0f} tokens/s")
        print("  Prefill scales with FLOPs, so batching it buys little - the GEMMs are")
        print("  already large enough to saturate the matrix cores.")
    else:
        print("  pass --peak-tflops to estimate the prefill (compute-bound) side")

    print()
    print("=" * 60)
    print("  All figures above are ANALYTICAL CEILINGS, not measurements.")
    print("  Measure with bench/profile_layer.py and bench/batch_scaling.py.")
    print("=" * 60)


def build_parser():
    p = argparse.ArgumentParser(
        description="Derive transformer params, FLOPs, KV-cache size, and the "
                    "memory-bound decode throughput ceiling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="presets: " + ", ".join(sorted(PRESETS))
               + "\naccelerators: " + ", ".join(sorted(ACCELERATORS)),
    )
    p.add_argument("--preset", choices=sorted(PRESETS),
                   help="start from a published config, then override any field")
    p.add_argument("--gpu", choices=sorted(ACCELERATORS),
                   help="fill bandwidth/capacity/peak-tflops from a vendor spec sheet")
    p.add_argument("-L", "--layers", type=int)
    p.add_argument("-d", "--hidden", type=int)
    p.add_argument("--heads", type=int)
    p.add_argument("--kv-heads", type=int,
                   help="< heads for GQA, 1 for MQA; defaults to heads (MHA)")
    p.add_argument("--ffn", type=int, help="FFN inner width; defaults to 4*d")
    p.add_argument("--glu", action="store_true",
                   help="SwiGLU-style FFN: 3 matrices instead of 2")
    p.add_argument("--vocab", type=int, help="vocabulary size (default 32000)")
    p.add_argument("--tie-embeddings", action="store_true",
                   help="share input embedding with lm_head")
    p.add_argument("--seq", type=int, help="context length N")
    p.add_argument("--batch", type=int, default=1, help="concurrent sequences")
    p.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="fp16")
    p.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES), default="fp16")
    p.add_argument("--bandwidth", type=float, default=5300.0,
                   help="HBM bandwidth in GB/s (default: MI300X 5300)")
    p.add_argument("--capacity", type=float, default=192.0,
                   help="HBM capacity in GB (default: MI300X 192)")
    p.add_argument("--peak-tflops", type=float, default=1307.4,
                   help="dense bf16/fp16 matrix-core peak (default: MI300X 1307.4)")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = dict(PRESETS[args.preset]) if args.preset else {}
    cfg.pop("source", None)

    # Explicit CLI values win over the preset; a preset supplies the rest.
    for field in ("layers", "hidden", "heads", "kv_heads", "ffn", "vocab", "seq"):
        val = getattr(args, field)
        if val is not None:
            cfg[field] = val
    if args.glu:
        cfg["glu"] = True
    if args.tie_embeddings:
        cfg["tie_embeddings"] = True

    if args.gpu:
        spec = ACCELERATORS[args.gpu]
        # Only override defaults the user did not set explicitly.
        defaults = parser.get_default
        if args.bandwidth == defaults("bandwidth"):
            args.bandwidth = spec["bandwidth"]
        if args.capacity == defaults("capacity"):
            args.capacity = spec["capacity"]
        if args.peak_tflops == defaults("peak_tflops"):
            args.peak_tflops = spec["tflops"]

    missing = [k for k in ("layers", "hidden", "heads") if not cfg.get(k)]
    if missing:
        parser.error(
            "need --preset, or all of -L/--layers -d/--hidden --heads "
            f"(missing: {', '.join(missing)})")

    cfg.setdefault("kv_heads", cfg["heads"])
    cfg.setdefault("ffn", 4 * cfg["hidden"])
    cfg.setdefault("glu", False)
    cfg.setdefault("vocab", 32000)
    cfg.setdefault("tie_embeddings", False)
    args.seq = cfg.pop("seq", None) or 2048

    try:
        model = Model(**cfg)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    report(model, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
