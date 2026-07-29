#!/usr/bin/env python3
"""Per-GPU training-memory planner for the four parallelism axes.

Everything printed here is an ANALYTIC MODEL evaluated in closed form -- no GPU
is touched and nothing is measured. Use it to rule configurations *out* before
you burn cluster hours, then confirm the survivors with a real run.

Formulas (see --explain for the full derivation):
  state bytes/param   16 for bf16 params + fp32 master + Adam    ZeRO, arXiv:1910.02054
  activations/layer   s*b*h*(34 + 5*a*s/h)                       arXiv:2205.05198 eq. 1
  selective recompute 34*s*b*h                                   arXiv:2205.05198 eq. 6
  full recompute      2*s*b*h                                    arXiv:1604.06174

Stdlib only: runs on any Python 3.8+ with no torch, no numpy.
"""

from __future__ import annotations

import argparse
import json
import math

GIB = 1024.0**3

# Published architectures, used only to spare you typing four numbers.
# Parameter counts are the headline figures from the papers below.
PRESETS = {
    "llama3-8b": dict(params=8.0e9, layers=32, hidden=4096, heads=32, seq=8192),
    "llama2-70b": dict(params=70.0e9, layers=80, hidden=8192, heads=64, seq=4096),
    "gpt3-175b": dict(params=175.0e9, layers=96, hidden=12288, heads=96, seq=2048),
    "mt-nlg-530b": dict(params=530.0e9, layers=105, hidden=20480, heads=128, seq=2048),
}

PRESET_SOURCES = {
    "llama3-8b": "Llama 3 herd of models, arXiv:2407.21783",
    "llama2-70b": "Llama 2, arXiv:2307.09288 (GQA: activation model is an upper bound)",
    "gpt3-175b": "GPT-3, arXiv:2005.14165",
    "mt-nlg-530b": "Megatron-Turing NLG 530B, arXiv:2201.11990",
}

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def state_bytes_per_param(dtype: str, optimizer: str) -> tuple[int, int, int]:
    """Return (param, grad, optimizer) resident bytes per parameter.

    Mixed precision keeps an fp32 master weight in addition to the low-precision
    compute copy; Adam adds two fp32 moments on top of that.
    """
    p = DTYPE_BYTES[dtype]
    g = p  # frameworks reduce gradients in the compute dtype by default
    master = 0 if dtype == "fp32" else 4
    moments = {"adam": 8, "adamw": 8, "sgd-momentum": 4, "sgd": 0}[optimizer]
    return p, g, master + moments


def activation_bytes(seq, micro_batch, hidden, heads, layers, tp, recompute, elem):
    """Per-GPU activation bytes for the worst-case pipeline stage.

    The per-layer term is Megatron's activation formula (arXiv:2205.05198). The
    layer count is L and *not* L/pp: under a 1F1B schedule the first stage holds
    ~pp in-flight micro-batches of L/pp layers each, so the product is L again.
    Pipeline parallelism therefore buys you parameter memory, not activation
    memory -- one of the least intuitive facts in this module.
    """
    s, b, h, a = seq, micro_batch, hidden, heads
    scale = elem / 2.0  # the published formula assumes 2-byte activations
    if recompute == "none":
        per_layer = s * b * h * (34.0 + 5.0 * a * s / h) / tp
    elif recompute == "selective":
        per_layer = 34.0 * s * b * h / tp
    elif recompute == "full":
        # Only the layer input is kept; TP does not shrink it without an extra
        # all-gather, which Megatron explicitly declines to pay.
        per_layer = 2.0 * s * b * h
    else:
        raise ValueError(f"unknown recompute mode: {recompute}")
    return per_layer * layers * scale


class Plan:
    """One candidate configuration and its analytic per-GPU footprint."""

    def __init__(self, name, cfg, tp, pp, zero, recompute):
        self.name = name
        self.tp = tp
        self.pp = pp
        self.zero = zero  # 0 = replicate, 1/2/3 = ZeRO stage sharded over dp
        self.recompute = recompute
        self.cfg = cfg

        gpus = cfg["gpus"]
        self.dp = max(1, gpus // (tp * pp))
        model_div = tp * pp  # TP splits matrices, PP splits layers

        pb, gb, ob = cfg["state_bytes"]
        psi = cfg["params"]
        opt_div = self.dp if zero >= 1 else 1
        grad_div = self.dp if zero >= 2 else 1
        par_div = self.dp if zero >= 3 else 1

        self.param_gb = psi * pb / (model_div * par_div) / GIB
        self.grad_gb = psi * gb / (model_div * grad_div) / GIB
        self.opt_gb = psi * ob / (model_div * opt_div) / GIB

        # ZeRO-3 must materialise the layer it is about to use (plus one
        # prefetched layer), so the "fully sharded" figure is never the whole story.
        self.transient_gb = 0.0
        if zero == 3:
            per_layer_params = psi / max(1, cfg["layers"])
            self.transient_gb = 2 * per_layer_params * pb / model_div / GIB

        self.act_gb = (
            activation_bytes(
                cfg["seq"],
                cfg["micro_batch"],
                cfg["hidden"],
                cfg["heads"],
                cfg["layers"],
                tp,
                recompute,
                cfg["act_elem"],
            )
            / GIB
        )
        self.total_gb = (
            self.param_gb
            + self.grad_gb
            + self.opt_gb
            + self.transient_gb
            + self.act_gb
            + cfg["reserve_gb"]
        )

    @property
    def state_gb(self):
        return self.param_gb + self.grad_gb + self.opt_gb + self.transient_gb

    @property
    def fits(self):
        return self.total_gb <= self.cfg["hbm_gb"]

    @property
    def disruption(self):
        """Heuristic ranking of engineering cost, NOT a measured slowdown.

        Ordering rationale: ZeRO stages add collectives on the existing DP mesh;
        recompute burns ~30% extra forward FLOPs; TP communicates activations
        every layer and pins you inside a node; PP adds a bubble plus scheduling
        complexity. Re-weight these for your own cluster.
        """
        score = {0: 0, 1: 1, 2: 2, 3: 4}[self.zero]
        score += {"none": 0, "selective": 2, "full": 4}[self.recompute]
        score += 6 * int(math.log2(self.tp))
        score += 5 * int(math.log2(self.pp))
        return score

    def axes(self):
        return f"dp={self.dp} tp={self.tp} pp={self.pp} z{self.zero}"

    def as_dict(self):
        return dict(
            name=self.name,
            dp=self.dp,
            tp=self.tp,
            pp=self.pp,
            zero=self.zero,
            recompute=self.recompute,
            param_gb=round(self.param_gb, 3),
            grad_gb=round(self.grad_gb, 3),
            optimizer_gb=round(self.opt_gb, 3),
            transient_gb=round(self.transient_gb, 3),
            activations_gb=round(self.act_gb, 3),
            reserve_gb=self.cfg["reserve_gb"],
            total_gb=round(self.total_gb, 3),
            fits=self.fits,
            disruption=self.disruption,
        )


def canonical_plans(cfg):
    """The single-axis strategies, each at the full GPU count, plus combinations."""
    n = cfg["gpus"]
    pow2 = [d for d in (1, 2, 4, 8, 16, 32, 64) if d <= n]
    tp_full = max(d for d in pow2 if d <= min(n, 8))  # TP is kept intra-node
    pp_full = max(d for d in pow2 if d <= min(n, cfg["layers"]))
    plans = [
        Plan("DDP (replicate everything)", cfg, 1, 1, 0, cfg["recompute"]),
        Plan("ZeRO-1 (shard optimizer)", cfg, 1, 1, 1, cfg["recompute"]),
        Plan("ZeRO-2 (+ shard grads)", cfg, 1, 1, 2, cfg["recompute"]),
        Plan("ZeRO-3 / FSDP (+ shard params)", cfg, 1, 1, 3, cfg["recompute"]),
        Plan(f"Tensor parallel tp={tp_full}", cfg, tp_full, 1, 0, cfg["recompute"]),
        Plan(f"Pipeline parallel pp={pp_full}", cfg, 1, pp_full, 0, cfg["recompute"]),
    ]
    if cfg["tp"] != 1 or cfg["pp"] != 1 or cfg["zero"] != 0:
        plans.append(
            Plan("Your flags (--tp/--pp/--zero)", cfg, cfg["tp"], cfg["pp"], cfg["zero"], cfg["recompute"])
        )
    return plans


def search(cfg):
    """Enumerate every (tp, pp, zero, recompute) that fits, cheapest first."""
    n = cfg["gpus"]
    divisors = [d for d in (1, 2, 4, 8, 16, 32, 64) if d <= n]
    found = []
    for tp in [d for d in divisors if d <= 8]:
        for pp in [d for d in divisors if d <= cfg["layers"] and tp * d <= n]:
            for zero in (0, 1, 2, 3):
                for rc in ("none", "selective", "full"):
                    plan = Plan(f"tp={tp} pp={pp} z{zero} rc={rc}", cfg, tp, pp, zero, rc)
                    if plan.fits:
                        found.append(plan)
    found.sort(key=lambda p: (p.disruption, p.total_gb))
    return found


def fmt_table(plans, cfg):
    head = (
        f"{'strategy':<32}{'axes':<22}{'state':>9}{'activ':>9}"
        f"{'reserve':>9}{'total':>9}  fits {cfg['hbm_gb']:.0f} GiB?"
    )
    lines = [head, "-" * len(head)]
    for p in plans:
        lines.append(
            f"{p.name:<32}{p.axes():<22}{p.state_gb:>9.1f}{p.act_gb:>9.1f}"
            f"{cfg['reserve_gb']:>9.1f}{p.total_gb:>9.1f}  "
            f"{'YES' if p.fits else 'NO':<4}"
        )
    lines.append("-" * len(head))
    lines.append("all figures GiB per GPU, analytic model (not measured)")
    return "\n".join(lines)


EXPLAIN = """
Derivation of every term
========================

1. Optimizer/parameter state -- bytes per parameter
   Mixed-precision training with Adam keeps FIVE copies of every weight:
     bf16 parameter            2 bytes
     bf16 gradient             2 bytes
     fp32 master weight        4 bytes
     fp32 Adam first moment    4 bytes
     fp32 Adam second moment   4 bytes
     -------------------------------------
     total                    16 bytes/parameter
   This is the "16 * psi" figure in the ZeRO paper (arXiv:1910.02054). A 70e9
   parameter model therefore needs 70e9 * 16 = 1.12e12 bytes ~= 1043 GiB of
   state before a single activation exists -- 5.4 GPUs of 192 GiB each.

2. Activations -- per transformer layer, per micro-batch
   Megatron's accounting (arXiv:2205.05198, eq. 1) for one layer with 2-byte
   activations, sequence s, micro-batch b, hidden h, heads a:
       s*b*h*(34 + 5*a*s/h)
   The 34 covers the linear/LayerNorm/dropout tensors; the 5*a*s/h term is the
   attention score matrix, which is QUADRATIC in s. That quadratic term is why
   long context breaks memory first and why selective recompute (eq. 6) targets
   exactly it, leaving 34*s*b*h/tp. Full recompute (arXiv:1604.06174) keeps only
   the layer input: 2*s*b*h.

3. Why the layer count is L and not L/pp
   Under 1F1B the deepest-loaded stage holds ~pp in-flight micro-batches of
   L/pp layers, and pp * L/pp = L. Pipeline parallelism shrinks parameter and
   optimizer memory by pp but leaves activation memory essentially unchanged.

4. Sharding divisors
   TP and PP divide parameters, gradients and optimizer state by tp*pp.
   ZeRO stage 1 divides optimizer state by dp; stage 2 also gradients; stage 3
   also parameters. TP additionally divides activations by tp (with sequence
   parallelism); ZeRO does not touch activations at all.

5. What this model does NOT include
   Fragmentation, framework/CUDA-or-HIP context, communication buffers beyond
   the ZeRO-3 prefetch, fused-kernel workspaces, and the loss/logit tensor
   (which is large when vocabulary is large). That is what --reserve is for;
   treat it as a fudge factor you calibrate against one real run.
"""


DEFAULT_SHAPE = dict(params=0.0, layers=32, hidden=4096, heads=32, seq=4096)


def build_config(args):
    """Merge preset and explicit flags. Any flag the user actually passed wins."""
    shape = dict(PRESETS[args.preset]) if args.preset else dict(DEFAULT_SHAPE)
    for key, value in (
        ("params", args.params * 1e9 if args.params else None),
        ("layers", args.layers),
        ("hidden", args.hidden),
        ("heads", args.heads),
        ("seq", args.seq),
    ):
        if value:
            shape[key] = value
    if not shape.get("params"):
        # Standard transformer estimate: ~12*h^2 parameters per layer.
        shape["params"] = 12.0 * shape["layers"] * shape["hidden"] ** 2

    cfg = dict(shape)
    cfg.update(
        micro_batch=args.micro_batch,
        gpus=args.gpus,
        tp=args.tp,
        pp=args.pp,
        zero=args.zero,
        recompute=args.recompute,
        hbm_gb=args.hbm,
        reserve_gb=args.reserve,
        state_bytes=state_bytes_per_param(args.dtype, args.optimizer),
        act_elem=DTYPE_BYTES[args.dtype],
    )
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analytic per-GPU training memory for DDP / ZeRO / TP / PP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--preset", choices=sorted(PRESETS), help="published model shape")
    ap.add_argument("--list-presets", action="store_true")
    # These default to 0/None so a --preset can supply them; an explicit flag wins.
    ap.add_argument("--params", type=float, default=0.0, help="parameter count in billions")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--heads", type=int, default=0)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--zero", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_BYTES))
    ap.add_argument(
        "--optimizer", default="adam", choices=("adam", "adamw", "sgd-momentum", "sgd")
    )
    ap.add_argument("--recompute", default="none", choices=("none", "selective", "full"))
    ap.add_argument("--hbm", type=float, default=192.0, help="GiB of HBM per GPU")
    ap.add_argument("--reserve", type=float, default=10.0, help="GiB held back for overhead")
    ap.add_argument("--suggest", action="store_true", help="rank every configuration that fits")
    ap.add_argument("--top", type=int, default=8, help="rows to show with --suggest")
    ap.add_argument("--explain", action="store_true", help="print the derivations and exit")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.list_presets:
        for name in sorted(PRESETS):
            p = PRESETS[name]
            print(f"{name:<14} {p['params']/1e9:>6.0f}B params  L={p['layers']:<4}"
                  f"h={p['hidden']:<6}a={p['heads']:<4}s={p['seq']:<6}"
                  f"  [{PRESET_SOURCES[name]}]")
        return 0
    if args.explain:
        print(EXPLAIN.strip())
        return 0

    # tp*pp*dp must fit in the world: without this guard a bad flag combination
    # would silently over-shard and report a footprint no cluster can deliver.
    if args.tp * args.pp > args.gpus:
        ap.error(f"--tp {args.tp} x --pp {args.pp} exceeds --gpus {args.gpus}")

    cfg = build_config(args)
    pb, gb, ob = cfg["state_bytes"]

    plans = search(cfg) if args.suggest else canonical_plans(cfg)
    if args.suggest:
        plans = plans[: args.top]

    if args.json:
        print(json.dumps(dict(config={k: v for k, v in cfg.items() if k != "state_bytes"},
                              state_bytes_per_param=dict(param=pb, grad=gb, optimizer=ob),
                              plans=[p.as_dict() for p in plans]), indent=2))
        return 0

    total_state_gb = cfg["params"] * (pb + gb + ob) / GIB
    print(f"model      : {cfg['params']/1e9:.1f}B params, L={cfg['layers']}, "
          f"h={cfg['hidden']}, heads={cfg['heads']}")
    print(f"step shape : seq={cfg['seq']}, micro-batch={cfg['micro_batch']}, "
          f"recompute={cfg['recompute']}")
    print(f"cluster    : {cfg['gpus']} GPUs x {cfg['hbm_gb']:.0f} GiB "
          f"(reserve {cfg['reserve_gb']:.0f} GiB, usable "
          f"{cfg['hbm_gb'] - cfg['reserve_gb']:.0f} GiB)")
    print(f"state      : {pb}+{gb}+{ob} = {pb+gb+ob} bytes/param -> "
          f"{total_state_gb:.0f} GiB unsharded "
          f"({total_state_gb/max(1e-9, cfg['hbm_gb']):.1f} GPUs of pure state)")
    print()
    print(fmt_table(plans, cfg))
    if args.suggest:
        best = plans[0] if plans else None
        print()
        if best is None:
            print("nothing fits: add GPUs, shorten the sequence, or cut the micro-batch")
        else:
            print(f"cheapest fit (heuristic score {best.disruption}): "
                  f"{best.axes()} recompute={best.recompute} "
                  f"-> {best.total_gb:.1f} GiB/GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
