#!/usr/bin/env python3
"""KV-cache capacity planner: how many sequences fit in HBM, and what wastes it.

Answers the question every serving deployment starts with: given this GPU and
this model, how many concurrent sequences can I hold, and what is the binding
resource? Then shows how much of that capacity three things buy back:

  1. paged (block) allocation instead of contiguous per-sequence preallocation
  2. grouped-query attention (fewer KV heads)
  3. KV-cache quantization (fewer bytes per element)

All arithmetic, no measurement. The model shapes are the published configs of
well-known open models; for anything you actually deploy, read `config.json`
(`num_hidden_layers`, `num_key_value_heads`, `hidden_size / num_attention_heads`)
rather than trusting a preset here.

Stdlib only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

GIB = 1024 ** 3

DTYPE_BYTES: Dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass(frozen=True)
class ModelShape:
    name: str
    params_b: float
    layers: int
    q_heads: int
    kv_heads: int
    head_dim: int

    @property
    def attention_kind(self) -> str:
        if self.kv_heads == 1:
            return "MQA"
        if self.kv_heads == self.q_heads:
            return "MHA"
        return f"GQA x{self.q_heads // self.kv_heads}"


PRESETS: Dict[str, ModelShape] = {
    "mha-7b": ModelShape("mha-7b", 7.0, 32, 32, 32, 128),
    "gqa-8b": ModelShape("gqa-8b", 8.0, 32, 32, 8, 128),
    "gqa-70b": ModelShape("gqa-70b", 70.0, 80, 64, 8, 128),
    "gqa-405b": ModelShape("gqa-405b", 405.0, 126, 128, 8, 128),
}

# Datasheet HBM capacities in decimal GB (10^9 bytes), as vendors quote them.
# Reported free memory is always lower -- confirm with `amd-smi static` or
# `nvidia-smi` on the machine you are planning for.
GPU_HBM_GB: Dict[str, float] = {
    "mi300x": 192.0,
    "mi325x": 256.0,
    "mi355x": 288.0,
    "h100-80": 80.0,
    "h200": 141.0,
    "a100-40": 40.0,
    "a100-80": 80.0,
}


@dataclass
class Plan:
    """One fully-resolved capacity plan."""

    shape: ModelShape
    tp: int
    hbm_bytes: float
    weight_dtype: str
    kv_dtype: str
    context: int
    mean_context: int
    block_tokens: int
    activation_bytes: float
    headroom_pct: float

    # -- per-GPU shard geometry -------------------------------------------- #
    @property
    def kv_heads_per_gpu(self) -> float:
        """KV heads are sharded across TP ranks; below tp they must be replicated."""
        return max(1.0, self.shape.kv_heads / self.tp)

    @property
    def kv_replicated(self) -> bool:
        return self.shape.kv_heads < self.tp

    @property
    def weight_bytes(self) -> float:
        total = self.shape.params_b * 1e9 * DTYPE_BYTES[self.weight_dtype]
        return total / self.tp

    @property
    def kv_bytes_per_token(self) -> float:
        """K and V, every layer, this GPU's share of the KV heads."""
        return (
            2.0
            * self.shape.layers
            * self.kv_heads_per_gpu
            * self.shape.head_dim
            * DTYPE_BYTES[self.kv_dtype]
        )

    # -- budget ------------------------------------------------------------ #
    @property
    def kv_pool_bytes(self) -> float:
        usable = self.hbm_bytes - self.weight_bytes - self.activation_bytes
        return max(0.0, usable * (1.0 - self.headroom_pct / 100.0))

    @property
    def headroom_bytes(self) -> float:
        usable = self.hbm_bytes - self.weight_bytes - self.activation_bytes
        return max(0.0, usable * (self.headroom_pct / 100.0))

    # -- concurrency ------------------------------------------------------- #
    @property
    def bytes_per_seq_contiguous(self) -> float:
        """One buffer per sequence, reserved for the maximum context up front."""
        return self.kv_bytes_per_token * self.context

    @property
    def blocks_per_seq_paged(self) -> int:
        return -(-self.mean_context // self.block_tokens)  # ceil

    @property
    def bytes_per_seq_paged(self) -> float:
        """Only the blocks a sequence has actually filled, rounded up to a block."""
        return self.kv_bytes_per_token * self.blocks_per_seq_paged * self.block_tokens

    @property
    def concurrency_contiguous(self) -> int:
        return int(self.kv_pool_bytes // self.bytes_per_seq_contiguous)

    @property
    def concurrency_paged(self) -> int:
        return int(self.kv_pool_bytes // self.bytes_per_seq_paged)

    @property
    def internal_frag_bytes(self) -> float:
        """Paged waste: the unused tail of each sequence's last block."""
        padded = self.blocks_per_seq_paged * self.block_tokens
        return self.kv_bytes_per_token * (padded - self.mean_context)

    @property
    def reservation_waste_bytes(self) -> float:
        """Contiguous waste: reserved for max context, used only mean context."""
        return self.kv_bytes_per_token * (self.context - self.mean_context)

    @property
    def fits(self) -> bool:
        return self.kv_pool_bytes > 0 and self.concurrency_paged >= 1


def gib(x: float) -> str:
    return f"{x / GIB:8.2f}"


def print_table(title: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    cells = [[f"{c}" for c in row] for row in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in cells)) for i in range(len(header))]
    print(title)
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
    print()


def report_budget(plan: Plan) -> None:
    rows = [
        ["HBM (per GPU)", gib(plan.hbm_bytes), "100.0"],
        [f"weights ({plan.weight_dtype}, tp={plan.tp})", gib(plan.weight_bytes),
         f"{100 * plan.weight_bytes / plan.hbm_bytes:.1f}"],
        ["activations + workspace", gib(plan.activation_bytes),
         f"{100 * plan.activation_bytes / plan.hbm_bytes:.1f}"],
        [f"fragmentation headroom ({plan.headroom_pct:.0f}%)", gib(plan.headroom_bytes),
         f"{100 * plan.headroom_bytes / plan.hbm_bytes:.1f}"],
        ["KV cache pool (what is left)", gib(plan.kv_pool_bytes),
         f"{100 * plan.kv_pool_bytes / plan.hbm_bytes:.1f}"],
    ]
    print_table("HBM budget", ["component", "GiB", "% of HBM"], rows)


def report_concurrency(plan: Plan) -> None:
    kv_per_tok_kib = plan.kv_bytes_per_token / 1024.0
    print(
        f"KV cache costs {kv_per_tok_kib:.1f} KiB per token per sequence "
        f"({plan.shape.attention_kind}, {plan.kv_heads_per_gpu:g} KV heads/GPU, "
        f"{plan.shape.layers} layers, {plan.kv_dtype})."
    )
    if plan.kv_replicated:
        print(
            f"  NOTE: kv_heads={plan.shape.kv_heads} < tp={plan.tp}, so the KV cache must be"
            " replicated across ranks -- TP stops shrinking it here."
        )
    print(
        f"A full {plan.context}-token context therefore costs "
        f"{plan.kv_bytes_per_token * plan.context / GIB:.2f} GiB per sequence.\n"
    )

    rows = [
        [
            "contiguous preallocation",
            f"{plan.context} (reserved)",
            gib(plan.bytes_per_seq_contiguous),
            plan.concurrency_contiguous,
            gib(plan.reservation_waste_bytes),
            f"{100 * (plan.context - plan.mean_context) / plan.context:.1f}",
        ],
        [
            f"paged, {plan.block_tokens}-token blocks",
            f"{plan.blocks_per_seq_paged * plan.block_tokens} (allocated)",
            gib(plan.bytes_per_seq_paged),
            plan.concurrency_paged,
            gib(plan.internal_frag_bytes),
            f"{100 * plan.internal_frag_bytes / plan.bytes_per_seq_paged:.1f}",
        ],
    ]
    print_table(
        f"Max concurrent sequences (mean actual length {plan.mean_context}, max context {plan.context})",
        ["allocation", "tokens held/seq", "GiB/seq", "max seqs", "wasted GiB/seq", "waste %"],
        rows,
    )
    if plan.concurrency_contiguous:
        gain = plan.concurrency_paged / plan.concurrency_contiguous
        print(
            f"Paging buys {gain:.1f}x the concurrency here, purely by not reserving for a maximum"
            " length that most requests never reach."
        )
    else:
        print(
            "Contiguous preallocation cannot fit even one sequence at this context length --"
            " the reservation model is the bug, not the GPU."
        )
    frag_bound_mib = plan.kv_bytes_per_token * (plan.block_tokens - 1) / (1024.0 ** 2)
    print(
        f"Paging's own waste is bounded: at most {plan.block_tokens - 1} unused tokens in the last"
        f" block, i.e. {frag_bound_mib:.2f} MiB per sequence"
        f" ({100 * (plan.block_tokens - 1) / plan.mean_context:.2f}% of a mean sequence)."
        " That bound is the whole argument for paging -- it converts an unbounded reservation"
        " waste into a waste you can name.\n"
    )


def report_gqa(plan: Plan) -> None:
    shape = plan.shape
    variants: List[int] = sorted(
        {shape.q_heads, max(1, shape.q_heads // 2), max(1, shape.q_heads // 4), shape.kv_heads, 1}
    )
    rows = []
    for kv_heads in variants:
        alt = Plan(**{**plan.__dict__, "shape": ModelShape(
            shape.name, shape.params_b, shape.layers, shape.q_heads, kv_heads, shape.head_dim)})
        rows.append([
            alt.shape.attention_kind,
            kv_heads,
            f"{alt.kv_bytes_per_token / 1024:.1f}",
            f"{alt.kv_bytes_per_token * plan.context / GIB:.2f}",
            alt.concurrency_paged,
            "<-- this model" if kv_heads == shape.kv_heads else "",
        ])
    print_table(
        "Effect of KV heads (GQA / MQA)",
        ["attention", "kv_heads", "KiB/token", "GiB @ max ctx", "max seqs", ""],
        rows,
    )
    print(
        "GQA shrinks the KV cache linearly in KV heads while leaving the query heads (and so most"
        " of the model quality) alone. This is why every recent open model ships with GQA.\n"
    )


def report_kv_quant(plan: Plan) -> None:
    rows = []
    for dtype in ("fp16", "fp8", "int4"):
        alt = Plan(**{**plan.__dict__, "kv_dtype": dtype})
        rows.append([
            dtype,
            DTYPE_BYTES[dtype],
            f"{alt.kv_bytes_per_token / 1024:.1f}",
            alt.concurrency_paged,
            f"{alt.concurrency_paged / plan.concurrency_paged:.2f}x" if plan.concurrency_paged else "n/a",
        ])
    print_table(
        "Effect of KV-cache quantization",
        ["kv dtype", "bytes/elem", "KiB/token", "max seqs", "vs current"],
        rows,
    )
    print(
        "KV quantization is the cheapest way to buy concurrency once GQA is already in place --"
        " but it is an accuracy decision, not a free one (see B04). Measure task quality, not"
        " just perplexity, before shipping fp8 or int4 KV.\n"
    )


def report_verdict(plan: Plan, compute_batch_cap: Optional[int]) -> None:
    kv_limit = plan.concurrency_paged
    if compute_batch_cap is None:
        print(
            f"Verdict: HBM allows {kv_limit} concurrent sequences. Pass --compute-batch-cap with the"
            " largest batch your TPOT budget tolerates to find out which of the two actually binds."
        )
        return
    binding = "KV cache (HBM)" if kv_limit <= compute_batch_cap else "per-step latency (TPOT SLO)"
    print(
        f"Verdict: HBM allows {kv_limit} sequences; the TPOT budget allows {compute_batch_cap}. "
        f"The binding resource is {binding}, so run at min = {min(kv_limit, compute_batch_cap)}."
    )
    if kv_limit <= compute_batch_cap:
        print(
            "  Memory-bound on capacity: spend effort on paging, GQA, KV quantization, prefix"
            " sharing, or more HBM / higher TP -- a faster kernel will not admit one extra request."
        )
    else:
        print(
            "  Latency-bound on capacity: you have spare HBM. Spend it on prefix caching or a"
            " larger prefill budget; raising the batch cap would only break the TPOT SLO."
        )


def build_plan(args: argparse.Namespace) -> Plan:
    shape = PRESETS[args.preset]
    if args.layers:
        shape = ModelShape(shape.name, shape.params_b, args.layers, shape.q_heads,
                           shape.kv_heads, shape.head_dim)
    if args.kv_heads:
        shape = ModelShape(shape.name, shape.params_b, shape.layers, shape.q_heads,
                           args.kv_heads, shape.head_dim)
    if args.head_dim:
        shape = ModelShape(shape.name, shape.params_b, shape.layers, shape.q_heads,
                           shape.kv_heads, args.head_dim)
    if args.params_b:
        shape = ModelShape(shape.name, args.params_b, shape.layers, shape.q_heads,
                           shape.kv_heads, shape.head_dim)

    hbm_gb = args.hbm_gb if args.hbm_gb else GPU_HBM_GB[args.gpu]
    mean_context = args.mean_context if args.mean_context else max(1, args.context // 4)

    return Plan(
        shape=shape,
        tp=args.tp,
        hbm_bytes=hbm_gb * 1e9,
        weight_dtype=args.weight_dtype,
        kv_dtype=args.kv_dtype,
        context=args.context,
        mean_context=min(mean_context, args.context),
        block_tokens=args.block_tokens,
        activation_bytes=args.activation_gib * GIB,
        headroom_pct=args.headroom_pct,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="gqa-70b")
    p.add_argument("--gpu", choices=sorted(GPU_HBM_GB), default="mi300x")
    p.add_argument("--hbm-gb", type=float, default=0.0,
                   help="override HBM capacity, decimal GB (10^9 bytes)")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel degree (shards weights and KV heads)")
    p.add_argument("--weight-dtype", choices=sorted(DTYPE_BYTES), default="fp16")
    p.add_argument("--kv-dtype", choices=sorted(DTYPE_BYTES), default="fp16")
    p.add_argument("--context", type=int, default=8192, help="maximum supported context length")
    p.add_argument("--mean-context", type=int, default=0,
                   help="mean actual sequence length (default: context/4)")
    p.add_argument("--block-tokens", type=int, default=16, help="KV block size, in tokens")
    p.add_argument("--activation-gib", type=float, default=4.0,
                   help="activations, workspace, CUDA/HIP graphs, allocator slack")
    p.add_argument("--headroom-pct", type=float, default=5.0,
                   help="reserve against allocator fragmentation and traffic spikes")
    p.add_argument("--compute-batch-cap", type=int, default=None,
                   help="largest batch your TPOT budget tolerates, to decide the binding resource")
    p.add_argument("--layers", type=int, default=0)
    p.add_argument("--kv-heads", type=int, default=0)
    p.add_argument("--head-dim", type=int, default=0)
    p.add_argument("--params-b", type=float, default=0.0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)

    print("=== KV cache capacity plan (arithmetic, not measured) ===")
    print(
        f"model {plan.shape.name}: {plan.shape.params_b:g}B params, {plan.shape.layers} layers, "
        f"{plan.shape.q_heads} q heads, {plan.shape.kv_heads} kv heads, head_dim {plan.shape.head_dim}"
    )
    print(
        f"gpu {args.gpu} ({plan.hbm_bytes / 1e9:.0f} GB = {plan.hbm_bytes / GIB:.1f} GiB) "
        f"x tp={plan.tp} | weights {plan.weight_dtype} | kv {plan.kv_dtype} | "
        f"block {plan.block_tokens} tok\n"
    )

    report_budget(plan)
    if plan.weight_bytes >= plan.hbm_bytes - plan.activation_bytes:
        print(
            "The weights alone do not leave room for a KV cache on this GPU. Raise --tp, quantize"
            " the weights, or pick a bigger GPU -- there is nothing to plan until then."
        )
        return 1

    report_concurrency(plan)
    report_gqa(plan)
    report_kv_quant(plan)
    report_verdict(plan, args.compute_batch_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
