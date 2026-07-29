#!/usr/bin/env python3
"""Recommender funnel budget calculator for module C07, case study 2.

Answers three questions with arithmetic instead of assertion:

  1. Why does the funnel exist? -- by costing the strawman that scores every
     item in the catalogue for every request, in both FLOPs and memory traffic.
  2. What does each stage cost? -- items scored, FLOPs, latency, and the bytes
     pulled out of the feature store, per stage and per second across the fleet.
  3. What breaks if you widen a stage? -- `--widen rank=2000` re-runs the whole
     budget and reports every constraint the change violates.

Every number is a derived estimate from the assumptions printed in the header,
not a measurement. The per-item latency coefficients in particular are the
things you must replace with load-test results before trusting a capacity plan.

Standard library only. Run `python funnel_budget.py --help` for all knobs.
"""

from __future__ import annotations

import argparse
import copy
import sys

GFLOP = 1e9
TFLOP = 1e12
TB = 1e12


class InputError(Exception):
    """Raised for user-supplied values that cannot produce a valid budget."""


class Stage:
    """One funnel stage.

    Latency is modelled as `fixed_ms + per_item_us * items` so that widening a
    stage actually moves the budget -- a constant per-stage latency would hide
    the very tradeoff this tool exists to show. `latency_driver` says which
    count drives it: ANN search scales with the candidates it returns, every
    other stage scales with the candidates it is handed.
    """

    def __init__(self, key, label, items_in, items_out, mflop_per_item,
                 fixed_mflop, fixed_ms, per_item_us, feature_bytes_per_item,
                 latency_driver="in"):
        self.key = key
        self.label = label
        self.items_in = items_in
        self.items_out = items_out
        self.mflop_per_item = mflop_per_item
        self.fixed_mflop = fixed_mflop
        self.fixed_ms = fixed_ms
        self.per_item_us = per_item_us
        self.feature_bytes_per_item = feature_bytes_per_item
        self.latency_driver = latency_driver

    @property
    def flops(self) -> float:
        return (self.mflop_per_item * self.items_in + self.fixed_mflop) * 1e6

    @property
    def latency_ms(self) -> float:
        driving = self.items_out if self.latency_driver == "out" else self.items_in
        return self.fixed_ms + self.per_item_us * driving / 1000.0

    @property
    def feature_bytes(self) -> float:
        return self.feature_bytes_per_item * self.items_in


def build_stages(a: argparse.Namespace) -> list[Stage]:
    """The default four-stage funnel, with each stage's input set by the
    previous stage's output."""
    return [
        Stage("retrieve", "retrieval  (two-tower + ANN)",
              items_in=a.catalogue, items_out=a.retrieve,
              # ANN is sublinear: cost is the probes, not the catalogue.
              mflop_per_item=0.0,
              fixed_mflop=a.tower_mflop + a.ann_probes * a.embed_dim * 2 / 1e6,
              fixed_ms=15.0, per_item_us=5.0,
              feature_bytes_per_item=0.0, latency_driver="out"),
        Stage("filter", "filtering  (seen / policy / eligibility)",
              items_in=a.retrieve, items_out=a.rank_in,
              mflop_per_item=0.0, fixed_mflop=0.0,
              fixed_ms=2.0, per_item_us=1.0,
              feature_bytes_per_item=0.0),
        Stage("fetch", "item feature fetch  (online store)",
              items_in=a.rank_in, items_out=a.rank_in,
              mflop_per_item=0.0, fixed_mflop=0.0,
              fixed_ms=3.0, per_item_us=20.0,
              feature_bytes_per_item=a.features_per_item * 4),
        Stage("rank", "ranking  (multi-task DNN)",
              items_in=a.rank_in, items_out=a.rank_in,
              mflop_per_item=a.rank_mflop, fixed_mflop=0.0,
              fixed_ms=4.0, per_item_us=35.0,
              feature_bytes_per_item=0.0),
        Stage("rerank", "re-ranking  (diversity + business rules)",
              items_in=a.rank_in, items_out=a.final,
              mflop_per_item=a.final * a.embed_dim * 2 / 1e6,
              fixed_mflop=0.0,
              fixed_ms=2.0, per_item_us=5.0,
              feature_bytes_per_item=0.0),
    ]


# Hops that are not funnel stages but sit inside the same latency budget.
FIXED_HOPS = [
    ("edge, auth, routing", 10.0),
    ("user feature fetch", 12.0),
    ("user tower forward", 2.0),
    ("response assembly", 5.0),
    ("reserve for jitter, GC, retry", 23.0),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="funnel_budget.py",
        description="Cost a recommender funnel stage by stage (C07 case study 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    scale = p.add_argument_group("scale")
    scale.add_argument("--catalogue", type=int, default=50_000_000,
                       help="eligible items in the catalogue")
    scale.add_argument("--qps", type=float, default=20_000.0,
                       help="peak recommendation requests per second")
    scale.add_argument("--budget-ms", type=float, default=150.0,
                       help="end-to-end p99 latency budget for the request")

    funnel = p.add_argument_group("funnel widths")
    funnel.add_argument("--retrieve", type=int, default=1000,
                        help="candidates returned by retrieval")
    funnel.add_argument("--rank-in", type=int, default=600,
                        help="candidates surviving the filter, i.e. scored by the ranker")
    funnel.add_argument("--final", type=int, default=20,
                        help="items returned to the client")

    model = p.add_argument_group("model and feature costs")
    model.add_argument("--rank-mflop", type=float, default=4.2,
                       help="MFLOP to score one (user, item) pair in the ranker")
    model.add_argument("--tower-mflop", type=float, default=5.0,
                       help="MFLOP for one user-tower forward pass")
    model.add_argument("--embed-dim", type=int, default=128,
                       help="retrieval embedding dimension")
    model.add_argument("--ann-probes", type=int, default=3000,
                       help="distance evaluations per ANN query at the target recall")
    model.add_argument("--features-per-item", type=int, default=200,
                       help="fp32 features fetched per candidate item")

    fleet = p.add_argument_group("fleet and cost")
    fleet.add_argument("--gpu-tflops", type=float, default=520.0,
                       help="achievable TFLOP/s per GPU for these small kernels")
    fleet.add_argument("--gpu-hbm-tbs", type=float, default=5.3,
                       help="peak HBM bandwidth per GPU, TB/s")
    fleet.add_argument("--req-per-replica", type=float, default=1500.0,
                       help="measured requests/s a single ranking replica sustains "
                            "inside its latency budget (this, not FLOPs, sizes the fleet)")
    fleet.add_argument("--cache-hit", type=float, default=0.80,
                       help="local item-feature cache hit rate on the ranking replicas")
    fleet.add_argument("--gpu-hour-usd", type=float, default=3.00,
                       help="fully-loaded cost per GPU-hour")

    p.add_argument("--widen", action="append", metavar="STAGE=N", default=[],
                   help="widen a funnel stage and report what breaks; repeatable. "
                        "Stages: retrieve (candidates returned by retrieval), "
                        "rank (candidates scored by the ranker), "
                        "final (items returned to the client)")

    return p.parse_args(argv)


def validate(a: argparse.Namespace) -> None:
    positive = {
        "--catalogue": a.catalogue, "--qps": a.qps, "--budget-ms": a.budget_ms,
        "--retrieve": a.retrieve, "--rank-in": a.rank_in, "--final": a.final,
        "--rank-mflop": a.rank_mflop, "--tower-mflop": a.tower_mflop,
        "--embed-dim": a.embed_dim, "--ann-probes": a.ann_probes,
        "--features-per-item": a.features_per_item, "--gpu-tflops": a.gpu_tflops,
        "--gpu-hbm-tbs": a.gpu_hbm_tbs, "--req-per-replica": a.req_per_replica,
        "--gpu-hour-usd": a.gpu_hour_usd,
    }
    for flag, value in positive.items():
        if value <= 0:
            raise InputError(f"{flag} must be greater than zero (got {value}).")

    if not 0.0 <= a.cache_hit < 1.0:
        raise InputError(f"--cache-hit must be in [0, 1) (got {a.cache_hit}).")

    if a.retrieve > a.catalogue:
        raise InputError(
            f"--retrieve {a.retrieve:,} exceeds --catalogue {a.catalogue:,}; "
            f"retrieval cannot return more candidates than the catalogue holds."
        )
    if a.rank_in > a.retrieve:
        raise InputError(
            f"--rank-in {a.rank_in:,} exceeds --retrieve {a.retrieve:,}; the filter "
            f"removes candidates, it cannot invent them."
        )
    if a.final > a.rank_in:
        raise InputError(
            f"--final {a.final:,} exceeds --rank-in {a.rank_in:,}; re-ranking selects "
            f"from the scored set."
        )


def apply_widen(a: argparse.Namespace, specs: list[str]) -> argparse.Namespace:
    """Translate `--widen rank=2000` into the funnel-width flags it implies.

    Only the three genuinely independent widths are addressable. The filter,
    feature-fetch and re-rank stages have no width of their own -- they inherit
    the width of the stage feeding them -- so exposing them here would let a
    caller write two names for the same knob.
    """
    field_for_stage = {
        "retrieve": "retrieve",
        "rank": "rank_in",
        "final": "final",
    }
    widened = copy.deepcopy(a)
    for spec in specs:
        if "=" not in spec:
            raise InputError(
                f"--widen expects STAGE=N (for example --widen rank=2000), got '{spec}'."
            )
        stage, _, raw = spec.partition("=")
        stage = stage.strip().lower()
        if stage not in field_for_stage:
            raise InputError(
                f"--widen: unknown stage '{stage}'. "
                f"Choose one of: {', '.join(sorted(field_for_stage))}."
            )
        try:
            count = int(raw)
        except ValueError:
            raise InputError(
                f"--widen {stage}=: '{raw}' is not an integer."
            ) from None
        if count <= 0:
            raise InputError(f"--widen {stage}={count}: count must be positive.")
        setattr(widened, field_for_stage[stage], count)
        # Keep the funnel monotone: a stage cannot emit more than it was handed.
        if stage == "rank" and count > widened.retrieve:
            widened.retrieve = count
        if stage == "final" and count > widened.rank_in:
            widened.rank_in = count
            if count > widened.retrieve:
                widened.retrieve = count
    return widened


def budget(a: argparse.Namespace) -> dict:
    stages = build_stages(a)
    stage_ms = sum(s.latency_ms for s in stages)
    fixed_ms = sum(ms for _, ms in FIXED_HOPS)
    total_ms = stage_ms + fixed_ms

    flops_per_req = sum(s.flops for s in stages)
    feature_bytes_per_req = sum(s.feature_bytes for s in stages)

    naive_flops_per_req = a.catalogue * a.rank_mflop * 1e6
    naive_embed_bytes_per_req = a.catalogue * a.embed_dim * 4

    replicas = a.qps / a.req_per_replica
    egress_bs = feature_bytes_per_req * a.qps * (1.0 - a.cache_hit)

    return {
        "stages": stages,
        "stage_ms": stage_ms,
        "fixed_ms": fixed_ms,
        "total_ms": total_ms,
        "flops_per_req": flops_per_req,
        "feature_bytes_per_req": feature_bytes_per_req,
        "naive_flops_per_req": naive_flops_per_req,
        "naive_embed_bytes_per_req": naive_embed_bytes_per_req,
        "replicas": replicas,
        "egress_bs": egress_bs,
        "fleet_usd_hour": replicas * a.gpu_hour_usd,
    }


def print_report(a: argparse.Namespace, b: dict) -> None:
    line = "-" * 78
    print(line)
    print("RECOMMENDER FUNNEL BUDGET  --  all figures are estimates derived from")
    print("the assumptions below, not measurements.")
    print(line)
    print("ASSUMPTIONS")
    print(f"  catalogue        {a.catalogue:,} eligible items")
    print(f"  traffic          {a.qps:,.0f} requests/s peak, "
          f"{a.budget_ms:g} ms end-to-end p99 budget")
    print(f"  funnel           {a.catalogue:,} -> {a.retrieve:,} -> {a.rank_in:,} "
          f"-> {a.final:,}")
    print(f"  ranker           {a.rank_mflop:g} MFLOP per (user, item) pair, "
          f"{a.features_per_item} fp32 features per item")
    print(f"  retrieval        {a.embed_dim}-d embeddings, "
          f"{a.ann_probes:,} ANN probes per query")
    print(f"  fleet            {a.req_per_replica:,.0f} req/s per ranking replica "
          f"(measured, not derived), ${a.gpu_hour_usd:.2f}/GPU-hour")

    print()
    print("WHY THE FUNNEL EXISTS  (cost of scoring the whole catalogue per request)")
    naive_tflop = b["naive_flops_per_req"] / TFLOP
    node_tflops = a.gpu_tflops * 8
    print(f"  FLOPs per request          {a.catalogue:,} x {a.rank_mflop:g} MFLOP "
          f"= {naive_tflop:,.1f} TFLOP")
    print(f"  fleet FLOP/s at {a.qps:,.0f} req/s  "
          f"{b['naive_flops_per_req'] * a.qps / 1e18:,.2f} EFLOP/s")
    print(f"  -> GPUs needed             "
          f"{b['naive_flops_per_req'] * a.qps / (a.gpu_tflops * TFLOP):,.0f} "
          f"({b['naive_flops_per_req'] * a.qps / (node_tflops * TFLOP):,.0f} 8-GPU nodes)")
    print(f"  latency, one request alone {naive_tflop / node_tflops * 1000:,.1f} ms "
          f"on a whole idle node")
    embed_bs = b["naive_embed_bytes_per_req"] * a.qps
    gpus_compute = b["naive_flops_per_req"] * a.qps / (a.gpu_tflops * TFLOP)
    gpus_bw = embed_bs / (a.gpu_hbm_tbs * TB)
    print(f"  embedding bytes read       "
          f"{b['naive_embed_bytes_per_req'] / 1e9:,.1f} GB per request, "
          f"{embed_bs / 1e12:,.0f} TB/s across the fleet")
    harder = "compute" if gpus_compute >= gpus_bw else "memory bandwidth"
    print(f"  -> GPUs needed for that BW {gpus_bw:,.0f} "
          f"({harder} is the harder wall)")
    print("  Batching across requests amortizes the embedding reads but not the FLOPs:")
    print("  every (user, item) pair still has to be scored. The compute wall is the")
    print("  irreducible one. Verdict: infeasible by orders of magnitude. Hence the funnel.")

    print()
    print("PER-STAGE BUDGET")
    print(f"  {'stage':<40}{'items in':>10}{'GFLOP':>9}{'ms':>8}")
    for s in b["stages"]:
        print(f"  {s.label:<40}{s.items_in:>10,}{s.flops / GFLOP:>9.3f}"
              f"{s.latency_ms:>8.1f}")
    print()
    for label, ms in FIXED_HOPS:
        print(f"  {label:<40}{'-':>10}{'-':>9}{ms:>8.1f}")
    print(f"  {'-' * 67}")
    print(f"  {'TOTAL':<40}{'':>10}{b['flops_per_req'] / GFLOP:>9.3f}"
          f"{b['total_ms']:>8.1f}")

    headroom = a.budget_ms - b["total_ms"]
    verdict = "WITHIN BUDGET" if headroom >= 0 else "OVER BUDGET"
    print(f"  {'budget':<40}{'':>10}{'':>9}{a.budget_ms:>8.1f}")
    print(f"  {'headroom':<40}{'':>10}{'':>9}{headroom:>8.1f}   {verdict}")

    print()
    print("SAVINGS FROM THE FUNNEL")
    ratio = b["naive_flops_per_req"] / b["flops_per_req"]
    print(f"  full scan                  {naive_tflop * 1000:,.0f} GFLOP per request")
    print(f"  funnel                     {b['flops_per_req'] / GFLOP:,.3f} GFLOP per request")
    print(f"  -> reduction               {ratio:,.0f}x")

    print()
    print("FLEET AND FEATURE I/O")
    raw = b["feature_bytes_per_req"] * a.qps
    print(f"  feature bytes per request  {b['feature_bytes_per_req'] / 1e3:,.0f} kB "
          f"({a.rank_in:,} items x {a.features_per_item} features x 4 B)")
    print(f"  feature-store egress       {raw / 1e9:,.1f} GB/s uncached, "
          f"{b['egress_bs'] / 1e9:,.1f} GB/s at a {a.cache_hit:.0%} local cache hit rate")
    print(f"  ranking replicas           {b['replicas']:,.1f} for capacity "
          f"(size for p99 and AZ redundancy, then double it)")
    print(f"  ranking fleet cost         ${b['fleet_usd_hour']:,.2f}/hour")
    cost_per_k = b["fleet_usd_hour"] / 3600.0 / a.qps * 1000
    print(f"  cost per 1,000 requests    ${cost_per_k:,.5f} (ranking GPUs only; "
          f"the feature store and ANN fleet usually cost more)")


def print_widen_report(base_a, base_b, new_a, new_b, specs) -> None:
    line = "=" * 78
    print()
    print(line)
    print(f"WHAT IF YOU WIDEN:  {', '.join(specs)}")
    print(line)

    print(f"  {'quantity':<34}{'before':>14}{'after':>14}{'change':>12}")

    def row(label, before, after, fmt="{:,.1f}", suffix=""):
        change = "n/a" if before == 0 else f"{after / before:,.2f}x"
        print(f"  {label:<34}{fmt.format(before) + suffix:>14}"
              f"{fmt.format(after) + suffix:>14}{change:>12}")

    row("candidates retrieved", base_a.retrieve, new_a.retrieve, "{:,.0f}")
    row("candidates ranked", base_a.rank_in, new_a.rank_in, "{:,.0f}")
    row("GFLOP per request",
        base_b["flops_per_req"] / GFLOP, new_b["flops_per_req"] / GFLOP, "{:,.3f}")
    row("p99 latency", base_b["total_ms"], new_b["total_ms"], "{:,.1f}", " ms")
    row("feature bytes per request",
        base_b["feature_bytes_per_req"] / 1e3,
        new_b["feature_bytes_per_req"] / 1e3, "{:,.0f}", " kB")
    row("feature-store egress",
        base_b["egress_bs"] / 1e9, new_b["egress_bs"] / 1e9, "{:,.1f}", " GB/s")

    print()
    print("WHAT BREAKS")
    broke = False
    over = new_b["total_ms"] - new_a.budget_ms
    if over > 0:
        broke = True
        print(f"  * p99 latency is {over:,.1f} ms over the {new_a.budget_ms:g} ms budget. "
              f"You must buy")
        print("    the time back somewhere: a cheaper first-pass ranker over the wide set,")
        print("    parallel feature fetch, or a smaller final candidate set.")
    egress_growth = new_b["egress_bs"] / base_b["egress_bs"] if base_b["egress_bs"] else 0
    if egress_growth > 1.5:
        broke = True
        print(f"  * feature-store egress grows {egress_growth:,.2f}x to "
              f"{new_b['egress_bs'] / 1e9:,.1f} GB/s. This is")
        print("    usually the real bill, and it scales linearly with candidates ranked,")
        print("    not with model quality.")
    flop_growth = new_b["flops_per_req"] / base_b["flops_per_req"]
    if flop_growth > 1.5:
        broke = True
        print(f"  * ranking FLOPs grow {flop_growth:,.2f}x. Compute is rarely the binding")
        print("    constraint at this scale, but it does set the floor on batch latency.")
    if not broke:
        print("  * Nothing exceeds a stated constraint. Widen it and measure the")
        print("    recall/NDCG gain; if the gain is inside the noise, revert.")

    print()
    print("  Widening is only worth it if recall@k at the new width improves the ONLINE")
    print("  metric. Verify with an A/B test, not with an offline NDCG delta.")


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    # Resolve every input before printing anything, so bad flags fail fast
    # instead of arriving after a page of otherwise-valid output.
    try:
        validate(a)
        widened_args = apply_widen(a, a.widen) if a.widen else None
        if widened_args is not None:
            validate(widened_args)
    except InputError as exc:
        print(f"funnel_budget.py: {exc}", file=sys.stderr)
        return 2

    base = budget(a)
    print_report(a, base)
    if widened_args is not None:
        print_widen_report(a, base, widened_args, budget(widened_args), a.widen)

    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
