#!/usr/bin/env python3
"""C04 - Serving cost model and optimization break-even.

An ANALYTIC MODEL, not a benchmark. It converts a serving configuration into
the only unit a business decision can be made in: dollars per 1,000 requests,
and dollars per month.

Model
-----
    tokens/s needed = qps * tokens_per_request * (1 - cache_hit_rate)
    GPUs needed     = ceil(tokens/s needed / (per_gpu_tokens_per_s * headroom))
    $/month         = GPUs * $/GPU-hour * 730
    $/1k requests   = $/month / (requests per month) * 1000

Then each optimization scenario applies a THROUGHPUT MULTIPLIER and/or a CACHE
HIT RATE that YOU supply from your own measurements, plus its engineering cost,
and the tool reports the monthly saving and the break-even point.

  The default multipliers are placeholders chosen to make the arithmetic
  legible. They are NOT published speedups and must be replaced with numbers
  you measured on your own model and hardware before anyone acts on them.

Prefill/decode split, batching mechanics and KV-cache paging belong to B06;
quantization numerics belong to B04. This tool only does the money.

Stdlib only. Run `python cost_model.py --help`.
"""

from __future__ import annotations

import argparse
import math
import sys

HOURS_PER_MONTH = 730.0


def die(message):
    print(f"error: {message}", file=sys.stderr)
    print("hint: run with --help to see accepted values.", file=sys.stderr)
    raise SystemExit(2)


def require_positive(name, value, allow_zero=False):
    if value != value or value in (float("inf"), float("-inf")):
        die(f"--{name} must be a finite number (got {value!r}).")
    if value < 0 or (value == 0 and not allow_zero):
        limit = "zero or greater" if allow_zero else "greater than zero"
        die(f"--{name} must be {limit} (got {value}).")
    return value


def require_fraction(name, value):
    if not 0.0 <= value < 1.0:
        die(f"--{name} must be in [0, 1) (got {value}).")
    return value


class Scenario:
    """One serving configuration and what it costs per month."""

    def __init__(self, name, args, throughput_multiplier=1.0, cache_hit_rate=0.0,
                 engineering_weeks=0.0, quality_risk="none"):
        self.name = name
        self.throughput_multiplier = throughput_multiplier
        self.cache_hit_rate = cache_hit_rate
        self.engineering_weeks = engineering_weeks
        self.quality_risk = quality_risk

        per_gpu_peak = args.per_gpu_tokens_per_second * throughput_multiplier
        per_gpu_usable = per_gpu_peak * args.headroom
        demand = args.qps * args.tokens_per_request * (1.0 - cache_hit_rate)
        self.gpus = max(1, math.ceil(demand / per_gpu_usable))
        self.monthly = self.gpus * args.gpu_hour_cost * HOURS_PER_MONTH
        requests_per_month = args.qps * 3600.0 * HOURS_PER_MONTH
        self.per_1k = self.monthly / requests_per_month * 1000.0
        # Against PEAK, so this reports the combined cost of spike headroom and
        # rounding up to whole GPUs - both of which you pay for and neither of
        # which is model inefficiency.
        self.utilization = demand / (self.gpus * per_gpu_peak)
        self.packing = demand / (self.gpus * per_gpu_usable)
        self.engineering_cost = engineering_weeks * args.engineer_weekly_cost


def build_scenarios(args):
    base = Scenario("baseline", args)
    return base, [
        Scenario("1. exact-match cache", args,
                 cache_hit_rate=args.cache_hit_rate,
                 engineering_weeks=args.cache_weeks,
                 quality_risk="none (identical outputs)"),
        Scenario("2. better batching / runtime", args,
                 throughput_multiplier=args.batch_multiplier,
                 engineering_weeks=args.batch_weeks,
                 quality_risk="none (same weights)"),
        Scenario("3. quantization", args,
                 throughput_multiplier=args.quantize_multiplier,
                 engineering_weeks=args.quantize_weeks,
                 quality_risk="small - needs a task-level eval"),
        Scenario("4. distilled student", args,
                 throughput_multiplier=args.distill_multiplier,
                 engineering_weeks=args.distill_weeks,
                 quality_risk="real - needs retraining and a full eval"),
        Scenario("cache + batching + quantization", args,
                 throughput_multiplier=args.batch_multiplier * args.quantize_multiplier,
                 cache_hit_rate=args.cache_hit_rate,
                 engineering_weeks=args.cache_weeks + args.batch_weeks
                                   + args.quantize_weeks,
                 quality_risk="small - the quantization gate still applies"),
    ]


def print_scenarios(base, scenarios):
    print("  scenario                          GPUs   $/1k req    $/month   "
          "saving/mo")
    print("  ------------------------------- ------ ---------- ---------- "
          "-----------")
    print(f"  {base.name:<31} {base.gpus:6d} {base.per_1k:10.4f} "
          f"{base.monthly:10,.0f} {'-':>11}")
    for s in scenarios:
        saving = base.monthly - s.monthly
        print(f"  {s.name:<31} {s.gpus:6d} {s.per_1k:10.4f} "
              f"{s.monthly:10,.0f} {saving:11,.0f}")
    print()


def print_break_even(base, scenarios, args):
    print("  scenario                         eng cost  saving/mo  break-even  "
          "quality risk")
    print("  ------------------------------- --------- ---------- ----------- "
          "-------------------------------")
    for s in scenarios:
        saving = base.monthly - s.monthly
        if saving <= 0:
            verdict = "never"
        else:
            months = s.engineering_cost / saving
            verdict = f"{months:.1f} mo" if months > 0 else "immediate"
        print(f"  {s.name:<31} {s.engineering_cost:9,.0f} {saving:10,.0f} "
              f"{verdict:>11}  {s.quality_risk}")
    print()
    print(f"  Break-even uses ${args.engineer_weekly_cost:,.0f} per engineer-week "
          "fully loaded.")
    print("  Anything past your planning horizon is a 'no' today, however elegant.")
    print("  Rounding up to whole GPUs is why savings arrive in steps, not smoothly:")
    print("  a 15% throughput win that does not remove a GPU saves nothing.")
    print()


def print_stopping_rule(base, args):
    """The cheapest optimization is the one you correctly decline to do."""
    floor = args.gpu_hour_cost * HOURS_PER_MONTH
    print("  Sanity checks before you optimize anything")
    print(f"  - One GPU costs ${floor:,.0f}/month. No optimization can save more")
    print(f"    than ${base.monthly:,.0f}/month, and the floor is one GPU.")
    print(f"  - Baseline hardware utilization is {base.utilization * 100:.0f}% of peak "
          f"({args.headroom:.0%} spike")
    print(f"    headroom x {base.packing * 100:.0f}% packing after rounding to whole GPUs).")
    print("    That gap is not model inefficiency and no model change will recover it -")
    print("    right-size the fleet or revisit the headroom policy first.")
    weeks_of_spend = base.monthly / args.engineer_weekly_cost
    print(f"  - The whole serving bill equals {weeks_of_spend:.1f} engineer-weeks per")
    print("    month. If a project costs more engineer-weeks than that, it cannot pay")
    print("    back inside a year even if it makes serving free.")
    print()


def build_parser():
    p = argparse.ArgumentParser(
        prog="cost_model.py",
        description="Serving cost per 1k requests and optimization break-even "
                    "(analytic, stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Multipliers are YOUR measurements. The defaults are illustrative "
               "placeholders, not published speedups.",
    )
    traffic = p.add_argument_group("traffic and hardware")
    traffic.add_argument("--qps", type=float, default=120.0)
    traffic.add_argument("--tokens-per-request", type=float, default=600.0,
                         help="compute-weighted tokens per request "
                              "(prefill + decode; see B06 for the split)")
    traffic.add_argument("--per-gpu-tokens-per-second", type=float, default=2200.0,
                         help="measured serving throughput of one GPU at your "
                              "latency target")
    traffic.add_argument("--gpu-hour-cost", type=float, default=3.20)
    traffic.add_argument("--headroom", type=float, default=0.70,
                         help="fraction of peak throughput you can actually run at "
                              "once you leave room for traffic spikes")
    traffic.add_argument("--engineer-weekly-cost", type=float, default=6000.0,
                         help="fully loaded cost of one engineer-week")

    opts = p.add_argument_group("optimization scenarios (supply measured values)")
    opts.add_argument("--cache-hit-rate", type=float, default=0.30)
    opts.add_argument("--cache-weeks", type=float, default=2.0)
    opts.add_argument("--batch-multiplier", type=float, default=1.6)
    opts.add_argument("--batch-weeks", type=float, default=3.0)
    opts.add_argument("--quantize-multiplier", type=float, default=1.8)
    opts.add_argument("--quantize-weeks", type=float, default=6.0)
    opts.add_argument("--distill-multiplier", type=float, default=3.0)
    opts.add_argument("--distill-weeks", type=float, default=20.0)
    return p


def validate(args):
    require_positive("qps", args.qps)
    require_positive("tokens-per-request", args.tokens_per_request)
    require_positive("per-gpu-tokens-per-second", args.per_gpu_tokens_per_second)
    require_positive("gpu-hour-cost", args.gpu_hour_cost)
    require_positive("engineer-weekly-cost", args.engineer_weekly_cost, allow_zero=True)
    require_fraction("cache-hit-rate", args.cache_hit_rate)
    if not 0.0 < args.headroom <= 1.0:
        die(f"--headroom must be in (0, 1] (got {args.headroom}).")
    for name in ("batch", "quantize", "distill"):
        value = getattr(args, f"{name}_multiplier")
        if value < 1.0:
            die(f"--{name}-multiplier must be at least 1.0 - a value below 1 means "
                f"the optimization made things slower (got {value}).")
        require_positive(f"{name}-weeks", getattr(args, f"{name}_weeks"),
                         allow_zero=True)
    require_positive("cache-weeks", args.cache_weeks, allow_zero=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)

    base, scenarios = build_scenarios(args)
    requests_per_month = args.qps * 3600.0 * HOURS_PER_MONTH

    print("Serving cost model - analytic")
    print("-" * 78)
    print(f"  traffic            : {args.qps:,.0f} qps "
          f"({requests_per_month / 1e6:,.1f}M requests/month)")
    print(f"  work per request   : {args.tokens_per_request:,.0f} tokens")
    print(f"  per-GPU throughput : {args.per_gpu_tokens_per_second:,.0f} tokens/s "
          f"at {args.headroom:.0%} usable headroom")
    print(f"  GPU price          : ${args.gpu_hour_cost:.2f}/h "
          f"(${args.gpu_hour_cost * HOURS_PER_MONTH:,.0f}/GPU/month)")
    print()
    print("  Multipliers below are placeholders. Replace them with numbers measured")
    print("  on your model, your hardware, at your latency target.")
    print()

    print("1. Cost per scenario")
    print_scenarios(base, scenarios)

    print("2. Does it pay for itself?")
    print_break_even(base, scenarios, args)

    print("3. When the answer is 'do not optimize'")
    print_stopping_rule(base, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
