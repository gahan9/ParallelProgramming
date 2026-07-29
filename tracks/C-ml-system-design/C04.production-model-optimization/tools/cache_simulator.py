#!/usr/bin/env python3
"""C04 - Exact-match cache simulator on a synthetic Zipf request trace.

A SEEDED SIMULATION, not production telemetry. It exists to show one thing:
real request traffic is skewed, and skew is what makes caching work.

Trace
-----
`--distinct-keys` distinct requests, drawn with probability proportional to
1 / rank**zipf. zipf=0 is uniform traffic (caching barely helps); zipf around
1.0 is the heavy skew typically reported for web and query workloads. The
generator is seeded, so runs are reproducible.

Cache
-----
Plain LRU over exact request keys, `--cache-size` entries, using an OrderedDict.
A hit costs `--cache-cost-per-1k`; a miss costs `--model-cost-per-1k`. Semantic
caching (embedding lookup, near-match) has a HIGHER hit rate and a real false-hit
risk - it is not modelled here, because its failure mode is a quality problem,
not an arithmetic one.

Stdlib only. Run `python cache_simulator.py --help`.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import OrderedDict


def die(message):
    print(f"error: {message}", file=sys.stderr)
    print("hint: run with --help to see accepted values.", file=sys.stderr)
    raise SystemExit(2)


def zipf_weights(n, exponent):
    return [1.0 / ((rank + 1) ** exponent) for rank in range(n)]


def cumulative(weights):
    total = 0.0
    out = []
    for w in weights:
        total += w
        out.append(total)
    return out, total


def generate_trace(args):
    """Keys drawn from a Zipf popularity distribution, seeded for reproducibility."""
    rng = random.Random(args.seed)
    cdf, total = cumulative(zipf_weights(args.distinct_keys, args.zipf))
    trace = []
    for _ in range(args.requests):
        target = rng.random() * total
        # Binary search the cumulative table; bisect would work too but this
        # keeps the popularity model visible.
        low, high = 0, len(cdf) - 1
        while low < high:
            mid = (low + high) // 2
            if cdf[mid] < target:
                low = mid + 1
            else:
                high = mid
        trace.append(low)
    return trace


def simulate_lru(trace, capacity):
    if capacity <= 0:
        return 0
    cache = OrderedDict()
    hits = 0
    for key in trace:
        if key in cache:
            cache.move_to_end(key)
            hits += 1
            continue
        cache[key] = True
        if len(cache) > capacity:
            cache.popitem(last=False)
    return hits


def head_share(args, top_n):
    weights = zipf_weights(args.distinct_keys, args.zipf)
    total = sum(weights)
    return sum(weights[:top_n]) / total


def cost_per_1k(hit_rate, args):
    return (hit_rate * args.cache_cost_per_1k
            + (1.0 - hit_rate) * args.model_cost_per_1k)


def build_parser():
    p = argparse.ArgumentParser(
        prog="cache_simulator.py",
        description="LRU exact-match cache on a seeded Zipf trace (stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Synthetic traffic - use it to reason about shape, then re-run the "
               "sweep against a real request log.",
    )
    p.add_argument("--requests", type=int, default=200_000)
    p.add_argument("--distinct-keys", type=int, default=20_000)
    p.add_argument("--zipf", type=float, default=1.0,
                   help="popularity skew: 0 is uniform, ~1.0 is typical web traffic")
    p.add_argument("--cache-size", type=int, default=2_000,
                   help="entries held by the cache (the headline configuration)")
    p.add_argument("--model-cost-per-1k", type=float, default=2.00,
                   help="$ per 1k requests served by the model")
    p.add_argument("--cache-cost-per-1k", type=float, default=0.01,
                   help="$ per 1k requests served from cache")
    p.add_argument("--monthly-requests", type=float, default=300e6,
                   help="production volume used to scale the monthly saving")
    p.add_argument("--seed", type=int, default=20260729)
    return p


def validate(args):
    if args.requests < 1000:
        die(f"--requests must be at least 1000 for a meaningful hit rate "
            f"(got {args.requests}).")
    if args.distinct_keys < 2:
        die(f"--distinct-keys must be at least 2 (got {args.distinct_keys}).")
    if args.zipf < 0:
        die(f"--zipf must be zero or greater (got {args.zipf}).")
    if args.cache_size < 0:
        die(f"--cache-size cannot be negative (got {args.cache_size}).")
    if args.cache_size > args.distinct_keys:
        die(f"--cache-size ({args.cache_size}) exceeds --distinct-keys "
            f"({args.distinct_keys}); the cache would hold the entire key space.")
    for name in ("model-cost-per-1k", "cache-cost-per-1k", "monthly-requests"):
        value = getattr(args, name.replace("-", "_"))
        if value < 0:
            die(f"--{name} cannot be negative (got {value}).")
    if args.cache_cost_per_1k > args.model_cost_per_1k:
        die("--cache-cost-per-1k is higher than --model-cost-per-1k; the cache would "
            "cost more than the model it replaces.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)

    print("Exact-match cache simulator - seeded synthetic trace")
    print("-" * 78)
    print(f"  requests       : {args.requests:,}")
    print(f"  distinct keys  : {args.distinct_keys:,}")
    print(f"  zipf exponent  : {args.zipf:g}")
    print(f"  seed           : {args.seed}")
    print()

    trace = generate_trace(args)
    unique_seen = len(set(trace))
    print(f"  distinct keys actually seen : {unique_seen:,} "
          f"({unique_seen / args.distinct_keys * 100:.1f}% of the key space)")
    for top in (10, 100, 1000):
        if top <= args.distinct_keys:
            print(f"  top {top:>4} keys hold          : "
                  f"{head_share(args, top) * 100:.1f}% of traffic")
    print()

    print("1. Hit rate and cost vs cache size")
    print("  cache size   % of keys   hit rate   $/1k req   saving vs no cache")
    print("  ---------- ----------- ---------- ---------- --------------------")
    sizes = sorted({max(1, args.distinct_keys // d)
                    for d in (1000, 200, 50, 20, 10, 4, 2)} | {args.cache_size})
    baseline = args.model_cost_per_1k
    for size in sizes:
        if size > args.distinct_keys:
            continue
        hits = simulate_lru(trace, size)
        hit_rate = hits / len(trace)
        cost = cost_per_1k(hit_rate, args)
        marker = "  <- configured" if size == args.cache_size else ""
        print(f"  {size:10,} {size / args.distinct_keys * 100:10.1f}% "
              f"{hit_rate * 100:9.1f}% {cost:10.4f} "
              f"{(1 - cost / baseline) * 100:18.1f}%{marker}")
    print()

    hits = simulate_lru(trace, args.cache_size)
    hit_rate = hits / len(trace)
    cost = cost_per_1k(hit_rate, args)
    monthly_base = args.monthly_requests / 1000.0 * baseline
    monthly_cached = args.monthly_requests / 1000.0 * cost
    print("2. At production volume")
    print(f"  hit rate at {args.cache_size:,} entries : {hit_rate * 100:.1f}%")
    print(f"  monthly cost without cache      : ${monthly_base:,.0f}")
    print(f"  monthly cost with cache         : ${monthly_cached:,.0f}")
    print(f"  monthly saving                  : ${monthly_base - monthly_cached:,.0f}")
    print()

    print("3. Why this works, and when it stops")
    print("  A cache holding a small slice of the key space captures a large slice of")
    print("  TRAFFIC, because traffic is skewed - that is the whole mechanism. Re-run")
    print("  with --zipf 0 for uniform traffic and watch the hit rate collapse to")
    print("  roughly cache_size / distinct_keys. Before you build a cache, measure the")
    print("  skew of your real request log; the exponent is the entire business case.")
    print()
    print("  What this model does not capture: staleness (how long a cached answer")
    print("  stays correct), personalisation and privacy (per-user keys destroy the")
    print("  skew, and a shared cache can leak one user's answer to another), and")
    print("  cold-start after every deploy. Those are the reasons real hit rates come")
    print("  in below the simulation, and they belong in the design review.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
