#!/usr/bin/env python3
"""Leaf oversubscription: the ratio, the concurrency cliff, and the implied bisection.

Everything here is an ANALYTIC MODEL of one leaf switch's capacity. Nothing is
measured. The arithmetic is exact and it is also trivial, which is the point: the
reason oversubscription surprises people is not that the model is hard, it is that
the test everybody runs sits in the region where the model predicts no degradation.

From module README section 2.3, for a leaf with H hosts at R_host and U uplinks at
R_up, and k hosts sending off-leaf at once:

    host-facing bandwidth   H * R_host
    fabric-facing bandwidth U * R_up
    oversubscription ratio  (H * R_host) / (U * R_up)
    per-host off-leaf rate  min(R_host * (1 - L), U * R_up / k)

with L the fraction of each host's bytes that stay inside the leaf. The uplink pool
first saturates at

    k_sat = floor(U * R_up / (R_host * (1 - L)))

which depends on the UPLINKS and not on the host count. A model whose saturation
point moves when you add hosts has described fairness, not capacity.

Stdlib only, and no site packages required:
    python -S -I tools/oversubscription.py --hosts-per-leaf 32 --uplinks 16
"""

from __future__ import annotations

import argparse
import json
import sys

BANNER = ("analytic fabric model, not measured: exact capacity arithmetic on the "
          "as-built uplink count")


class LeafError(ValueError):
    """Bad input, carrying a one-line message the CLI can print without a traceback."""


def validate(hosts_per_leaf: int, uplinks: int, host_gbs: float, uplink_gbs: float,
             local_fraction: float) -> None:
    if hosts_per_leaf < 1:
        raise LeafError(f"--hosts-per-leaf must be at least 1, got {hosts_per_leaf}")
    if uplinks < 1:
        raise LeafError(f"--uplinks must be at least 1, got {uplinks}")
    if host_gbs <= 0.0:
        raise LeafError(f"--host-gbs must be positive, got {host_gbs}")
    if uplink_gbs <= 0.0:
        raise LeafError(f"--uplink-gbs must be positive, got {uplink_gbs}")
    if local_fraction < 0.0:
        raise LeafError(f"--local-fraction must be in [0, 1), got {local_fraction}")
    if local_fraction >= 1.0:
        raise LeafError(
            f"--local-fraction must be below 1, got {local_fraction}: at 1.0 nothing "
            f"leaves the leaf and there is no off-leaf rate to model")
    ratio = (hosts_per_leaf * host_gbs) / (uplinks * uplink_gbs)
    if ratio < 1.0:
        raise LeafError(
            f"oversubscription ratio is {ratio:.3f}:1, below 1:1 -- this leaf has more "
            f"fabric-facing than host-facing bandwidth, so the uplinks can never "
            f"saturate; reduce --uplinks or --uplink-gbs")


def oversubscription_ratio(hosts_per_leaf: int, uplinks: int, host_gbs: float,
                           uplink_gbs: float) -> float:
    """Host-facing bandwidth divided by fabric-facing bandwidth, as built."""
    return (hosts_per_leaf * host_gbs) / (uplinks * uplink_gbs)


def achievable_rate(hosts_per_leaf: int, uplinks: int, host_gbs: float,
                    uplink_gbs: float, concurrent_senders: int,
                    local_fraction: float = 0.0) -> float:
    """Per-sender off-leaf rate in GB/s with `concurrent_senders` hosts sending.

    Below saturation every sender gets the rate it offered, because the idle uplinks
    absorb it. Above saturation the uplink pool is divided evenly. `hosts_per_leaf`
    appears here only to reject a sender count the leaf cannot host.
    """
    if concurrent_senders < 1:
        raise LeafError(
            f"concurrent senders must be at least 1, got {concurrent_senders}")
    if concurrent_senders > hosts_per_leaf:
        raise LeafError(
            f"{concurrent_senders} concurrent senders on a leaf with "
            f"{hosts_per_leaf} hosts")
    offered = host_gbs * (1.0 - local_fraction)
    pool = uplinks * uplink_gbs
    return min(offered, pool / concurrent_senders)


def saturation_point(uplinks: int, host_gbs: float, uplink_gbs: float,
                     local_fraction: float = 0.0) -> int:
    """Largest sender count the uplink pool still carries at the offered rate.

    Deliberately independent of the host count: capacity is a property of the uplinks.
    With equal host and uplink rates and no locality this is exactly the uplink count,
    and the first degraded sender count is one more than it.
    """
    offered = host_gbs * (1.0 - local_fraction)
    return int((uplinks * uplink_gbs) // offered)


def implied_bisection_fraction(hosts_per_leaf: int, uplinks: int, host_gbs: float,
                               uplink_gbs: float) -> float:
    """Fraction of full bisection left for cross-leaf traffic: 1 / ratio, capped at 1.

    Oversubscription and bisection are the same fact at two scales (README section
    2.3): a 2:1 leaf halves the bisection of the whole fabric for anything that leaves
    the leaf. Computing this from the switch radix instead of the as-built uplink count
    is the standard way a design document is wrong by exactly 2x.
    """
    ratio = oversubscription_ratio(hosts_per_leaf, uplinks, host_gbs, uplink_gbs)
    return min(1.0, 1.0 / ratio)


def classify(hosts_per_leaf: int, uplinks: int, host_gbs: float, uplink_gbs: float,
             concurrency: int, local_fraction: float = 0.0) -> str:
    """Verdict for one traffic pattern on one leaf.

    Three outcomes, because there are three genuinely different situations: a leaf that
    cannot be oversubscribed, a leaf that is oversubscribed but not by enough to hurt
    THIS pattern, and a leaf that this pattern will drive into its uplink pool.
    """
    ratio = oversubscription_ratio(hosts_per_leaf, uplinks, host_gbs, uplink_gbs)
    if ratio <= 1.0:
        return "non-blocking"
    if concurrency <= saturation_point(uplinks, host_gbs, uplink_gbs, local_fraction):
        return "safe for this pattern"
    return "oversubscribed and exposed"


def sender_ladder(hosts_per_leaf: int, saturation: int, sweep: bool) -> list[int]:
    """Sender counts to tabulate: everything with --sweep-senders, else the interesting
    ones -- powers of two, the saturation point, the first count past it, and a full
    leaf. The cliff is one integer wide, so a powers-of-two-only table walks past it."""
    if sweep:
        return list(range(1, hosts_per_leaf + 1))
    counts = set()
    k = 1
    while k <= hosts_per_leaf:
        counts.add(k)
        k *= 2
    for candidate in (saturation, saturation + 1, hosts_per_leaf):
        if 1 <= candidate <= hosts_per_leaf:
            counts.add(candidate)
    return sorted(counts)


def rows_for(args, saturation: int) -> list[dict]:
    offered = args.host_gbs * (1.0 - args.local_fraction)
    rows = []
    for k in sender_ladder(args.hosts_per_leaf, saturation, args.sweep_senders):
        rate = achievable_rate(args.hosts_per_leaf, args.uplinks, args.host_gbs,
                               args.uplink_gbs, k, args.local_fraction)
        rows.append(dict(
            senders=k,
            uplink_demand_gbs=k * offered,
            per_host_gbs=rate,
            fraction_of_offered=(rate / offered) if offered else 0.0,
            saturated=k > saturation,
        ))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leaf oversubscription ratio, the concurrency at which the uplink "
                    "pool saturates, and the bisection fraction that implies.")
    parser.add_argument("--hosts-per-leaf", type=int, default=32,
                        help="host-facing ports in use on the leaf (default 32)")
    parser.add_argument("--uplinks", type=int, default=16,
                        help="as-built uplinks to the spine layer (default 16)")
    parser.add_argument("--host-gbs", type=float, default=50.0,
                        help="one host link's bandwidth in GB/s (default 50, which is "
                             "400 Gb/s)")
    parser.add_argument("--uplink-gbs", type=float, default=50.0,
                        help="one uplink's bandwidth in GB/s (default 50)")
    parser.add_argument("--local-fraction", type=float, default=0.0,
                        help="fraction of each host's bytes that stay inside the leaf "
                             "and never touch an uplink (default 0)")
    parser.add_argument("--sweep-senders", action="store_true",
                        help="tabulate every sender count from 1 to the host count")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        validate(args.hosts_per_leaf, args.uplinks, args.host_gbs, args.uplink_gbs,
                 args.local_fraction)
        saturation = saturation_point(args.uplinks, args.host_gbs, args.uplink_gbs,
                                      args.local_fraction)
        rows = rows_for(args, saturation)
    except LeafError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    host_capacity = args.hosts_per_leaf * args.host_gbs
    uplink_capacity = args.uplinks * args.uplink_gbs
    ratio = oversubscription_ratio(args.hosts_per_leaf, args.uplinks, args.host_gbs,
                                   args.uplink_gbs)
    fraction = implied_bisection_fraction(args.hosts_per_leaf, args.uplinks,
                                          args.host_gbs, args.uplink_gbs)
    verdict = classify(args.hosts_per_leaf, args.uplinks, args.host_gbs,
                       args.uplink_gbs, args.hosts_per_leaf, args.local_fraction)
    tolerable = 1.0 / (1.0 - args.local_fraction)

    if args.json:
        print(json.dumps(dict(banner=BANNER, hosts_per_leaf=args.hosts_per_leaf,
                              uplinks=args.uplinks, host_gbs=args.host_gbs,
                              uplink_gbs=args.uplink_gbs,
                              local_fraction=args.local_fraction,
                              host_capacity_gbs=host_capacity,
                              uplink_capacity_gbs=uplink_capacity,
                              oversubscription_ratio=ratio,
                              saturation_senders=saturation,
                              implied_bisection_fraction=fraction,
                              verdict_at_full_concurrency=verdict,
                              tolerable_ratio_at_this_locality=tolerable,
                              senders=rows), indent=2))
        return 0

    print(BANNER)
    print(f"leaf: {args.hosts_per_leaf} hosts x {args.host_gbs:g} GB/s down, "
          f"{args.uplinks} uplinks x {args.uplink_gbs:g} GB/s up")
    print(f"  host-facing    {host_capacity:g} GB/s")
    print(f"  fabric-facing  {uplink_capacity:g} GB/s")
    print(f"  ratio          {ratio:g}:1"
          f"{' (non-blocking)' if ratio <= 1.0 else ''}")
    if args.local_fraction:
        print(f"  locality       {args.local_fraction * 100:g}% of each host's bytes "
              f"stay on the leaf, so each host offers "
              f"{args.host_gbs * (1.0 - args.local_fraction):g} GB/s to the uplinks")
    print()
    header = (f"{'senders k':>10}{'uplink demand':>15}{'per-host GB/s':>15}"
              f"{'of offered':>12}  state")
    print(header)
    print("-" * (len(header) + 8))
    for row in rows:
        print(f"{row['senders']:>10}{row['uplink_demand_gbs']:>15g}"
              f"{row['per_host_gbs']:>15.2f}"
              f"{row['fraction_of_offered'] * 100:>11.1f}%"
              f"  {'sharing the pool' if row['saturated'] else 'full offered rate'}")
    print("-" * (len(header) + 8))
    if saturation >= args.hosts_per_leaf:
        print(f"uplink pool would saturate at k = {saturation} concurrent off-leaf "
              f"senders, more than the leaf's {args.hosts_per_leaf} hosts, so this "
              f"pattern cannot reach the cliff at all")
    else:
        print(f"uplink pool first saturates at k = {saturation} concurrent off-leaf "
              f"senders; the per-host rate first falls below the offered rate at "
              f"k = {saturation + 1}")
    print(f"implied bisection for cross-leaf traffic: "
          f"{fraction * 100:.1f}% of full -- oversubscription and bisection are the "
          f"same fact at two scales")
    print(f"at full concurrency ({args.hosts_per_leaf} senders): {verdict}")
    print(f"a pattern with {args.local_fraction * 100:g}% locality tolerates a ratio up "
          f"to {tolerable:g}:1 before its off-leaf demand exceeds the uplinks")
    print("nothing above k = the saturation point degrades at all, which is why a "
          "two-rank bandwidth test cannot tell a 1:1 leaf from a 4:1 leaf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
