#!/usr/bin/env python3
"""Flat ring against a two-level hierarchical all-reduce, and where they cross.

Everything here is an ANALYTIC MODEL of two ring-based algorithms. No GPU is touched
and nothing is measured; the tier bandwidths and the per-step latencies are yours to
justify, and the times below are only as good as they are.

For P = N*G ranks, m bytes per rank, from module README section 3.6:

  flat ring over all P ranks -- one chain, so it runs at the pace of its SLOWEST link,
  and any link that leaves a node is the slow tier:
      bytes per rank   2m(P-1)/P          all of it on the slow tier when N > 1
      steps            2(P-1)
  two-level: reduce-scatter inside the node, one all-reduce per rail across nodes,
  all-gather inside the node:
      intra bytes      2m(G-1)/G          on the fast link, 2(G-1) steps
      inter bytes      2(m/G)(N-1)/N      on the slow tier, 2(N-1) steps

The identity worth checking, and the point of Lab 2:

      2m(G-1)/G + 2m(N-1)/(NG) == 2m(NG-1)/(NG) == 2m(P-1)/P

so the two algorithms move EXACTLY the same bytes per rank. Patarasuk and Yuan 2009
proved that count is a lower bound, and the hierarchical algorithm does not beat it --
it moves (G-1)/G of those bytes onto the tier where bandwidth is cheap.

In this idealized model, whenever the intra-node tier is at least as fast as the
network the two-level algorithm never loses: it ties at N = 1 (where its cross-node
phase is empty and it degenerates to the same intra-node ring) and wins for every
N > 1. A real crossover comes from the fixed cost of running three phases instead of
one -- extra launches and synchronization -- which is what --phase-overhead-us models.

Stdlib only, and no site packages required:
    python -S -I tools/collective_hierarchy.py --gpus-per-node 8 --max-nodes 512
"""

from __future__ import annotations

import argparse
import json
import sys

BANNER = ("analytic collective model, not measured: exact ring arithmetic on assumed "
          "tier bandwidths")

US = 1e-6
GIB = 1024.0**3

# Times that differ by less than this are a tie, not a win. Comparing two sums of
# floating-point products for strict inequality is how a "winner" column ends up
# reporting the last bit of a division as an engineering result.
TIE_TOLERANCE_S = 1e-15

SEARCH_CEILING = 1 << 20


class ModelError(ValueError):
    """Bad input, carrying a one-line message the CLI can print without a traceback."""


def validate(gpus_per_node: int, nbytes: float, intra_gbs: float, inter_gbs: float,
             alpha_intra_us: float, alpha_inter_us: float, max_nodes: int,
             phase_overhead_us: float) -> None:
    if gpus_per_node < 1:
        raise ModelError(f"--gpus-per-node must be at least 1, got {gpus_per_node}")
    if nbytes <= 0.0:
        raise ModelError(f"--bytes must be positive, got {nbytes}")
    if intra_gbs <= 0.0:
        raise ModelError(f"--intra-gbs must be positive, got {intra_gbs}")
    if inter_gbs <= 0.0:
        raise ModelError(f"--inter-gbs must be positive, got {inter_gbs}")
    if alpha_intra_us < 0.0:
        raise ModelError(f"--alpha-intra-us must be non-negative, got {alpha_intra_us}")
    if alpha_inter_us < 0.0:
        raise ModelError(f"--alpha-inter-us must be non-negative, got {alpha_inter_us}")
    if max_nodes < 1:
        raise ModelError(f"--max-nodes must be at least 1, got {max_nodes}")
    if phase_overhead_us < 0.0:
        raise ModelError(
            f"--phase-overhead-us must be non-negative, got {phase_overhead_us}")


def flat_ring_cost(nodes: int, gpus_per_node: int, nbytes: float, intra_bw: float,
                   inter_bw: float, alpha_intra: float, alpha_inter: float,
                   phase_overhead: float = 0.0) -> dict:
    """One ring over all P = N*G ranks, paced by its slowest link.

    At N = 1 every link is intra-node, so the ring runs at the fast rate; at N > 1 at
    least one link leaves a node and the whole chain proceeds at the slow rate. That
    all-or-nothing pacing is the point: (G-1)/G of the hops could have been fast and
    none of them get to be.
    """
    if nodes < 1:
        raise ModelError(f"nodes must be at least 1, got {nodes}")
    ranks = nodes * gpus_per_node
    if ranks <= 1:
        return dict(ranks=ranks, total_bytes=0.0, slow_bytes=0.0, slow_steps=0,
                    fast_bytes=0.0, fast_steps=0, seconds=0.0, phases=1)
    steps = 2 * (ranks - 1)
    volume = 2.0 * nbytes * (ranks - 1) / ranks
    crosses_nodes = nodes > 1
    bandwidth = inter_bw if crosses_nodes else intra_bw
    alpha = alpha_inter if crosses_nodes else alpha_intra
    seconds = volume / bandwidth + steps * alpha + phase_overhead
    return dict(
        ranks=ranks,
        total_bytes=volume,
        slow_bytes=volume if crosses_nodes else 0.0,
        slow_steps=steps if crosses_nodes else 0,
        fast_bytes=0.0 if crosses_nodes else volume,
        fast_steps=0 if crosses_nodes else steps,
        seconds=seconds,
        phases=1,
    )


def two_level_cost(nodes: int, gpus_per_node: int, nbytes: float, intra_bw: float,
                   inter_bw: float, alpha_intra: float, alpha_inter: float,
                   phase_overhead: float = 0.0) -> dict:
    """Reduce-scatter in node, all-reduce per rail across nodes, all-gather in node.

    The cross-node phase is G concurrent N-rank all-reduces of m/G bytes each, one per
    rail, so its cost is that of a single one: the rails are disjoint switch trees and
    do not contend (README section 3.5). At N = 1 that phase is empty -- zero bytes and
    zero steps, never a negative -- and the algorithm degenerates to an intra-node ring.
    """
    if nodes < 1:
        raise ModelError(f"nodes must be at least 1, got {nodes}")
    ranks = nodes * gpus_per_node
    intra_steps = 2 * (gpus_per_node - 1)
    intra_bytes = 2.0 * nbytes * (gpus_per_node - 1) / gpus_per_node
    if nodes > 1:
        inter_steps = 2 * (nodes - 1)
        inter_bytes = 2.0 * (nbytes / gpus_per_node) * (nodes - 1) / nodes
        phases = 3 if gpus_per_node > 1 else 1
    else:
        inter_steps, inter_bytes = 0, 0.0
        phases = 2 if gpus_per_node > 1 else 0
    seconds = (intra_bytes / intra_bw + intra_steps * alpha_intra
               + inter_bytes / inter_bw + inter_steps * alpha_inter
               + phases * phase_overhead)
    return dict(
        ranks=ranks,
        total_bytes=intra_bytes + inter_bytes,
        slow_bytes=inter_bytes,
        slow_steps=inter_steps,
        fast_bytes=intra_bytes,
        fast_steps=intra_steps,
        seconds=seconds,
        phases=phases,
    )


def node_ladder(max_nodes: int) -> list[int]:
    """Powers of two up to max_nodes, with max_nodes appended if it is not one."""
    ladder, nodes = [], 1
    while nodes <= max_nodes:
        ladder.append(nodes)
        nodes *= 2
    if not ladder:
        ladder = [max_nodes]
    elif ladder[-1] != max_nodes:
        ladder.append(max_nodes)
    return ladder


def compare_at(nodes: int, args, intra_bw: float, inter_bw: float,
               alpha_intra: float, alpha_inter: float, overhead: float) -> dict:
    flat = flat_ring_cost(nodes, args.gpus_per_node, args.bytes, intra_bw, inter_bw,
                          alpha_intra, alpha_inter, overhead)
    two = two_level_cost(nodes, args.gpus_per_node, args.bytes, intra_bw, inter_bw,
                         alpha_intra, alpha_inter, overhead)
    gap = flat["seconds"] - two["seconds"]
    if abs(gap) <= TIE_TOLERANCE_S:
        winner, speedup = "tie", 1.0
    elif gap > 0.0:
        winner = "two-level"
        speedup = flat["seconds"] / two["seconds"] if two["seconds"] else float("inf")
    else:
        winner = "flat"
        speedup = two["seconds"] / flat["seconds"] if flat["seconds"] else float("inf")
    return dict(nodes=nodes, ranks=flat["ranks"], flat=flat, two_level=two,
                winner=winner, speedup=speedup,
                bytes_agree=abs(flat["total_bytes"] - two["total_bytes"]) <= 1e-6)


def crossover_nodes(args, intra_bw: float, inter_bw: float, alpha_intra: float,
                    alpha_inter: float, overhead: float,
                    ceiling: int = SEARCH_CEILING) -> int | None:
    """Smallest node count at which the two-level algorithm strictly wins.

    `flat - two_level` is increasing in the node count -- the flat ring adds 2G slow
    steps and the two-level version adds 2 per extra node, and the flat ring's byte
    term is G times larger -- so bracket the sign change on the power-of-two ladder
    and then bisect. Returns None when the two-level version never wins inside the
    range, which is the honest answer when the tiers are equally fast and the phase
    overhead is real.
    """
    def wins(nodes: int) -> bool:
        row = compare_at(nodes, args, intra_bw, inter_bw, alpha_intra, alpha_inter,
                         overhead)
        return row["winner"] == "two-level"

    low, high = 1, None
    for nodes in node_ladder(ceiling):
        if wins(nodes):
            high = nodes
            break
        low = nodes
    if high is None:
        return None
    while high - low > 1:
        mid = (low + high) // 2
        if wins(mid):
            high = mid
        else:
            low = mid
    return high


def byte_unit(largest: float) -> tuple[str, float]:
    """Pick one unit for a whole column: a 4 KiB run must not print 0.0000 GiB."""
    for name, size in (("GiB", GIB), ("MiB", 1024.0**2), ("KiB", 1024.0)):
        if largest >= size:
            return name, size
    return "B", 1.0


def format_bytes(nbytes: float) -> str:
    name, size = byte_unit(nbytes)
    return f"{nbytes / size:g} {name}"


def print_sweep(rows: list[dict], args) -> None:
    print(BANNER)
    print(f"G = {args.gpus_per_node} GPUs per node, m = {format_bytes(args.bytes)} "
          f"per rank, intra {args.intra_gbs:g} GB/s / {args.alpha_intra_us:g} us, "
          f"inter {args.inter_gbs:g} GB/s / {args.alpha_inter_us:g} us")
    if args.phase_overhead_us:
        print(f"per-phase overhead {args.phase_overhead_us:g} us "
              f"(flat pays it once, two-level once per phase)")
    print()
    unit, size = byte_unit(max(row["flat"]["total_bytes"] for row in rows))
    header = (f"{'nodes':>6}{'ranks':>7}{'flat slow ' + unit:>15}{'steps':>7}"
              f"{'flat ms':>10}{'2L slow ' + unit:>13}{'steps':>7}{'2L ms':>9}"
              f"{'winner':>11}{'by':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        flat, two = row["flat"], row["two_level"]
        speed = "--" if row["winner"] == "tie" else f"{row['speedup']:.2f}x"
        print(f"{row['nodes']:>6}{row['ranks']:>7}{flat['slow_bytes'] / size:>15.4f}"
              f"{flat['slow_steps']:>7}{flat['seconds'] * 1e3:>10.3f}"
              f"{two['slow_bytes'] / size:>13.4f}{two['slow_steps']:>7}"
              f"{two['seconds'] * 1e3:>9.3f}{row['winner']:>11}{speed:>8}")
    print("-" * len(header))
    last = rows[-1]
    print(f"total bytes per rank at N = {last['nodes']}: "
          f"flat {last['flat']['total_bytes'] / size:.6f} {unit}, two-level "
          f"{last['two_level']['total_bytes'] / size:.6f} {unit} -- "
          f"{'identical, as the identity requires' if last['bytes_agree'] else 'DISAGREE, which is a bug'}")
    print("the slow-tier columns, not the totals, are what predict the time")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flat ring against a two-level hierarchical all-reduce: "
                    "slow-tier bytes, step counts, modelled times and the crossover.")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                        help="GPUs per node, G (default 8)")
    parser.add_argument("--bytes", type=float, default=1073741824.0,
                        help="all-reduce buffer per rank, m (default 1 GiB)")
    parser.add_argument("--intra-gbs", type=float, default=100.0,
                        help="effective intra-node bandwidth in GB/s (default 100)")
    parser.add_argument("--inter-gbs", type=float, default=25.0,
                        help="effective inter-node bandwidth per GPU in GB/s "
                             "(default 25)")
    parser.add_argument("--alpha-intra-us", type=float, default=1.0,
                        help="per-step intra-node latency in microseconds (default 1)")
    parser.add_argument("--alpha-inter-us", type=float, default=3.0,
                        help="per-step inter-node latency in microseconds (default 3)")
    parser.add_argument("--max-nodes", type=int, default=512,
                        help="largest node count to tabulate (default 512)")
    parser.add_argument("--phase-overhead-us", type=float, default=0.0,
                        help="fixed cost per collective phase in microseconds; "
                             "the only term that can make the flat ring win")
    parser.add_argument("--find-crossover", action="store_true",
                        help="print only the smallest node count where two-level wins")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        validate(args.gpus_per_node, args.bytes, args.intra_gbs, args.inter_gbs,
                 args.alpha_intra_us, args.alpha_inter_us, args.max_nodes,
                 args.phase_overhead_us)
    except ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    intra_bw = args.intra_gbs * 1e9
    inter_bw = args.inter_gbs * 1e9
    alpha_intra = args.alpha_intra_us * US
    alpha_inter = args.alpha_inter_us * US
    overhead = args.phase_overhead_us * US

    crossover = crossover_nodes(args, intra_bw, inter_bw, alpha_intra, alpha_inter,
                                overhead)
    ratio = args.intra_gbs / args.inter_gbs

    if args.find_crossover:
        rows = []
    else:
        rows = [compare_at(nodes, args, intra_bw, inter_bw, alpha_intra, alpha_inter,
                           overhead)
                for nodes in node_ladder(args.max_nodes)]

    if args.json:
        print(json.dumps(dict(banner=BANNER, gpus_per_node=args.gpus_per_node,
                              bytes=args.bytes, intra_gbs=args.intra_gbs,
                              inter_gbs=args.inter_gbs,
                              tier_ratio=ratio,
                              phase_overhead_us=args.phase_overhead_us,
                              crossover_nodes=crossover, sweep=rows), indent=2))
        return 0

    if rows:
        print_sweep(rows, args)
        print()
    else:
        print(BANNER)
        print(f"G = {args.gpus_per_node}, m = {format_bytes(args.bytes)}, "
              f"intra {args.intra_gbs:g} GB/s, inter {args.inter_gbs:g} GB/s, "
              f"tier ratio {ratio:.2f}x")

    if crossover is None:
        print("no crossover: the flat ring wins at every node count in range, which "
              "happens when the tiers are equally fast or the phase overhead dominates")
    elif crossover <= 1:
        print("crossover at N = 1: the two-level algorithm wins even on a single node, "
              "which means the model has been handed a flat ring that cannot use the "
              "fast tier")
    else:
        print(f"crossover: two-level first wins at N = {crossover} nodes "
              f"({crossover * args.gpus_per_node} ranks); at N = {crossover - 1} the "
              f"flat ring is at least as good")
    if ratio > 1.0:
        print(f"the advantage is the tier ratio intra/inter = {ratio:.2f}x. Raise "
              f"--inter-gbs to {args.intra_gbs:g} and it collapses: equal tiers make "
              f"the two algorithms move identical bytes over identical-rate links.")
    else:
        print(f"tier ratio intra/inter = {ratio:.2f}x, so the byte advantage is gone: "
              f"both algorithms move identical bytes over identical-rate links and "
              f"only the step count separates them.")
    if not args.phase_overhead_us:
        print("with --phase-overhead-us 0 the two-level algorithm cannot lose above "
              "N = 1; give the extra phases a cost to see a crossover move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
