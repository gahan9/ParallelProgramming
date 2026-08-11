#!/usr/bin/env python3
"""What a rank-to-device mapping costs a flat ring all-reduce, in fabric load.

Everything here is an ANALYTIC MODEL. The link counts are exact combinatorics on the
mapping -- you can check every one of them by hand on four nodes and two GPUs -- and
the times are those counts multiplied by bandwidths you supplied. Nothing is measured.

A flat ring over P = N*G ranks uses P links, one per consecutive pair including the
wrap. For each link this tool asks two different questions:

  cross-node   do the two ranks sit on different nodes? Then the link is on the
               network instead of the intra-node fast link, and it carries about 2m
               bytes over the run (2m(P-1)/P exactly).
  cross-rail   given that it is cross-node, do the two ranks sit on different rails?
               On a rail-optimized fabric (README section 3.5) GPU i of every node
               hangs off leaf switch i, so a same-index cross-node flow is ONE leaf
               hop, and a mismatched-index flow must climb to the spine -- the tier
               where oversubscription lives.

The three mappings, with rank r -> (node, local index):

  block    (r // G, r % G)     consecutive ranks share a node except at boundaries
  cyclic   (r % N, r // N)     consecutive ranks are always on different nodes
  random   a seeded shuffle of the (node, local) slots

Stdlib only, and no site packages required:
    python -S -I tools/rank_placement.py --nodes 64 --gpus-per-node 8 --compare-all
"""

from __future__ import annotations

import argparse
import json
import random
import sys

BANNER = ("analytic placement model, not measured: exact link counts on assumed tier "
          "bandwidths")

MAPPINGS = ("block", "cyclic", "random")
US = 1e-6
GIB = 1024.0**3


class PlacementError(ValueError):
    """Bad input, carrying a one-line message the CLI can print without a traceback."""


def validate(nodes: int, gpus_per_node: int, rails: int, nbytes: float,
             intra_gbs: float, inter_gbs: float) -> None:
    if nodes < 1:
        raise PlacementError(f"--nodes must be at least 1, got {nodes}")
    if gpus_per_node < 1:
        raise PlacementError(
            f"--gpus-per-node must be at least 1, got {gpus_per_node}")
    if rails < 1:
        raise PlacementError(f"--rails must be at least 1, got {rails}")
    if nbytes <= 0.0:
        raise PlacementError(f"--bytes must be positive, got {nbytes}")
    if intra_gbs <= 0.0:
        raise PlacementError(f"--intra-gbs must be positive, got {intra_gbs}")
    if inter_gbs <= 0.0:
        raise PlacementError(f"--inter-gbs must be positive, got {inter_gbs}")


def placement(mapping: str, nodes: int, gpus_per_node: int,
              seed: int = 0) -> list[tuple[int, int]]:
    """Rank-indexed list of (node, local index) for one mapping.

    The random mapping is a seeded permutation of the same slot set, not independent
    draws: every device gets exactly one rank, which is what a scheduler produces when
    nothing has asked for locality.
    """
    if mapping == "block":
        return [(r // gpus_per_node, r % gpus_per_node)
                for r in range(nodes * gpus_per_node)]
    if mapping == "cyclic":
        return [(r % nodes, r // nodes) for r in range(nodes * gpus_per_node)]
    if mapping == "random":
        slots = [(n, g) for n in range(nodes) for g in range(gpus_per_node)]
        random.Random(seed).shuffle(slots)
        return slots
    raise PlacementError(
        f"unknown mapping {mapping!r}: choose from {', '.join(MAPPINGS)}")


def ring_metrics(slots: list[tuple[int, int]], rails: int, nbytes: float,
                 intra_bw: float, inter_bw: float, alpha_intra: float,
                 alpha_inter: float) -> dict:
    """Link counts and the modelled all-reduce time for a flat ring over these slots.

    A cross-rail link is counted only when it is also cross-node: two GPUs in the same
    chassis talk over the intra-node link and never reach a leaf switch, so asking
    which rail they are on is asking about a switch neither of them used.
    """
    ranks = len(slots)
    if ranks <= 1:
        return dict(ranks=ranks, links=0, cross_node=0, cross_rail=0,
                    bytes_per_link=0.0, cross_node_bytes=0.0, cross_rail_bytes=0.0,
                    seconds=0.0, paced_by="nothing to do")
    cross_node = cross_rail = 0
    for r in range(ranks):
        node_a, local_a = slots[r]
        node_b, local_b = slots[(r + 1) % ranks]
        if node_a != node_b:
            cross_node += 1
            if (local_a % rails) != (local_b % rails):
                cross_rail += 1
    steps = 2 * (ranks - 1)
    bytes_per_link = 2.0 * nbytes * (ranks - 1) / ranks
    bandwidth = inter_bw if cross_node else intra_bw
    alpha = alpha_inter if cross_node else alpha_intra
    return dict(
        ranks=ranks,
        links=ranks,
        cross_node=cross_node,
        cross_rail=cross_rail,
        bytes_per_link=bytes_per_link,
        cross_node_bytes=cross_node * bytes_per_link,
        cross_rail_bytes=cross_rail * bytes_per_link,
        seconds=steps * (nbytes / ranks / bandwidth + alpha),
        paced_by=("the inter-node rate" if cross_node else "the intra-node rate"),
    )


def evaluate(mapping: str, args, intra_bw: float, inter_bw: float,
             alpha_intra: float, alpha_inter: float) -> dict:
    slots = placement(mapping, args.nodes, args.gpus_per_node, args.seed)
    row = ring_metrics(slots, args.rails, args.bytes, intra_bw, inter_bw,
                       alpha_intra, alpha_inter)
    row["mapping"] = mapping
    return row


def expected_counts(mapping: str, nodes: int, gpus_per_node: int) -> str:
    """The derived count for this mapping, so the table can be checked, not trusted."""
    if nodes < 2 or gpus_per_node < 2:
        return "the derivation assumes N > 1 and G > 1"
    if mapping == "block":
        return (f"derived: {nodes} cross-node links (one per node boundary), all "
                f"{nodes} of them cross-rail, since the boundary joins local index "
                f"{gpus_per_node - 1} to local index 0")
    if mapping == "cyclic":
        return (f"derived: {nodes * gpus_per_node} cross-node links (every link), of "
                f"which {gpus_per_node} are cross-rail -- consecutive ranks share a "
                f"local index except once per pass around the nodes")
    return ("derived: about P*(1 - (G-1)/(P-1)) cross-node links and about "
            "P*(G-1)/G cross-rail, so a random mapping approaches cyclic on fabric "
            "load and is far worse than either on cross-rail")


def print_one(row: dict, args, baseline: dict | None) -> None:
    print(f"mapping {row['mapping']}"
          f"{f' (seed {args.seed})' if row['mapping'] == 'random' else ''}: "
          f"{args.nodes} nodes x {args.gpus_per_node} GPUs = {row['ranks']} ranks, "
          f"{args.rails} rails")
    print(f"  ring links          {row['links']}")
    print(f"  cross-node links    {row['cross_node']}"
          f"  ({row['cross_node'] / row['links'] * 100:.1f}% of the ring)")
    print(f"  cross-rail links    {row['cross_rail']}"
          f"  (of the cross-node ones; these climb to the spine)")
    print(f"  bytes per ring link {row['bytes_per_link'] / GIB:.4f} GiB "
          f"(2m(P-1)/P over the run)")
    print(f"  cross-node bytes    {row['cross_node_bytes'] / GIB:.2f} GiB pushed into "
          f"the fabric")
    print(f"  cross-rail bytes    {row['cross_rail_bytes'] / GIB:.2f} GiB of that on "
          f"the spine tier")
    print(f"  modelled ring pace  {row['seconds'] * 1e3:.2f} ms, set by "
          f"{row['paced_by']}")
    if baseline and baseline["cross_node_bytes"] > 0.0:
        print(f"  vs block baseline   "
              f"{row['cross_node_bytes'] / baseline['cross_node_bytes']:.2f}x the "
              f"cross-node bytes, "
              f"{row['seconds'] / baseline['seconds']:.2f}x the modelled pace")
    print(f"  {expected_counts(row['mapping'], args.nodes, args.gpus_per_node)}")


def print_compare(rows: list[dict], args) -> None:
    baseline = rows[0]
    header = (f"{'mapping':<10}{'ranks':>7}{'cross-node':>12}{'cross-rail':>12}"
              f"{'fabric GiB':>13}{'spine GiB':>12}{'pace ms':>10}{'load vs block':>15}")
    print(header)
    print("-" * len(header))
    for row in rows:
        load = (row["cross_node_bytes"] / baseline["cross_node_bytes"]
                if baseline["cross_node_bytes"] else float("inf"))
        print(f"{row['mapping']:<10}{row['ranks']:>7}{row['cross_node']:>12}"
              f"{row['cross_rail']:>12}{row['cross_node_bytes'] / GIB:>13.2f}"
              f"{row['cross_rail_bytes'] / GIB:>12.2f}{row['seconds'] * 1e3:>10.2f}"
              f"{load:>14.2f}x")
    print("-" * len(header))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-node and cross-rail ring links, fabric load and modelled "
                    "pace for block, cyclic and random rank mappings.")
    parser.add_argument("--nodes", type=int, default=64, help="node count N (default 64)")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                        help="GPUs per node, G (default 8)")
    parser.add_argument("--mapping", default="block",
                        help="block, cyclic or random (default block)")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random mapping, so a run is reproducible")
    parser.add_argument("--rails", type=int, default=8,
                        help="rails in the fabric; GPU i lands on leaf i mod rails "
                             "(default 8)")
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
    parser.add_argument("--compare-all", action="store_true",
                        help="tabulate all three mappings side by side")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of a table")
    args = parser.parse_args(argv)

    intra_bw = args.intra_gbs * 1e9
    inter_bw = args.inter_gbs * 1e9
    alpha_intra = args.alpha_intra_us * US
    alpha_inter = args.alpha_inter_us * US

    try:
        validate(args.nodes, args.gpus_per_node, args.rails, args.bytes,
                 args.intra_gbs, args.inter_gbs)
        if args.compare_all:
            names = list(MAPPINGS)
        else:
            if args.mapping not in MAPPINGS:
                raise PlacementError(
                    f"unknown mapping {args.mapping!r}: choose from "
                    f"{', '.join(MAPPINGS)}")
            names = [args.mapping]
        rows = [evaluate(name, args, intra_bw, inter_bw, alpha_intra, alpha_inter)
                for name in names]
    except PlacementError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(dict(banner=BANNER, nodes=args.nodes,
                              gpus_per_node=args.gpus_per_node, rails=args.rails,
                              bytes=args.bytes, seed=args.seed, mappings=rows),
                         indent=2))
        return 0

    print(BANNER)
    if args.compare_all:
        print(f"{args.nodes} nodes x {args.gpus_per_node} GPUs = "
              f"{args.nodes * args.gpus_per_node} ranks, {args.rails} rails, "
              f"m = {args.bytes / GIB:g} GiB, random seed {args.seed}")
        print()
        print_compare(rows, args)
        for row in rows:
            print(f"{row['mapping']:<7} {expected_counts(row['mapping'], args.nodes, args.gpus_per_node)}")
    else:
        print_one(rows[0], args, None)
    print()
    print("the pace barely moves -- a flat ring was already running at the inter-node "
          "rate -- so the damage is in the fabric and spine columns, which is why the "
          "production symptom is congestion and variance rather than a clean slowdown")
    print("a cross-rail count of zero is achievable for block mapping only if the ring "
          "is ordered to enter and leave each node on the SAME local index (a snaked "
          "ring); the plain block order joins local G-1 to local 0 at every boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
