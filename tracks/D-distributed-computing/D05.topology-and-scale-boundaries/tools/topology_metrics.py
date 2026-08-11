#!/usr/bin/env python3
"""Diameter, bisection bandwidth and the all-to-all floor for four fabric shapes.

Everything here is an ANALYTIC MODEL. The graph properties -- diameter, bisection,
links per node -- are exact and can be re-derived by hand; nothing about them was
measured. The *times* are the product of an exact formula and a link rate you
supplied, which makes them assumptions wearing a formula's clothes.

The two characterizing numbers, from module README section 2.1:

    diameter   worst-case hop count on a shortest path; bounds the alpha term
    bisection  total link bandwidth across the WORST cut into two equal halves;
               a hard ceiling on every all-to-all pattern (section 2.2):

                   bytes crossing the cut = N * m / 2
                   time >= (N * m / 2) / B_bisect

Per-shape formulas, all in units of one link rate B_link:

    fat-tree (folded Clos, radix k, L levels, oversubscription o)
        endpoint capacity  2 * (k//2)**L
        diameter           2L-1 switch traversals, 2L link traversals, flat in N
        bisection          N / (2*o)          o=1 is "full bisection", N/2
    dragonfly (a routers/group, p terminals/router, h global channels/router)
        groups             g <= a*h + 1 so every pair of groups is directly joined
        diameter           3 inter-router hops minimal (local, global, local),
                           up to 5 non-minimal; 5 link traversals end to end
        bisection          (g//2) * (g - g//2) * floor(a*h / (g-1)) global channels
    torus / mesh (d dimensions, side n = N**(1/d))
        diameter           d * (n//2) torus, d * (n-1) mesh
        bisection          2 * n**(d-1) torus, n**(d-1) mesh

Stdlib only, and no site packages required:
    python -S -I tools/topology_metrics.py --topology torus --nodes 4096 --dims 3
"""

from __future__ import annotations

import argparse
import json
import sys

BANNER = ("analytic topology model, not measured: the graph properties are exact, "
          "the link rate is your assumption")

TOPOLOGIES = ("fat-tree", "dragonfly", "torus", "mesh")

# A fat-tree that needs more levels than this is not a fabric anyone builds; the cap
# exists so a radix that cannot reach the endpoint count fails with a message instead
# of looping forever (radix 2 leaves one downlink per switch and never grows).
MAX_LEVELS = 16

GIB = 1024.0**3


class TopologyError(ValueError):
    """Bad input, carrying a one-line message the CLI can print without a traceback."""


def validate_common(nodes: int, link_gbs: float, nbytes: float) -> None:
    if nodes < 1:
        raise TopologyError(f"--nodes must be at least 1, got {nodes}")
    if link_gbs <= 0.0:
        raise TopologyError(f"--link-gbs must be positive, got {link_gbs}")
    if nbytes <= 0.0:
        raise TopologyError(f"--bytes must be positive, got {nbytes}")


def integer_root(value: int, degree: int) -> int | None:
    """Exact integer d-th root of `value`, or None when there is not one.

    Done by search around the float estimate rather than by `round(value ** (1/d))`
    alone: at 4096 and d=6 the float root is 3.9999999999999996, and a topology tool
    that silently accepts a side length of 3 has produced a different machine.
    """
    if value < 1 or degree < 1:
        return None
    estimate = int(round(value ** (1.0 / degree)))
    for candidate in range(max(1, estimate - 2), estimate + 3):
        if candidate**degree == value:
            return candidate
    return None


def fat_tree_metrics(nodes: int, radix: int, levels: int = 0,
                     oversubscription: float = 1.0) -> dict:
    """Folded-Clos metrics. `levels` of 0 derives the smallest level count that fits.

    Leiserson's fat-tree keeps aggregate bandwidth constant as you climb, so no level
    is a bottleneck (see docs/REFERENCES.md, Leiserson 1985). A folded Clos implements
    that with parallel switches instead of a physically fatter cable: each switch puts
    k//2 ports down and k//2 up, so L levels reach 2*(k//2)**L endpoints and every
    endpoint pair is exactly 2L link traversals apart.
    """
    if radix < 2:
        raise TopologyError(f"--radix must be at least 2, got {radix}")
    if oversubscription < 1.0:
        raise TopologyError(
            f"--oversubscription must be at least 1 (1 means non-blocking), "
            f"got {oversubscription}")
    if levels < 0:
        raise TopologyError(f"--levels must be non-negative, got {levels}")

    down = radix // 2
    if levels == 0:
        levels = 1
        while 2 * down**levels < nodes:
            levels += 1
            if levels > MAX_LEVELS:
                raise TopologyError(
                    f"no fat-tree of radix {radix} reaches {nodes} endpoints within "
                    f"{MAX_LEVELS} levels: raise --radix")
    capacity = 2 * down**levels
    if capacity < nodes:
        raise TopologyError(
            f"{levels} levels of radix {radix} hold {capacity} endpoints, "
            f"not {nodes}: raise --levels or --radix")

    # Bisection is the as-built uplink capacity crossing the cut, never the radix --
    # computing it from the radix is the standard way a design document lands 2x high
    # (module README section 2.3).
    bisection = nodes / (2.0 * oversubscription)
    return dict(
        name=f"fat-tree {levels}-level radix {radix} {oversubscription:g}:1",
        family="fat-tree",
        nodes=nodes,
        switch_hops=2 * levels - 1,
        link_traversals=2 * levels,
        diameter_grows_with_n=False,
        bisection_links=bisection,
        links_per_node=1,
        cost_driver="optics and spine ports, then switch radix",
        detail=(f"{levels} levels, capacity {capacity} endpoints, "
                f"{down} down and {radix - down} up per switch"),
        notes=[
            "diameter is flat in N within a level count: adding endpoints under the "
            "same three levels does not add a hop",
            "oversubscription changes the bisection and nothing else, which is why it "
            "is invisible to every latency and small-message test",
        ],
    )


def dragonfly_metrics(nodes: int, routers_per_group: int, terminals_per_router: int,
                      global_channels_per_router: int) -> dict:
    """Dragonfly metrics for the smallest group count that holds `nodes` terminals.

    Kim et al. 2008 collect routers into groups that are richly connected internally,
    so a group behaves as one very-high-radix virtual router, then join the groups so
    that every pair has a direct global channel. That gives the minimal route
    local-global-local: three inter-router hops at any machine size.

    The bisection reported here is the count of global channels severed by a balanced
    cut of the GROUPS. With every pair of groups directly joined by c channels, a cut
    into halves of g//2 and g-g//2 groups severs (g//2)*(g-g//2)*c of them -- about
    half of all global channels, since g/(2*(g-1)) -> 1/2.
    """
    a, p, h = routers_per_group, terminals_per_router, global_channels_per_router
    if a < 1:
        raise TopologyError(f"--routers-per-group must be at least 1, got {a}")
    if p < 1:
        raise TopologyError(f"--terminals-per-router must be at least 1, got {p}")
    if h < 1:
        raise TopologyError(f"--global-channels-per-router must be at least 1, got {h}")

    max_groups = a * h + 1
    per_group = a * p
    groups = -(-nodes // per_group)  # ceil, in integers
    if groups > max_groups:
        raise TopologyError(
            f"a={a} p={p} h={h} holds {max_groups * per_group} terminals in at most "
            f"{max_groups} groups, not {nodes}: raise --routers-per-group, "
            f"--terminals-per-router or --global-channels-per-router")

    if groups > 1:
        channels_per_pair = (a * h) // (groups - 1)
        half = groups // 2
        bisection = float(half * (groups - half) * channels_per_pair)
        spare = a * h - (groups - 1) * channels_per_pair
    else:
        channels_per_pair = 0
        bisection = 0.0
        spare = a * h

    balanced = (a == 2 * p) and (a == 2 * h)
    notes = [
        f"{groups} groups of {a} routers x {p} terminals; every pair of groups joined "
        f"by {channels_per_pair} global channel(s), {spare} global port(s) per group "
        f"left spare",
        "diameter 3 is the MINIMAL route; adaptive (Valiant) routing via an "
        "intermediate group takes up to 5 inter-router hops and two global channels",
        "bisection here is a balanced cut of the groups, and it assumes the global "
        "channels are the cut's narrowest part rather than the local ones",
    ]
    notes.append(
        "balanced per Kim et al.: a = 2p = 2h" if balanced else
        f"NOT balanced: Kim et al. size a group as a = 2p = 2h, this is "
        f"a={a} p={p} h={h}")
    return dict(
        name=f"dragonfly a={a} p={p} h={h}",
        family="dragonfly",
        nodes=nodes,
        switch_hops=4,
        link_traversals=5,
        diameter_grows_with_n=False,
        bisection_links=bisection,
        links_per_node=1,
        cost_driver="global (long, optical) channels",
        detail=(f"{groups} groups, minimal route local-global-local = 3 inter-router "
                f"hops, up to 5 non-minimal"),
        notes=notes,
    )


def torus_metrics(nodes: int, dims: int, mesh: bool = False) -> dict:
    """Torus or mesh metrics for N nodes arranged as a cube of side n = N**(1/dims).

    Cutting one dimension of a torus severs two sheets of n**(d-1) links -- two,
    because the wraparound closes the ring -- so the bisection is 2*n**(d-1) and the
    ratio to full bisection falls as N**(-1/d). A mesh has no wraparound: one sheet,
    and a diameter of d*(n-1) instead of d*(n//2).
    """
    if dims < 1:
        raise TopologyError(f"--dims must be at least 1, got {dims}")
    side = integer_root(nodes, dims)
    if side is None:
        raise TopologyError(
            f"no integer side length satisfies n**{dims} == {nodes}: --nodes {nodes} "
            f"cannot form a {dims}-dimensional cube")

    if mesh:
        diameter = dims * (side - 1)
        bisection = float(side ** (dims - 1))
        family, label = "mesh", "mesh"
    else:
        diameter = dims * (side // 2)
        bisection = 2.0 * side ** (dims - 1)
        family, label = "torus", "torus"

    notes = [
        "there are no fabric switches: every node forwards for its neighbours, so a "
        "hop is a node traversal and the diameter grows as N**(1/d)",
        "unbeatable for nearest-neighbour halo exchange -- the wiring IS the "
        "communication graph, one hop, no contention, at any machine size",
    ]
    if side <= 2 and not mesh:
        notes.append(
            f"side n = {side}: the wraparound link coincides with the direct link, so "
            f"a real machine wires one link where this formula counts two -- treat the "
            f"bisection as the doubled-link idealization")
    return dict(
        name=f"{label} {dims}D {'x'.join([str(side)] * dims)}",
        family=family,
        nodes=nodes,
        switch_hops=diameter,
        link_traversals=diameter,
        diameter_grows_with_n=True,
        bisection_links=bisection,
        # A torus is 2d links everywhere. A mesh is 2d only in the interior: an edge node
        # has fewer and a corner node has d, so this is the maximum rather than the norm.
        links_per_node=2 * dims,
        cost_driver="per-node port count, and the dimension count",
        detail=f"side n = {side}, {2 * dims} links per node",
        notes=notes,
    )


def full_bisection_links(nodes: int) -> float:
    """Full bisection in link units: half the endpoints each pushing one link rate."""
    return nodes / 2.0


def all_to_all_floor_seconds(nodes: int, nbytes: float, bisection_links: float,
                             link_bytes_per_s: float) -> float:
    """Section 2.2's lower bound: (N*m/2) / B_bisect. Routing-independent."""
    crossing = nodes * nbytes / 2.0
    capacity = bisection_links * link_bytes_per_s
    if capacity <= 0.0:
        return float("inf")
    return crossing / capacity


def enrich(row: dict, link_gbs: float, nbytes: float) -> dict:
    """Add the derived bandwidth and time columns to one topology record."""
    link_bytes = link_gbs * 1e9
    full = full_bisection_links(row["nodes"])
    row["bisection_gbs"] = row["bisection_links"] * link_gbs
    row["bisection_fraction"] = (row["bisection_links"] / full) if full else 0.0
    row["all_to_all_floor_s"] = all_to_all_floor_seconds(
        row["nodes"], nbytes, row["bisection_links"], link_bytes)
    return row


def build_row(args, topology: str) -> dict:
    if topology == "fat-tree":
        row = fat_tree_metrics(args.nodes, args.radix, args.levels,
                               args.oversubscription)
    elif topology == "dragonfly":
        row = dragonfly_metrics(args.nodes, args.routers_per_group,
                                args.terminals_per_router,
                                args.global_channels_per_router)
    elif topology in ("torus", "mesh"):
        row = torus_metrics(args.nodes, args.dims, mesh=(topology == "mesh"))
    else:
        raise TopologyError(
            f"unknown topology {topology!r}: choose from {', '.join(TOPOLOGIES)}")
    return enrich(row, args.link_gbs, args.bytes)


def comparison_rows(args) -> tuple[list[dict], list[str]]:
    """The five Lab 1 configurations plus the 4D torus from README section 3.3.

    A configuration that cannot exist at this node count -- 4,096 is a perfect cube
    and a perfect 6th power, 4,000 is neither -- is skipped with its reason rather
    than failing the whole table.
    """
    plans = [
        ("fat-tree, non-blocking",
         lambda: fat_tree_metrics(args.nodes, args.radix, args.levels, 1.0)),
        ("fat-tree, 2:1 oversubscribed",
         lambda: fat_tree_metrics(args.nodes, args.radix, args.levels, 2.0)),
        ("dragonfly, balanced",
         lambda: dragonfly_metrics(args.nodes, args.routers_per_group,
                                   args.terminals_per_router,
                                   args.global_channels_per_router)),
        ("torus 3D", lambda: torus_metrics(args.nodes, 3)),
        ("torus 4D", lambda: torus_metrics(args.nodes, 4)),
        ("torus 6D", lambda: torus_metrics(args.nodes, 6)),
    ]
    rows, skipped = [], []
    for label, build in plans:
        try:
            row = enrich(build(), args.link_gbs, args.bytes)
        except TopologyError as exc:
            skipped.append(f"{label}: {exc}")
            continue
        row["label"] = label
        rows.append(row)
    return rows, skipped


def format_bytes(nbytes: float) -> str:
    if nbytes >= GIB:
        return f"{nbytes / GIB:g} GiB"
    if nbytes >= 1024.0:
        return f"{nbytes / 1024.0:g} KiB"
    return f"{nbytes:g} B"


def print_single(row: dict, args) -> None:
    print(BANNER)
    print(f"{row['name']}: N = {row['nodes']} endpoints, "
          f"B_link = {args.link_gbs:g} GB/s")
    print(f"  shape             {row['detail']}")
    if row["family"] in ("torus", "mesh"):
        print(f"  diameter          {row['link_traversals']} hops, each one a node "
              f"traversal (grows as N**(1/d))")
    else:
        print(f"  diameter          {row['switch_hops']} switch traversals, "
              f"{row['link_traversals']} link traversals"
              f"{'' if row['diameter_grows_with_n'] else ' (flat in N)'}")
    print(f"  bisection         {row['bisection_links']:g} link-units = "
          f"{row['bisection_gbs']:g} GB/s")
    print(f"  fraction of full  {row['bisection_fraction'] * 100:.1f}% of "
          f"(N/2) * B_link = {full_bisection_links(row['nodes']):g} link-units")
    print(f"  links per node    {row['links_per_node']}")
    print(f"  cost driver       {row['cost_driver']}")
    print(f"  all-to-all floor  {format_bytes(args.bytes)} per endpoint -> "
          f"{row['all_to_all_floor_s'] * 1e3:.3f} ms  "
          f"(N*m/2 / B_bisect, section 2.2)")
    print("notes:")
    for note in row["notes"]:
        print(f"  - {note}")


def print_comparison(rows: list[dict], skipped: list[str], args) -> None:
    print(BANNER)
    print(f"N = {args.nodes} endpoints, B_link = {args.link_gbs:g} GB/s, "
          f"all-to-all message {format_bytes(args.bytes)} per endpoint")
    print()
    header = (f"{'configuration':<28}{'hops':>6}{'links/node':>12}"
              f"{'bisect (link-units)':>21}{'of full':>9}{'a2a floor ms':>14}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['label']:<28}{row['link_traversals']:>6}"
              f"{row['links_per_node']:>12}{row['bisection_links']:>21g}"
              f"{row['bisection_fraction'] * 100:>8.1f}%"
              f"{row['all_to_all_floor_s'] * 1e3:>14.3f}")
    print("-" * len(header))
    print("hops = link traversals end to end for the worst pair (minimal routing), "
          "counting the two terminal links")
    # Worth spelling out: the dragonfly's famous "three hops" counts inter-router hops,
    # so it is 5 in this column, and 5 is ALSO the usual non-minimal inter-router figure.
    # Two different quantities that collide at one number is exactly how a reader is
    # misled, so name both rather than leaving the column to be misread.
    print("dragonfly: 5 here is the minimal route counted in links "
          "(3 inter-router hops + 2 terminal links), not the non-minimal route")
    for reason in skipped:
        print(f"skipped {reason}")
    print("the fat-tree rows differ only in bisection: oversubscription is invisible "
          "to a hop count, and therefore to a latency test")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diameter, bisection bandwidth and the all-to-all lower bound "
                    "for a fat-tree, a dragonfly, a torus or a mesh.")
    parser.add_argument("--topology", default="fat-tree",
                        help="fat-tree, dragonfly, torus or mesh (default fat-tree)")
    parser.add_argument("--nodes", type=int, default=4096,
                        help="endpoint count N (default 4096)")
    parser.add_argument("--radix", type=int, default=64,
                        help="switch port count for the fat-tree (default 64)")
    parser.add_argument("--levels", type=int, default=0,
                        help="fat-tree switch levels; 0 derives the smallest that fits")
    parser.add_argument("--oversubscription", type=float, default=1.0,
                        help="leaf host-to-uplink bandwidth ratio, 1 = non-blocking")
    parser.add_argument("--routers-per-group", type=int, default=16,
                        help="dragonfly routers per group, `a` (default 16)")
    parser.add_argument("--terminals-per-router", type=int, default=8,
                        help="dragonfly terminals per router, `p` (default 8)")
    parser.add_argument("--global-channels-per-router", type=int, default=8,
                        help="dragonfly global channels per router, `h` (default 8)")
    parser.add_argument("--dims", type=int, default=3,
                        help="torus or mesh dimension count d (default 3)")
    parser.add_argument("--link-gbs", type=float, default=50.0,
                        help="one link's bandwidth in GB/s, 1 GB = 1e9 bytes "
                             "(default 50)")
    parser.add_argument("--bytes", type=float, default=1073741824.0,
                        help="all-to-all message size per endpoint (default 1 GiB)")
    parser.add_argument("--compare", action="store_true",
                        help="tabulate every shape at this node count instead of one")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        validate_common(args.nodes, args.link_gbs, args.bytes)
        if args.compare:
            rows, skipped = comparison_rows(args)
            if not rows:
                raise TopologyError(
                    f"no topology in the comparison set is buildable at "
                    f"--nodes {args.nodes}: {skipped[0]}")
        else:
            if args.topology not in TOPOLOGIES:
                raise TopologyError(
                    f"unknown topology {args.topology!r}: choose from "
                    f"{', '.join(TOPOLOGIES)}")
            rows, skipped = [build_row(args, args.topology)], []
    except TopologyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(dict(banner=BANNER, nodes=args.nodes,
                              link_gbs=args.link_gbs, bytes=args.bytes,
                              full_bisection_links=full_bisection_links(args.nodes),
                              topologies=rows, skipped=skipped), indent=2))
        return 0

    if args.compare:
        print_comparison(rows, skipped, args)
    else:
        print_single(rows[0], args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
