#!/usr/bin/env python3
"""Multi-node all-reduce: flat ring vs hierarchical, as an ANALYTIC MODEL (D03 5.3).

Everything this tool prints is arithmetic. This repository has no multi-node
cluster, so it does not publish multi-node measurements; it publishes the model
and labels it, in a banner, on every run. The single-node numbers in this module
ARE measured, by hip/busbw_sweep.cpp on your own machine.

Two schedules for the same collective over N = gpus_per_node * nodes ranks:

  FLAT RING over all N ranks
    steps  2(N-1), every step moving m/N bytes on every edge
    the ring crosses each node boundary exactly ONCE, so exactly one NIC per node
    carries inter-node traffic and the slowest edge sets the pace:
      T = 2(N-1) * ( max(alpha_intra, alpha_inter) + (m/N) / min(B_intra, B_inter) )

  HIERARCHICAL reduce-scatter / inter-node all-reduce / all-gather
    A  intra-node reduce-scatter over G ranks: (G-1) steps of m/G bytes on B_intra
    B  inter-node all-reduce of the m/G slice over `nodes` ranks: 2(nodes-1) steps
       of (m/G)/nodes bytes; all G GPUs do this concurrently, so each gets
       B_inter * min(1, nics_per_node / G) of injection
    C  intra-node all-gather over G ranks: (G-1) steps of m/G bytes on B_intra

The interesting result is that the two schedules push nearly the SAME number of
bytes across each node boundary, about 2m. The hierarchy does not win on volume;
it wins by spreading that volume over every NIC instead of one, and by replacing
2(N-1) network-latency steps with 2(nodes-1) of them. That is why NIC count per
node is an architectural decision (README 3.4) and why libraries reduce inside a
node first (3.6).

Standard library only:
  python -S -I tools/multinode_allreduce_model.py --gpus-per-node 8 --nodes 16 \
      --intra-gbs 50 --inter-gbs 25 --nics-per-node 8 --bytes 1073741824
"""

from __future__ import annotations

import argparse
import json
import math
import sys

PROGRAM = "multinode_allreduce_model"

BANNER = "ANALYTIC MODEL of two all-reduce schedules - not a measurement"

SUFFIXES = {
    "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3,
}


def die(message: str) -> None:
    """One line on stderr, non-zero exit, never a traceback."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def parse_size(text: str) -> float:
    s = text.strip().lower()
    scale = 1.0
    for suffix in ("kib", "mib", "gib", "kb", "mb", "gb", "b"):
        if s.endswith(suffix):
            s, scale = s[: -len(suffix)].strip(), float(SUFFIXES[suffix])
            break
    try:
        value = float(s) * scale
    except ValueError:
        raise ValueError("cannot parse %r as a size; write a number with an optional "
                         "B/KB/MB/GB/KiB/MiB/GiB suffix" % text)
    if value <= 0 or not math.isfinite(value):
        raise ValueError("size must be finite and positive: %s" % text)
    return value


def human_bytes(nbytes: float) -> str:
    if nbytes < 1024:
        return "%.0f B" % nbytes
    for unit, scale in (("KiB", 1024.0), ("MiB", 1024.0**2), ("GiB", 1024.0**3)):
        if nbytes < scale * 1024 or unit == "GiB":
            return "%.3g %s" % (nbytes / scale, unit)
    return "%.0f B" % nbytes


def human_time(seconds: float) -> str:
    if seconds < 1e-3:
        return "%.2f us" % (seconds * 1e6)
    if seconds < 1.0:
        return "%.3f ms" % (seconds * 1e3)
    return "%.3f s" % seconds


def flat_ring(nbytes, gpus_per_node, nodes, b_intra, b_inter, a_intra, a_inter):
    """One ring over every rank. The slowest edge sets every step's cost."""
    n = gpus_per_node * nodes
    if n < 2:
        return {"steps": 0, "latency_s": 0.0, "wire_s": 0.0, "seconds": 0.0,
                "inter_bytes_per_node": 0.0, "nics_used": 0}
    steps = 2 * (n - 1)
    per_step_bytes = nbytes / n
    bw = min(b_intra, b_inter) if nodes > 1 else b_intra
    alpha = max(a_intra, a_inter) if nodes > 1 else a_intra
    latency = steps * alpha
    wire = steps * per_step_bytes / bw
    return {
        "steps": steps,
        "latency_s": latency,
        "wire_s": wire,
        "seconds": latency + wire,
        "bottleneck_bw": bw,
        # One ring edge crosses each node boundary, so this is what a single NIC
        # carries for the whole collective.
        "inter_bytes_per_node": (steps * per_step_bytes) if nodes > 1 else 0.0,
        "nics_used": 1 if nodes > 1 else 0,
    }


def hierarchical(nbytes, gpus_per_node, nodes, b_intra, b_inter, a_intra, a_inter,
                 nics_per_node):
    """Reduce-scatter inside the node, all-reduce across, all-gather back down."""
    g, k = gpus_per_node, nodes
    slice_bytes = nbytes / g if g > 0 else nbytes

    intra_steps = 2 * (g - 1) if g > 1 else 0
    intra_latency = intra_steps * a_intra
    intra_wire = (intra_steps * (nbytes / g) / b_intra) if g > 1 else 0.0

    inter_steps = 2 * (k - 1) if k > 1 else 0
    # Concurrency is the whole point: G GPUs per node inject at once, sharing the
    # node's NICs. With fewer NICs than GPUs each flow gets a fraction of a NIC,
    # which is the injection ceiling no channel count can raise (README 3.4).
    per_flow_bw = b_inter * min(1.0, float(nics_per_node) / g) if g > 0 else b_inter
    inter_latency = inter_steps * a_inter
    inter_wire = (inter_steps * (slice_bytes / k) / per_flow_bw) if k > 1 else 0.0

    return {
        "steps": intra_steps + inter_steps,
        "intra_steps": intra_steps,
        "inter_steps": inter_steps,
        "latency_s": intra_latency + inter_latency,
        "wire_s": intra_wire + inter_wire,
        "seconds": intra_latency + intra_wire + inter_latency + inter_wire,
        "per_flow_inter_bw": per_flow_bw,
        "slice_bytes": slice_bytes,
        "inter_bytes_per_node": (g * inter_steps * (slice_bytes / k)) if k > 1 else 0.0,
        "nics_used": min(nics_per_node, g) if k > 1 else 0,
    }


def busbw(nbytes, ranks, seconds):
    """The number a benchmark would print for this modelled time (README 2.4)."""
    if seconds <= 0 or ranks < 2:
        return 0.0
    return (nbytes / seconds) * 2.0 * (ranks - 1) / ranks


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analytic model: flat-ring vs hierarchical multi-node all-reduce.",
        epilog="Every number printed is a model. The single-node equivalents are "
               "measurable with hip/busbw_sweep.cpp; measure those instead of "
               "quoting these.",
    )
    ap.add_argument("--gpus-per-node", type=int, default=8,
                    help="GPUs per node, i.e. the intra-node ring size (default 8)")
    ap.add_argument("--nodes", type=int, default=16, help="nodes (default 16)")
    ap.add_argument("--intra-gbs", type=float, default=50.0,
                    help="per-link intra-node bandwidth in GB/s. ILLUSTRATIVE "
                         "default: substitute your own from D04 or a measurement")
    ap.add_argument("--inter-gbs", type=float, default=25.0,
                    help="per-NIC inter-node bandwidth in GB/s. ILLUSTRATIVE default")
    ap.add_argument("--nics-per-node", type=int, default=8,
                    help="network adapters per node; caps concurrent inter-node "
                         "flows (default 8, i.e. one per GPU)")
    ap.add_argument("--bytes", "-b", default="1GiB",
                    help="all-reduce buffer size, with an optional unit suffix")
    ap.add_argument("--intra-alpha-us", type=float, default=2.0,
                    help="per-step intra-node latency in us. ILLUSTRATIVE default")
    ap.add_argument("--inter-alpha-us", type=float, default=8.0,
                    help="per-step inter-node latency in us. ILLUSTRATIVE default")
    ap.add_argument("--sizes", default="",
                    help="comma-separated sizes for a sweep instead of one size")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.gpus_per_node < 1:
        ap.error("--gpus-per-node must be at least 1")
    if args.nodes < 1:
        ap.error("--nodes must be at least 1")
    if args.nics_per_node < 1:
        ap.error("--nics-per-node must be at least 1")
    for name, value in (("--intra-gbs", args.intra_gbs),
                        ("--inter-gbs", args.inter_gbs),
                        ("--intra-alpha-us", args.intra_alpha_us),
                        ("--inter-alpha-us", args.inter_alpha_us)):
        if value <= 0 or not math.isfinite(value):
            ap.error("%s must be positive and finite" % name)

    try:
        sizes = ([parse_size(s) for s in args.sizes.split(",") if s.strip()]
                 if args.sizes else [parse_size(args.bytes)])
    except ValueError as exc:
        ap.error("bad size: %s" % exc)
    if not sizes:
        ap.error("no sizes to evaluate")

    g, k = args.gpus_per_node, args.nodes
    n = g * k
    b_intra, b_inter = args.intra_gbs * 1e9, args.inter_gbs * 1e9
    a_intra, a_inter = args.intra_alpha_us * 1e-6, args.inter_alpha_us * 1e-6

    rows = []
    for nbytes in sizes:
        flat = flat_ring(nbytes, g, k, b_intra, b_inter, a_intra, a_inter)
        hier = hierarchical(nbytes, g, k, b_intra, b_inter, a_intra, a_inter,
                            args.nics_per_node)
        rows.append({
            "bytes": nbytes,
            "flat": flat,
            "hierarchical": hier,
            "speedup": (flat["seconds"] / hier["seconds"]) if hier["seconds"] > 0 else 0.0,
            "flat_busbw_gbs": busbw(nbytes, n, flat["seconds"]) / 1e9,
            "hier_busbw_gbs": busbw(nbytes, n, hier["seconds"]) / 1e9,
        })

    out = {
        "disclaimer": BANNER,
        "ranks": n,
        "gpus_per_node": g,
        "nodes": k,
        "nics_per_node": args.nics_per_node,
        "intra_bytes_per_s": b_intra,
        "inter_bytes_per_s": b_inter,
        "intra_alpha_s": a_intra,
        "inter_alpha_s": a_inter,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print("=" * 78)
    print(BANNER)
    print("=" * 78)
    print("N = %d ranks (%d GPUs x %d nodes), %d NIC(s) per node" %
          (n, g, k, args.nics_per_node))
    print("intra-node %g GB/s per link, alpha %g us   |   inter-node %g GB/s per "
          "NIC, alpha %g us" % (args.intra_gbs, args.intra_alpha_us,
                                args.inter_gbs, args.inter_alpha_us))
    print("Bandwidths and latencies above are PARAMETERS YOU SUPPLIED (or the")
    print("illustrative defaults). Per-fabric figures with sources and retrieval")
    print("dates are in D04; nothing here is measured.")
    print("")

    header = ("%12s %8s %11s %11s %11s %8s %11s %11s" %
              ("bytes", "schedule", "steps", "alpha term", "wire term", "total",
               "busbw GB/s", "NICs used"))
    print(header)
    print("-" * len(header))
    for row in rows:
        for label, key in (("flat", "flat"), ("hier", "hierarchical")):
            entry = row[key]
            print("%12s %8s %11d %11s %11s %8s %11.2f %11d"
                  % (human_bytes(row["bytes"]) if label == "flat" else "",
                     label, entry["steps"], human_time(entry["latency_s"]),
                     human_time(entry["wire_s"]), human_time(entry["seconds"]),
                     row["flat_busbw_gbs"] if label == "flat" else row["hier_busbw_gbs"],
                     entry["nics_used"]))
        print("%12s %8s %11s %11s %11s %8.2fx" % ("", "ratio", "", "", "",
                                                  row["speedup"]))
    print("-" * len(header))
    print("")

    row = rows[-1]
    flat, hier = row["flat"], row["hierarchical"]
    if k == 1:
        print("One node, so there is no network and both schedules degenerate to the")
        print("same intra-node ring: the two rows above are the same arithmetic. The")
        print("single-node case is MEASURABLE - run hip/busbw_sweep.cpp instead of")
        print("reading a model of it.")
        print("")
    else:
        print("Where the difference comes from, at %s:" % human_bytes(row["bytes"]))
        # In the flat ring every step is paced by the slowest edge, which crosses
        # the network, so all 2(N-1) of its steps pay a network latency. The
        # hierarchy pays one only in phase B.
        print("  steps paying a network latency   flat %d   vs hierarchical %d"
              % (flat["steps"], hier["inter_steps"]))
        print("  bytes across one node boundary   flat %s   vs hierarchical %s"
              % (human_bytes(flat["inter_bytes_per_node"]),
                 human_bytes(hier["inter_bytes_per_node"])))
        print("  NICs carrying that traffic       flat %d   vs hierarchical %d"
              % (flat["nics_used"], hier["nics_used"]))
        volume_ratio = (flat["inter_bytes_per_node"] / hier["inter_bytes_per_node"]
                        if hier["inter_bytes_per_node"] > 0 else 0.0)
        print("")
        if volume_ratio > 0:
            print("Note the volume ratio: %.2fx. The hierarchy is NOT moving fewer "
                  "bytes" % volume_ratio)
            print("across the network - it moves about 2m either way, which is the")
            print("information bound.")
        if hier["nics_used"] > flat["nics_used"]:
            print("It wins by using %d NICs instead of %d and by paying %d network"
                  % (hier["nics_used"], flat["nics_used"], hier["inter_steps"]))
            print("latencies instead of %d." % flat["steps"])
        if row["speedup"] < 1.0:
            print("Here the hierarchy LOSES (%.2fx), and the reason is worth having:"
                  % row["speedup"])
            print("its phases A and C add intra-node traffic that the flat ring never")
            print("pays, and that cost is only recovered when the network was the")
            print("bottleneck. With %d NIC(s) for %d GPUs the network stays the"
                  % (args.nics_per_node, g))
            print("bottleneck either way, so the extra local work is pure loss. A real")
            print("library makes this same call from measured thresholds.")
    if args.nics_per_node < g and k > 1:
        print("")
        print("--nics-per-node %d is below --gpus-per-node %d, so each inter-node"
              % (args.nics_per_node, g))
        print("flow gets %.2f GB/s of a %g GB/s adapter. That ceiling is LogP's gap"
              % (hier["per_flow_inter_bw"] / 1e9, args.inter_gbs))
        print("parameter and no channel count raises it: it is why large training")
        print("clusters are built with one NIC per GPU.")

    print("")
    print("Where this model stops being true, stated because the table hides it:")
    print("  * It assumes a contention-free fabric: no oversubscription, no")
    print("    congestion, no sharing with other jobs. A real fat tree with 2:1")
    print("    oversubscription violates that at exactly the sizes that matter.")
    print("  * It assumes perfect overlap within a phase and none between phases.")
    print("    A real library pipelines the boundary, so it beats this model's")
    print("    hierarchical number and the shape stays right while the value does not.")
    print("  * It ignores the reduction arithmetic (gamma) and buffer registration.")
    print("  * It assumes the ring ordering respects the topology. When placement")
    print("    is wrong the flat ring can cross node boundaries far more than once")
    print("    per node, which makes the real gap larger than the one modelled here.")
    print("  * alpha and beta are per-fabric properties. Substitute measured or")
    print("    cited values (D04) before showing this to anyone.")
    print("")
    print(BANNER)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
