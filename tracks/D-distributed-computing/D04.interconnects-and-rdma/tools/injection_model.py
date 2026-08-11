#!/usr/bin/env python3
"""Node injection bandwidth, the per-GPU share, and the cliff against the scale-up tier.

Everything this tool prints is an ANALYTIC MODEL of README section 3.6. No node is
inspected and nothing is measured. It is deliberately trivial arithmetic:

    node injection bandwidth = n_nic * B_nic
    per-GPU share            = n_nic * B_nic / n_gpu
    ratio                    = per-GPU share / scale-up bandwidth per GPU

The arithmetic is trivial and the consequence is not. Demand for the network scales with
GPU count; supply scales with adapter count; those are different numbers, and a node
where they differ has a ceiling that no single-node benchmark can see. This is LogP's
gap parameter g made concrete -- D01 section 3.2 introduces it as a model parameter, and
here it is a line item on a purchase order.

What this model does not know, and what therefore bounds how far you can push it: real
adapters do not sustain their line rate, PCIe sits between the GPU and the adapter and
can be narrower than either (README 3.7), the fabric may be oversubscribed above the
node, and a collective's traffic pattern decides how much of the injection budget it can
actually use. Every one of those makes the real number worse than this one, so treat the
output as an upper bound on the node and never as a prediction of a job.

Stdlib only:
    python -S -I tools/injection_model.py --gpus-per-node 8 --nics-per-node 1 \\
        --nic-gbs 25 --scaleup-gbs 400
"""

from __future__ import annotations

import argparse
import json
import sys

PROGRAM = "injection_model"
BANNER = ("ANALYTIC MODEL, NOT A MEASUREMENT: line-rate adapters, no PCIe limit between\n"
          "GPU and adapter, no fabric oversubscription, and every GPU able to use its\n"
          "full share. The real number is lower for all four reasons.")

GB = 1e9


def validate(n_nic, nic_bps, n_gpu, scaleup_bps=None):
    """Reject configurations the model has no meaning for, naming the bad value."""
    if n_gpu < 1:
        raise ValueError("a node needs at least one GPU, got %d" % n_gpu)
    if n_nic < 0:
        raise ValueError("adapter count cannot be negative, got %d" % n_nic)
    if n_nic == 0:
        # A defined error rather than a division by zero or an infinity: a node with no
        # adapter has no inter-node bandwidth at all, which is a configuration answer,
        # not a number.
        raise ValueError("a node with zero network adapters cannot leave the node at "
                         "all; there is no injection bandwidth to divide")
    if nic_bps <= 0:
        raise ValueError("adapter bandwidth must be positive, got %g B/s" % nic_bps)
    if scaleup_bps is not None and scaleup_bps <= 0:
        raise ValueError("scale-up bandwidth must be positive, got %g B/s" % scaleup_bps)


def node_injection(n_nic, nic_bps):
    """Aggregate bytes per second the node can push onto the fabric."""
    return n_nic * nic_bps


def per_gpu_injection(n_nic, nic_bps, n_gpu):
    """Bytes per second of network bandwidth each GPU gets, if shared evenly."""
    validate(n_nic, nic_bps, n_gpu)
    return node_injection(n_nic, nic_bps) / n_gpu


def scaleup_ratio(n_nic, nic_bps, n_gpu, scaleup_bps):
    """Per-GPU network share as a fraction of that GPU's scale-up bandwidth.

    This ratio, not either bandwidth on its own, is the number that decides whether a
    flow can cross the node boundary without falling off README section 2.1's ladder.
    """
    validate(n_nic, nic_bps, n_gpu, scaleup_bps)
    return per_gpu_injection(n_nic, nic_bps, n_gpu) / scaleup_bps


def min_nics_for_ratio(target, nic_bps, n_gpu, scaleup_bps):
    """Fewest adapters whose per-GPU share reaches `target` times the scale-up tier.

    Closed form: n_nic >= target * scaleup * n_gpu / nic. Returned as an integer count
    because you cannot buy a fraction of an adapter, and the ceiling is taken so the
    answer meets the target rather than approaching it.
    """
    if target <= 0:
        raise ValueError("target ratio must be positive, got %g" % target)
    validate(1, nic_bps, n_gpu, scaleup_bps)
    exact = target * scaleup_bps * n_gpu / nic_bps
    count = int(exact)
    if count < exact:
        count += 1
    return max(count, 1)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute a node's injection ceiling, the per-GPU share, and the "
                    "ratio against the scale-up tier.",
        epilog="Every number printed is a model, never a measurement.")
    parser.add_argument("--gpus-per-node", type=int, default=8,
                        help="GPUs in the node (default 8)")
    parser.add_argument("--nics-per-node", type=int, default=1,
                        help="network adapters in the node (default 1)")
    parser.add_argument("--nic-gbs", type=float, default=25.0,
                        help="sustained bandwidth per adapter in GB/s, 1 GB = 1e9 B "
                             "(default 25)")
    parser.add_argument("--scaleup-gbs", type=float, default=400.0,
                        help="scale-up bandwidth available to one GPU in GB/s; get "
                             "yours from B01 section 3.5 or measure it with lab 2 "
                             "(default 400)")
    parser.add_argument("--target-ratio", type=float, default=1.0,
                        help="ratio of scale-up bandwidth to solve the adapter count "
                             "for (default 1.0, i.e. parity)")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    nic_bps = args.nic_gbs * GB
    scaleup_bps = args.scaleup_gbs * GB

    try:
        validate(args.nics_per_node, nic_bps, args.gpus_per_node, scaleup_bps)
        if args.target_ratio <= 0:
            raise ValueError("--target-ratio must be positive, got %g"
                             % args.target_ratio)
    except ValueError as exc:
        sys.stderr.write("%s: %s\n" % (PROGRAM, exc))
        return 2

    node = node_injection(args.nics_per_node, nic_bps)
    share = per_gpu_injection(args.nics_per_node, nic_bps, args.gpus_per_node)
    ratio = scaleup_ratio(args.nics_per_node, nic_bps, args.gpus_per_node, scaleup_bps)
    needed = min_nics_for_ratio(args.target_ratio, nic_bps, args.gpus_per_node,
                                scaleup_bps)

    # Linear in adapter count with no diminishing returns, which is the whole point:
    # print it rather than assert it. The ladder stops at twice the GPU count, because
    # one adapter per GPU is the design this argument is heading towards and rows beyond
    # it describe a machine nobody builds.
    top = max(args.gpus_per_node, args.nics_per_node) * 2
    ladder = []
    count = 1
    while count <= top:
        ladder.append(count)
        count *= 2
    ladder.append(args.gpus_per_node)
    ladder.append(args.nics_per_node)
    if needed <= top:
        ladder.append(needed)
    ladder = sorted(set(ladder))
    sweep = [{"nics": n,
              "node_gbs": node_injection(n, nic_bps) / GB,
              "per_gpu_gbs": per_gpu_injection(n, nic_bps, args.gpus_per_node) / GB,
              "ratio": scaleup_ratio(n, nic_bps, args.gpus_per_node, scaleup_bps)}
             for n in ladder]

    result = {
        "banner": BANNER.replace("\n", " "),
        "gpus_per_node": args.gpus_per_node,
        "nics_per_node": args.nics_per_node,
        "nic_bytes_per_s": nic_bps,
        "scaleup_bytes_per_s": scaleup_bps,
        "node_injection_bytes_per_s": node,
        "per_gpu_injection_bytes_per_s": share,
        "ratio_against_scaleup": ratio,
        "target_ratio": args.target_ratio,
        "nics_needed_for_target": needed,
        "sweep": sweep,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(BANNER)
    print("=" * 78)
    print("node: %d GPU(s), %d adapter(s) at %.4g GB/s each, scale-up %.4g GB/s per GPU"
          % (args.gpus_per_node, args.nics_per_node, args.nic_gbs, args.scaleup_gbs))
    print("=" * 78)
    print("")
    print("node injection bandwidth   n_nic * B_nic          = %10.2f GB/s"
          % (node / GB))
    print("per-GPU share              n_nic * B_nic / n_gpu  = %10.2f GB/s"
          % (share / GB))
    print("scale-up bandwidth per GPU (you supplied it)      = %10.2f GB/s"
          % args.scaleup_gbs)
    print("ratio, network : scale-up                         = %10.4f"
          % ratio)
    if ratio < 1.0:
        print("                                             that is   1 : %.1f"
              % (1.0 / ratio))
    else:
        print("                                             that is %.1f : 1" % ratio)
    print("")

    print("adapters against the per-GPU share (linear in n_nic, no diminishing returns):")
    header = "%8s %14s %16s %14s" % ("adapters", "node GB/s", "per-GPU GB/s",
                                     "vs scale-up")
    print(header)
    print("-" * len(header))
    for row in sweep:
        marker = "  <- this node" if row["nics"] == args.nics_per_node else ""
        print("%8d %14.2f %16.3f %13.4f%s"
              % (row["nics"], row["node_gbs"], row["per_gpu_gbs"], row["ratio"], marker))
    print("-" * len(header))
    print("")

    print("to reach %.4g x the scale-up tier per GPU you need %d adapter(s) in this node."
          % (args.target_ratio, needed))
    if needed > args.gpus_per_node:
        print("  That is more adapters than GPUs, which usually means the target ratio")
        print("  is asking the network to match a link that was never meant to be")
        print("  matched. Re-examine the parallelism plan before the purchase order.")
    print("")

    if ratio < 0.1:
        print("This node is injection-bound by roughly an order of magnitude or more.")
        print("Two consequences follow, and both are arithmetic rather than opinion:")
        print("  * adding GPUs to this node adds demand and no supply, so per-GPU")
        print("    efficiency falls with every GPU you add;")
        print("  * a single-node bandwidth sweep cannot see any of it, because inside")
        print("    the node the binding constraint is the scale-up link instead.")
        print("Keep the traffic that must stay fast inside the node -- tensor")
        print("parallelism and anything on the token critical path (README 3.2).")
    elif ratio < 1.0:
        print("This node can cross its boundary at a usable fraction of its in-node")
        print("bandwidth. Data-parallel gradient traffic, which is large, once per step")
        print("and overlappable, is what this shape of node is good at.")
    else:
        print("Per-GPU injection meets or exceeds the scale-up figure you supplied.")
        print("Check that figure before trusting the conclusion: it should be the")
        print("bandwidth ONE GPU sees to its neighbours, not the node's aggregate, and")
        print("not a bidirectional specification compared against a unidirectional")
        print("share. That factor-of-two trap is B01 section 3.5's.")
    print("")
    print("The next lever after adapter count is which switch each adapter lands on:")
    print("one adapter per GPU, wired so the n-th GPU of every node shares a rail, is")
    print("the rail-optimized design D05 derives.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
