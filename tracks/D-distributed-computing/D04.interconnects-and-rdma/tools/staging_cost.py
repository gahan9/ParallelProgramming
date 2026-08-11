#!/usr/bin/env python3
"""Staged-through-host versus direct device-to-device transfer, as an analytic model.

Everything this tool prints is an ANALYTIC MODEL of README section 2.3. No device is
opened, no fabric is contacted, and nothing is measured. It exists so that the
staged-versus-direct comparison can be made with arithmetic instead of with folklore,
and so that the answer comes out as "it depends on which link is narrow" rather than as
a confident multiplier.

The two paths, counted segment by segment (README 2.3):

    staged   GPU -> host, host -> pinned/NIC, wire, NIC -> host, host -> GPU
             four PCIe traversals, two per side, plus the wire
    direct   NIC reads GPU memory, wire, remote NIC writes GPU memory
             two PCIe traversals, one per side, plus the wire

Two bounds, because they answer different questions:

    serial     every segment waits for the one before it. This is the honest bound for a
               single un-chunked transfer, and it is what a small message actually pays.
    pipelined  the transfer is chunked and the segments overlap, so the cost is set by
               the busiest single resource. The staged path puts 2m through the local
               PCIe complex on each side, so its bound is max(2m/pcie, m/net); the
               direct path's is max(m/pcie, m/net).

The pipelined pair is where the interesting result lives: they are EQUAL whenever PCIe
is at least twice the wire, which is why "GPUDirect gives you 2x" is usually wrong and
why the defensible claim is about latency, host memory and interference instead.

What this model deliberately does not have: a latency term, host memory bandwidth as a
finite resource, CPU cost, chunk size, or contention from anything else on the node.
Those are exactly the costs the direct path removes, so this model UNDERSTATES the case
for the direct path. It is a bandwidth argument only, and it says so.

Stdlib only:
    python -S -I tools/staging_cost.py --bytes 1073741824 --pcie-gbs 20 --net-gbs 10
"""

from __future__ import annotations

import argparse
import json
import sys

PROGRAM = "staging_cost"
BANNER = ("ANALYTIC MODEL, NOT A MEASUREMENT: bandwidth bounds only, with no latency "
          "term,\nno host-memory ceiling, no CPU cost and no contention. Section 4's "
          "labs measure your case.")

GB = 1e9

# PCIe traversals per path, counting both sides. README 2.3's table, condensed.
STAGED_PCIE_HOPS = 4
DIRECT_PCIE_HOPS = 2
# Host memory traffic the staged path generates, in multiples of the message size.
# README 2.3: "roughly 4m versus zero".
STAGED_HOST_TRAFFIC_MULTIPLE = 4.0


def validate(nbytes, pcie_bps, net_bps):
    """Reject inputs the model has no meaning for, naming the offending value."""
    if nbytes <= 0:
        raise ValueError("message size must be positive, got %g bytes" % nbytes)
    if pcie_bps <= 0:
        raise ValueError("PCIe bandwidth must be positive, got %g B/s" % pcie_bps)
    if net_bps <= 0:
        raise ValueError("network bandwidth must be positive, got %g B/s" % net_bps)


def staged_cost(nbytes, pcie_bps, net_bps):
    """Return (serial_seconds, pipelined_seconds) for the host-staged path."""
    validate(nbytes, pcie_bps, net_bps)
    pcie_leg = nbytes / pcie_bps
    wire = nbytes / net_bps
    serial = STAGED_PCIE_HOPS * pcie_leg + wire
    pipelined = max(2.0 * pcie_leg, wire)
    return serial, pipelined


def direct_cost(nbytes, pcie_bps, net_bps):
    """Return (serial_seconds, pipelined_seconds) for the device-to-device path."""
    validate(nbytes, pcie_bps, net_bps)
    pcie_leg = nbytes / pcie_bps
    wire = nbytes / net_bps
    serial = DIRECT_PCIE_HOPS * pcie_leg + wire
    pipelined = max(pcie_leg, wire)
    return serial, pipelined


def crossover_pcie_bps(net_bps):
    """PCIe bandwidth below which the staged path becomes bandwidth-limited.

    The staged path's pipelined bound is max(2m/pcie, m/net). It stops being set by the
    wire, and starts being set by the doubled PCIe demand, exactly at pcie = 2*net.
    Above that, staging costs latency, host memory bandwidth and a CPU core -- but not
    throughput. Below it, staging is directly and arithmetically slower.
    """
    if net_bps <= 0:
        raise ValueError("network bandwidth must be positive, got %g B/s" % net_bps)
    return 2.0 * net_bps


def human_bytes(nbytes):
    if nbytes < 1024:
        return "%.0f B" % nbytes
    for unit, scale in (("KiB", 1024.0), ("MiB", 1024.0 ** 2), ("GiB", 1024.0 ** 3)):
        if nbytes < scale * 1024 or unit == "GiB":
            return "%.3g %s" % (nbytes / scale, unit)
    return "%.0f B" % nbytes


def seconds(value):
    if value >= 1.0:
        return "%.4f s" % value
    if value >= 1e-3:
        return "%.3f ms" % (value * 1e3)
    return "%.3f us" % (value * 1e6)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Model the staged-through-host and direct device-to-device paths "
                    "for one transfer, under serial and pipelined bounds.",
        epilog="Every number printed is a model, never a measurement.")
    parser.add_argument("--bytes", type=float, default=1073741824.0,
                        help="message size in bytes (default 1073741824, i.e. 1 GiB)")
    parser.add_argument("--pcie-gbs", type=float, default=20.0,
                        help="sustained PCIe bandwidth in GB/s, 1 GB = 1e9 B "
                             "(default 20)")
    parser.add_argument("--net-gbs", type=float, default=10.0,
                        help="sustained network bandwidth per adapter in GB/s "
                             "(default 10)")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    nbytes = args.bytes
    pcie_bps = args.pcie_gbs * GB
    net_bps = args.net_gbs * GB

    try:
        validate(nbytes, pcie_bps, net_bps)
    except ValueError as exc:
        sys.stderr.write("%s: %s\n" % (PROGRAM, exc))
        return 2

    staged_serial, staged_pipe = staged_cost(nbytes, pcie_bps, net_bps)
    direct_serial, direct_pipe = direct_cost(nbytes, pcie_bps, net_bps)
    knee = crossover_pcie_bps(net_bps)

    # The same model at four PCIe widths, so the shape is visible without re-running.
    sweep = []
    for factor in (0.5, 1.0, 2.0, 4.0):
        probe = knee * factor
        _s_serial, s_pipe = staged_cost(nbytes, probe, net_bps)
        _d_serial, d_pipe = direct_cost(nbytes, probe, net_bps)
        sweep.append({"pcie_gbs": probe / GB, "staged_pipelined_s": s_pipe,
                      "direct_pipelined_s": d_pipe, "ratio": s_pipe / d_pipe})

    result = {
        "banner": BANNER.replace("\n", " "),
        "bytes": nbytes,
        "pcie_bytes_per_s": pcie_bps,
        "net_bytes_per_s": net_bps,
        "staged": {"serial_s": staged_serial, "pipelined_s": staged_pipe,
                   "pcie_traversals": STAGED_PCIE_HOPS,
                   "host_memory_bytes": STAGED_HOST_TRAFFIC_MULTIPLE * nbytes},
        "direct": {"serial_s": direct_serial, "pipelined_s": direct_pipe,
                   "pcie_traversals": DIRECT_PCIE_HOPS, "host_memory_bytes": 0.0},
        "serial_ratio": staged_serial / direct_serial,
        "pipelined_ratio": staged_pipe / direct_pipe,
        "crossover_pcie_bytes_per_s": knee,
        "pcie_is_narrow": pcie_bps < knee,
        "sweep": sweep,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(BANNER)
    print("=" * 78)
    print("message %s over PCIe %.3g GB/s and a wire at %.3g GB/s"
          % (human_bytes(nbytes), args.pcie_gbs, args.net_gbs))
    print("=" * 78)
    print("")

    header = "%-10s %8s %14s %16s %20s" % (
        "path", "PCIe hops", "serial bound", "pipelined bound", "host memory moved")
    print(header)
    print("-" * len(header))
    print("%-10s %8d %14s %16s %20s"
          % ("staged", STAGED_PCIE_HOPS, seconds(staged_serial), seconds(staged_pipe),
             human_bytes(STAGED_HOST_TRAFFIC_MULTIPLE * nbytes)))
    print("%-10s %8d %14s %16s %20s"
          % ("direct", DIRECT_PCIE_HOPS, seconds(direct_serial), seconds(direct_pipe),
             "0 B"))
    print("-" * len(header))
    print("staged / direct    %14.2fx %15.2fx"
          % (staged_serial / direct_serial, staged_pipe / direct_pipe))
    print("")

    print("crossover: the staged path stops being bandwidth-limited at a PCIe bandwidth")
    print("of %.3g GB/s, which is twice the wire. Yours is %.3g GB/s, so PCIe is %s."
          % (knee / GB, args.pcie_gbs, "the narrow link here" if pcie_bps < knee
             else "wide enough"))
    print("")

    print("the same model across PCIe widths, pipelined bound only:")
    print("%12s %18s %18s %10s" % ("PCIe GB/s", "staged", "direct", "ratio"))
    print("-" * 62)
    for row in sweep:
        print("%12.3g %18s %18s %9.2fx"
              % (row["pcie_gbs"], seconds(row["staged_pipelined_s"]),
                 seconds(row["direct_pipelined_s"]), row["ratio"]))
    print("-" * 62)
    print("")

    if pcie_bps >= knee:
        print("Read the pipelined column before quoting a speedup. At this PCIe width the")
        print("two paths have the SAME bandwidth bound, so a well-pipelined staged")
        print("transfer will land close to the direct one, and a claim of 'GPUDirect is")
        print("k times faster' would not survive a measurement. What the direct path")
        print("still buys you here is real but is not in this model: two fewer")
        print("store-and-forward hops of latency per side, %s of host memory traffic that"
              % human_bytes(STAGED_HOST_TRAFFIC_MULTIPLE * nbytes))
        print("never happens, a CPU core left alone, and no interference with the data")
        print("loader or with other ranks on the node.")
    else:
        print("Here PCIe IS the narrow link, so staging costs throughput directly: you")
        print("doubled demand on the narrowest resource in the path. This is the case")
        print("where the direct path shows up as a bandwidth number, and it is also the")
        print("case README 3.7 says to check first -- lane count, negotiated generation")
        print("and a shared switch uplink all land you here without any error message.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
