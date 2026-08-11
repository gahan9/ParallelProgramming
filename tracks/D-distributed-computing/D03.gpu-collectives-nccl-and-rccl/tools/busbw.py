#!/usr/bin/env python3
"""Convert a MEASURED collective timing into algbw and busbw (D03 section 2.4).

This tool does not model anything. You give it bytes and a time that you or a
benchmark measured; it applies the collective's traffic factor and shows the
arithmetic. That is why there is no model disclaimer in the output: the only
numbers printed are the ones you supplied and the algebra applied to them.

  algbw = m / t                     application-visible throughput
  busbw = algbw * f(N)              bytes that crossed the wire, per second

  collective        bytes each rank moves (ring)   f(N)
  all-reduce        2m(N-1)/N                      2(N-1)/N
  reduce-scatter    m(N-1)/N                       (N-1)/N
  all-gather        m(N-1)/N                       (N-1)/N
  all-to-all        m(N-1)/N                       (N-1)/N
  broadcast         m                              1
  reduce            m                              1

Compare busbw against a link rate and algbw against your application's step
budget. Quoting one where the other is meant is the most common way a collective
benchmark misleads, and for all-reduce the error always flatters the suspicion
that something is broken.

Standard library only:
  python -S -I tools/busbw.py --collective allreduce --ranks 4 \
      --bytes 1073741824 --time-ms 40
"""

from __future__ import annotations

import argparse
import json
import math
import sys

PROGRAM = "busbw"

# f(N) per collective, straight out of README section 2.2's byte counts. Each
# entry is (traffic factor, bytes each rank moves) as callables of N.
FACTORS = {
    "allreduce": lambda n: 2.0 * (n - 1) / n,
    "reducescatter": lambda n: (n - 1) / n,
    "allgather": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}

ALIASES = {
    "allreduce": "allreduce",
    "reducescatter": "reducescatter",
    "scatterreduce": "reducescatter",
    "allgather": "allgather",
    "alltoall": "alltoall",
    "broadcast": "broadcast",
    "bcast": "broadcast",
    "reduce": "reduce",
    "gather": None,
    "scatter": None,
}

SUFFIXES = {
    "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3,
}


def die(message: str) -> None:
    """One line on stderr, non-zero exit, never a traceback."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def traffic_factor(collective: str, ranks: int) -> float:
    """f(N) for one collective, with a single decision about N == 1.

    The tabulated factors are derived for N >= 2. At N == 1 no byte crosses any
    wire for ANY collective, including broadcast, whose factor is otherwise the
    constant 1. Returning 0 there keeps one rule for every collective and refuses
    to claim a fabric that was not involved; returning algbw would quote local
    memory bandwidth as interconnect bandwidth, which is the exact mistake this
    tool exists to prevent.
    """
    if ranks <= 1:
        return 0.0
    return FACTORS[collective](float(ranks))


def normalize(name: str) -> str:
    """Fold 'all_reduce', 'All-Reduce' and 'allreduce' onto one key."""
    key = name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key in ALIASES and ALIASES[key] is not None:
        return ALIASES[key]
    if key in ("gather", "scatter"):
        die("%s is a rooted collective whose traffic factor depends on the root's "
            "link, not on a ring; this tool covers %s"
            % (key, ", ".join(sorted(FACTORS))))
    die("unknown collective %r; expected one of %s"
        % (name, ", ".join(sorted(FACTORS))))
    return ""  # unreachable; die() raises


def parse_size(text: str) -> float:
    """Accept 4096, 4KiB, 256MiB, 1GB. Raises ValueError with a usable message."""
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
    if value < 0 or not math.isfinite(value):
        raise ValueError("size must be finite and non-negative: %s" % text)
    return value


def human_bytes(nbytes: float) -> str:
    if nbytes < 1024:
        return "%.0f B" % nbytes
    for unit, scale in (("KiB", 1024.0), ("MiB", 1024.0**2), ("GiB", 1024.0**3)):
        if nbytes < scale * 1024 or unit == "GiB":
            return "%.3g %s" % (nbytes / scale, unit)
    return "%.0f B" % nbytes


def convert(collective: str, ranks: int, nbytes: float, seconds: float) -> dict:
    """algbw and busbw from a measured (bytes, seconds) pair.

    ranks == 1 is accepted and gives a factor of 0. That is the correct answer
    rather than an error: with one rank nothing crosses a wire, so the bus
    bandwidth of the fabric really is zero and a single-rank number measures local
    memory, not the interconnect. The tool says so rather than hiding it behind an
    exception, because the mistake in the field is quoting a one-GPU benchmark as
    a fabric figure.
    """
    if ranks < 1:
        die("--ranks must be at least 1, got %d" % ranks)
    if seconds <= 0 or not math.isfinite(seconds):
        die("the measured time must be positive and finite; a collective cannot "
            "take zero seconds")
    if nbytes <= 0 or not math.isfinite(nbytes):
        die("--bytes must be positive and finite")
    factor = traffic_factor(collective, ranks)
    algbw = nbytes / seconds
    wire_bytes = nbytes * factor
    return {
        "collective": collective,
        "ranks": ranks,
        "bytes": nbytes,
        "seconds": seconds,
        "factor": factor,
        "algbw_bytes_per_s": algbw,
        "busbw_bytes_per_s": algbw * factor,
        "wire_bytes_per_rank": wire_bytes,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Convert a measured collective timing into algbw and busbw.",
        epilog="Measured input, measured output: this tool models nothing. "
               "1 GB = 1e9 bytes, matching nccl-tests and rccl-tests.",
    )
    ap.add_argument("--collective", "-c", default="allreduce",
                    help="allreduce, reducescatter, allgather, alltoall, "
                         "broadcast or reduce (default allreduce)")
    ap.add_argument("--ranks", "-N", type=int, required=True,
                    help="ranks in the collective; f(N) moves with it, so a "
                         "busbw without its rank count is uninterpretable")
    ap.add_argument("--bytes", "-b", default="",
                    help="message size, with an optional B/KB/MB/GB/KiB/MiB/GiB "
                         "suffix")
    time_group = ap.add_mutually_exclusive_group()
    time_group.add_argument("--time-ms", type=float, default=None,
                            help="measured per-iteration time in milliseconds")
    time_group.add_argument("--time-us", type=float, default=None,
                            help="measured per-iteration time in microseconds")
    time_group.add_argument("--algbw-gbs", type=float, default=None,
                            help="skip the timing: convert a reported algbw "
                                 "(GB/s) into busbw")
    time_group.add_argument("--busbw-gbs", type=float, default=None,
                            help="invert: convert a reported busbw (GB/s) back "
                                 "into algbw and a time")
    ap.add_argument("--link-gbs", type=float, default=None,
                    help="per-direction link rate in GB/s to compare busbw "
                         "against; prints the fraction of the wire in use")
    ap.add_argument("--table", action="store_true",
                    help="print f(N) for every collective at this rank count")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    collective = normalize(args.collective)
    ranks = args.ranks
    if ranks < 1:
        ap.error("--ranks must be at least 1")
    if args.link_gbs is not None and (args.link_gbs <= 0
                                      or not math.isfinite(args.link_gbs)):
        ap.error("--link-gbs must be positive")

    if args.table:
        rows = [{"collective": name, "factor": traffic_factor(name, ranks)}
                for name in sorted(FACTORS)]
        if args.json:
            print(json.dumps({"ranks": ranks, "factors": rows}, indent=2))
        else:
            print("traffic factor f(N) at N = %d" % ranks)
            print("-" * 40)
            for row in rows:
                print("  %-15s %8.4f" % (row["collective"], row["factor"]))
            print("-" * 40)
            print("busbw = algbw * f(N). For all-reduce f > 1, so busbw exceeds")
            print("algbw; for reduce-scatter f < 1, so it is smaller.")
        return 0

    if not args.bytes:
        ap.error("--bytes is required (or use --table for the factors alone)")
    try:
        nbytes = parse_size(args.bytes)
    except ValueError as exc:
        ap.error("bad --bytes: %s" % exc)

    supplied = [args.time_ms, args.time_us, args.algbw_gbs, args.busbw_gbs]
    if all(x is None for x in supplied):
        ap.error("give one of --time-ms, --time-us, --algbw-gbs or --busbw-gbs")

    factor = traffic_factor(collective, ranks)

    if args.time_ms is not None:
        seconds = args.time_ms * 1e-3
        source = "measured time %g ms" % args.time_ms
    elif args.time_us is not None:
        seconds = args.time_us * 1e-6
        source = "measured time %g us" % args.time_us
    elif args.algbw_gbs is not None:
        if args.algbw_gbs <= 0 or not math.isfinite(args.algbw_gbs):
            ap.error("--algbw-gbs must be positive")
        seconds = nbytes / (args.algbw_gbs * 1e9)
        source = "reported algbw %g GB/s" % args.algbw_gbs
    else:
        if args.busbw_gbs <= 0 or not math.isfinite(args.busbw_gbs):
            ap.error("--busbw-gbs must be positive")
        if factor <= 0:
            die("cannot invert a busbw at N = 1: the traffic factor for %s is 0, "
                "so no time reproduces a nonzero busbw" % collective)
        seconds = nbytes * factor / (args.busbw_gbs * 1e9)
        source = "reported busbw %g GB/s" % args.busbw_gbs

    result = convert(collective, ranks, nbytes, seconds)
    result["input"] = source
    if args.link_gbs is not None:
        result["link_bytes_per_s"] = args.link_gbs * 1e9
        result["wire_utilization"] = result["busbw_bytes_per_s"] / (args.link_gbs * 1e9)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 72)
    print("CONVERTED from %s; nothing here is modelled" % source)
    print("=" * 72)
    print("collective      %s over %d ranks" % (collective, ranks))
    print("message         %s (%.0f bytes)" % (human_bytes(nbytes), nbytes))
    print("time            %.3f us per iteration" % (result["seconds"] * 1e6))
    print("")
    print("algbw = m / t              %10.2f GB/s   application-visible"
          % (result["algbw_bytes_per_s"] / 1e9))
    print("f(N)  = traffic factor     %10.4f        for %s at N = %d"
          % (factor, collective, ranks))
    print("busbw = algbw * f(N)       %10.2f GB/s   comparable to a link rate"
          % (result["busbw_bytes_per_s"] / 1e9))
    print("bytes on the wire per rank %10s        = %.4f x the buffer"
          % (human_bytes(result["wire_bytes_per_rank"]), factor))
    if ranks == 1:
        print("")
        print("N = 1, so f(N) = 0 for every collective: nothing crossed a wire and")
        print("busbw is 0 by definition. A single-rank collective benchmark measures")
        print("local memory, not the fabric. Do not quote it as a fabric number.")
    if args.link_gbs is not None:
        pct = result["wire_utilization"] * 100.0
        print("")
        print("against a %g GB/s per-direction link: %.1f%% of the wire"
              % (args.link_gbs, pct))
        if pct > 100.0:
            print("  Above 100%: either the link figure is per-direction where the")
            print("  measurement is bidirectional, the rank count is wrong, or the")
            print("  path is not the one you think it is. Check before celebrating.")
        elif pct > 85.0:
            print("  At this level the collective is bandwidth-bound and near the")
            print("  wire. There is nothing left to win in the algorithm.")
    print("")
    print("Whichever number you quote, quote the rank count with it: f(N) moves")
    print("with N, so a busbw on its own cannot be checked by anyone.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:  # `| head` should not print a traceback
        sys.exit(0)
