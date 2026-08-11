#!/usr/bin/env python3
"""Alpha-beta-gamma cost model for the MPI collective catalogue (D02 section 3.2).

Everything printed here is an ANALYTIC MODEL, not a measurement. It evaluates the
closed-form cost expressions that Thakur, Rabenseifner and Gropp (2005) derived for
MPICH's collective algorithms, under the same two assumptions their paper makes:

  * N is a power of two (the halving/doubling algorithms need extra steps otherwise)
  * the network is contention-free, so beta is a constant

Parameters, all of which you should measure rather than guess (module lab 2):

  alpha  per-message startup latency,          seconds
  beta   inverse bandwidth,                    seconds per byte
  gamma  local reduction arithmetic per byte,  seconds per byte

Message-size convention, restated because mixing it up invalidates every formula:
Bcast/Reduce/Allreduce/Scan take m as the operand buffer size, identical on every
rank. Allgather takes m as the ASSEMBLED result (each rank contributes m/N).
Reduce_scatter takes m as the input, of which each rank keeps m/N. Alltoall takes m
as the total each rank sends.

Standard library only: runs under `python -S -I tools/collective_cost.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

BANNER = "ANALYTIC MODEL of published collective algorithms - not a measurement"

# Each entry: (semantics, small-m closed form, large-m closed form).
# steps() returns (latency_terms, beta_bytes, gamma_bytes) so the table below stays
# readable as arithmetic rather than as prose.
OPS = (
    "barrier",
    "bcast",
    "reduce",
    "allreduce",
    "allgather",
    "reduce_scatter",
    "alltoall",
    "scan",
)

SEMANTICS = {
    "barrier": "no data; no rank leaves until all arrive",
    "bcast": "root's buffer replicated to all ranks",
    "reduce": "elementwise op across ranks, result on root only",
    "allreduce": "reduce, result on every rank",
    "allgather": "every rank's contribution concatenated on every rank",
    "reduce_scatter": "reduce, then rank i keeps only block i",
    "alltoall": "full transpose: distinct data from every rank to every rank",
    "scan": "prefix reduction: rank i gets the op over ranks 0..i",
}

# The algorithm each closed form describes, so the output names its own source.
ALGORITHMS = {
    "barrier": ("dissemination / recursive doubling", "same; there is no data term"),
    "bcast": ("binomial tree", "scatter then ring allgather"),
    "reduce": ("binomial tree", "reduce-scatter then gather"),
    "allreduce": ("recursive doubling", "Rabenseifner: reduce-scatter + allgather"),
    "allgather": ("recursive doubling", "ring"),
    "reduce_scatter": ("recursive halving", "ring"),
    "alltoall": ("Bruck", "pairwise exchange"),
    "scan": ("recursive doubling", "same shape; rarely bandwidth-critical"),
}


def _log2_steps(ranks: int) -> int:
    """Steps in a halving/doubling algorithm.

    ceil() rather than log2() so a non-power-of-two rank count at least pays for the
    extra round it really costs. The published formulas assume a power of two, so the
    caller is warned separately when N is not one.
    """
    return int(math.ceil(math.log2(ranks)))


def cost(op: str, nbytes: float, ranks: int, alpha: float, beta: float,
         gamma: float, regime: str) -> float:
    """Modelled seconds for one collective. regime is 'small' or 'large'."""
    n = ranks
    m = float(nbytes)
    if n < 2:
        return 0.0
    lg = _log2_steps(n)
    frac = (n - 1.0) / n          # the (N-1)/N that shows up in every ring algorithm
    small = regime == "small"

    if op == "barrier":
        return lg * alpha
    if op == "bcast":
        if small:
            return lg * (alpha + m * beta)
        return (lg + n - 1) * alpha + 2.0 * m * beta * frac
    if op == "reduce":
        if small:
            return lg * (alpha + m * beta + m * gamma)
        return (lg + n - 1) * alpha + 2.0 * m * beta * frac + m * gamma * frac
    if op == "allreduce":
        if small:
            return lg * (alpha + m * beta + m * gamma)
        return 2.0 * lg * alpha + 2.0 * m * beta * frac + m * gamma * frac
    if op == "allgather":
        if small:
            return lg * alpha + m * beta * frac
        return (n - 1) * alpha + m * beta * frac
    if op == "reduce_scatter":
        if small:
            return lg * alpha + m * beta * frac + m * gamma * frac
        return (n - 1) * alpha + m * beta * frac + m * gamma * frac
    if op == "alltoall":
        if small:
            return lg * alpha + (m / 2.0) * lg * beta
        return (n - 1) * alpha + m * beta
    if op == "scan":
        return lg * (alpha + m * beta + m * gamma)
    raise ValueError("unknown collective: %s" % op)


def best(op: str, nbytes: float, ranks: int, alpha: float, beta: float, gamma: float):
    """What a decision-tree implementation would pick: the cheaper of the two."""
    s = cost(op, nbytes, ranks, alpha, beta, gamma, "small")
    l = cost(op, nbytes, ranks, alpha, beta, gamma, "large")
    return (s, "small-message", l) if s <= l else (l, "large-message", s)


def crossover_bytes(op: str, ranks: int, alpha: float, beta: float, gamma: float,
                    hi: float = 1 << 40) -> float:
    """Smallest m at which the large-m algorithm becomes the cheaper choice.

    Bisection rather than algebra: the two expressions differ per collective, and one
    numerical routine that works for all of them is less code to get wrong than eight
    hand-solved inequalities. Returns nan when the crossing is outside [0, hi].
    """
    def diff(m):
        return (cost(op, m, ranks, alpha, beta, gamma, "large")
                - cost(op, m, ranks, alpha, beta, gamma, "small"))

    if diff(0.0) <= 0.0:          # large-m form already wins at zero bytes
        return 0.0
    if diff(hi) > 0.0:            # never wins in the range we searched
        return float("nan")
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if diff(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


SUFFIXES = {
    "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3,
}


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


def human_time(seconds: float) -> str:
    if seconds < 1e-3:
        return "%.2f us" % (seconds * 1e6)
    if seconds < 1.0:
        return "%.3f ms" % (seconds * 1e3)
    return "%.3f s" % seconds


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analytic alpha-beta-gamma cost model for the MPI collectives.",
        epilog="Every number printed is a model. Measure alpha and beta with lab 2.",
    )
    ap.add_argument("--ranks", "-N", type=int, default=64,
                    help="ranks in the communicator (default 64)")
    ap.add_argument("--alpha", type=float, default=2.5,
                    help="per-message latency in microseconds (default 2.5)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--beta", type=float, default=None,
                       help="inverse bandwidth in ns per byte (default 0.04 = 25 GB/s)")
    group.add_argument("--bandwidth", type=float, default=None,
                       help="bandwidth in GB/s (1 GB = 1e9 bytes); sets beta = 1/bw")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="local reduction cost in ns per byte (default 0)")
    ap.add_argument("--sizes", default="1KiB,64KiB,1MiB,64MiB,256MiB",
                    help="comma-separated message sizes, suffixes B/KB/MB/GB/KiB/MiB/GiB")
    ap.add_argument("--op", default="all", choices=("all",) + OPS,
                    help="one collective, or all of them (default all)")
    ap.add_argument("--crossover", action="store_true",
                    help="also print where the large-m algorithm overtakes the small-m one")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.ranks < 1:
        ap.error("--ranks must be at least 1")
    if not math.isfinite(args.alpha) or args.alpha < 0:
        ap.error("--alpha must be finite and non-negative")
    if args.bandwidth is not None:
        if args.bandwidth <= 0 or not math.isfinite(args.bandwidth):
            ap.error("--bandwidth must be positive")
        beta = 1.0 / (args.bandwidth * 1e9)
    else:
        beta_ns = 0.04 if args.beta is None else args.beta
        if beta_ns <= 0 or not math.isfinite(beta_ns):
            ap.error("--beta must be positive")
        beta = beta_ns * 1e-9
    if args.gamma < 0 or not math.isfinite(args.gamma):
        ap.error("--gamma must be finite and non-negative")

    try:
        sizes = [parse_size(s) for s in args.sizes.split(",") if s.strip()]
    except ValueError as exc:
        ap.error("bad --sizes: %s" % exc)
    if not sizes:
        ap.error("--sizes parsed to an empty list")

    alpha = args.alpha * 1e-6
    gamma = args.gamma * 1e-9
    ops = OPS if args.op == "all" else (args.op,)
    power_of_two = args.ranks >= 1 and (args.ranks & (args.ranks - 1)) == 0

    out = {
        "disclaimer": BANNER,
        "ranks": args.ranks,
        "alpha_s": alpha,
        "beta_s_per_byte": beta,
        "gamma_s_per_byte": gamma,
        "power_of_two_ranks": power_of_two,
        "bandwidth_gb_s": 1e-9 / beta,
        "rows": [],
    }
    for op in ops:
        for nbytes in sizes:
            seconds, which, other = best(op, nbytes, args.ranks, alpha, beta, gamma)
            row = {
                "op": op,
                "bytes": nbytes,
                "seconds": seconds,
                "algorithm_class": which,
                "algorithm": ALGORITHMS[op][0 if which == "small-message" else 1],
                "other_seconds": other,
            }
            if args.crossover:
                row["crossover_bytes"] = crossover_bytes(op, args.ranks, alpha, beta, gamma)
            out["rows"].append(row)

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print("=" * 78)
    print(BANNER)
    print("=" * 78)
    print("ranks N = %d%s   alpha = %g us   beta = %g ns/byte (%.3g GB/s)   gamma = %g ns/byte"
          % (args.ranks, "" if power_of_two else " (NOT a power of two)",
             args.alpha, beta * 1e9, 1e-9 / beta, gamma * 1e9))
    if not power_of_two:
        print("WARNING: the published formulas assume a power-of-two rank count.")
        print("         Steps are rounded up with ceil(log2 N); treat this as a shape, not a value.")
    print("m follows the section 3.2 convention: operand size for reduce-like collectives,")
    print("assembled result for allgather, total sent per rank for alltoall.")
    print("")
    print("%-15s %10s %12s %14s  %s" % ("collective", "m", "modelled", "regime", "algorithm"))
    print("-" * 78)
    last_op = None
    for row in out["rows"]:
        if last_op is not None and row["op"] != last_op:
            print("")
        last_op = row["op"]
        print("%-15s %10s %12s %14s  %s"
              % (row["op"], human_bytes(row["bytes"]), human_time(row["seconds"]),
                 row["algorithm_class"], row["algorithm"]))
    print("-" * 78)

    if args.crossover:
        print("")
        print("Where the large-m algorithm overtakes the small-m one (model, at N = %d):"
              % args.ranks)
        for op in ops:
            x = crossover_bytes(op, args.ranks, alpha, beta, gamma)
            if math.isnan(x):
                note = "never below 1 TiB (latency-optimal algorithm wins everywhere)"
            elif x == 0.0:
                note = "at every size (the bandwidth-optimal algorithm always wins)"
            else:
                note = "at m = %s" % human_bytes(x)
            print("  %-15s %s" % (op, note))

    print("")
    print("semantics:")
    for op in ops:
        print("  %-15s %s" % (op, SEMANTICS[op]))
    print("")
    print("Where this model stops being useful, stated because the output does not show it:")
    print("  * For allgather and reduce_scatter the two published forms share the SAME")
    print("    bandwidth term, m*beta*(N-1)/N, so the logarithmic-step form is cheaper at")
    print("    every size and this model reports no crossover. Real implementations still")
    print("    switch to the ring at large m, for reasons the alpha-beta model does not")
    print("    represent: network contention, and recursive doubling's non-contiguous")
    print("    buffer handling. Read that as missing from the model, not absent in reality.")
    print("  * Scan's two forms are identical here, so its regime label carries no")
    print("    information.")
    print("  * A loaded fabric does not deliver a constant beta, and a non-power-of-two N")
    print("    forces extra steps. Both assumptions come from the source paper.")
    print("")
    print(BANNER)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:                       # `| head` should not print a traceback
        sys.exit(0)
