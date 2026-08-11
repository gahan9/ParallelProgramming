#!/usr/bin/env python3
"""The three all-reduce algorithm families, priced side by side (D02 section 3.3).

Everything printed here is an ANALYTIC MODEL, not a measurement. It evaluates the
step and byte counts derived in the module README:

  recursive doubling  log2(N) steps      m*log2(N) bytes per rank   latency-optimal
  ring                2(N-1) steps       2m(N-1)/N bytes per rank   bandwidth-optimal
  Rabenseifner        2*log2(N) steps    2m(N-1)/N bytes per rank   both, if N is 2^k

Times, with alpha the per-message latency, beta the inverse bandwidth and gamma the
per-byte cost of the local reduction arithmetic:

  recursive doubling  log2(N) * (alpha + m*beta + m*gamma)
  ring                2(N-1)*alpha + 2m(N-1)/N*beta + m(N-1)/N*gamma
  Rabenseifner        2*log2(N)*alpha + 2m(N-1)/N*beta + m(N-1)/N*gamma

The 2m(N-1)/N bandwidth term is a proven lower bound (Patarasuk and Yuan, 2009), so
the ring and Rabenseifner cannot be beaten on bytes moved; they differ only in step
count. Sources are cited in the module's section 8.

Run with no arguments to reproduce the comparison table in README section 3.3 under
its stated illustrative parameters. Standard library only, so:
  python -S -I tools/allreduce_algorithms.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys

BANNER = "ANALYTIC MODEL of published all-reduce algorithms - not a measurement"

ALGORITHMS = ("recursive_doubling", "ring", "rabenseifner")

LABELS = {
    "recursive_doubling": "recursive doubling",
    "ring": "ring",
    "rabenseifner": "Rabenseifner",
}


def _log2_steps(ranks: int) -> int:
    """ceil(log2 N): a non-power-of-two rank count really does pay the extra round."""
    return int(math.ceil(math.log2(ranks)))


def model(algorithm: str, nbytes: float, ranks: int, alpha: float, beta: float,
          gamma: float) -> dict:
    """Steps, bytes moved per rank, and the three cost terms, for one algorithm."""
    m = float(nbytes)
    n = ranks
    if n < 2:
        return dict(algorithm=algorithm, steps=0, bytes_per_rank=0.0,
                    latency_s=0.0, transfer_s=0.0, reduce_s=0.0, seconds=0.0)
    lg = _log2_steps(n)
    frac = (n - 1.0) / n

    if algorithm == "recursive_doubling":
        steps = lg
        moved = m * lg
        reduce_bytes = m * lg
    elif algorithm == "ring":
        steps = 2 * (n - 1)
        moved = 2.0 * m * frac
        reduce_bytes = m * frac
    elif algorithm == "rabenseifner":
        steps = 2 * lg
        moved = 2.0 * m * frac
        reduce_bytes = m * frac
    else:
        raise ValueError("unknown algorithm: %s" % algorithm)

    latency = steps * alpha
    transfer = moved * beta
    reduce_s = reduce_bytes * gamma
    return dict(
        algorithm=algorithm,
        steps=steps,
        bytes_per_rank=moved,
        latency_s=latency,
        transfer_s=transfer,
        reduce_s=reduce_s,
        seconds=latency + transfer + reduce_s,
    )


def compare(nbytes: float, ranks: int, alpha: float, beta: float, gamma: float) -> dict:
    rows = [model(a, nbytes, ranks, alpha, beta, gamma) for a in ALGORITHMS]
    winner = min(rows, key=lambda r: r["seconds"])["algorithm"] if ranks > 1 else None
    return {"bytes": nbytes, "winner": winner, "algorithms": rows}


def crossover_bytes(left: str, right: str, ranks: int, alpha: float, beta: float,
                    gamma: float, hi: float = 1 << 40) -> float:
    """Smallest m at which `right` becomes at least as cheap as `left`.

    Both cost functions are affine and increasing in m, so bisection on the
    difference is exact to the tolerance of the search and needs no algebra per pair.
    Returns nan when the crossing lies outside [0, hi].
    """
    def diff(m):
        return (model(right, m, ranks, alpha, beta, gamma)["seconds"]
                - model(left, m, ranks, alpha, beta, gamma)["seconds"])

    if diff(0.0) <= 0.0:
        return 0.0
    if diff(hi) > 0.0:
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
        return "%.2f ms" % (seconds * 1e3)
    return "%.3f s" % seconds


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analytic comparison of the three all-reduce algorithm families.",
        epilog="Defaults reproduce the README section 3.3 table. Model output, never measured.",
    )
    ap.add_argument("--ranks", "-N", type=int, default=64,
                    help="ranks in the all-reduce (default 64)")
    ap.add_argument("--alpha", type=float, default=2.5,
                    help="per-message latency in microseconds (default 2.5)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--beta", type=float, default=None,
                       help="inverse bandwidth in ns per byte (default 0.04 = 25 GB/s)")
    group.add_argument("--bandwidth", type=float, default=None,
                       help="bandwidth in GB/s (1 GB = 1e9 bytes); sets beta = 1/bw")
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="local reduction cost in ns per byte (default 0, i.e. ignored)")
    ap.add_argument("--sizes", default="4KiB,256MiB",
                    help="comma-separated buffer sizes, suffixes B/KB/MB/GB/KiB/MiB/GiB")
    ap.add_argument("--crossover", action="store_true",
                    help="print the sizes at which the ranking changes")
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
    power_of_two = (args.ranks & (args.ranks - 1)) == 0

    out = {
        "disclaimer": BANNER,
        "ranks": args.ranks,
        "alpha_s": alpha,
        "beta_s_per_byte": beta,
        "gamma_s_per_byte": gamma,
        "power_of_two_ranks": power_of_two,
        "comparisons": [compare(m, args.ranks, alpha, beta, gamma) for m in sizes],
    }
    if args.crossover:
        out["crossovers"] = {
            "ring_beats_recursive_doubling":
                crossover_bytes("recursive_doubling", "ring", args.ranks, alpha, beta, gamma),
            "rabenseifner_beats_recursive_doubling":
                crossover_bytes("recursive_doubling", "rabenseifner", args.ranks, alpha, beta, gamma),
            "ring_beats_rabenseifner":
                crossover_bytes("rabenseifner", "ring", args.ranks, alpha, beta, gamma),
        }

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print("=" * 76)
    print(BANNER)
    print("=" * 76)
    print("ranks N = %d%s   alpha = %g us   beta = %g ns/byte (%.3g GB/s)   gamma = %g ns/byte"
          % (args.ranks, "" if power_of_two else " (NOT a power of two)",
             args.alpha, beta * 1e9, 1e-9 / beta, gamma * 1e9))
    if not power_of_two:
        print("WARNING: Rabenseifner's halving/doubling assumes N = 2^k. Steps are rounded up")
        print("         with ceil(log2 N); a real implementation pays more than that here.")
    print("")

    for comp in out["comparisons"]:
        print("m = %s" % human_bytes(comp["bytes"]))
        print("  %-20s %8s %14s %12s %12s %10s"
              % ("algorithm", "steps", "bytes/rank", "latency", "transfer", "total"))
        print("  " + "-" * 74)
        for row in comp["algorithms"]:
            mark = " <=" if row["algorithm"] == comp["winner"] else ""
            print("  %-20s %8d %14s %12s %12s %10s%s"
                  % (LABELS[row["algorithm"]], row["steps"],
                     human_bytes(row["bytes_per_rank"]),
                     human_time(row["latency_s"]), human_time(row["transfer_s"]),
                     human_time(row["seconds"]), mark))
        if comp["winner"] is not None:
            print("  cheapest: %s" % LABELS[comp["winner"]])
        print("")

    if args.crossover:
        print("Where the ranking changes (model, at N = %d):" % args.ranks)
        for key, label in (
            ("ring_beats_recursive_doubling", "ring overtakes recursive doubling"),
            ("rabenseifner_beats_recursive_doubling", "Rabenseifner overtakes recursive doubling"),
            ("ring_beats_rabenseifner", "ring overtakes Rabenseifner"),
        ):
            x = out["crossovers"][key]
            if math.isnan(x):
                print("  %-42s never below 1 TiB" % label)
            elif x == 0.0:
                print("  %-42s at every size" % label)
            else:
                print("  %-42s at m = %s" % (label, human_bytes(x)))
        print("")

    print("Ring and Rabenseifner move the same, provably minimal, 2m(N-1)/N bytes per rank;")
    print("they differ only in step count, which is why Rabenseifner is the usual default.")
    print(BANNER)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
