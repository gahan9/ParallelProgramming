#!/usr/bin/env python3
"""Amdahl and Gustafson curves, and the rank count where efficiency stops paying.

Everything here is an ANALYTIC MODEL of two closed-form scaling laws. No cluster is
contacted and nothing is measured. Both laws omit the same term -- communication --
so the speedup they report is an upper bound on what a real fabric will deliver.
Section 2.4 of the module README puts that term back and shows what it costs.

For a fixed problem with serial fraction s, parallel fraction p = 1 - s, N ranks:

    amdahl(N)     = 1 / (s + p/N)           ceiling 1/s, approached but never reached
    efficiency(N) = amdahl(N) / N           share of the machine doing useful work
    gustafson(N)  = s + p*N = N - s*(N-1)   the problem grows with the machine

Efficiency, not speedup, is the number a capacity request should carry. Speedup keeps
rising while efficiency collapses, so a table that shows only speedup can make a
mostly-idle cluster look healthy.

Stdlib only, and no site packages required:
    python -S -I tools/scaling_model.py --serial-fraction 0.02 --ranks 1024
"""

from __future__ import annotations

import argparse
import json
import sys

BANNER = "analytic scaling model, not measured: both laws assume communication is free"

# Efficiency targets land on exact ties -- s=0.02 against a 50% target ties at N=51 --
# and binary floating point puts those ties a few ulps on the wrong side. Compare with
# slack so the answer does not depend on the last bit of a division.
TOL = 1e-12

# Bisection needs a bracket. s = 1e-9 against a 50% target breaks near N = 1e9, so a
# ceiling of 2**40 covers every serial fraction anyone will type without overflowing
# into rank counts that mean nothing.
SEARCH_CEILING = 1 << 40


def validate(serial_fraction: float, ranks: int, target_efficiency: float) -> None:
    """Reject inputs the model has no meaning for, with the offending value named."""
    if not 0.0 <= serial_fraction <= 1.0:
        raise ValueError(f"--serial-fraction must be in [0, 1], got {serial_fraction}")
    if ranks < 1:
        raise ValueError(f"--ranks must be at least 1, got {ranks}")
    if not 0.0 < target_efficiency <= 1.0:
        raise ValueError(
            f"--target-efficiency must be in (0, 1], got {target_efficiency}")


def amdahl_speedup(serial_fraction: float, ranks: int) -> float:
    """Fixed-problem speedup: the parallel part divides, the serial part does not."""
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / ranks)


def parallel_efficiency(serial_fraction: float, ranks: int) -> float:
    """Speedup per rank. This is what you bought divided by what you paid for."""
    return amdahl_speedup(serial_fraction, ranks) / ranks


def gustafson_speedup(serial_fraction: float, ranks: int) -> float:
    """Scaled speedup when the problem grows with the machine and time is held fixed.

    Linear in N rather than asymptotic, because the serial part is now a shrinking
    share of a growing job instead of a fixed share of a fixed one.
    """
    return ranks - serial_fraction * (ranks - 1)


def largest_ranks_at_efficiency(serial_fraction: float, target: float) -> int | None:
    """Largest rank count whose Amdahl efficiency still meets `target`, by bisection.

    Efficiency is 1/(s*N + p), strictly decreasing in N whenever s > 0, so bisection on
    the integer N is well defined. Returns None when the target never breaks inside the
    search range; s = 0 is the honest case, where efficiency is 1 at every rank count.

    A closed form exists -- N <= (1/target - 1 + s)/s -- and bisection is used anyway,
    because it survives being handed a cost model that has no closed form, which is
    exactly what happens once the communication term goes back in.
    """
    if parallel_efficiency(serial_fraction, SEARCH_CEILING) >= target - TOL:
        return None
    low, high = 1, SEARCH_CEILING
    while high - low > 1:
        mid = (low + high) // 2
        if parallel_efficiency(serial_fraction, mid) >= target - TOL:
            low = mid
        else:
            high = mid
    return low


def rank_ladder(max_ranks: int) -> list[int]:
    """Powers of two up to max_ranks, with max_ranks itself appended if it is not one."""
    ladder = []
    ranks = 1
    while ranks <= max_ranks:
        ladder.append(ranks)
        ranks *= 2
    if ladder[-1] != max_ranks:
        ladder.append(max_ranks)
    return ladder


def curve(serial_fraction: float, max_ranks: int) -> list[dict]:
    rows = []
    for ranks in rank_ladder(max_ranks):
        scaled = gustafson_speedup(serial_fraction, ranks)
        rows.append(
            dict(
                ranks=ranks,
                amdahl_speedup=amdahl_speedup(serial_fraction, ranks),
                efficiency=parallel_efficiency(serial_fraction, ranks),
                gustafson_speedup=scaled,
                gustafson_efficiency=scaled / ranks,
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Amdahl and Gustafson curves with the efficiency column, "
                    "plus the rank count where efficiency falls through a target.")
    parser.add_argument("--serial-fraction", type=float, default=0.02,
                        help="fraction s of the run that cannot parallelize (default 0.02)")
    parser.add_argument("--ranks", type=int, default=1024,
                        help="largest rank count to tabulate (default 1024)")
    parser.add_argument("--target-efficiency", type=float, default=0.5,
                        help="efficiency floor to solve for, as a fraction (default 0.5)")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        validate(args.serial_fraction, args.ranks, args.target_efficiency)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    serial = args.serial_fraction
    rows = curve(serial, args.ranks)
    knee = largest_ranks_at_efficiency(serial, args.target_efficiency)
    ceiling = None if serial == 0.0 else 1.0 / serial
    final = rows[-1]

    if args.json:
        print(json.dumps(dict(banner=BANNER, serial_fraction=serial, ranks=args.ranks,
                              target_efficiency=args.target_efficiency,
                              amdahl_ceiling=ceiling, target_ranks=knee, curve=rows),
                         indent=2))
        return 0

    print(BANNER)
    print(f"serial fraction s = {serial:g}   parallel fraction p = {1.0 - serial:g}"
          f"   target efficiency {args.target_efficiency * 100:g}%")
    if ceiling is None:
        print("Amdahl ceiling: unbounded, because s = 0 makes speedup exactly N")
    else:
        print(f"Amdahl ceiling 1/s = {ceiling:.1f}x -- no rank count ever beats it")
    print()

    header = (f"{'ranks':>10}{'amdahl':>11}{'efficiency':>13}"
              f"{'gustafson':>12}{'scaled eff':>13}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['ranks']:>10}{row['amdahl_speedup']:>10.2f}x"
              f"{row['efficiency'] * 100:>12.1f}%"
              f"{row['gustafson_speedup']:>11.1f}x"
              f"{row['gustafson_efficiency'] * 100:>12.1f}%")
    print("-" * len(header))

    if knee is None:
        print(f"efficiency never falls below {args.target_efficiency * 100:g}% in this model")
    else:
        print(f"efficiency holds at or above {args.target_efficiency * 100:g}% up to "
              f"N = {knee} ranks, and falls below it at N = {knee + 1}")
    print(f"at N = {final['ranks']}: a {final['amdahl_speedup']:.1f}x speedup is "
          f"{final['efficiency'] * 100:.1f}% efficiency -- the remaining "
          f"{100.0 - final['efficiency'] * 100:.1f}% is capacity you rented and did "
          f"not use")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
