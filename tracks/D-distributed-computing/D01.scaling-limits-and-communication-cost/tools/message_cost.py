#!/usr/bin/env python3
"""Hockney alpha-beta cost of one message, and the size where the two terms swap places.

ANALYTIC MODEL, nothing measured: one message on an idle link. No contention, no
congestion, no injection-rate limit, no protocol switch at some size threshold. Real
fabrics have all four, so read this for the shape of the curve, not for a time you can
quote in a report.

    T(m) = alpha + m * beta,   beta = 1 / bandwidth

The output that matters is not a time, it is the crossover where the two terms are
equal:

    m* = alpha / beta = alpha * bandwidth

Below m* you are paying for the privilege of sending at all and the payload rides along
nearly free, which is why merging small messages is close to pure profit. Above m*,
latency is a rounding error and only bandwidth buys anything. A 4 KiB message sits below
m* on every fabric ever built -- that is the entire justification for gradient bucketing
and for message aggregation in graph workloads.

Bandwidth is in GB/s with 1 GB = 1e9 bytes, the convention link vendors quote.

Stdlib only, and no site packages required:
    python -S -I tools/message_cost.py --alpha-us 2.5 --bandwidth-gbs 25
"""

from __future__ import annotations

import argparse
import json
import sys

BANNER = ("analytic alpha-beta model, not measured: one message, idle link, "
          "no contention")

US = 1e-6
GB = 1e9
KIB = 1024.0

# Nine decades, 1 byte to 1 GB. Wide enough that every fabric's crossover lands inside
# the sweep, so the flat region and the linear region are both visible in one table.
DECADE_BYTES = tuple(float(10 ** power) for power in range(10))

# Within a factor of two of the crossover neither term can be ignored, so the honest
# label is "balanced" rather than picking a side. Algorithm choices are genuinely close
# in this band, which is where a library's tuned thresholds live.
BALANCED_BAND = 2.0


def validate(alpha_us: float, bandwidth_gbs: float, nbytes: float) -> None:
    """Reject inputs the model has no meaning for, with the offending value named."""
    if alpha_us < 0.0:
        raise ValueError(f"--alpha-us must be non-negative, got {alpha_us}")
    if bandwidth_gbs <= 0.0:
        raise ValueError(f"--bandwidth-gbs must be positive, got {bandwidth_gbs}")
    if nbytes < 0.0:
        raise ValueError(f"--bytes must be non-negative, got {nbytes}")


def crossover_bytes(alpha_s: float, bandwidth_bytes_per_s: float) -> float:
    """Message size at which the latency and transfer terms are equal.

    Setting alpha = m * beta and substituting beta = 1/bandwidth gives
    m* = alpha * bandwidth, so m* is a property of the *link*, not of the workload.
    Two fabrics with the same bandwidth and different latency have different m*, which
    is why a fused-message threshold tuned on one cluster is wrong on the next.
    """
    return alpha_s * bandwidth_bytes_per_s


def message_time(nbytes: float, alpha_s: float,
                 bandwidth_bytes_per_s: float) -> tuple[float, float, float]:
    """Return (total seconds, latency seconds, transfer seconds) for one message."""
    transfer = nbytes / bandwidth_bytes_per_s
    return alpha_s + transfer, alpha_s, transfer


def classify(nbytes: float, crossover: float) -> str:
    """Name the regime a message sits in, which decides what optimization can help."""
    if crossover <= 0.0:
        return "bandwidth-bound"
    if nbytes < crossover / BALANCED_BAND:
        return "latency-bound"
    if nbytes > crossover * BALANCED_BAND:
        return "bandwidth-bound"
    return "balanced"


def human_bytes(nbytes: float) -> str:
    if nbytes < KIB:
        return f"{nbytes:.0f} B"
    for limit, unit in ((KIB ** 2, "KiB"), (KIB ** 3, "MiB")):
        if nbytes < limit:
            return f"{nbytes / (limit / KIB):.1f} {unit}"
    return f"{nbytes / KIB ** 3:.2f} GiB"


def sweep(alpha_s: float, bandwidth_bytes_per_s: float,
          sizes: tuple[float, ...]) -> tuple[list[dict], float]:
    """Cost every size in `sizes` plus the crossover itself, sorted by size."""
    crossover = crossover_bytes(alpha_s, bandwidth_bytes_per_s)
    rows = []
    for nbytes in sorted(set(sizes) | {crossover}):
        total, latency, transfer = message_time(nbytes, alpha_s, bandwidth_bytes_per_s)
        rows.append(
            dict(
                bytes=nbytes,
                seconds=total,
                latency_share=latency / total if total else 0.0,
                effective_gbs=(nbytes / total) / GB if total else 0.0,
                regime=classify(nbytes, crossover),
                is_crossover=nbytes == crossover,
            )
        )
    return rows, crossover


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Alpha-beta message cost swept across nine decades of size, "
                    "marking the latency/bandwidth crossover m* = alpha * bandwidth.")
    parser.add_argument("--alpha-us", type=float, default=2.5,
                        help="startup latency alpha in microseconds (default 2.5)")
    parser.add_argument("--bandwidth-gbs", type=float, default=25.0,
                        help="achievable bandwidth in GB/s, 1 GB = 1e9 bytes (default 25)")
    parser.add_argument("--bytes", type=float, default=0.0,
                        help="also cost and classify one specific message size in bytes")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        validate(args.alpha_us, args.bandwidth_gbs, args.bytes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    alpha = args.alpha_us * US
    bandwidth = args.bandwidth_gbs * GB
    rows, crossover = sweep(alpha, bandwidth, DECADE_BYTES)

    single = None
    if args.bytes:
        total, latency, transfer = message_time(args.bytes, alpha, bandwidth)
        single = dict(bytes=args.bytes, seconds=total, latency_s=latency,
                      transfer_s=transfer, regime=classify(args.bytes, crossover))

    if args.json:
        out = dict(banner=BANNER, alpha_us=args.alpha_us,
                   bandwidth_gbs=args.bandwidth_gbs, crossover_bytes=crossover,
                   sweep=rows)
        if single is not None:
            out["single"] = single
        print(json.dumps(out, indent=2))
        return 0

    beta_ns_per_byte = (1.0 / bandwidth) * 1e9 if bandwidth else float("inf")
    print(BANNER)
    print(f"alpha = {args.alpha_us:g} us   bandwidth = {args.bandwidth_gbs:g} GB/s"
          f"   beta = 1/bandwidth = {beta_ns_per_byte:g} ns/byte")
    print(f"crossover m* = alpha * bandwidth = {crossover:.0f} bytes "
          f"({human_bytes(crossover)})")
    print()

    header = (f"{'bytes':>13}{'size':>11}{'time us':>12}{'latency %':>11}"
              f"{'effective GB/s':>16}  regime")
    print(header)
    print("-" * (len(header) + 10))
    for row in rows:
        marker = "  <- m* crossover" if row["is_crossover"] else ""
        print(f"{row['bytes']:>13.0f}{human_bytes(row['bytes']):>11}"
              f"{row['seconds'] / US:>12.3f}{row['latency_share'] * 100:>11.1f}"
              f"{row['effective_gbs']:>16.3f}  {row['regime']}{marker}")
    print("-" * (len(header) + 10))

    # The flat region is the actionable part of the curve, and its extent is exactly a
    # factor of two: a zero-byte message costs alpha, a message of m* costs 2*alpha.
    at_crossover, _, _ = message_time(crossover, alpha, bandwidth)
    if alpha > 0.0:
        print(f"a message of m* costs {at_crossover / alpha:.2f}x a zero-byte message, "
              f"so every size below m* -- however many decades that spans -- fits inside "
              f"that one factor")
    print(f"latency-bound below {crossover / BALANCED_BAND:.0f} bytes, bandwidth-bound "
          f"above {crossover * BALANCED_BAND:.0f} bytes, balanced in between")

    if single is not None:
        print(f"\n{single['bytes']:.0f} bytes ({human_bytes(single['bytes'])}): "
              f"{single['seconds'] / US:.3f} us = {single['latency_s'] / US:.3f} us "
              f"latency + {single['transfer_s'] / US:.3f} us wire -> {single['regime']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
