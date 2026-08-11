#!/usr/bin/env python3
"""Recursive doubling against ring for one all-reduce, and which one a library would pick.

ANALYTIC MODEL, nothing measured: alpha-beta costs on an idle, uniform, contention-free
fabric. No topology, no congestion, no protocol switch, and only two of the several
algorithms a production library carries.

For N ranks and m bytes, with alpha per step and beta = 1 / bandwidth:

    recursive doubling   steps = ceil(log2 N)    bytes per rank = m * steps
    ring                 steps = 2 (N - 1)       bytes per rank = 2 m (N - 1) / N

Those are the opposite corners of one tradeoff. The ring's byte count is optimal --
Patarasuk and Yuan proved 2m(N-1)/N is a lower bound on what any rank must move -- and
its step count is the worst of any practical algorithm. Recursive doubling is the mirror
image: log2(N) steps, but every step ships the whole buffer, so its byte count grows
with rank count. Neither wins everywhere, which is why a collective library holds a
decision tree rather than an implementation, and picks per call from m and N.

The step counts are the textbook power-of-two ones. A real library handles a
non-power-of-two rank count with extra steps, and switches to a reduce-scatter plus
all-gather pair on large messages, so treat this as the shape of the decision rather
than a model of any particular library.

Bandwidth is in GB/s with 1 GB = 1e9 bytes, the convention link vendors quote.

Stdlib only, and no site packages required:
    python -S -I tools/collective_choice.py --ranks 64 --bytes 4096
"""

from __future__ import annotations

import argparse
import json
import sys

BANNER = ("analytic collective model, not measured: alpha-beta costs, no contention "
          "or topology")

US = 1e-6
GB = 1e9
KIB = 1024.0


def validate(ranks: int, nbytes: float, alpha_us: float, bandwidth_gbs: float) -> None:
    """Reject inputs the model has no meaning for, with the offending value named."""
    if ranks < 1:
        raise ValueError(f"--ranks must be at least 1, got {ranks}")
    if nbytes < 0.0:
        raise ValueError(f"--bytes must be non-negative, got {nbytes}")
    if alpha_us < 0.0:
        raise ValueError(f"--alpha-us must be non-negative, got {alpha_us}")
    if bandwidth_gbs <= 0.0:
        raise ValueError(f"--bandwidth-gbs must be positive, got {bandwidth_gbs}")


def doubling_steps(ranks: int) -> int:
    """ceil(log2(ranks)) in exact integer arithmetic, so 2**k lands on k and not k+1."""
    return (ranks - 1).bit_length()


def recursive_doubling_cost(ranks: int, nbytes: float, alpha_s: float,
                            beta_s_per_byte: float) -> dict:
    """log2(N) rounds, each exchanging the whole buffer with a partner N/2 away.

    Cheapest possible step count, and the price is that the byte count grows with
    log2(N) instead of staying flat. One rank is a degenerate all-reduce: nothing is
    exchanged, so the cost is zero rather than negative.
    """
    if ranks <= 1:
        return dict(name="recursive doubling", steps=0, bytes_per_rank=0.0,
                    latency_s=0.0, transfer_s=0.0, total_s=0.0)
    steps = doubling_steps(ranks)
    volume = nbytes * steps
    return dict(name="recursive doubling", steps=steps, bytes_per_rank=volume,
                latency_s=steps * alpha_s, transfer_s=volume * beta_s_per_byte,
                total_s=steps * alpha_s + volume * beta_s_per_byte)


def ring_cost(ranks: int, nbytes: float, alpha_s: float,
              beta_s_per_byte: float) -> dict:
    """Reduce-scatter then all-gather around a ring: 2(N-1) steps of m/N bytes each.

    The byte count converges to 2m as soon as N is above a handful, which is why a
    correctly-behaving ring all-reduce takes almost the same wall-clock time at 64 ranks
    as at 512. The step count is what grows.
    """
    if ranks <= 1:
        return dict(name="ring", steps=0, bytes_per_rank=0.0, latency_s=0.0,
                    transfer_s=0.0, total_s=0.0)
    steps = 2 * (ranks - 1)
    volume = 2.0 * nbytes * (ranks - 1) / ranks
    return dict(name="ring", steps=steps, bytes_per_rank=volume,
                latency_s=steps * alpha_s, transfer_s=volume * beta_s_per_byte,
                total_s=steps * alpha_s + volume * beta_s_per_byte)


def tie_bytes(ranks: int, alpha_s: float, beta_s_per_byte: float) -> float | None:
    """Message size at which the two algorithms cost the same, or None if they never do.

    Equating the two costs and solving for m:

        m = alpha * (2(N-1) - log2N) / (beta * (log2N - 2(N-1)/N))

    Both factors are positive for N >= 3, so exactly one tie exists and it moves with
    rank count -- which is why an algorithm threshold cannot be a compile-time constant.
    At N = 2 the two algorithms move the same bytes in one step versus two, so recursive
    doubling wins at every size and there is nothing to solve for.
    """
    if ranks <= 2:
        return None
    steps_doubling = doubling_steps(ranks)
    steps_ring = 2 * (ranks - 1)
    ring_volume_factor = 2.0 * (ranks - 1) / ranks
    latency_gap = steps_ring - steps_doubling
    volume_gap = steps_doubling - ring_volume_factor
    if latency_gap <= 0 or volume_gap <= 0 or beta_s_per_byte <= 0.0:
        return None
    return alpha_s * latency_gap / (beta_s_per_byte * volume_gap)


def human_bytes(nbytes: float) -> str:
    if nbytes < KIB:
        return f"{nbytes:.0f} B"
    for limit, unit in ((KIB ** 2, "KiB"), (KIB ** 3, "MiB")):
        if nbytes < limit:
            return f"{nbytes / (limit / KIB):.1f} {unit}"
    return f"{nbytes / KIB ** 3:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cost recursive doubling against ring for one all-reduce under the "
                    "alpha-beta model, and name the winner.")
    parser.add_argument("--ranks", type=int, default=64,
                        help="ranks participating in the all-reduce (default 64)")
    parser.add_argument("--bytes", type=float, default=4096.0,
                        help="all-reduce buffer size in bytes (default 4096)")
    parser.add_argument("--alpha-us", type=float, default=2.5,
                        help="per-step latency alpha in microseconds (default 2.5)")
    parser.add_argument("--bandwidth-gbs", type=float, default=25.0,
                        help="achievable bandwidth in GB/s, 1 GB = 1e9 bytes (default 25)")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        validate(args.ranks, args.bytes, args.alpha_us, args.bandwidth_gbs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    alpha = args.alpha_us * US
    beta = 1.0 / (args.bandwidth_gbs * GB)
    candidates = [recursive_doubling_cost(args.ranks, args.bytes, alpha, beta),
                  ring_cost(args.ranks, args.bytes, alpha, beta)]
    winner, runner_up = sorted(candidates, key=lambda row: row["total_s"])
    tie = tie_bytes(args.ranks, alpha, beta)

    if args.json:
        print(json.dumps(dict(banner=BANNER, ranks=args.ranks, bytes=args.bytes,
                              alpha_us=args.alpha_us, bandwidth_gbs=args.bandwidth_gbs,
                              candidates=candidates, winner=winner["name"],
                              tie_bytes=tie), indent=2))
        return 0

    print(BANNER)
    print(f"ranks = {args.ranks}   buffer = {args.bytes:.0f} bytes "
          f"({human_bytes(args.bytes)})   alpha = {args.alpha_us:g} us   "
          f"bandwidth = {args.bandwidth_gbs:g} GB/s")
    print()

    header = (f"{'algorithm':<20}{'steps':>7}{'bytes/rank':>14}{'latency us':>13}"
              f"{'wire us':>12}{'total us':>12}")
    print(header)
    print("-" * len(header))
    for row in candidates:
        print(f"{row['name']:<20}{row['steps']:>7}{row['bytes_per_rank']:>14.0f}"
              f"{row['latency_s'] / US:>13.3f}{row['transfer_s'] / US:>12.3f}"
              f"{row['total_s'] / US:>12.3f}")
    print("-" * len(header))

    if args.ranks == 1:
        print("at 1 rank an all-reduce is a no-op: both algorithms cost zero, and there "
              "is no choice to make")
        return 0
    if winner["total_s"] == runner_up["total_s"]:
        print(f"tie: both cost {winner['total_s'] / US:.3f} us at this size")
    else:
        margin = (runner_up["total_s"] / winner["total_s"]
                  if winner["total_s"] > 0 else float("inf"))
        print(f"winner: {winner['name']} at {winner['total_s'] / US:.3f} us, "
              f"{margin:.2f}x cheaper than {runner_up['name']}")

    if tie is None:
        print(f"no crossover exists at {args.ranks} rank(s): recursive doubling moves "
              f"the same bytes in fewer steps, so it wins at every size")
    else:
        print(f"the two tie at {tie:.0f} bytes ({human_bytes(tie)}) for "
              f"{args.ranks} ranks -- recursive doubling wins below that size, the ring "
              f"above it")
        print("re-run with a different --ranks: the tie moves, which is why this choice "
              "cannot be a compile-time constant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
