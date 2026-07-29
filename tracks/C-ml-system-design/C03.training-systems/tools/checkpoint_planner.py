#!/usr/bin/env python3
"""C03 - Checkpoint interval planner.

An ANALYTIC MODEL, not a measurement. It answers one question:

    given how often the job dies (MTBF) and how expensive a checkpoint is,
    how often should I checkpoint?

Model (Young 1974 / Daly 2006, first-order form). Let
    T = seconds of useful compute between checkpoints
    C = seconds the training step loop is BLOCKED by a checkpoint
    M = mean time between failures, seconds
    R = seconds to detect a failure, respawn, and reload, seconds
then the fraction of wall time that is not useful work is

    waste(T) = C / T            (you paid to write)
             + (T/2 + R) / M    (on average you lose half an interval, plus restart)

Minimising waste gives the classic  T* = sqrt(2 * C * M).
The two terms are the whole tradeoff: write often and the first term dominates,
write rarely and the second does.

Everything printed here follows from your inputs. Replace the defaults with
numbers measured on your own cluster before you make a real decision.

Stdlib only. Run `python checkpoint_planner.py --help`.
"""

from __future__ import annotations

import argparse
import math
import sys

HOUR = 3600.0


def die(message):
    print(f"error: {message}", file=sys.stderr)
    print("hint: run with --help to see accepted values.", file=sys.stderr)
    raise SystemExit(2)


def require_positive(name, value, allow_zero=False):
    if value != value or value in (float("inf"), float("-inf")):
        die(f"--{name} must be a finite number (got {value!r}).")
    if value < 0 or (value == 0 and not allow_zero):
        limit = "zero or greater" if allow_zero else "greater than zero"
        die(f"--{name} must be {limit} (got {value}).")
    return value


def waste_fraction(interval_s, blocking_s, mtbf_s, restart_s):
    """Fraction of wall time lost to checkpoint writes plus re-done work."""
    write = blocking_s / interval_s
    lost = (0.5 * interval_s + restart_s) / mtbf_s
    return write, lost, write + lost


def optimal_interval(blocking_s, mtbf_s):
    """Young/Daly first-order optimum. Valid while C << M."""
    return math.sqrt(2.0 * blocking_s * mtbf_s)


def fmt_duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / HOUR:.2f}h"


def strategies(args):
    """Blocking cost of one checkpoint under three write strategies.

    Sharding divides the bytes each rank writes; async overlaps the write with
    the next steps, so the loop only stalls for the device->host snapshot.
    """
    full = args.checkpoint_seconds
    sharded = full / args.shards
    return [
        ("synchronous, single writer", full, full),
        (f"sharded across {args.shards} ranks", sharded, sharded),
        (
            f"sharded + async (stall {args.async_stall_fraction:.0%})",
            sharded * args.async_stall_fraction,
            sharded,
        ),
    ]


def print_header(args):
    print("Checkpoint interval planner - analytic model (Young/Daly first order)")
    print("-" * 78)
    print(f"  MTBF                      : {args.mtbf_hours:.2f} h")
    print(f"  full checkpoint write     : {args.checkpoint_seconds:.1f} s")
    print(f"  restart + reload          : {args.restart_minutes:.1f} min")
    print(f"  step time                 : {args.step_seconds:.3f} s")
    print(f"  useful compute in the run : {args.run_hours:.1f} h")
    print(f"  GPUs x $/GPU-hour         : {args.gpus} x ${args.gpu_hour_cost:.2f}")
    print()


def sweep(args, blocking_s):
    mtbf_s = args.mtbf_hours * HOUR
    restart_s = args.restart_minutes * 60.0
    print("  interval    steps    write%   lost%   total%   goodput%")
    print("  ---------- -------- -------- ------- -------- ---------")
    for minutes in (1, 2, 5, 10, 20, 30, 60, 120, 240, 480):
        interval_s = minutes * 60.0
        write, lost, total = waste_fraction(interval_s, blocking_s, mtbf_s, restart_s)
        steps = interval_s / args.step_seconds
        goodput = max(0.0, 1.0 - total)
        print(
            f"  {fmt_duration(interval_s):>10} {steps:8.0f} "
            f"{write * 100:7.2f}% {lost * 100:6.2f}% {total * 100:7.2f}% "
            f"{goodput * 100:8.2f}%"
        )
    print()
    print("  Rows where write% exceeds 100% are the first-order model outside its")
    print("  validity range (C must be small next to T): they mean 'never do this',")
    print("  not a literal figure.")
    print()


def report_strategy(args, label, blocking_s, io_s):
    mtbf_s = args.mtbf_hours * HOUR
    restart_s = args.restart_minutes * 60.0
    best = optimal_interval(blocking_s, mtbf_s)
    write, lost, total = waste_fraction(best, blocking_s, mtbf_s, restart_s)
    goodput = 1.0 - total

    print(f"* {label}")
    print(f"    blocking cost per checkpoint : {blocking_s:.1f} s "
          f"(bytes still take {io_s:.1f} s of I/O)")
    print(f"    optimal interval T*          : {fmt_duration(best)} "
          f"({best / args.step_seconds:.0f} steps)")
    print(f"    overhead at T*               : {write * 100:.2f}% write + "
          f"{lost * 100:.2f}% lost = {total * 100:.2f}%")

    if goodput <= 0:
        print("    VERDICT: this configuration cannot make progress - failures and")
        print("             writes consume the whole budget. Reduce checkpoint cost")
        print("             or fix the hardware before tuning the interval.")
        print()
        return

    if io_s > best:
        print(f"    WARNING: the background write ({io_s:.0f} s) is longer than the")
        print("             interval - checkpoints would queue up. Shard harder or")
        print("             checkpoint less often than the write takes.")

    wall_h = args.run_hours / goodput
    wasted_h = wall_h - args.run_hours
    dollars = wasted_h * args.gpus * args.gpu_hour_cost
    print(f"    run of {args.run_hours:.0f} useful GPU-h -> {wall_h:.1f} h wall, "
          f"{wasted_h:.1f} h wasted (${dollars:,.0f} across {args.gpus} GPUs)")
    print()


def sensitivity(args, blocking_s):
    """Being wrong about the interval is cheap; being wrong about MTBF is not."""
    mtbf_s = args.mtbf_hours * HOUR
    restart_s = args.restart_minutes * 60.0
    best = optimal_interval(blocking_s, mtbf_s)
    _, _, best_total = waste_fraction(best, blocking_s, mtbf_s, restart_s)
    print("  How much does missing T* cost you?")
    print("    factor off T*   interval    total overhead   penalty vs T*")
    print("    ------------- ----------- ---------------- ---------------")
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        interval_s = best * factor
        _, _, total = waste_fraction(interval_s, blocking_s, mtbf_s, restart_s)
        penalty = (total - best_total) * 100
        print(f"    {factor:12.2f}x {fmt_duration(interval_s):>11} "
              f"{total * 100:15.2f}% {penalty:+14.2f} pp")
    print()
    print("  The curve is flat near the optimum: a 2x error costs little, which is")
    print("  why a rough MTBF estimate is good enough to act on.")
    print()


def build_parser():
    p = argparse.ArgumentParser(
        prog="checkpoint_planner.py",
        description="Analytic checkpoint-interval planner (stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="All output is derived from these inputs - it is a model, not a benchmark.",
    )
    p.add_argument("--mtbf-hours", type=float, default=6.0,
                   help="mean time between job-killing failures")
    p.add_argument("--checkpoint-seconds", type=float, default=180.0,
                   help="wall seconds to write one full checkpoint from a single writer")
    p.add_argument("--restart-minutes", type=float, default=12.0,
                   help="detect + respawn + reload time after a failure")
    p.add_argument("--step-seconds", type=float, default=0.75,
                   help="wall seconds per training step (used to report intervals in steps)")
    p.add_argument("--run-hours", type=float, default=336.0,
                   help="useful compute hours the run needs (2 weeks = 336)")
    p.add_argument("--shards", type=int, default=8,
                   help="ranks writing distinct shards of the checkpoint in parallel")
    p.add_argument("--async-stall-fraction", type=float, default=0.1,
                   help="fraction of the write the step loop still blocks on "
                        "(the device->host snapshot) when writing asynchronously")
    p.add_argument("--gpus", type=int, default=64, help="GPUs held by the job")
    p.add_argument("--gpu-hour-cost", type=float, default=2.50,
                   help="all-in $ per GPU-hour")
    return p


def validate(args):
    require_positive("mtbf-hours", args.mtbf_hours)
    require_positive("checkpoint-seconds", args.checkpoint_seconds)
    require_positive("restart-minutes", args.restart_minutes, allow_zero=True)
    require_positive("step-seconds", args.step_seconds)
    require_positive("run-hours", args.run_hours)
    require_positive("gpu-hour-cost", args.gpu_hour_cost, allow_zero=True)
    if args.shards < 1:
        die(f"--shards must be at least 1 (got {args.shards}).")
    if args.gpus < 1:
        die(f"--gpus must be at least 1 (got {args.gpus}).")
    if not 0.0 < args.async_stall_fraction <= 1.0:
        die("--async-stall-fraction must be in (0, 1]; it is the fraction of the "
            f"write the loop still blocks on (got {args.async_stall_fraction}).")
    if args.checkpoint_seconds >= args.mtbf_hours * HOUR:
        die("--checkpoint-seconds is larger than the MTBF: the job would spend all "
            "its time checkpointing. Check your units (seconds vs hours).")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)

    print_header(args)
    print("1. Overhead vs interval (synchronous single-writer checkpoint)")
    sweep(args, args.checkpoint_seconds)

    print("2. The same job under three checkpoint strategies")
    for label, blocking_s, io_s in strategies(args):
        report_strategy(args, label, blocking_s, io_s)

    print("3. Sensitivity around the optimum (synchronous case)")
    sensitivity(args, args.checkpoint_seconds)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
