#!/usr/bin/env python3
"""C04 - Two-tier cascade analyzer: blended accuracy and cost vs escalation rate.

An ANALYTIC MODEL, not a benchmark. A small cheap model answers everything; the
uncertain fraction `e` is escalated to a large expensive model.

Model
-----
    cost(e)  = small_cost + e * large_cost          (the small model always runs)

    Errors. Let E_s = 1 - small_accuracy. A confidence signal of quality g
    captures a fraction of those errors inside the escalated slice:

        capture(e) = 1 - (1 - e) ** g

    g = 1 is escalating at random (capture == e). g > 1 means the confidence
    signal really does find the cases the small model gets wrong. Then

        error(e) = E_s * (1 - capture(e))   errors kept, never escalated
                 + e * (1 - large_accuracy) errors the large model still makes

    accuracy(e) = 1 - error(e)

Assumption worth stating out loud in a review: the large model is assumed to
hit its headline accuracy on the escalated slice too. Those are the HARD cases,
so in reality it does worse there and the curve below is optimistic. Use
--large-hardness to see how much that assumption is worth.

Stdlib only. Run `python cascade_analyzer.py --help`.
"""

from __future__ import annotations

import argparse
import sys


def die(message):
    print(f"error: {message}", file=sys.stderr)
    print("hint: run with --help to see accepted values.", file=sys.stderr)
    raise SystemExit(2)


def require_fraction(name, value, low=0.0, high=1.0):
    if value != value:
        die(f"--{name} must be a number (got {value!r}).")
    if not low <= value <= high:
        die(f"--{name} must be between {low} and {high} (got {value}).")
    return value


def capture(escalation, gain):
    """Fraction of the small model's errors that land inside the escalated slice."""
    return 1.0 - (1.0 - escalation) ** gain


def evaluate(escalation, args):
    small_errors = 1.0 - args.small_accuracy
    large_error_rate = min(1.0, (1.0 - args.large_accuracy) * args.large_hardness)

    kept_errors = small_errors * (1.0 - capture(escalation, args.confidence_gain))
    escalated_errors = escalation * large_error_rate
    accuracy = 1.0 - (kept_errors + escalated_errors)

    cost = args.small_cost + escalation * args.large_cost
    latency = args.small_latency_ms + escalation * args.large_latency_ms
    return accuracy, cost, latency


def sweep(args, steps=21):
    rows = []
    for i in range(steps):
        e = i / (steps - 1)
        accuracy, cost, latency = evaluate(e, args)
        rows.append((e, accuracy, cost, latency))
    return rows


def find_operating_point(args, floor, resolution=2001):
    """Cheapest escalation rate that still clears the quality floor."""
    best = None
    for i in range(resolution):
        e = i / (resolution - 1)
        accuracy, cost, latency = evaluate(e, args)
        if accuracy + 1e-12 < floor:
            continue
        if best is None or cost < best[2]:
            best = (e, accuracy, cost, latency)
    return best


def build_parser():
    p = argparse.ArgumentParser(
        prog="cascade_analyzer.py",
        description="Two-tier model cascade: blended accuracy and cost "
                    "(analytic, stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Accuracies and costs are YOUR measurements; the defaults only make "
               "the arithmetic legible.",
    )
    p.add_argument("--small-accuracy", type=float, default=0.90)
    p.add_argument("--large-accuracy", type=float, default=0.96)
    p.add_argument("--small-cost", type=float, default=0.10,
                   help="cost per 1k requests when the small model runs on all traffic")
    p.add_argument("--large-cost", type=float, default=2.00,
                   help="cost per 1k requests if the large model ran on ALL traffic")
    p.add_argument("--small-latency-ms", type=float, default=25.0)
    p.add_argument("--large-latency-ms", type=float, default=400.0)
    p.add_argument("--confidence-gain", type=float, default=3.0,
                   help="quality of the escalation signal: 1.0 is random, higher "
                        "means the signal really finds the small model's mistakes")
    p.add_argument("--large-hardness", type=float, default=1.0,
                   help="multiplier on the large model's error rate over the "
                        "escalated (hard) slice; 1.0 is the optimistic assumption")
    p.add_argument("--quality-floor", type=float, default=0.95,
                   help="minimum blended accuracy the product requires")
    return p


def validate(args):
    require_fraction("small-accuracy", args.small_accuracy)
    require_fraction("large-accuracy", args.large_accuracy)
    require_fraction("quality-floor", args.quality_floor)
    for name in ("small-cost", "large-cost", "small-latency-ms", "large-latency-ms"):
        value = getattr(args, name.replace("-", "_"))
        if value < 0:
            die(f"--{name} cannot be negative (got {value}).")
    if args.confidence_gain < 1.0:
        die("--confidence-gain must be at least 1.0; below 1 would mean the "
            f"confidence signal is worse than random (got {args.confidence_gain}).")
    if args.large_hardness < 1.0:
        die("--large-hardness must be at least 1.0; the escalated slice cannot be "
            f"easier than average traffic (got {args.large_hardness}).")
    if args.large_accuracy < args.small_accuracy:
        die("--large-accuracy is below --small-accuracy: there is nothing to "
            "escalate to. Ship the small model.")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)

    print("Two-tier cascade - analytic model")
    print("-" * 78)
    print(f"  small model : accuracy {args.small_accuracy:.3f}, "
          f"${args.small_cost:.3f}/1k, {args.small_latency_ms:.0f} ms")
    print(f"  large model : accuracy {args.large_accuracy:.3f}, "
          f"${args.large_cost:.3f}/1k, {args.large_latency_ms:.0f} ms (if used alone)")
    print(f"  confidence  : gain {args.confidence_gain:g} "
          f"(1.0 = escalate at random)")
    print(f"  hard-slice  : large-model errors x{args.large_hardness:g} on escalated "
          "traffic")
    print()

    print("1. Sweep the escalation rate")
    print("  escalate   accuracy   errors caught   $/1k req   avg latency   $ vs large-only")
    print("  -------- ---------- --------------- ---------- ------------- -----------------")
    for e, accuracy, cost, latency in sweep(args):
        caught = capture(e, args.confidence_gain)
        saving = args.large_cost - cost
        print(f"  {e * 100:7.0f}% {accuracy:10.4f} {caught * 100:14.1f}% "
              f"{cost:10.4f} {latency:11.0f} ms {saving:17.4f}")
    print()

    print(f"  Large model on 100% of traffic: accuracy {args.large_accuracy:.4f} at "
          f"${args.large_cost:.3f}/1k.")
    print("  The cascade also pays the small model on every request, which is why the")
    print("  100% row costs more than large-only. A cascade is only worth it if you")
    print("  stop well short of escalating everything.")

    peak_e, peak_acc = max(
        ((i / 2000.0, evaluate(i / 2000.0, args)[0]) for i in range(2001)),
        key=lambda pair: pair[1],
    )
    if peak_e < 0.995 and peak_acc > args.large_accuracy + 1e-6:
        print()
        print(f"  Worth noticing: accuracy PEAKS at {peak_e * 100:.0f}% escalation "
              f"({peak_acc:.4f}), above")
        print("  the large model alone. Past that point you are escalating requests the")
        print("  small model already had right, and the large model gets some of them")
        print("  wrong. A good confidence signal beats a bigger model on its own.")
    print()

    print("2. Cost-optimal operating point")
    best = find_operating_point(args, args.quality_floor)
    if best is None:
        reachable, _, _ = evaluate(1.0, args)
        print(f"  No escalation rate reaches the {args.quality_floor:.3f} floor.")
        print(f"  The best this cascade can do is {reachable:.4f} (escalate everything).")
        print("  Options: raise the small model's accuracy, improve the confidence")
        print("  signal (--confidence-gain), get a better large model, or renegotiate")
        print("  the floor. Adding escalation will not fix it.")
    else:
        e, accuracy, cost, latency = best
        if cost >= args.large_cost:
            print("  The cheapest way to clear this floor is to escalate (nearly)")
            print("  everything, which costs MORE than just serving the large model.")
            print("  This cascade does not earn its complexity - drop it, or fix the")
            print("  confidence signal (--confidence-gain) first.")
            print()
        print(f"  escalate {e * 100:.1f}% of traffic")
        print(f"    blended accuracy : {accuracy:.4f} (floor {args.quality_floor:.4f})")
        print(f"    cost             : ${cost:.4f}/1k "
              f"(vs ${args.large_cost:.4f} large-only, "
              f"{(1 - cost / args.large_cost) * 100:.1f}% cheaper)")
        print(f"    average latency  : {latency:.0f} ms "
              f"(the escalated {e * 100:.1f}% still waits "
              f"{args.small_latency_ms + args.large_latency_ms:.0f} ms)")
        print()
        print("  Two things to say in the design review: the tail latency is the")
        print("  cascade's real cost - escalated requests pay BOTH models - and the")
        print("  escalation rate drifts with traffic, so it must be monitored, not set")
        print("  once. Budget for the day the small model's confidence goes stale.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
