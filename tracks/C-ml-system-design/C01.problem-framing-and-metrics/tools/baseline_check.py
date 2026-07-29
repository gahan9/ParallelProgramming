#!/usr/bin/env python3
"""Trivial baselines for a binary label distribution.

Answers one question: "what score is unimpressive?" Before you report a model number,
you need to know what majority / uniform-random / stratified-random already achieve on
the same label distribution.

Supply the distribution one of three ways:
  --positives N --negatives M
  --prevalence P --n N
  --labels FILE            (CSV whose first column is 0/1, header optional)

Standard library only.
"""

import argparse
import csv
import random
import sys

EXIT_BAD_INPUT = 2


def fail(message):
    print("error: " + message, file=sys.stderr)
    raise SystemExit(EXIT_BAD_INPUT)


def safe_div(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def fmt(value, width=8, places=4):
    if value is None:
        return "undef".rjust(width)
    return ("%.*f" % (places, value)).rjust(width)


def expected_metrics(positives, negatives, flag_rate):
    """Expected confusion counts for a label-blind policy that flags at `flag_rate`.

    Label-blind means the flag decision is independent of the label, so precision is
    pinned to the prevalence no matter what rate you choose. That invariance is the
    lesson: a label-blind policy can buy recall but never precision.
    """
    total = positives + negatives
    tp = flag_rate * positives
    fp = flag_rate * negatives
    fn = positives - tp
    tn = negatives - fp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, positives)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def simulate_metrics(positives, negatives, flag_rate, trials, seed):
    """Monte-Carlo check that the closed-form expectations above are right."""
    rng = random.Random(seed)
    totals = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    counted = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    for _ in range(trials):
        tp = fp = 0
        for _ in range(positives):
            if rng.random() < flag_rate:
                tp += 1
        for _ in range(negatives):
            if rng.random() < flag_rate:
                fp += 1
        fn = positives - tp
        tn = negatives - fp
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, positives)
        f1 = None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        for name, value in (("accuracy", safe_div(tp + tn, positives + negatives)),
                            ("precision", precision), ("recall", recall), ("f1", f1)):
            if value is not None:
                totals[name] += value
                counted[name] += 1
    return {name: safe_div(totals[name], counted[name]) for name in totals}


def resolve_distribution(args):
    given = [args.labels is not None,
             args.positives is not None or args.negatives is not None,
             args.prevalence is not None]
    if sum(1 for flag in given if flag) != 1:
        fail("give exactly one of: --labels FILE | --positives/--negatives | --prevalence with --n")

    if args.labels is not None:
        return read_labels(args.labels)

    if args.prevalence is not None:
        if args.n is None:
            fail("--prevalence requires --n (the number of examples)")
        if not 0.0 < args.prevalence < 1.0:
            fail("--prevalence must be strictly between 0 and 1, got %g" % args.prevalence)
        if args.n < 2:
            fail("--n must be >= 2, got %d" % args.n)
        positives = int(round(args.prevalence * args.n))
        positives = min(max(positives, 1), args.n - 1)
        return positives, args.n - positives

    if args.positives is None or args.negatives is None:
        fail("--positives and --negatives must be given together")
    if args.positives < 1 or args.negatives < 1:
        fail("--positives and --negatives must both be >= 1 (got %d and %d); a single-class "
             "distribution has no meaningful precision/recall baseline"
             % (args.positives, args.negatives))
    return args.positives, args.negatives


def read_labels(path):
    handle = sys.stdin if path == "-" else None
    if handle is None:
        try:
            handle = open(path, "r", newline="", encoding="utf-8")
        except OSError as error:
            fail("cannot read labels file %r: %s" % (path, error.strerror or error))

    positives = 0
    negatives = 0
    try:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or row[0].strip() == "":
                continue
            token = row[0].strip()
            if line_number == 1 and token not in ("0", "1"):
                continue  # header row
            if token == "1":
                positives += 1
            elif token == "0":
                negatives += 1
            else:
                fail("line %d: label %r must be 0 or 1" % (line_number, token))
    finally:
        if handle is not sys.stdin:
            handle.close()

    if positives + negatives == 0:
        fail("no labels found in %r" % path)
    if positives == 0 or negatives == 0:
        fail("file %r contains only one class (%d positives, %d negatives); baselines need both"
             % (path, positives, negatives))
    return positives, negatives


def main(argv):
    parser = argparse.ArgumentParser(
        prog="baseline_check.py",
        description="Show what trivial baselines score on a given label distribution.",
    )
    parser.add_argument("--positives", type=int)
    parser.add_argument("--negatives", type=int)
    parser.add_argument("--prevalence", type=float, help="positive base rate; use with --n")
    parser.add_argument("--n", type=int, help="total examples; use with --prevalence")
    parser.add_argument("--labels", help="CSV whose first column is 0/1, or - for stdin")
    parser.add_argument("--simulate", type=int, default=0, metavar="TRIALS",
                        help="also Monte-Carlo the random baselines this many times")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    if args.simulate < 0:
        fail("--simulate must be >= 0, got %d" % args.simulate)

    positives, negatives = resolve_distribution(args)
    total = positives + negatives
    prevalence = positives / float(total)

    print("label distribution")
    print("  n = %d   positives = %d   negatives = %d   prevalence = %s"
          % (total, positives, negatives, fmt(prevalence)))
    print()

    baselines = [
        ("always-negative (majority if rare)", 0.0),
        ("always-positive", 1.0),
        ("uniform random (flag 50%)", 0.5),
        ("stratified random (flag at prevalence)", prevalence),
    ]

    print("label-blind baselines")
    print("  policy                                   accuracy  precision   recall        F1")
    for name, rate in baselines:
        metrics = expected_metrics(positives, negatives, rate)
        print("  %-40s %s %s %s %s"
              % (name, fmt(metrics["accuracy"]), fmt(metrics["precision"]),
                 fmt(metrics["recall"]), fmt(metrics["f1"])))

    majority_accuracy = max(prevalence, 1.0 - prevalence)
    print()
    print("ranking baselines (threshold-free)")
    print("  random scores:  ROC-AUC = %s   PR-AUC = %s  (PR-AUC baseline is the prevalence)"
          % (fmt(0.5), fmt(prevalence)))
    print()
    print("the numbers your model must beat to be interesting")
    print("  accuracy  > %s   (majority class; anything below this is worse than a constant)"
          % fmt(majority_accuracy))
    print("  precision > %s   (any label-blind policy already gets this)" % fmt(prevalence))
    print("  F1        > %s   (stratified random)" % fmt(prevalence))
    print("  PR-AUC    > %s   (random ranking)" % fmt(prevalence))
    print("  ROC-AUC   > %s" % fmt(0.5))

    if prevalence <= 0.05:
        print()
        print("  At %.2f%% prevalence, accuracy is nearly useless as a headline metric: the"
              % (100.0 * prevalence))
        print("  always-negative policy already scores %s while catching zero positives."
              % fmt(1.0 - prevalence))

    if args.simulate:
        print()
        print("simulated check (%d trials, seed %d) - should match the analytic rows above"
              % (args.simulate, args.seed))
        print("  policy                                   accuracy  precision   recall        F1")
        for name, rate in baselines:
            if rate in (0.0, 1.0):
                continue  # deterministic; nothing to simulate
            metrics = simulate_metrics(positives, negatives, rate, args.simulate, args.seed)
            print("  %-40s %s %s %s %s"
                  % (name, fmt(metrics["accuracy"]), fmt(metrics["precision"]),
                     fmt(metrics["recall"]), fmt(metrics["f1"])))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        raise SystemExit(0)
    except KeyboardInterrupt:
        raise SystemExit(130)
