#!/usr/bin/env python3
"""Metric explorer for classification decisions.

Three subcommands:
  confusion   metrics from a confusion matrix (TP/FP/FN/TN)
  sweep       metrics + cost-weighted optimum across every threshold of scored data
  imbalance   how ROC-AUC and PR-AUC diverge as prevalence falls

Standard library only. Run `--help` on any subcommand for its arguments.
"""

import argparse
import csv
import math
import random
import sys

EXIT_BAD_INPUT = 2


def fail(message):
    print("error: " + message, file=sys.stderr)
    raise SystemExit(EXIT_BAD_INPUT)


# ---------------------------------------------------------------- metrics ----

def safe_div(numerator, denominator):
    """Return None rather than raising, so callers can print 'undefined'."""
    return None if denominator == 0 else numerator / denominator


def fmt(value, width=7, places=4):
    if value is None:
        return "undef".rjust(width)
    return ("%.*f" % (places, value)).rjust(width)


def confusion_metrics(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    return {
        "accuracy": safe_div(tp + tn, total),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "fpr": safe_div(fp, fp + tn),
        "prevalence": safe_div(tp + fn, total),
        "f1": f1_score(tp, fp, fn),
    }


def f1_score(tp, fp, fn):
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def roc_auc(pairs):
    """Rank-based (Mann-Whitney) ROC-AUC with mid-rank correction for ties."""
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None

    ordered = sorted(pairs, key=lambda pair: pair[0])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        # Ranks are 1-based; tied scores all receive the average of their ranks.
        average_rank = (index + 1 + stop + 1) / 2.0
        for position in range(index, stop + 1):
            if ordered[position][1] == 1:
                rank_sum_positive += average_rank
        index = stop + 1

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(pairs):
    """PR-AUC as average precision: sum over thresholds of (dRecall * precision)."""
    positives = sum(1 for _, label in pairs if label == 1)
    if positives == 0:
        return None

    ordered = sorted(pairs, key=lambda pair: -pair[0])
    tp = 0
    fp = 0
    previous_recall = 0.0
    total = 0.0
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        for position in range(index, stop + 1):
            if ordered[position][1] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / positives
        total += (recall - previous_recall) * precision
        previous_recall = recall
        index = stop + 1

    return total


def sweep_thresholds(pairs, fp_cost, fn_cost):
    """One row per distinct score: predict positive when score >= that score."""
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    ordered = sorted(pairs, key=lambda pair: -pair[0])

    rows = []
    tp = 0
    fp = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == threshold:
            stop += 1
        for position in range(index, stop + 1):
            if ordered[position][1] == 1:
                tp += 1
            else:
                fp += 1
        fn = positives - tp
        tn = negatives - fp
        rows.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": safe_div(tp, tp + fp),
            "recall": safe_div(tp, tp + fn),
            "f1": f1_score(tp, fp, fn),
            "cost": fp * fp_cost + fn * fn_cost,
        })
        index = stop + 1
    return rows


# ------------------------------------------------------------------- input ----

def read_pairs(path):
    """Read `score,label` rows. Accepts an optional header and `-` for stdin."""
    handle = sys.stdin if path == "-" else None
    if handle is None:
        try:
            handle = open(path, "r", newline="", encoding="utf-8")
        except OSError as error:
            fail("cannot read scores file %r: %s" % (path, error.strerror or error))

    pairs = []
    try:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(field.strip() == "" for field in row):
                continue
            if len(row) < 2:
                fail("line %d: expected 'score,label', got %r" % (line_number, ",".join(row)))
            score_text, label_text = row[0].strip(), row[1].strip()
            if line_number == 1:
                try:
                    float(score_text)
                except ValueError:
                    continue  # header row
            try:
                score = float(score_text)
            except ValueError:
                fail("line %d: score %r is not a number" % (line_number, score_text))
            if label_text not in ("0", "1"):
                fail("line %d: label %r must be 0 or 1" % (line_number, label_text))
            pairs.append((score, int(label_text)))
    finally:
        if handle is not sys.stdin:
            handle.close()

    if not pairs:
        fail("no usable rows found (expected CSV lines of 'score,label')")
    labels = set(label for _, label in pairs)
    if len(labels) < 2:
        fail("all %d rows have label %d; both classes are required" % (len(pairs), labels.pop()))
    return pairs


def synthetic_pairs(positives, negatives, separation, seed):
    """Two Gaussians a fixed number of standard deviations apart, mapped to (0,1)."""
    rng = random.Random(seed)
    pairs = []
    for _ in range(positives):
        pairs.append((logistic(rng.gauss(separation, 1.0)), 1))
    for _ in range(negatives):
        pairs.append((logistic(rng.gauss(0.0, 1.0)), 0))
    return pairs


def logistic(value):
    if value < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


# ---------------------------------------------------------------- commands ----

def print_confusion_block(tp, fp, fn, tn):
    metrics = confusion_metrics(tp, fp, fn, tn)
    print("confusion matrix")
    print("                 predicted 1   predicted 0")
    print("  actual 1     %12d  %12d" % (tp, fn))
    print("  actual 0     %12d  %12d" % (fp, tn))
    print()
    print("  n            = %d" % (tp + fp + fn + tn))
    print("  prevalence   = %s   (base rate of the positive class)" % fmt(metrics["prevalence"]))
    print("  accuracy     = %s" % fmt(metrics["accuracy"]))
    print("  precision    = %s   (of those flagged, how many were right)" % fmt(metrics["precision"]))
    print("  recall       = %s   (of the real positives, how many we caught)" % fmt(metrics["recall"]))
    print("  specificity  = %s" % fmt(metrics["specificity"]))
    print("  FPR          = %s" % fmt(metrics["fpr"]))
    print("  F1           = %s" % fmt(metrics["f1"]))
    return metrics


def command_confusion(args):
    for name, value in (("tp", args.tp), ("fp", args.fp), ("fn", args.fn), ("tn", args.tn)):
        if value < 0:
            fail("--%s must be >= 0, got %d" % (name, value))
    if args.tp + args.fp + args.fn + args.tn == 0:
        fail("all four counts are zero; there is nothing to score")

    metrics = print_confusion_block(args.tp, args.fp, args.fn, args.tn)

    majority_accuracy = max(metrics["prevalence"], 1.0 - metrics["prevalence"])
    print()
    print("baseline comparison")
    print("  always-predict-majority accuracy = %s" % fmt(majority_accuracy))
    if metrics["accuracy"] is not None and metrics["accuracy"] <= majority_accuracy:
        print("  VERDICT: this model does not beat the trivial majority baseline on accuracy.")
    else:
        print("  VERDICT: beats majority accuracy by %s" % fmt(metrics["accuracy"] - majority_accuracy))
    print("  Accuracy is a weak signal at this prevalence. Report precision/recall per slice.")

    cost = args.fp * args.fp_cost + args.fn * args.fn_cost
    print()
    print("cost at this operating point (FP=%g, FN=%g)" % (args.fp_cost, args.fn_cost))
    print("  total cost   = %g   (%d FP x %g + %d FN x %g)"
          % (cost, args.fp, args.fp_cost, args.fn, args.fn_cost))
    print("  cost / case  = %s" % fmt(cost / (args.tp + args.fp + args.fn + args.tn)))
    return 0


def command_sweep(args):
    if args.fp_cost < 0 or args.fn_cost < 0:
        fail("costs must be >= 0 (got fp=%g fn=%g)" % (args.fp_cost, args.fn_cost))
    if args.fp_cost == 0 and args.fn_cost == 0:
        fail("both costs are zero; every threshold is equally 'optimal'")

    if args.scores:
        pairs = read_pairs(args.scores)
        source = args.scores
    else:
        pairs = synthetic_pairs(args.positives, args.negatives, args.separation, args.seed)
        source = ("synthetic: %d positives, %d negatives, separation=%.2f sd, seed=%d"
                  % (args.positives, args.negatives, args.separation, args.seed))

    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    rows = sweep_thresholds(pairs, args.fp_cost, args.fn_cost)

    best_f1 = max(rows, key=lambda row: (-1.0 if row["f1"] is None else row["f1"]))
    best_cost = min(rows, key=lambda row: row["cost"])
    reject_all_cost = positives * args.fn_cost
    accept_all_cost = negatives * args.fp_cost

    print("source: %s" % source)
    print("n = %d  positives = %d  negatives = %d  prevalence = %s"
          % (len(pairs), positives, negatives, fmt(positives / len(pairs))))
    print()
    print("ranking quality (threshold-free)")
    print("  ROC-AUC = %s" % fmt(roc_auc(pairs)))
    print("  PR-AUC  = %s   (random baseline = prevalence = %s)"
          % (fmt(average_precision(pairs)), fmt(positives / len(pairs))))
    print()

    shown = select_rows(rows, args.max_rows, {id(best_f1), id(best_cost)})
    print("threshold sweep  (predict positive when score >= threshold)")
    print("  thresh      TP     FP     FN     TN  precision  recall      F1        cost")
    for row in shown:
        marks = []
        if row is best_f1:
            marks.append("<- max F1")
        if row is best_cost:
            marks.append("<- min cost")
        print("  %.4f %6d %6d %6d %6d  %s %s %s %11.2f %s"
              % (row["threshold"], row["tp"], row["fp"], row["fn"], row["tn"],
                 fmt(row["precision"], 9), fmt(row["recall"], 7), fmt(row["f1"], 7),
                 row["cost"], " ".join(marks)))
    if len(shown) < len(rows):
        print("  (%d of %d distinct thresholds shown; use --max-rows 0 for all)"
              % (len(shown), len(rows)))

    ratio = args.fn_cost / args.fp_cost if args.fp_cost > 0 else float("inf")
    analytic = args.fp_cost / (args.fp_cost + args.fn_cost)
    print()
    print("cost-weighted decision (FP=%g, FN=%g, FN/FP ratio = %s)"
          % (args.fp_cost, args.fn_cost, "inf" if ratio == float("inf") else "%.2f" % ratio))
    print("  empirical best threshold = %.4f  ->  cost %.2f (%.4f per case)"
          % (best_cost["threshold"], best_cost["cost"], best_cost["cost"] / len(pairs)))
    print("  F1-optimal threshold     = %.4f  ->  cost %.2f"
          % (best_f1["threshold"], best_f1["cost"]))
    print("  analytic optimum for a *calibrated* probability p* = C_fp / (C_fp + C_fn) = %.4f"
          % analytic)
    print("  flag nothing (reject all) cost = %.2f | flag everything cost = %.2f"
          % (reject_all_cost, accept_all_cost))
    print()
    if abs(analytic - best_cost["threshold"]) > 0.05:
        print("  The empirical and analytic thresholds disagree by more than 0.05: these scores")
        print("  are not calibrated probabilities, so p* cannot be used directly. Either calibrate")
        print("  the scores or keep picking the threshold empirically on a held-out set.")
    else:
        print("  Empirical and analytic thresholds agree, which is what calibrated scores look like.")
    if best_cost["cost"] >= min(reject_all_cost, accept_all_cost):
        print("  WARNING: no threshold beats a trivial always/never-flag policy at these costs.")
    return 0


def select_rows(rows, max_rows, must_keep_ids):
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    step = (len(rows) - 1) / float(max_rows - 1)
    keep = {int(round(index * step)) for index in range(max_rows)}
    keep |= {index for index, row in enumerate(rows) if id(row) in must_keep_ids}
    return [rows[index] for index in sorted(keep)]


def command_imbalance(args):
    prevalences = parse_float_list(args.prevalences, "--prevalences")
    for value in prevalences:
        if not 0.0 < value < 1.0:
            fail("--prevalences entries must be strictly between 0 and 1, got %g" % value)
    if args.positives < 2:
        fail("--positives must be >= 2, got %d" % args.positives)

    print("Fixed ranking quality (separation = %.2f sd), varying only the class balance."
          % args.separation)
    print("Positives held at %d; negatives added to hit each prevalence.\n" % args.positives)
    print("  prevalence  negatives   ROC-AUC   PR-AUC   PR-AUC baseline   PR-AUC lift")
    for prevalence in sorted(prevalences, reverse=True):
        negatives = int(round(args.positives * (1.0 - prevalence) / prevalence))
        pairs = synthetic_pairs(args.positives, negatives, args.separation, args.seed)
        actual_prevalence = args.positives / float(args.positives + negatives)
        pr = average_precision(pairs)
        print("  %10.4f %10d  %s  %s  %s          %s"
              % (actual_prevalence, negatives, fmt(roc_auc(pairs)), fmt(pr),
                 fmt(actual_prevalence), fmt(safe_div(pr, actual_prevalence), 7, 2)))
    print()
    print("ROC-AUC barely moves because it is a ratio of rates (TPR vs FPR) and both rates are")
    print("computed within a class. PR-AUC collapses because precision has the negative count in")
    print("its denominator. On rare-positive problems, report PR-AUC (and its lift over the")
    print("prevalence baseline); an ROC-AUC of 0.9 can coexist with a useless precision.")
    return 0


def parse_float_list(text, flag):
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            fail("%s: %r is not a number" % (flag, token))
    if not values:
        fail("%s: no values given" % flag)
    return values


# ------------------------------------------------------------------- main ----

def build_parser():
    parser = argparse.ArgumentParser(
        prog="metric_explorer.py",
        description="Compute, sweep, and cost-weight classification metrics (stdlib only).",
    )
    subparsers = parser.add_subparsers(dest="command")

    confusion = subparsers.add_parser("confusion", help="metrics from a confusion matrix")
    confusion.add_argument("--tp", type=int, required=True)
    confusion.add_argument("--fp", type=int, required=True)
    confusion.add_argument("--fn", type=int, required=True)
    confusion.add_argument("--tn", type=int, required=True)
    confusion.add_argument("--fp-cost", type=float, default=1.0)
    confusion.add_argument("--fn-cost", type=float, default=1.0)
    confusion.set_defaults(handler=command_confusion)

    sweep = subparsers.add_parser("sweep", help="threshold sweep with a cost-weighted optimum")
    sweep.add_argument("--scores", help="CSV of 'score,label' rows, or - for stdin")
    sweep.add_argument("--positives", type=int, default=200, help="synthetic positives if no --scores")
    sweep.add_argument("--negatives", type=int, default=9800, help="synthetic negatives if no --scores")
    sweep.add_argument("--separation", type=float, default=2.0,
                       help="synthetic class separation in standard deviations")
    sweep.add_argument("--seed", type=int, default=7)
    sweep.add_argument("--fp-cost", type=float, default=1.0)
    sweep.add_argument("--fn-cost", type=float, default=20.0)
    sweep.add_argument("--max-rows", type=int, default=18, help="0 shows every threshold")
    sweep.set_defaults(handler=command_sweep)

    imbalance = subparsers.add_parser("imbalance", help="ROC-AUC vs PR-AUC as prevalence falls")
    imbalance.add_argument("--positives", type=int, default=200)
    imbalance.add_argument("--prevalences", default="0.5,0.1,0.01,0.001")
    imbalance.add_argument("--separation", type=float, default=2.0)
    imbalance.add_argument("--seed", type=int, default=7)
    imbalance.set_defaults(handler=command_imbalance)

    return parser


def main(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return EXIT_BAD_INPUT
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        raise SystemExit(0)
    except KeyboardInterrupt:
        raise SystemExit(130)
