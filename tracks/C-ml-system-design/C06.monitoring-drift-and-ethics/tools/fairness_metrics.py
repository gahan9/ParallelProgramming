#!/usr/bin/env python3
"""Group fairness metrics, and a demonstration that they conflict.

Standard library only (math, random, csv, argparse, sys). No numpy, no scipy.

Given (score, label, group) triples this computes, per group:
  selection rate, TPR, FPR, precision (PPV), NPV, accuracy, base rate,
and across groups:
  demographic parity difference, equal-opportunity difference,
  equalized-odds difference, and calibration by group.

Then it does the part that matters: it SEARCHES for the per-group thresholds
that equalize each criterion in turn, and shows what that does to the others.

Why this is the point
---------------------
When base rates differ between groups, several of these criteria cannot hold
simultaneously except in degenerate cases -- this is a theorem, not an
engineering shortfall (Chouldechova 2017; Kleinberg, Mullainathan & Raghavan
2016). A calibrated score with unequal base rates cannot also equalize false
positive and false negative rates.

The practical consequence is not "fairness is impossible". It is that
"is this model fair?" has no answer until someone states WHICH criterion
applies and WHY. That choice is a decision about who bears the cost of the
model's errors, it belongs to the people accountable for the domain, and it
must be written down and justified -- not selected because it happened to
produce a passing number.

This tool takes no position on which criterion is right. It shows you the
tradeoff surface and the size of each cost so the choice can be made with
numbers in front of it.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys


# --------------------------------------------------------------------------
# Confusion metrics
# --------------------------------------------------------------------------


def confusion(rows, threshold):
    """rows: list of (score, label). Returns tp, fp, tn, fn."""
    tp = fp = tn = fn = 0
    for score, label in rows:
        predicted = score >= threshold
        if label and predicted:
            tp += 1
        elif label and not predicted:
            fn += 1
        elif not label and predicted:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def group_metrics(rows, threshold):
    tp, fp, tn, fn = confusion(rows, threshold)
    n = tp + fp + tn + fn
    return {
        "n": n,
        "threshold": threshold,
        "base_rate": _ratio(tp + fn, n),
        "selection_rate": _ratio(tp + fp, n),
        "tpr": _ratio(tp, tp + fn),
        "fpr": _ratio(fp, fp + tn),
        "precision": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "accuracy": _ratio(tp + tn, n),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _spread(values):
    live = [v for v in values if not math.isnan(v)]
    return max(live) - min(live) if len(live) >= 2 else 0.0


def summarize(per_group):
    """Cross-group fairness differences. All are max-minus-min across groups."""
    return {
        "demographic_parity_diff": _spread([m["selection_rate"] for m in per_group.values()]),
        "equal_opportunity_diff": _spread([m["tpr"] for m in per_group.values()]),
        "fpr_diff": _spread([m["fpr"] for m in per_group.values()]),
        "equalized_odds_diff": max(_spread([m["tpr"] for m in per_group.values()]),
                                   _spread([m["fpr"] for m in per_group.values()])),
        "precision_diff": _spread([m["precision"] for m in per_group.values()]),
        "accuracy_diff": _spread([m["accuracy"] for m in per_group.values()]),
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibration(rows, bins=5):
    """Per-bin mean predicted score vs observed positive rate, plus ECE."""
    buckets = [[] for _ in range(bins)]
    for score, label in rows:
        index = min(bins - 1, max(0, int(score * bins)))
        buckets[index].append((score, label))

    table = []
    ece = 0.0
    total = len(rows)
    for index, bucket in enumerate(buckets):
        if not bucket:
            table.append({"bin": index, "n": 0, "mean_score": float("nan"),
                          "observed": float("nan")})
            continue
        mean_score = sum(s for s, _ in bucket) / len(bucket)
        observed = sum(1 for _, y in bucket if y) / len(bucket)
        table.append({"bin": index, "n": len(bucket), "mean_score": mean_score,
                      "observed": observed})
        ece += (len(bucket) / total) * abs(mean_score - observed)
    return table, ece


# --------------------------------------------------------------------------
# Threshold search
# --------------------------------------------------------------------------


def candidate_thresholds(rows, steps=400):
    """Thresholds to search. Score-derived, so the search adapts to the data."""
    if not rows:
        return [0.5]
    scores = sorted(s for s, _ in rows)
    lo, hi = scores[0], scores[-1]
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / steps for i in range(steps + 1)]


def thresholds_matching(groups, target, metric_name):
    """Per-group thresholds whose `metric_name` is closest to `target`."""
    chosen = {}
    for name, rows in groups.items():
        best_threshold, best_error = 0.5, float("inf")
        for threshold in candidate_thresholds(rows):
            value = group_metrics(rows, threshold)[metric_name]
            if math.isnan(value):
                continue
            error = abs(value - target)
            if error < best_error:
                best_threshold, best_error = threshold, error
        chosen[name] = best_threshold
    return chosen


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_group_table(per_group, title):
    print()
    print("  %s" % title)
    header = ("    %-10s %7s %8s %8s %7s %7s %8s %7s %8s"
              % ("group", "n", "thresh", "base", "sel", "TPR", "FPR", "PPV", "acc"))
    print(header)
    print("    " + "-" * (len(header) - 4))
    for name in sorted(per_group):
        m = per_group[name]
        print("    %-10s %7d %8.3f %8.3f %7.3f %7.3f %8.3f %7.3f %8.3f"
              % (name, m["n"], m["threshold"], m["base_rate"], m["selection_rate"],
                 m["tpr"], m["fpr"], m["precision"], m["accuracy"]))


def print_summary(summary, label):
    print()
    print("    %-38s %s" % ("cross-group differences (max - min)", label))
    for key, pretty in (("demographic_parity_diff", "demographic parity difference"),
                        ("equal_opportunity_diff", "equal-opportunity difference (TPR)"),
                        ("fpr_diff", "false-positive-rate difference"),
                        ("equalized_odds_diff", "equalized-odds difference"),
                        ("precision_diff", "precision (PPV) difference"),
                        ("accuracy_diff", "accuracy difference")):
        print("      %-40s %.4f" % (pretty, summary[key]))


def print_calibration(groups, bins):
    print()
    print("  Calibration by group -- mean predicted score vs observed positive rate")
    print("    %-10s %6s %14s %16s %8s" % ("group", "bin", "mean score", "observed rate", "n"))
    print("    " + "-" * 58)
    for name in sorted(groups):
        table, ece = calibration(groups[name], bins)
        for row in table:
            if row["n"] == 0:
                continue
            print("    %-10s %6d %14.3f %16.3f %8d"
                  % (name, row["bin"], row["mean_score"], row["observed"], row["n"]))
        print("    %-10s %s ECE = %.4f" % (name, " " * 30, ece))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def synthetic_data(seed, n_a, n_b, base_rate_a, base_rate_b, noise_a, noise_b):
    """Two groups with different base rates and a slightly noisier score for group B.

    The score is deliberately built to be roughly calibrated overall -- it is
    the *probability* used to draw the label. That matters: the conflict this
    tool demonstrates is not caused by a broken model. It appears for a
    well-calibrated, unbiased-by-construction score, purely because the base
    rates differ.
    """
    rng = random.Random(seed)
    groups = {}
    for name, n, base, noise in (("A", n_a, base_rate_a, noise_a),
                                 ("B", n_b, base_rate_b, noise_b)):
        rows = []
        for _ in range(n):
            latent = rng.gauss(0.0, 1.0)
            logit = math.log(base / (1.0 - base)) + 1.6 * latent
            probability = 1.0 / (1.0 + math.exp(-logit))
            label = rng.random() < probability
            # The model observes the true probability through measurement noise.
            observed = min(0.999, max(0.001, probability + rng.gauss(0.0, noise)))
            rows.append((observed, label))
        groups[name] = rows
    return groups


def load_csv(path):
    """CSV with columns score,label,group."""
    groups = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"score", "label", "group"}
            if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
                raise ValueError("CSV must have a header with columns: score,label,group "
                                 "(found: %s)" % (reader.fieldnames,))
            for line_number, row in enumerate(reader, start=2):
                try:
                    score = float(row["score"])
                    label = str(row["label"]).strip().lower() in ("1", "true", "yes", "y")
                except (TypeError, ValueError):
                    raise ValueError("line %d: score must be a number and label must be "
                                     "0/1 or true/false" % line_number)
                groups.setdefault(row["group"], []).append((score, label))
    except FileNotFoundError:
        raise ValueError("no such file: %s" % path)
    if len(groups) < 2:
        raise ValueError("need at least two groups to compute a group-fairness difference; "
                         "found %d" % len(groups))
    return groups


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(args):
    if args.data:
        groups = load_csv(args.data)
        source = args.data
    else:
        groups = synthetic_data(args.seed, args.n_a, args.n_b,
                                args.base_rate_a, args.base_rate_b,
                                args.noise_a, args.noise_b)
        source = "synthetic, seed %d (SIMULATED DATA)" % args.seed

    for name, rows in groups.items():
        if len(rows) < 100:
            print("WARNING: group %r has only %d rows. Group-fairness metrics on small "
                  "slices are extremely noisy; treat these numbers as indicative only."
                  % (name, len(rows)))

    print("=" * 84)
    print("GROUP FAIRNESS METRICS AND THEIR CONFLICTS")
    print("  data: %s" % source)
    print("=" * 84)

    # --- Scenario 1: one shared threshold -----------------------------------
    shared = {name: group_metrics(rows, args.threshold) for name, rows in groups.items()}
    print()
    print("SCENARIO 1 -- one threshold for everyone (threshold = %.2f)" % args.threshold)
    print("  The default. Often called 'fairness through unawareness'; note that it")
    print("  equalizes the DECISION RULE, not any outcome.")
    print_group_table(shared, "per-group outcomes")
    print_summary(summarize(shared), "")
    print_calibration(groups, args.calibration_bins)

    # --- Scenario 2: equalize TPR (equal opportunity) -----------------------
    target_tpr = sum(m["tpr"] for m in shared.values()) / len(shared)
    eo_thresholds = thresholds_matching(groups, target_tpr, "tpr")
    equal_opportunity = {name: group_metrics(groups[name], eo_thresholds[name])
                         for name in groups}
    print()
    print("SCENARIO 2 -- equalize TPR across groups (EQUAL OPPORTUNITY, target TPR = %.3f)"
          % target_tpr)
    print("  'Among those who truly qualify, the same share are approved in every group.'")
    print_group_table(equal_opportunity, "per-group outcomes with per-group thresholds")
    print_summary(summarize(equal_opportunity), "")

    # --- Scenario 3: equalize selection rate (demographic parity) -----------
    target_selection = sum(m["selection_rate"] for m in shared.values()) / len(shared)
    dp_thresholds = thresholds_matching(groups, target_selection, "selection_rate")
    demographic_parity = {name: group_metrics(groups[name], dp_thresholds[name])
                          for name in groups}
    print()
    print("SCENARIO 3 -- equalize selection rate (DEMOGRAPHIC PARITY, target = %.3f)"
          % target_selection)
    print("  'The same share of each group is approved, regardless of outcome.'")
    print_group_table(demographic_parity, "per-group outcomes with per-group thresholds")
    print_summary(summarize(demographic_parity), "")

    # --- The comparison -----------------------------------------------------
    scenarios = [("1. shared threshold", shared),
                 ("2. equal opportunity", equal_opportunity),
                 ("3. demographic parity", demographic_parity)]

    print()
    print("=" * 84)
    print("THE TENSION, SIDE BY SIDE")
    print("=" * 84)
    header = ("  %-24s %10s %10s %10s %10s %10s"
              % ("criterion enforced", "DP diff", "EO diff", "FPR diff", "PPV diff", "accuracy"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, per_group in scenarios:
        s = summarize(per_group)
        total_n = sum(m["n"] for m in per_group.values())
        overall_accuracy = sum(m["accuracy"] * m["n"] for m in per_group.values()) / total_n
        print("  %-24s %10.4f %10.4f %10.4f %10.4f %10.4f"
              % (label, s["demographic_parity_diff"], s["equal_opportunity_diff"],
                 s["fpr_diff"], s["precision_diff"], overall_accuracy))

    print()
    print("READ THE DIAGONAL. Each row drives ONE column to (near) zero and moves the")
    print("others. Enforcing equal opportunity (row 2) zeroes the EO column and generally")
    print("worsens demographic parity and precision parity; enforcing demographic parity")
    print("(row 3) zeroes the DP column and moves TPR apart. There is no row that zeroes")
    print("every column, and that is not a limitation of this search -- it is the")
    print("impossibility result.")

    rates = {name: shared[name]["base_rate"] for name in shared}
    print()
    print("WHY -- the base rates differ: %s"
          % ", ".join("%s = %.3f" % (k, v) for k, v in sorted(rates.items())))
    print("  For a calibrated score with unequal base rates, you cannot simultaneously")
    print("  equalize false-positive and false-negative rates AND keep calibration within")
    print("  groups, except when the base rates are equal or the classifier is perfect")
    print("  (Chouldechova 2017; Kleinberg, Mullainathan & Raghavan 2016).")
    print()
    print("WHAT TO DO WITH THIS")
    print("-" * 84)
    print("  1. Pick the criterion from the DECISION, not from the dashboard. Ask: who")
    print("     bears the cost of each error type here?")
    print("       * A false positive that costs the INDIVIDUAL (wrongful denial, wrongful")
    print("         flag) argues for equalizing false-positive rates.")
    print("       * A false negative that denies an OPPORTUNITY (loan, interview, treatment)")
    print("         argues for equal opportunity -- equal TPR.")
    print("       * A legally-defined protected decision may impose demographic parity")
    print("         regardless of what you would choose. This is where you involve counsel;")
    print("         this tool cannot and does not tell you what the law requires.")
    print("  2. WRITE DOWN the choice and the reasoning, in the model card, before you")
    print("     measure. Choosing the criterion after seeing which one passes is the")
    print("     fairness equivalent of peeking.")
    print("  3. Report the cost. Row 2 and row 3 both change overall accuracy and change")
    print("     who is affected. Present that to the accountable owner as a tradeoff, not")
    print("     as a compliance checkbox.")
    print("  4. Check whether the disparity is upstream. Different base rates and different")
    print("     score noise across groups often come from measurement or representation")
    print("     bias in the data, and fixing THAT beats any post-hoc threshold adjustment.")
    print("  5. Per-group thresholds are themselves a consequential choice: in some")
    print("     jurisdictions and domains, using group membership in the decision rule is")
    print("     restricted. Again: counsel, not this tool.")


def build_parser():
    p = argparse.ArgumentParser(
        description="Per-group fairness metrics and a demonstration of their conflicts.")
    p.add_argument("--data", help="CSV with columns score,label,group (default: synthetic demo)")
    p.add_argument("--threshold", type=float, default=0.5, help="shared decision threshold")
    p.add_argument("--calibration-bins", type=int, default=5)
    p.add_argument("--seed", type=int, default=20240729)
    p.add_argument("--n-a", type=int, default=6000)
    p.add_argument("--n-b", type=int, default=3000)
    p.add_argument("--base-rate-a", type=float, default=0.38)
    p.add_argument("--base-rate-b", type=float, default=0.18)
    p.add_argument("--noise-a", type=float, default=0.05)
    p.add_argument("--noise-b", type=float, default=0.09)
    return p


def main(argv=None):
    if sys.version_info < (3, 8):
        sys.stderr.write("ERROR: this tool needs Python 3.8 or newer.\n")
        return 2
    args = build_parser().parse_args(argv)
    try:
        for name in ("threshold", "base_rate_a", "base_rate_b"):
            value = getattr(args, name)
            if not 0.0 < value < 1.0:
                raise ValueError("--%s must be strictly between 0 and 1, got %r"
                                 % (name.replace("_", "-"), value))
        if args.calibration_bins < 2:
            raise ValueError("--calibration-bins must be at least 2")
        run(args)
    except ValueError as exc:
        sys.stderr.write("ERROR: %s\n" % exc)
        sys.stderr.write("Run with -h for usage.\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
