#!/usr/bin/env python3
"""Flag features that could not have been known at prediction time.

Three independent checks:
  1. TIME TRAVEL  - a feature whose own timestamp is at or after the label timestamp.
  2. SEPARATION   - a single feature that separates the label almost perfectly.
  3. NAMING       - a feature whose name suggests it encodes the outcome.

None of these is proof of leakage, and the tool says so: they are the three cheap screens
that catch the overwhelming majority of real cases before a model is ever trained.

  python leakage_detector.py --demo
  python leakage_detector.py --data features.csv --label-col label --label-time-col label_time
  python leakage_detector.py --write-demo _demo

Data convention: for a feature column `X`, an optional companion column `X_ts` (configurable
via --ts-suffix) carries the time that feature's value became observable.

Exit status: 0 clean, 1 findings, 2 bad usage.
Standard library only.
"""

import argparse
import csv
import math
import os
import random
import sys
from datetime import datetime, timedelta

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_BAD_INPUT = 2

MAX_EXAMPLES = 3
DEFAULT_TS_SUFFIX = "_ts"
DEFAULT_AUC_WARN = 0.99
# Substrings that, in practice, almost always indicate a column derived from the outcome.
SUSPICIOUS_NAME_PARTS = (
    "label", "target", "outcome", "is_fraud", "fraud_flag", "chargeback", "refund",
    "cancel", "churn", "final", "resolution", "closed", "settled", "reversed",
    "confirmed", "disposition", "verdict", "future", "next_", "post_",
)


def fail(message):
    print("error: " + message, file=sys.stderr)
    raise SystemExit(EXIT_BAD_INPUT)


# --------------------------------------------------------------------- metrics ----

def roc_auc(pairs):
    """Rank-based ROC-AUC with mid-rank tie correction; None if one class is missing."""
    positives = sum(1 for _, label in pairs if label == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average_rank = (index + 1 + stop + 1) / 2.0
        for position in range(index, stop + 1):
            if ordered[position][1] == 1:
                rank_sum += average_rank
        index = stop + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def category_purity(values_and_labels):
    """Largest (coverage, purity) group: a category that is almost always one label."""
    groups = {}
    for value, label in values_and_labels:
        bucket = groups.setdefault(value, [0, 0])
        bucket[label] += 1
    total = float(len(values_and_labels)) or 1.0
    worst = None
    for value, (negatives, positives) in groups.items():
        count = negatives + positives
        purity = max(negatives, positives) / float(count)
        coverage = count / total
        if worst is None or (purity, coverage) > (worst[2], worst[1]):
            worst = (value, coverage, purity, count)
    return worst, len(groups)


# ----------------------------------------------------------------------- input ----

def parse_timestamp(value):
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def read_table(path):
    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as error:
        fail("cannot read data file %r: %s" % (path, error.strerror or error))
    if len(rows) < 2:
        fail("data file %r needs a header row plus at least one data row" % path)
    header = [field.strip() for field in rows[0]]
    records = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or all(field.strip() == "" for field in row):
            continue
        if len(row) != len(header):
            fail("data file %r line %d has %d fields but the header declares %d"
                 % (path, line_number, len(row), len(header)))
        records.append((line_number, dict(zip(header, [field.strip() for field in row]))))
    if not records:
        fail("data file %r has a header but no data rows" % path)
    return header, records


# ------------------------------------------------------------------ demo data ----

DEMO_ROWS = 240
DEMO_SEED = 11


def build_demo_rows():
    """A fraud-style table with three honest features and three planted leaks.

    Each planted leak is caught by a *different* check, which is the point: no single
    screen would have found all three.
    """
    rng = random.Random(DEMO_SEED)
    header = [
        "claim_id", "label", "label_time",
        "amount", "amount_ts",
        "account_age_days", "account_age_days_ts",
        "prior_disputes_90d", "prior_disputes_90d_ts",
        "risk_bucket", "risk_bucket_ts",
        "investigator_notes_len", "investigator_notes_len_ts",
        "chargeback_amount", "chargeback_amount_ts",
        "case_status_at_review", "case_status_at_review_ts",
    ]
    rows = [header]
    base = datetime(2026, 4, 1, 8, 0, 0)
    for index in range(DEMO_ROWS):
        decision_time = base + timedelta(hours=index)
        label_time = decision_time + timedelta(days=30)
        latent = rng.gauss(0.0, 1.0)
        label = 1 if rng.random() < 1.0 / (1.0 + math.exp(-(latent - 1.4))) else 0

        amount = round(abs(rng.gauss(120.0, 60.0)) + 40.0 * label, 2)
        account_age = rng.randint(5, 2500)
        prior_disputes = max(0, int(round(rng.gauss(0.4 + 1.6 * label, 0.8))))
        risk_bucket = rng.choice(["low", "medium", "high"])
        # Leak 1: written by an investigator only after the case resolves. Caught by the
        # timestamp check alone; its name looks innocent.
        notes_len = rng.randint(400, 1200) if label else rng.randint(0, 60)
        # Leak 2: non-zero only for confirmed fraud. Caught by all three checks.
        chargeback = round(amount, 2) if label else 0.0
        # Leak 3: stamped at decision time, so the timestamp check passes, and the name
        # trips no keyword. Only the separation check catches this one.
        case_status = ("settled" if label else rng.choice(["open", "escalated"]))

        rows.append([
            "c%04d" % (index + 1), str(label), label_time.isoformat(),
            "%.2f" % amount, decision_time.isoformat(),
            str(account_age), decision_time.isoformat(),
            str(prior_disputes), decision_time.isoformat(),
            risk_bucket, decision_time.isoformat(),
            str(notes_len), (label_time + timedelta(days=2)).isoformat(),
            "%.2f" % chargeback, (label_time + timedelta(hours=1)).isoformat(),
            case_status, decision_time.isoformat(),
        ])
    return rows


def write_demo(directory):
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "features.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(build_demo_rows())
    except OSError as error:
        fail("cannot write demo file into %r: %s" % (directory, error.strerror or error))
    print("wrote %s" % path)
    print("now run:")
    print("  python %s --data %s --label-col label --label-time-col label_time"
          % (os.path.basename(__file__), path))
    return EXIT_OK


# ------------------------------------------------------------------- checking ----

def find_feature_columns(header, label_col, label_time_col, ts_suffix):
    reserved = {label_col, label_time_col}
    timestamp_columns = {name for name in header if name.endswith(ts_suffix)}
    features = []
    for name in header:
        if name in reserved or name in timestamp_columns:
            continue
        features.append((name, name + ts_suffix if name + ts_suffix in timestamp_columns else None))
    orphans = sorted(name for name in timestamp_columns
                     if name[:-len(ts_suffix)] not in header and name not in reserved)
    return features, orphans


def check_time_travel(records, feature, ts_column, label_time_col):
    violations = []
    unparsed = 0
    compared = 0
    for line_number, record in records:
        label_time = parse_timestamp(record.get(label_time_col, ""))
        feature_time = parse_timestamp(record.get(ts_column, ""))
        if label_time is None or feature_time is None:
            unparsed += 1
            continue
        compared += 1
        if feature_time >= label_time:
            violations.append("line %d: %s at %s >= label at %s"
                              % (line_number, feature, feature_time.isoformat(),
                                 label_time.isoformat()))
    return violations, compared, unparsed


def collect_feature_values(records, feature, labels):
    numeric = []
    categorical = []
    numeric_ok = True
    for (line_number, record), label in zip(records, labels):
        if label is None:
            continue
        raw = record.get(feature, "")
        if raw == "":
            continue
        categorical.append((raw, label))
        if numeric_ok:
            try:
                numeric.append((float(raw), label))
            except ValueError:
                numeric_ok = False
    return (numeric if numeric_ok else None), categorical


def main(argv):
    parser = argparse.ArgumentParser(
        prog="leakage_detector.py",
        description="Flag time-travel leakage and suspiciously perfect features in a CSV.",
    )
    parser.add_argument("--data", help="path to the feature CSV")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--label-time-col", default="label_time")
    parser.add_argument("--ts-suffix", default=DEFAULT_TS_SUFFIX,
                        help="suffix naming a feature's observability timestamp column")
    parser.add_argument("--auc-warn", type=float, default=DEFAULT_AUC_WARN,
                        help="single-feature ROC-AUC at or above this is reported as suspicious")
    parser.add_argument("--demo", action="store_true", help="run against built-in sample data")
    parser.add_argument("--write-demo", metavar="DIR", help="write the sample CSV to DIR")
    args = parser.parse_args(argv)

    if args.write_demo:
        return write_demo(args.write_demo)
    if not 0.5 < args.auc_warn <= 1.0:
        fail("--auc-warn must be in (0.5, 1.0], got %g" % args.auc_warn)
    if not args.ts_suffix:
        fail("--ts-suffix must not be empty")

    if args.demo:
        if args.data:
            fail("--demo uses built-in data; do not combine it with --data")
        rows = build_demo_rows()
        header = rows[0]
        records = [(index, dict(zip(header, row))) for index, row in enumerate(rows[1:], start=2)]
        source = "built-in demo table (%d rows, 3 planted leaks)" % DEMO_ROWS
    else:
        if not args.data:
            fail("give --data PATH, or use --demo (see --help)")
        header, records = read_table(args.data)
        source = args.data

    if args.label_col not in header:
        fail("label column %r not found; columns are: %s" % (args.label_col, ", ".join(header)))
    if args.label_time_col not in header:
        fail("label time column %r not found; columns are: %s"
             % (args.label_time_col, ", ".join(header)))

    labels = []
    bad_labels = []
    for line_number, record in records:
        token = record.get(args.label_col, "")
        if token in ("0", "1"):
            labels.append(int(token))
        else:
            labels.append(None)
            bad_labels.append("line %d: %r" % (line_number, token))
    usable = sum(1 for label in labels if label is not None)
    if usable == 0:
        fail("no usable 0/1 values in label column %r (checked %d rows)"
             % (args.label_col, len(records)))
    if len(set(label for label in labels if label is not None)) < 2:
        fail("label column %r has only one class; separation checks need both" % args.label_col)

    features, orphans = find_feature_columns(header, args.label_col, args.label_time_col,
                                             args.ts_suffix)
    if not features:
        fail("no feature columns found besides %r and %r" % (args.label_col, args.label_time_col))

    print("leakage report")
    print("  source        : %s" % source)
    print("  rows          : %d (%d with a usable 0/1 label)" % (len(records), usable))
    print("  label column  : %s   label time column: %s" % (args.label_col, args.label_time_col))
    print("  features      : %d   timestamp suffix: %r" % (len(features), args.ts_suffix))
    if bad_labels:
        print("  NOTE: %d row(s) had an unusable label and were skipped, e.g. %s"
              % (len(bad_labels), "; ".join(bad_labels[:MAX_EXAMPLES])))
    if orphans:
        print("  NOTE: timestamp column(s) with no matching feature: %s" % ", ".join(orphans))
    print()

    findings = []
    no_timestamp = []
    for feature, ts_column in features:
        severities = []

        if ts_column is None:
            no_timestamp.append(feature)
        else:
            violations, compared, unparsed = check_time_travel(records, feature, ts_column,
                                                              args.label_time_col)
            if violations:
                severities.append(("LEAK", "time-travel",
                                   "%d of %d comparable rows have %s observed at or after the label"
                                   % (len(violations), compared, feature),
                                   violations[:MAX_EXAMPLES]))
            if unparsed:
                severities.append(("WARN", "unparsed-time",
                                   "%d row(s) had an unparseable %s or %s timestamp (ISO 8601 "
                                   "expected), so those rows were not checked"
                                   % (unparsed, ts_column, args.label_time_col), []))

        numeric, categorical = collect_feature_values(records, feature, labels)
        if numeric and len(numeric) >= 10:
            auc = roc_auc(numeric)
            if auc is not None:
                extreme = max(auc, 1.0 - auc)
                if extreme >= args.auc_warn:
                    severities.append(("SUSPECT", "separation",
                                       "single-feature ROC-AUC %.4f (>= %.2f) - one threshold "
                                       "almost perfectly separates the label"
                                       % (auc, args.auc_warn), []))
        elif categorical and len(categorical) >= 10:
            worst, group_count = category_purity(categorical)
            if worst and group_count > 1:
                value, coverage, purity, count = worst
                if purity >= args.auc_warn and coverage >= 0.05:
                    severities.append(("SUSPECT", "separation",
                                       "category %r covers %.1f%% of rows and is %.1f%% one label"
                                       % (value, 100 * coverage, 100 * purity), []))

        lowered = feature.lower()
        matched = [part for part in SUSPICIOUS_NAME_PARTS if part in lowered]
        if matched:
            severities.append(("WARN", "naming",
                               "name contains %s - names like this usually mark a column derived "
                               "from the outcome" % ", ".join(repr(part) for part in matched), []))

        if severities:
            findings.append((feature, severities))

    if not findings:
        print("  no findings")
    for feature, severities in findings:
        for severity, check, message, examples in severities:
            print("  [%-7s] %-14s %-24s %s" % (severity, check, feature, message))
            for example in examples:
                print("               %s" % example)
    if no_timestamp:
        print()
        print("  NOT CHECKED for time travel (no %r companion column): %s"
              % (args.ts_suffix, ", ".join(no_timestamp)))
        print("  A feature with no recorded observability time cannot be audited. Adding that")
        print("  timestamp to the pipeline is the fix, not tuning this tool.")

    leaks = sum(1 for _, severities in findings
                for severity, _, _, _ in severities if severity in ("LEAK", "SUSPECT"))
    warnings = sum(1 for _, severities in findings
                   for severity, _, _, _ in severities if severity == "WARN")
    print()
    print("summary: %d leak/suspect finding(s), %d warning(s) across %d feature(s)"
          % (leaks, warnings, len(features)))
    if leaks:
        print("GATE: REJECT - resolve or drop these features before training")
        print("Reminder: SUSPECT is a screen, not a verdict. A feature can be legitimately")
        print("near-perfect (a hard business rule). Confirm by asking whether its value was")
        print("available at the decision moment - not by looking at the metric.")
        return EXIT_FINDINGS
    print("GATE: PASS - no time-travel or separation findings")
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        raise SystemExit(EXIT_OK)
    except KeyboardInterrupt:
        raise SystemExit(130)
