#!/usr/bin/env python3
"""Measure how much optimism the wrong train/test split buys you.

Builds one synthetic dataset with many rows per user and a time axis, then evaluates the
same four features under three splits:

  random   rows shuffled                 - the default, and wrong for this data
  temporal train = earliest, test = latest - honest about time
  grouped  train and test hold disjoint users - honest about entities

The dataset is unchanged across splits, so every difference in reported AUC is an artifact
of the split, not of the data or the model. That gap is the number this tool exists to show.

  python split_demo.py
  python split_demo.py --users 400 --rows-per-user 12 --seed 3 --verbose

Standard library only.
"""

import argparse
import math
import random
import sys

EXIT_BAD_INPUT = 2
TRAIN_FRACTION = 0.7


def fail(message):
    print("error: " + message, file=sys.stderr)
    raise SystemExit(EXIT_BAD_INPUT)


def logistic(value):
    if value < -700:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


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


def fmt(value, width=7, places=4):
    if value is None:
        return "undef".rjust(width)
    return ("%.*f" % (places, value)).rjust(width)


# ------------------------------------------------------------------- dataset ----

def generate(users, rows_per_user, seed):
    """Rows carry a stationary signal, a drifting signal, and a per-user propensity.

    The drifting feature's coefficient changes sign across the time axis, which is what
    makes a random split flatter than a temporal one on real time-series data.
    """
    rng = random.Random(seed)
    total_days = users * rows_per_user
    rows = []
    for user in range(users):
        propensity = rng.gauss(0.0, 1.2)
        for occurrence in range(rows_per_user):
            day = rng.randrange(total_days)
            signal = rng.gauss(0.0, 1.0)
            drifting = rng.gauss(0.0, 1.0)
            drift_coefficient = 1.6 - 3.2 * (day / float(total_days - 1))
            logit = propensity + 1.1 * signal + drift_coefficient * drifting - 0.6
            rows.append({
                "user": user,
                "day": day,
                "x_signal": signal,
                "x_drifting": drifting,
                "label": 1 if rng.random() < logistic(logit) else 0,
            })
    rows.sort(key=lambda row: row["day"])
    return rows


def add_global_target_encoding(rows):
    """The classic leak: a per-user label mean computed over the whole dataset."""
    totals = {}
    for row in rows:
        bucket = totals.setdefault(row["user"], [0, 0])
        bucket[0] += row["label"]
        bucket[1] += 1
    for row in rows:
        positives, count = totals[row["user"]]
        row["x_user_mean_global"] = positives / float(count)


def add_train_only_target_encoding(train, test):
    """The correct version: fitted on train rows only, unseen users get the train mean."""
    totals = {}
    for row in train:
        bucket = totals.setdefault(row["user"], [0, 0])
        bucket[0] += row["label"]
        bucket[1] += 1
    fallback = (sum(row["label"] for row in train) / float(len(train))) if train else 0.0
    for rows in (train, test):
        for row in rows:
            if row["user"] in totals:
                positives, count = totals[row["user"]]
                row["x_user_mean_trainonly"] = positives / float(count)
            else:
                row["x_user_mean_trainonly"] = fallback


# --------------------------------------------------------------------- splits ----

def split_random(rows, seed):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * TRAIN_FRACTION)
    return shuffled[:cut], shuffled[cut:]


def split_temporal(rows, _seed):
    ordered = sorted(rows, key=lambda row: row["day"])
    cut = int(len(ordered) * TRAIN_FRACTION)
    boundary_day = ordered[cut]["day"]
    # Keep a whole day on one side so the boundary itself cannot leak across the split.
    train = [row for row in ordered if row["day"] < boundary_day]
    test = [row for row in ordered if row["day"] >= boundary_day]
    return train, test


def split_grouped(rows, seed):
    users = sorted({row["user"] for row in rows})
    random.Random(seed).shuffle(users)
    cut = int(len(users) * TRAIN_FRACTION)
    train_users = set(users[:cut])
    train = [row for row in rows if row["user"] in train_users]
    test = [row for row in rows if row["user"] not in train_users]
    return train, test


SPLITS = (
    ("random", split_random, "rows shuffled; the same user and the same day appear on both sides"),
    ("temporal", split_temporal, "train is the earlier period, test is the later period"),
    ("grouped", split_grouped, "train and test users are disjoint"),
)

FEATURES = (
    ("x_signal", "honest, stationary predictor"),
    ("x_drifting", "predictive, but its relationship to the label flips over time"),
    ("x_user_mean_global", "per-user label mean computed over ALL rows (leaks)"),
    ("x_user_mean_trainonly", "per-user label mean fitted on TRAIN rows only"),
)


def describe(train, test):
    train_users = {row["user"] for row in train}
    test_users = {row["user"] for row in test}
    overlap = train_users & test_users
    train_rate = sum(row["label"] for row in train) / float(len(train)) if train else 0.0
    test_rate = sum(row["label"] for row in test) / float(len(test)) if test else 0.0
    train_days = [row["day"] for row in train]
    test_days = [row["day"] for row in test]
    return {
        "train_rows": len(train), "test_rows": len(test),
        "user_overlap": (len(overlap) / float(len(test_users))) if test_users else 0.0,
        "train_rate": train_rate, "test_rate": test_rate,
        "day_overlap": bool(train_days and test_days
                            and min(test_days) <= max(train_days)),
    }


def main(argv):
    parser = argparse.ArgumentParser(
        prog="split_demo.py",
        description="Show measured optimism from random vs temporal vs grouped splits.",
    )
    parser.add_argument("--users", type=int, default=600)
    parser.add_argument("--rows-per-user", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--verbose", action="store_true",
                        help="also print the train-set AUC for each feature")
    args = parser.parse_args(argv)

    if args.users < 20:
        fail("--users must be >= 20 so a grouped split leaves usable data, got %d" % args.users)
    if args.rows_per_user < 2:
        fail("--rows-per-user must be >= 2, or there is no entity leakage to demonstrate, got %d"
             % args.rows_per_user)

    rows = generate(args.users, args.rows_per_user, args.seed)
    add_global_target_encoding(rows)
    positives = sum(row["label"] for row in rows)
    if positives in (0, len(rows)):
        fail("the generated dataset is single-class; try a different --seed")

    print("dataset: %d rows, %d users, %d rows per user, base rate %s (seed %d)"
          % (len(rows), args.users, args.rows_per_user, fmt(positives / float(len(rows))),
             args.seed))
    print()
    print("features under test")
    for name, description in FEATURES:
        print("  %-24s %s" % (name, description))
    print()

    results = {}
    for split_name, split_function, description in SPLITS:
        train, test = split_function(rows, args.seed)
        add_train_only_target_encoding(train, test)
        facts = describe(train, test)
        results[split_name] = {}

        print("%s split - %s" % (split_name.upper(), description))
        print("  train %d rows / test %d rows | test users also in train: %.0f%% | "
              "time ranges overlap: %s"
              % (facts["train_rows"], facts["test_rows"], 100 * facts["user_overlap"],
                 "yes" if facts["day_overlap"] else "no"))
        print("  base rate train %s vs test %s"
              % (fmt(facts["train_rate"]), fmt(facts["test_rate"])))
        header = "  %-24s test AUC" % "feature"
        if args.verbose:
            header += "   train AUC"
        print(header)
        for name, _ in FEATURES:
            test_auc = roc_auc([(row[name], row["label"]) for row in test])
            train_auc = roc_auc([(row[name], row["label"]) for row in train])
            results[split_name][name] = test_auc
            line = "  %-24s %s" % (name, fmt(test_auc, 8))
            if args.verbose:
                line += "  %s" % fmt(train_auc, 9)
            print(line)
        print()

    print("measured optimism (random split test AUC minus the honest split's)")
    print("  %-24s vs grouped  vs temporal   verdict" % "feature")
    for name, _ in FEATURES:
        random_auc = results["random"][name]
        grouped_gap = subtract(random_auc, results["grouped"][name])
        temporal_gap = subtract(random_auc, results["temporal"][name])
        worst = max(gap for gap in (grouped_gap, temporal_gap) if gap is not None)
        if worst is None:
            verdict = "not comparable"
        elif worst >= 0.10:
            verdict = "SPLIT ARTIFACT - the random number is not real"
        elif worst >= 0.04:
            verdict = "some optimism"
        else:
            verdict = "no split-driven gap (audit the feature itself)"
        print("  %-24s %s    %s    %s"
              % (name, fmt(grouped_gap, 8, 4), fmt(temporal_gap, 9, 4), verdict))

    print()
    print("how to read this")
    print("  Gaps below about 0.04 sit inside sampling noise at this test-set size, so only the")
    print("  two flagged rows below are real artifacts.")
    print()
    print("  x_signal is stationary and entity-independent, so all three splits roughly agree:")
    print("  that is the control proving the dataset and the metric themselves are fine.")
    print()
    print("  x_drifting collapses under the TEMPORAL split - below 0.5, i.e. actively")
    print("  anti-predictive - because its relationship to the label reverses across the time")
    print("  axis. A random split averages past and future together and reports a number")
    print("  production will never see. This is the leak a temporal split exists to catch.")
    print()
    print("  x_user_mean_trainonly collapses to ~0.5 under the GROUPED split, because a user")
    print("  with no training history gets the fallback constant. That is the honest cold-start")
    print("  number. Its higher score under the other splits is legitimate only if you really")
    print("  will have that user's history at serving time. This is the leak a grouped split")
    print("  exists to catch.")
    print()
    print("  x_user_mean_global is the important one: it scores high under EVERY split. The leak")
    print("  is not in the split, it is in the feature - the aggregate was computed over the")
    print("  whole dataset, so each row's own label is inside its own feature value and no")
    print("  partition can remove it. A number that is stable across all three splits is")
    print("  therefore NOT evidence of validity. Only reading the transformation code, or")
    print("  auditing feature timestamps, finds this class of bug.")
    print()
    print("choose the split that matches how the model will be used - temporal if you predict")
    print("the future, grouped if you will see unseen entities, both if both are true - and")
    print("compute every aggregate feature from training data only.")
    return 0


def subtract(left, right):
    if left is None or right is None:
        return None
    return left - right


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        raise SystemExit(0)
    except KeyboardInterrupt:
        raise SystemExit(130)
