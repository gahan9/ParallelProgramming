#!/usr/bin/env python3
"""Drift detection: PSI, two-sample KS, chi-square, and a drift-type classifier.

Standard library only (math, random, json, argparse, sys). No numpy, no scipy.

What it does
------------
  * Population Stability Index (PSI) on a numeric feature, using reference
    quantile bins.
  * Two-sample Kolmogorov-Smirnov statistic with the asymptotic p-value.
  * Chi-square test of homogeneity for a categorical feature, with the p-value
    from a locally-implemented regularized incomplete gamma function.
  * A drift-TYPE classifier: given reference/current samples of the inputs,
    the predictions, and (optionally) the labels or observed accuracy, it
    names the most likely kind of shift -- covariate, prior/label, or concept
    -- or says the evidence is insufficient, which is frequently the honest
    answer.

What it refuses to do
---------------------
It will not report a test result it does not believe. Every test carries an
explicit sample-adequacy check, and undersized samples are reported as
INCONCLUSIVE rather than as a number someone will paste into a slide. A
drift dashboard that fires constantly is worse than no drift dashboard,
because people turn it off.

Two warnings the tool prints and you should internalise:
  * Statistical significance is not practical significance. With a million
    rows, every feature drifts at p < 0.001 and almost none of it matters.
    Report effect size (PSI, KS D) alongside, and threshold on that.
  * Testing many features multiplies false positives. Correct for it, or
    accept that a 200-feature dashboard at alpha = 0.05 shows you ten
    "drifted" features every single day by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys

# Industry rules of thumb from credit-risk practice, not derived thresholds.
# Calibrate them against your own retraining history before trusting them.
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25

MIN_SAMPLES_PSI = 200          # below this, quantile bins are too noisy to mean anything
MIN_EXPECTED_CELL = 5.0        # standard chi-square validity condition
MIN_SAMPLES_KS = 30


# --------------------------------------------------------------------------
# Distribution functions, implemented locally
# --------------------------------------------------------------------------


def _gamma_series(a, x, iterations=300, tol=1e-12):
    """Lower regularized incomplete gamma P(a, x) by series expansion."""
    if x <= 0.0:
        return 0.0
    term = 1.0 / a
    total = term
    n = a
    for _ in range(iterations):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * tol:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(a, x, iterations=300, tol=1e-12):
    """Upper regularized incomplete gamma Q(a, x) by continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(statistic, dof):
    """Upper tail P(X > statistic) for a chi-square with `dof` degrees of freedom."""
    if statistic <= 0.0:
        return 1.0
    if dof <= 0:
        raise ValueError("degrees of freedom must be positive")
    a = dof / 2.0
    x = statistic / 2.0
    # The series converges quickly below the mean, the continued fraction above it.
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_continued_fraction(a, x)


def ks_sf(lam, terms=120):
    """Kolmogorov distribution upper tail Q(lambda) = 2 * sum (-1)^(j-1) exp(-2 j^2 lam^2)."""
    if lam <= 0.0:
        return 1.0
    total = 0.0
    for j in range(1, terms + 1):
        term = 2.0 * ((-1.0) ** (j - 1)) * math.exp(-2.0 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return max(0.0, min(1.0, total))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def quantile_edges(values, bins):
    """Bin edges at equal-count quantiles of the reference sample."""
    ordered = sorted(values)
    n = len(ordered)
    edges = []
    for i in range(1, bins):
        position = i * n / bins
        low = int(math.floor(position)) - 1
        low = max(0, min(n - 1, low))
        edges.append(ordered[low])
    # Collapse duplicate edges: a feature with heavy point mass (lots of zeros)
    # cannot support `bins` distinct quantiles, and silently keeping duplicate
    # edges produces empty bins and a meaningless PSI.
    unique = []
    for e in edges:
        if not unique or e > unique[-1]:
            unique.append(e)
    return unique


def bucketize(values, edges):
    counts = [0] * (len(edges) + 1)
    for v in values:
        index = 0
        while index < len(edges) and v > edges[index]:
            index += 1
        counts[index] += 1
    return counts


def psi(reference, current, bins=10):
    """Population Stability Index with per-bin contributions."""
    if len(reference) < MIN_SAMPLES_PSI or len(current) < MIN_SAMPLES_PSI:
        return {
            "psi": None,
            "verdict": "INCONCLUSIVE",
            "reason": "need >= %d samples in each window for %d quantile bins; got %d / %d"
                      % (MIN_SAMPLES_PSI, bins, len(reference), len(current)),
            "bins": [],
        }

    edges = quantile_edges(reference, bins)
    if len(edges) < 2:
        return {
            "psi": None,
            "verdict": "INCONCLUSIVE",
            "reason": "reference has too little variation to form quantile bins "
                      "(heavy point mass?); PSI is not meaningful here",
            "bins": [],
        }

    ref_counts = bucketize(reference, edges)
    cur_counts = bucketize(current, edges)
    n_ref = float(len(reference))
    n_cur = float(len(current))

    # Laplace-style floor: an empty bin makes the log term infinite, which would
    # otherwise let one unseen bucket dominate the whole index.
    floor = 0.5 / max(n_ref, n_cur)

    total = 0.0
    detail = []
    for i, (rc, cc) in enumerate(zip(ref_counts, cur_counts)):
        p_ref = max(rc / n_ref, floor)
        p_cur = max(cc / n_cur, floor)
        contribution = (p_cur - p_ref) * math.log(p_cur / p_ref)
        total += contribution
        detail.append({"bin": i, "ref_pct": 100 * rc / n_ref, "cur_pct": 100 * cc / n_cur,
                       "contribution": contribution})

    if total >= PSI_MAJOR:
        verdict = "MAJOR SHIFT"
    elif total >= PSI_MODERATE:
        verdict = "MODERATE SHIFT"
    else:
        verdict = "STABLE"
    return {"psi": total, "verdict": verdict, "reason": "", "bins": detail, "edges": edges}


def ks_two_sample(reference, current):
    """Two-sample Kolmogorov-Smirnov statistic and asymptotic p-value."""
    n1, n2 = len(reference), len(current)
    if n1 < MIN_SAMPLES_KS or n2 < MIN_SAMPLES_KS:
        return {"d": None, "p_value": None, "verdict": "INCONCLUSIVE",
                "reason": "need >= %d samples per window; got %d / %d"
                          % (MIN_SAMPLES_KS, n1, n2)}

    a = sorted(reference)
    b = sorted(current)
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    while i < n1 and j < n2:
        if a[i] <= b[j]:
            value = a[i]
            while i < n1 and a[i] == value:
                i += 1
            cdf_a = i / n1
        else:
            value = b[j]
            while j < n2 and b[j] == value:
                j += 1
            cdf_b = j / n2
        # Advance any ties in the other sample so the comparison is at the same x.
        while i < n1 and a[i] == value:
            i += 1
            cdf_a = i / n1
        while j < n2 and b[j] == value:
            j += 1
            cdf_b = j / n2
        d = max(d, abs(cdf_a - cdf_b))

    ne = math.sqrt(n1 * n2 / float(n1 + n2))
    lam = (ne + 0.12 + 0.11 / ne) * d
    p = ks_sf(lam)
    return {"d": d, "p_value": p,
            "verdict": "DIFFERENT" if p < 0.01 else "not distinguishable",
            "reason": ""}


def chi_square_categorical(ref_counts, cur_counts):
    """Chi-square test of homogeneity between two categorical count dicts."""
    categories = sorted(set(ref_counts) | set(cur_counts))
    n_ref = float(sum(ref_counts.values()))
    n_cur = float(sum(cur_counts.values()))
    if n_ref == 0 or n_cur == 0:
        return {"statistic": None, "p_value": None, "verdict": "INCONCLUSIVE",
                "reason": "one of the windows is empty"}

    total = n_ref + n_cur
    statistic = 0.0
    small_cells = []
    for c in categories:
        row_total = ref_counts.get(c, 0) + cur_counts.get(c, 0)
        expected_ref = row_total * n_ref / total
        expected_cur = row_total * n_cur / total
        if min(expected_ref, expected_cur) < MIN_EXPECTED_CELL:
            small_cells.append(c)
        if expected_ref > 0:
            statistic += (ref_counts.get(c, 0) - expected_ref) ** 2 / expected_ref
        if expected_cur > 0:
            statistic += (cur_counts.get(c, 0) - expected_cur) ** 2 / expected_cur

    dof = len(categories) - 1
    if dof < 1:
        return {"statistic": None, "p_value": None, "verdict": "INCONCLUSIVE",
                "reason": "only one category present"}

    p = chi2_sf(statistic, dof)
    reason = ""
    if small_cells:
        reason = ("expected count < %.0f in %d categories (%s) -- the chi-square "
                  "approximation is unreliable; merge rare categories into 'other'"
                  % (MIN_EXPECTED_CELL, len(small_cells), ", ".join(map(str, small_cells[:5]))))
    verdict = "INCONCLUSIVE" if small_cells else ("DIFFERENT" if p < 0.01 else "not distinguishable")

    # Cramer's V: the effect size. This is the number to threshold on, because
    # the p-value is a function of sample size and will go to zero on any real
    # production volume regardless of whether anyone should care.
    cramers_v = math.sqrt(statistic / (total * dof)) if total > 0 else 0.0
    return {"statistic": statistic, "dof": dof, "p_value": p, "cramers_v": cramers_v,
            "verdict": verdict, "reason": reason}


# --------------------------------------------------------------------------
# Drift-type classification
# --------------------------------------------------------------------------


def classify_drift(inputs_drifted, predictions_drifted, labels_drifted, accuracy_dropped):
    """Name the most likely drift type from the evidence available.

    Arguments are True / False / None, where None means "not measured". The
    None cases matter: the delayed-label problem means `accuracy_dropped` is
    usually None in the moment, and this function's job is to say so rather
    than to guess.
    """
    if accuracy_dropped is None:
        if inputs_drifted and predictions_drifted:
            return ("LIKELY COVARIATE SHIFT",
                    "P(x) moved and the predictions followed, which is what a well-behaved "
                    "model does under new inputs. Cannot rule out concept drift without "
                    "labels. Action: check whether the new input region is one the model "
                    "was trained on; sample and label it if not.")
        if inputs_drifted and not predictions_drifted:
            return ("INPUT SHIFT WITHOUT PREDICTION IMPACT",
                    "The drifting features are probably ones the model barely uses. Verify "
                    "against feature importance before dismissing it -- and check that the "
                    "feature is actually reaching the model at all.")
        if not inputs_drifted and predictions_drifted:
            return ("INVESTIGATE THE SERVING PATH FIRST",
                    "Inputs stable but outputs moved means something in the pipeline changed, "
                    "not the world: a model version rollout, a feature-encoding change, a "
                    "preprocessing bug, or a silently different default for a missing value. "
                    "This is an incident, not drift.")
        return ("NO DRIFT DETECTED",
                "Inputs and predictions both stable. Note that concept drift can be invisible "
                "here -- P(y|x) can move with P(x) unchanged. Only labels can rule it out.")

    if accuracy_dropped:
        if not inputs_drifted and predictions_drifted:
            return ("SERVING-PATH INCIDENT, NOT DRIFT",
                    "Inputs unchanged, predictions moved, accuracy fell. The world did not "
                    "change -- your pipeline did. Check, in order: a model version rollout, "
                    "a feature-encoding or preprocessing change, an upstream join failing "
                    "silently to a default value, and a schema change. Retraining would be "
                    "the wrong response and would bake the bug into the next model.")
        if not inputs_drifted and not predictions_drifted:
            return ("CONCEPT DRIFT",
                    "Same inputs, same predictions, worse outcomes: the relationship P(y|x) "
                    "itself moved. This is the case that NECESSARILY requires retraining -- "
                    "no amount of reweighting or recalibration fixes a changed target "
                    "function. Retrain on data from the new regime.")
        if labels_drifted and not inputs_drifted:
            return ("PRIOR / LABEL SHIFT",
                    "P(y) moved while P(x|y) looks stable. Often fixable without retraining: "
                    "adjust the decision threshold or reweight to the new prior, then verify. "
                    "Cheaper and faster than a retrain -- try it first.")
        if inputs_drifted:
            return ("COVARIATE SHIFT WITH PERFORMANCE IMPACT",
                    "P(x) moved into a region the model handles badly -- extrapolation, not a "
                    "changed target function. Importance weighting or targeted data collection "
                    "in the new region may fix it without a full retrain. Check whether the "
                    "new region is genuinely out of the training distribution.")
        return ("UNEXPLAINED DEGRADATION",
                "Accuracy fell but no distribution moved measurably. Suspect the label pipeline, "
                "the evaluation window, or a bug in the metric itself before suspecting the model.")

    return ("STABLE",
            "No measured degradation. Keep monitoring; absence of evidence over a short window "
            "is weak evidence of absence, especially with delayed labels.")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def print_numeric_report(name, reference, current, bins):
    print("  feature: %s" % name)
    print("    reference n = %d, current n = %d" % (len(reference), len(current)))

    result = psi(reference, current, bins)
    if result["psi"] is None:
        print("    PSI       : INCONCLUSIVE -- %s" % result["reason"])
    else:
        print("    PSI       : %.4f  [%s]   (>%.2f moderate, >%.2f major -- industry rules "
              "of thumb, calibrate on your own data)"
              % (result["psi"], result["verdict"], PSI_MODERATE, PSI_MAJOR))
        worst = max(result["bins"], key=lambda b: b["contribution"])
        print("                top bin %d contributes %.4f (ref %.1f%% -> cur %.1f%%)"
              % (worst["bin"], worst["contribution"], worst["ref_pct"], worst["cur_pct"]))

    ks = ks_two_sample(reference, current)
    if ks["d"] is None:
        print("    KS        : INCONCLUSIVE -- %s" % ks["reason"])
    else:
        print("    KS        : D = %.4f, p = %s  [%s]"
              % (ks["d"], _fmt_p(ks["p_value"]), ks["verdict"]))
        if ks["p_value"] < 0.01 and (result["psi"] or 0.0) < PSI_MODERATE:
            print("                NOTE: significant but small. With this sample size the KS")
            print("                test detects differences too small to act on. Trust the PSI.")
    return result


def _fmt_p(p):
    if p is None:
        return "n/a"
    if p < 1e-12:
        return "<1e-12"
    return "%.3g" % p


def print_categorical_report(name, ref_counts, cur_counts):
    print("  feature: %s (categorical)" % name)
    result = chi_square_categorical(ref_counts, cur_counts)
    if result["statistic"] is None:
        print("    chi-square: INCONCLUSIVE -- %s" % result["reason"])
        return result
    print("    chi-square: %.2f on %d dof, p = %s, Cramer's V = %.4f  [%s]"
          % (result["statistic"], result["dof"], _fmt_p(result["p_value"]),
             result["cramers_v"], result["verdict"]))
    if result["reason"]:
        print("                WARNING: %s" % result["reason"])
    print("                Threshold on Cramer's V (effect size), not on p. At production")
    print("                volume p is a function of n and will always be tiny.")
    return result


# --------------------------------------------------------------------------
# Demo scenarios
# --------------------------------------------------------------------------


def scenario_samples(seed, kind, n=4000):
    """Generate a reference and current window for one drift scenario."""
    rng = random.Random(seed)
    reference = [rng.gauss(50.0, 12.0) for _ in range(n)]
    ref_cat = {}
    for _ in range(n):
        c = rng.choices(["mobile", "desktop", "tablet"], weights=[0.55, 0.40, 0.05])[0]
        ref_cat[c] = ref_cat.get(c, 0) + 1
    # The model reads more than the one feature we are monitoring, so its score
    # is not a pure monotone transform of it. Without this the prediction PSI
    # would be numerically identical to the feature PSI (quantile binning is
    # invariant under monotone transforms), which is true but misleading.
    def score(x):
        return _sigmoid((x - 50.0) / 10.0 + rng.gauss(0.0, 0.45))

    ref_pred = [score(x) for x in reference]

    if kind == "stable":
        current = [rng.gauss(50.0, 12.0) for _ in range(n)]
        cur_weights = [0.55, 0.40, 0.05]
        cur_pred = [score(x) for x in current]
        accuracy_dropped = False

    elif kind == "covariate":
        # The population moved: an older, higher-value cohort arrived.
        current = [rng.gauss(58.0, 14.0) for _ in range(n)]
        cur_weights = [0.35, 0.58, 0.07]
        cur_pred = [score(x) for x in current]
        accuracy_dropped = True

    elif kind == "concept":
        # Inputs identical, but the mapping from input to outcome changed --
        # a competitor launched, so the same customer now behaves differently.
        # Nothing observable moves; only the outcomes do.
        current = [rng.gauss(50.0, 12.0) for _ in range(n)]
        cur_weights = [0.55, 0.40, 0.05]
        cur_pred = [score(x) for x in current]
        accuracy_dropped = True

    elif kind == "pipeline_bug":
        # Inputs identical, predictions collapsed: a feature is arriving as its
        # default value because an upstream join started failing.
        current = [rng.gauss(50.0, 12.0) for _ in range(n)]
        cur_weights = [0.55, 0.40, 0.05]
        cur_pred = [_sigmoid(0.0 + rng.gauss(0.0, 0.02)) for _ in current]
        accuracy_dropped = True

    else:
        raise ValueError("unknown scenario %r" % kind)

    cur_cat = {}
    for _ in range(n):
        c = rng.choices(["mobile", "desktop", "tablet"], weights=cur_weights)[0]
        cur_cat[c] = cur_cat.get(c, 0) + 1

    return {"reference": reference, "current": current, "ref_cat": ref_cat, "cur_cat": cur_cat,
            "ref_pred": ref_pred, "cur_pred": cur_pred, "accuracy_dropped": accuracy_dropped}


def _sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def run_scenario(title, note, data, bins):
    print()
    print("-" * 84)
    print("SCENARIO: %s" % title)
    print("  %s" % note)
    print("-" * 84)

    feature = print_numeric_report("session_value", data["reference"], data["current"], bins)
    print()
    cat = print_categorical_report("device_type", data["ref_cat"], data["cur_cat"])
    print()
    prediction = print_numeric_report("model_score (prediction distribution)",
                                      data["ref_pred"], data["cur_pred"], bins)

    inputs_drifted = ((feature["psi"] or 0.0) >= PSI_MODERATE
                      or (cat.get("cramers_v") or 0.0) >= 0.05)
    predictions_drifted = (prediction["psi"] or 0.0) >= PSI_MODERATE

    print()
    for label, accuracy in (("labels available", data["accuracy_dropped"]),
                            ("labels NOT yet available (the usual case)", None)):
        verdict, explanation = classify_drift(inputs_drifted, predictions_drifted,
                                              None, accuracy)
        print("  classification (%s):" % label)
        print("    -> %s" % verdict)
        for line in _wrap(explanation, 76):
            print("       %s" % line)
    return verdict


def _wrap(text, width):
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines


def run_demo(args):
    print("=" * 84)
    print("DRIFT DETECTOR -- seeded demo (SIMULATED DATA, seed %d)" % args.seed)
    print("=" * 84)
    print("Four scenarios that a monitoring dashboard must tell apart. Only one of")
    print("them requires retraining; one of them is not drift at all.")

    run_scenario("No drift",
                 "Same generating process in both windows. The control case.",
                 scenario_samples(args.seed, "stable", args.n), args.bins)
    run_scenario("Covariate shift -- P(x) changed",
                 "An older, higher-value cohort arrived. The relationship is unchanged.",
                 scenario_samples(args.seed + 1, "covariate", args.n), args.bins)
    run_scenario("Concept drift -- P(y|x) changed",
                 "Inputs identical; a competitor launched and the same customers now behave "
                 "differently.",
                 scenario_samples(args.seed + 2, "concept", args.n), args.bins)
    run_scenario("Not drift at all -- a broken pipeline",
                 "Inputs identical, predictions collapsed: an upstream join is failing and a "
                 "feature arrives as its default.",
                 scenario_samples(args.seed + 3, "pipeline_bug", args.n), args.bins)

    print()
    print("=" * 84)
    print("WHAT THE FOUR SCENARIOS TEACH")
    print("=" * 84)
    print("  1. Only CONCEPT DRIFT necessarily requires retraining. Covariate shift may be")
    print("     fixable with reweighting or targeted data collection; prior shift is often")
    print("     just a threshold adjustment. Diagnosing before acting saves a training run.")
    print("  2. Concept drift is INVISIBLE to input monitoring. Scenario 3 shows stable")
    print("     inputs and stable predictions while accuracy falls. If your dashboard only")
    print("     watches input distributions, you will never see it.")
    print("  3. Scenario 4 looks like severe drift and is a BUG. Stable inputs plus moved")
    print("     predictions almost always means the serving path changed, not the world.")
    print("     Check deploys and upstream jobs before you schedule a retrain.")
    print("  4. Note how the classification changes when labels are unavailable. That is the")
    print("     delayed-label problem, and it is the normal operating condition.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run_from_config(args):
    try:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError:
        raise ValueError("no such config file: %s" % args.config)
    except json.JSONDecodeError as exc:
        raise ValueError("%s is not valid JSON: %s" % (args.config, exc))

    if not isinstance(config, dict) or "features" not in config:
        raise ValueError("config must be an object with a 'features' list; "
                         "see --print-example")

    print("=" * 84)
    print("DRIFT REPORT -- %s" % config.get("name", "(unnamed)"))
    print("=" * 84)
    numeric_drifted = False
    for feature in config["features"]:
        name = feature.get("name", "(unnamed)")
        print()
        if feature.get("type") == "categorical":
            result = print_categorical_report(name, feature["reference"], feature["current"])
            numeric_drifted = numeric_drifted or (result.get("cramers_v") or 0.0) >= 0.05
        else:
            result = print_numeric_report(name, feature["reference"], feature["current"],
                                          args.bins)
            numeric_drifted = numeric_drifted or (result.get("psi") or 0.0) >= PSI_MODERATE

    print()
    print("  %d features tested. At alpha = 0.01 you expect %.1f false positives from"
          % (len(config["features"]), 0.01 * len(config["features"])))
    print("  chance alone -- correct for multiplicity or threshold on effect size.")


EXAMPLE_CONFIG = {
    "name": "checkout model, week 32 vs week 31",
    "features": [
        {"name": "basket_value", "type": "numeric",
         "reference": [12.0, 15.5, 9.1, 22.0], "current": [13.0, 16.5, 10.1, 25.0]},
        {"name": "device_type", "type": "categorical",
         "reference": {"mobile": 5500, "desktop": 4000, "tablet": 500},
         "current": {"mobile": 6100, "desktop": 3400, "tablet": 500}}
    ]
}


def main(argv=None):
    if sys.version_info < (3, 8):
        sys.stderr.write("ERROR: this tool needs Python 3.8 or newer.\n")
        return 2

    parser = argparse.ArgumentParser(
        description="PSI, KS, chi-square and drift-type classification (stdlib only).")
    parser.add_argument("--config", help="JSON file with reference/current samples")
    parser.add_argument("--print-example", action="store_true",
                        help="print an example config and exit")
    parser.add_argument("--bins", type=int, default=10, help="quantile bins for PSI")
    parser.add_argument("--n", type=int, default=4000, help="demo sample size per window")
    parser.add_argument("--seed", type=int, default=20240729)
    args = parser.parse_args(argv)

    try:
        if args.print_example:
            print(json.dumps(EXAMPLE_CONFIG, indent=2))
            return 0
        if args.bins < 2:
            raise ValueError("--bins must be at least 2")
        if args.config:
            run_from_config(args)
        else:
            run_demo(args)
    except ValueError as exc:
        sys.stderr.write("ERROR: %s\n" % exc)
        sys.stderr.write("Run with --print-example to see the expected config shape.\n")
        return 2
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrupted.\n")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
