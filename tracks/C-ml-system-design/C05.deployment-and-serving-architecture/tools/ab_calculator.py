#!/usr/bin/env python3
"""A/B test calculator: sample size, significance, and the peeking problem.

Standard library only (math, random, argparse, sys). No numpy, no scipy.

Sub-commands
------------
  size      users per arm needed to detect a given minimum detectable effect
  evaluate  two-proportion z-test, p-value and confidence interval
  peeking   seeded simulation showing how repeated significance testing on an
            A/A test inflates the false-positive rate far above alpha

Run with no sub-command for a demo of all three.

Everything here uses the normal approximation to the binomial, which is the
standard practice for web-scale conversion experiments. It is accurate when
n*p and n*(1-p) are both comfortably above ~10; the tool warns when they are
not. The normal CDF and its inverse are implemented below rather than
imported, so the file has zero third-party dependencies.
"""

from __future__ import annotations

import argparse
import math
import random
import sys

# --------------------------------------------------------------------------
# Normal distribution: CDF and inverse CDF, implemented locally
# --------------------------------------------------------------------------


def normal_cdf(z: float) -> float:
    """Standard normal CDF.

    Uses erfc rather than erf for the upper tail: erf(x) -> 1 as x grows, so
    1 - erf loses all significant digits past about z = 5, while erfc stays
    accurate. p-values in large experiments routinely land in that tail.
    """
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def normal_sf(z: float) -> float:
    """Upper-tail probability P(Z > z), accurate deep into the tail."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# Acklam's rational approximation to the inverse normal CDF. Relative error
# below 1.15e-9 before refinement; the Halley step below drives it to machine
# precision, which matters because a wrong z_alpha silently mis-sizes every
# experiment the tool is asked about.
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (the quantile function)."""
    if not 0.0 < p < 1.0:
        raise ValueError("normal_ppf needs a probability strictly between 0 and 1, got %r" % (p,))

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
              + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / (
            (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)
    elif p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        x = (((((_ACKLAM_A[0] * r + _ACKLAM_A[1]) * r + _ACKLAM_A[2]) * r + _ACKLAM_A[3]) * r
              + _ACKLAM_A[4]) * r + _ACKLAM_A[5]) * q / (
            ((((_ACKLAM_B[0] * r + _ACKLAM_B[1]) * r + _ACKLAM_B[2]) * r + _ACKLAM_B[3]) * r
             + _ACKLAM_B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_ACKLAM_C[0] * q + _ACKLAM_C[1]) * q + _ACKLAM_C[2]) * q + _ACKLAM_C[3]) * q
               + _ACKLAM_C[4]) * q + _ACKLAM_C[5]) / (
            (((_ACKLAM_D[0] * q + _ACKLAM_D[1]) * q + _ACKLAM_D[2]) * q + _ACKLAM_D[3]) * q + 1.0)

    # One Halley refinement against our own CDF.
    err = normal_cdf(x) - p
    u = err * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


# --------------------------------------------------------------------------
# Sample size
# --------------------------------------------------------------------------


def sample_size_per_arm(baseline: float, mde_abs: float, alpha: float, power: float) -> int:
    """Users per arm for a two-sided two-proportion test.

    Textbook form: the null variance uses the pooled rate (that is the
    distribution the test statistic is compared against), the alternative
    variance uses the two separate rates.
    """
    treatment = baseline + mde_abs
    if not 0.0 < treatment < 1.0:
        raise ValueError(
            "baseline %.4f + MDE %.4f = %.4f is not a valid rate; pick a smaller MDE"
            % (baseline, mde_abs, treatment))

    z_alpha = normal_ppf(1.0 - alpha / 2.0)
    z_beta = normal_ppf(power)
    pooled = (baseline + treatment) / 2.0

    null_sd = math.sqrt(2.0 * pooled * (1.0 - pooled))
    alt_sd = math.sqrt(baseline * (1.0 - baseline) + treatment * (1.0 - treatment))
    n = ((z_alpha * null_sd + z_beta * alt_sd) / mde_abs) ** 2
    return int(math.ceil(n))


def cmd_size(args: argparse.Namespace) -> None:
    baseline = args.baseline
    if args.mde_rel is not None:
        mde_abs = baseline * args.mde_rel
        mde_label = "%.4f absolute (%.1f%% relative)" % (mde_abs, 100.0 * args.mde_rel)
    else:
        mde_abs = args.mde_abs
        mde_label = "%.4f absolute (%.1f%% relative)" % (mde_abs, 100.0 * mde_abs / baseline)

    n = sample_size_per_arm(baseline, mde_abs, args.alpha, args.power)

    print("Sample size (two-sided two-proportion test)")
    print("-" * 62)
    print("  baseline rate        : %.4f" % baseline)
    print("  minimum detectable   : %s" % mde_label)
    print("  significance (alpha) : %.3f   -> z = %.4f" % (args.alpha, normal_ppf(1 - args.alpha / 2)))
    print("  power (1 - beta)     : %.3f   -> z = %.4f" % (args.power, normal_ppf(args.power)))
    print()
    print("  users per arm        : %s" % f"{n:,}")
    print("  users total (2 arms) : %s" % f"{2 * n:,}")

    if args.daily_users is not None:
        if args.daily_users <= 0:
            raise ValueError("--daily-users must be positive")
        exposed = args.daily_users * args.traffic_fraction
        days = 2.0 * n / exposed
        print("  at %s eligible users/day x %.0f%% in the test: %.1f days"
              % (f"{args.daily_users:,}", 100 * args.traffic_fraction, days))
        if days < 7:
            print("  NOTE: run at least one full week anyway. Weekday and weekend")
            print("        populations differ; a 3-day test measures Tue-Thu users.")

    if baseline * n < 10 or (1 - baseline) * n < 10:
        print()
        print("  WARNING: n*p = %.1f. The normal approximation is unreliable below" % (baseline * n))
        print("           about 10 expected events per arm. Treat this as a rough guide.")

    print()
    print("  Sensitivity: halving the MDE quadruples the sample size.")
    for factor in (2.0, 1.0, 0.5, 0.25):
        try:
            n_f = sample_size_per_arm(baseline, mde_abs * factor, args.alpha, args.power)
            print("    MDE x %-5.2f -> %14s users/arm" % (factor, f"{n_f:,}"))
        except ValueError:
            print("    MDE x %-5.2f -> invalid (rate out of range)" % factor)


# --------------------------------------------------------------------------
# Evaluating an observed result
# --------------------------------------------------------------------------


def two_proportion_z(n_a: int, c_a: int, n_b: int, c_b: int):
    """Return (p_a, p_b, z, p_value, ci_low, ci_high) for B - A at 95%."""
    if n_a <= 0 or n_b <= 0:
        raise ValueError("arm sizes must be positive")
    if not 0 <= c_a <= n_a or not 0 <= c_b <= n_b:
        raise ValueError("conversions must satisfy 0 <= conversions <= users")

    p_a = c_a / n_a
    p_b = c_b / n_b
    diff = p_b - p_a

    pooled = (c_a + c_b) / (n_a + n_b)
    se_null = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if se_null == 0.0:
        raise ValueError("both arms have identical degenerate rates (0 or 1); the z-test is undefined")
    z = diff / se_null
    p_value = 2.0 * normal_sf(abs(z))

    # The interval uses the unpooled SE: under the alternative the two arms
    # genuinely have different variances, and the pooled SE would understate
    # the width.
    se_alt = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z95 = normal_ppf(0.975)
    return p_a, p_b, z, p_value, diff - z95 * se_alt, diff + z95 * se_alt


def cmd_evaluate(args: argparse.Namespace) -> None:
    p_a, p_b, z, p_value, lo, hi = two_proportion_z(
        args.n_control, args.c_control, args.n_treatment, args.c_treatment)

    print("Observed result (two-sided two-proportion z-test)")
    print("-" * 62)
    print("  control   : %10s users, %8s conv -> %.4f" % (f"{args.n_control:,}", f"{args.c_control:,}", p_a))
    print("  treatment : %10s users, %8s conv -> %.4f" % (f"{args.n_treatment:,}", f"{args.c_treatment:,}", p_b))
    print()
    print("  absolute lift : %+.5f" % (p_b - p_a))
    if p_a > 0:
        print("  relative lift : %+.2f%%" % (100.0 * (p_b - p_a) / p_a))
    print("  z statistic   : %+.4f" % z)
    print("  p-value       : %.5f" % p_value)
    print("  95%% CI (abs)  : [%+.5f, %+.5f]" % (lo, hi))
    print()

    verdict = "significant at alpha=%.3f" % args.alpha if p_value < args.alpha else \
        "NOT significant at alpha=%.3f" % args.alpha
    print("  verdict: %s" % verdict)
    if lo < 0 < hi:
        print("  the interval spans zero: a real regression is still consistent with this data")
    if min(args.c_control, args.c_treatment) < 10:
        print("  WARNING: fewer than 10 conversions in an arm; normal approximation is shaky")
    print()
    print("  Reminder: this p-value is only valid if you fixed the sample size")
    print("  in advance and looked once. If you have been watching a dashboard,")
    print("  run the 'peeking' sub-command to see what that costs you.")


# --------------------------------------------------------------------------
# The peeking problem
# --------------------------------------------------------------------------


def _binomial(rng: random.Random, n: int, p: float) -> int:
    """Binomial draw, exact where the runtime offers it."""
    sampler = getattr(rng, "binomialvariate", None)
    if sampler is not None:  # Python 3.12+
        return sampler(n, p)
    # Fallback: normal approximation with continuity correction. Only reached
    # on older interpreters, and only affects the simulation's fidelity, not
    # its qualitative conclusion.
    mean = n * p
    sd = math.sqrt(n * p * (1.0 - p))
    return max(0, min(n, int(round(rng.gauss(mean, sd)))))


def simulate_peeking(trials: int, n_per_arm: int, looks: int, rate: float,
                     alpha: float, seed: int) -> dict:
    """A/A simulation: both arms have identical true rate, so every 'win' is false.

    Returns the false-positive rate when stopping at the first significant
    look, and when honouring the fixed sample size and testing once.
    """
    rng = random.Random(seed)
    step = n_per_arm // looks
    if step < 1:
        raise ValueError("n_per_arm (%d) must be at least the number of looks (%d)" % (n_per_arm, looks))

    stopped_early = 0
    final_only = 0
    stop_look_hist = [0] * looks

    for _ in range(trials):
        c_a = c_b = 0
        n = 0
        already_stopped = False
        for look in range(looks):
            c_a += _binomial(rng, step, rate)
            c_b += _binomial(rng, step, rate)
            n += step
            pooled = (c_a + c_b) / (2 * n)
            se = math.sqrt(pooled * (1 - pooled) * (2.0 / n)) if 0 < pooled < 1 else 0.0
            if se > 0:
                z = (c_b - c_a) / n / se
                significant = 2.0 * normal_sf(abs(z)) < alpha
            else:
                significant = False
            if significant and not already_stopped:
                stopped_early += 1
                stop_look_hist[look] += 1
                already_stopped = True
            if look == looks - 1 and significant:
                final_only += 1

    return {
        "trials": trials,
        "looks": looks,
        "fpr_peeking": stopped_early / trials,
        "fpr_single_look": final_only / trials,
        "stop_look_hist": stop_look_hist,
    }


def cmd_peeking(args: argparse.Namespace) -> None:
    print("The peeking problem (seeded A/A simulation -- SIMULATED DATA, not measured)")
    print("-" * 74)
    print("  Both arms are drawn from the SAME true rate (%.3f), so any 'significant'" % args.rate)
    print("  result is a false positive by construction. alpha = %.3f, so an honest" % args.alpha)
    print("  test looked at once should be wrong about %.1f%% of the time." % (100 * args.alpha))
    print("  trials = %s, users/arm = %s, seed = %d" % (f"{args.trials:,}", f"{args.n_per_arm:,}", args.seed))
    print()
    print("  %-8s %-22s %-22s" % ("looks", "stop-at-first-signif.", "fixed-n, test once"))
    print("  %-8s %-22s %-22s" % ("-" * 8, "-" * 22, "-" * 22))

    for looks in args.look_counts:
        res = simulate_peeking(args.trials, args.n_per_arm, looks, args.rate, args.alpha, args.seed)
        print("  %-8d %-22s %-22s"
              % (looks,
                 "%.1f%%" % (100 * res["fpr_peeking"]),
                 "%.1f%%" % (100 * res["fpr_single_look"])))

    print()
    print("  Read the left column: it is the chance of shipping a change that does")
    print("  nothing, purely because someone refreshed the dashboard. It climbs with")
    print("  the number of looks while the right column stays near alpha.")
    print()
    print("  Fixes, in rough order of practicality:")
    print("    1. Fix the sample size in advance and do not stop early. Boring, correct.")
    print("    2. Alpha-spending / group-sequential boundaries (O'Brien-Fleming,")
    print("       Pocock) if you need pre-planned interim looks.")
    print("    3. Always-valid inference / sequential tests (Johari et al. 2015) if")
    print("       you genuinely need a continuously-monitored dashboard.")
    print("    4. Keep a separate, always-on guardrail alert for harm. Stopping EARLY")
    print("       FOR HARM is legitimate; stopping early for a win is not.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def run_demo() -> None:
    print("=" * 74)
    print("A/B CALCULATOR DEMO -- run with -h for the individual sub-commands")
    print("=" * 74)
    print()

    print(">>> size --baseline 0.12 --mde-rel 0.02")
    print()
    cmd_size(argparse.Namespace(baseline=0.12, mde_abs=None, mde_rel=0.02, alpha=0.05,
                                power=0.80, daily_users=500_000, traffic_fraction=0.20))
    print()
    print(">>> evaluate --n-control 120000 --c-control 14320 "
          "--n-treatment 120000 --c-treatment 14690")
    print()
    cmd_evaluate(argparse.Namespace(n_control=120_000, c_control=14_320,
                                    n_treatment=120_000, c_treatment=14_690, alpha=0.05))
    print()
    print(">>> peeking")
    print()
    cmd_peeking(argparse.Namespace(trials=4000, n_per_arm=40_000, rate=0.12, alpha=0.05,
                                   seed=20240729, look_counts=[1, 2, 5, 10, 20]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B test sample size, significance, and peeking simulator (stdlib only).")
    sub = parser.add_subparsers(dest="command")

    p_size = sub.add_parser("size", help="users per arm for a target minimum detectable effect")
    p_size.add_argument("--baseline", type=float, required=True, help="baseline conversion rate, 0-1")
    group = p_size.add_mutually_exclusive_group(required=True)
    group.add_argument("--mde-abs", type=float, help="minimum detectable effect, absolute (e.g. 0.005)")
    group.add_argument("--mde-rel", type=float, help="minimum detectable effect, relative (e.g. 0.02 = 2%%)")
    p_size.add_argument("--alpha", type=float, default=0.05)
    p_size.add_argument("--power", type=float, default=0.80)
    p_size.add_argument("--daily-users", type=int, default=None, help="eligible users per day")
    p_size.add_argument("--traffic-fraction", type=float, default=1.0,
                        help="fraction of eligible users allocated to the experiment")
    p_size.set_defaults(func=cmd_size)

    p_eval = sub.add_parser("evaluate", help="z-test, p-value and CI for an observed result")
    p_eval.add_argument("--n-control", type=int, required=True)
    p_eval.add_argument("--c-control", type=int, required=True)
    p_eval.add_argument("--n-treatment", type=int, required=True)
    p_eval.add_argument("--c-treatment", type=int, required=True)
    p_eval.add_argument("--alpha", type=float, default=0.05)
    p_eval.set_defaults(func=cmd_evaluate)

    p_peek = sub.add_parser("peeking", help="seeded A/A simulation of repeated significance testing")
    p_peek.add_argument("--trials", type=int, default=4000)
    p_peek.add_argument("--n-per-arm", type=int, default=40_000)
    p_peek.add_argument("--rate", type=float, default=0.12)
    p_peek.add_argument("--alpha", type=float, default=0.05)
    p_peek.add_argument("--seed", type=int, default=20240729)
    p_peek.add_argument("--look-counts", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p_peek.set_defaults(func=cmd_peeking)

    return parser


def validate_common(args: argparse.Namespace) -> None:
    for name in ("alpha", "power", "baseline", "rate"):
        value = getattr(args, name, None)
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError("--%s must be strictly between 0 and 1, got %r"
                             % (name.replace("_", "-"), value))
    # A fraction of 1.0 is legitimate: the whole eligible population is in the test.
    fraction = getattr(args, "traffic_fraction", None)
    if fraction is not None and not 0.0 < fraction <= 1.0:
        raise ValueError("--traffic-fraction must be in (0, 1], got %r" % fraction)
    for name in ("mde_abs", "mde_rel"):
        value = getattr(args, name, None)
        if value is not None and value <= 0.0:
            raise ValueError("--%s must be positive (this tool tests for a lift)"
                             % name.replace("_", "-"))


def main(argv=None) -> int:
    if sys.version_info < (3, 8):
        sys.stderr.write("ERROR: this tool needs Python 3.8 or newer.\n")
        return 2

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if getattr(args, "command", None) is None:
            run_demo()
        else:
            validate_common(args)
            args.func(args)
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
