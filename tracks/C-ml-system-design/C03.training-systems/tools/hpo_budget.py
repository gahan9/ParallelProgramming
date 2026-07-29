#!/usr/bin/env python3
"""C03 - Hyperparameter search under a fixed GPU-hour budget.

A SEEDED SIMULATION of a synthetic search space, not a benchmark of any real
model. It compares three strategies spending the SAME budget:

  grid              every dimension gets the same number of values, so the
                    number of distinct values per dimension is n**(1/d)
  random            n independent draws, all distinct in every dimension
  successive halving  start many configs cheaply, keep the top 1/eta at each
                    rung, and only the survivors pay for a full run (ASHA)

Search space model
------------------
Each config's final quality is  q = a + b  where
    a ~ Normal(0, w)      contribution of the ONE dimension that matters
    b ~ Normal(0, 1 - w)  everything else, which behaves like noise
Grid draws only n**(1/d) distinct values of `a` and repeats each of them,
which is Bergstra & Bengio's argument for random search in one line.
A partial run of fraction f observes  q + Normal(0, early_noise / sqrt(f)) -
cheap rungs rank configs, they do not measure them.

The headline output is "best true quality found", averaged over --trials
independent runs of the whole search. Absolute values are arbitrary units;
the RANKING and the configs-explored column are the lesson.

Stdlib only. Run `python hpo_budget.py --help`.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys


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


def draw_quality(rng, weight):
    """True final quality of one config, split into 'the dimension that matters'
    and everything else."""
    important = rng.gauss(0.0, math.sqrt(weight))
    rest = rng.gauss(0.0, math.sqrt(1.0 - weight))
    return important, important + rest


def observe(rng, true_q, fraction, early_noise):
    """What a partial run of `fraction` of full training reports."""
    if fraction >= 1.0:
        return true_q
    return true_q + rng.gauss(0.0, early_noise / math.sqrt(fraction))


def run_random(rng, args, budget_units):
    n = int(budget_units)
    if n < 1:
        return None
    best = max(draw_quality(rng, args.important_weight)[1] for _ in range(n))
    return {"explored": n, "full_runs": n, "best": best, "distinct_important": n}


def run_grid(rng, args, budget_units):
    """A grid of d dimensions with k values each; only k of them vary the
    dimension that matters, so the same `a` is retried k**(d-1) times."""
    n = int(budget_units)
    if n < 1:
        return None
    k = max(1, int(round(n ** (1.0 / args.dimensions))))
    lattice = [rng.gauss(0.0, math.sqrt(args.important_weight)) for _ in range(k)]
    best = -float("inf")
    for i in range(n):
        important = lattice[i % k]
        rest = rng.gauss(0.0, math.sqrt(1.0 - args.important_weight))
        best = max(best, important + rest)
    return {"explored": n, "full_runs": n, "best": best, "distinct_important": k}


def run_successive_halving(rng, args, budget_units):
    """Geometric rungs: n configs at 1/eta**(rungs-1) of full cost, survivors
    promoted by a factor of eta each rung until the last rung trains fully.

    Each rung costs about the same in aggregate, which is what makes the
    budget arithmetic easy: budget_units / rungs per rung.
    """
    rungs = args.rungs
    per_rung_units = budget_units / rungs
    smallest_fraction = 1.0 / (args.eta ** (rungs - 1))
    n = int(per_rung_units / smallest_fraction)
    if n < 1:
        return None

    pool = [draw_quality(rng, args.important_weight) for _ in range(n)]
    survivors = [(observe(rng, q, smallest_fraction, args.early_noise), q)
                 for _, q in pool]
    fraction = smallest_fraction
    for _ in range(rungs - 1):
        survivors.sort(key=lambda pair: pair[0], reverse=True)
        keep = max(1, int(len(survivors) / args.eta))
        survivors = survivors[:keep]
        fraction = min(1.0, fraction * args.eta)
        survivors = [(observe(rng, true_q, fraction, args.early_noise), true_q)
                     for _, true_q in survivors]

    best = max(true_q for _, true_q in survivors)
    return {"explored": n, "full_runs": len(survivors), "best": best,
            "distinct_important": n}


def simulate(args):
    budget_units = args.budget_gpu_hours / args.full_run_gpu_hours
    if budget_units < 1:
        die(f"the budget buys {budget_units:.2f} full runs - raise "
            "--budget-gpu-hours or lower --full-run-gpu-hours.")

    strategies = {
        "grid": run_grid,
        "random": run_random,
        "successive halving": run_successive_halving,
    }
    results = {name: [] for name in strategies}
    shapes = {}

    for trial in range(args.trials):
        for name, fn in strategies.items():
            # Same seed per trial across strategies: they face the same luck.
            rng = random.Random(args.seed + trial)
            out = fn(rng, args, budget_units)
            if out is None:
                die(f"the {name} strategy cannot fit a single run in the budget.")
            results[name].append(out["best"])
            shapes[name] = out

    return budget_units, results, shapes


def build_parser():
    p = argparse.ArgumentParser(
        prog="hpo_budget.py",
        description="Grid vs random vs successive halving under one GPU-hour budget "
                    "(seeded simulation, stdlib only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Synthetic search space - the ranking is the lesson, not the units.",
    )
    p.add_argument("--budget-gpu-hours", type=float, default=2048.0)
    p.add_argument("--full-run-gpu-hours", type=float, default=64.0,
                   help="cost of training one config to completion")
    p.add_argument("--dimensions", type=int, default=4,
                   help="hyperparameters being searched (grid pays for all of them)")
    p.add_argument("--important-weight", type=float, default=0.7,
                   help="share of quality variance owned by the one dimension "
                        "that actually matters")
    p.add_argument("--eta", type=float, default=3.0,
                   help="successive-halving reduction factor: keep top 1/eta")
    p.add_argument("--rungs", type=int, default=4,
                   help="number of successive-halving rungs")
    p.add_argument("--early-noise", type=float, default=0.35,
                   help="how misleading a partial run is at fraction 1.0; scaled "
                        "by 1/sqrt(fraction) at shorter rungs")
    p.add_argument("--trials", type=int, default=400,
                   help="independent repetitions of the whole search")
    p.add_argument("--seed", type=int, default=20260729)
    return p


def validate(args):
    require_positive("budget-gpu-hours", args.budget_gpu_hours)
    require_positive("full-run-gpu-hours", args.full_run_gpu_hours)
    require_positive("early-noise", args.early_noise, allow_zero=True)
    if args.dimensions < 1:
        die(f"--dimensions must be at least 1 (got {args.dimensions}).")
    if not 0.0 < args.important_weight < 1.0:
        die("--important-weight must be strictly between 0 and 1 "
            f"(got {args.important_weight}).")
    if args.eta <= 1.0:
        die(f"--eta must be greater than 1 (got {args.eta}).")
    if args.rungs < 2:
        die(f"--rungs must be at least 2 (got {args.rungs}).")
    if args.trials < 1:
        die(f"--trials must be at least 1 (got {args.trials}).")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)

    budget_units, results, shapes = simulate(args)

    print("Hyperparameter search under a fixed budget - seeded simulation")
    print("-" * 78)
    print(f"  budget            : {args.budget_gpu_hours:,.0f} GPU-hours "
          f"= {budget_units:.0f} full runs")
    print(f"  full run          : {args.full_run_gpu_hours:,.0f} GPU-hours")
    print(f"  search space      : {args.dimensions} dimensions, "
          f"{args.important_weight:.0%} of the variance in one of them")
    print(f"  halving           : eta={args.eta:g}, {args.rungs} rungs, "
          f"early-run noise {args.early_noise:g}")
    print(f"  repetitions       : {args.trials} (seed {args.seed})")
    print()

    print("  strategy             configs  full runs  distinct values")
    print("                      explored   completed  on the dim that matters")
    print("  ------------------- --------- ----------- -----------------------")
    for name in ("grid", "random", "successive halving"):
        s = shapes[name]
        print(f"  {name:<19} {s['explored']:9d} {s['full_runs']:11d} "
              f"{s['distinct_important']:23d}")
    print()

    print("  strategy             mean best   median   p10      p90")
    print("  ------------------- ---------- -------- -------- --------")
    for name in ("grid", "random", "successive halving"):
        vals = sorted(results[name])
        mean = statistics.fmean(vals)
        median = statistics.median(vals)
        p10 = vals[int(0.10 * (len(vals) - 1))]
        p90 = vals[int(0.90 * (len(vals) - 1))]
        print(f"  {name:<19} {mean:10.3f} {median:8.3f} {p10:8.3f} {p90:8.3f}")
    print()

    print("  How to read this")
    print("  - Grid and random spend the identical budget, but grid retries the same")
    print("    values of the dimension that matters, so it effectively samples far")
    print("    fewer distinct settings. That gap widens with --dimensions.")
    print("  - Successive halving explores far more configs for the same money by")
    print("    killing bad ones early. Its risk is --early-noise: if short runs do")
    print("    not rank configs correctly, it discards the eventual winner. Raise")
    print("    --early-noise and watch its advantage disappear.")
    print("  - p10 is the number to defend in a review: it is the outcome when the")
    print("    search gets unlucky, and it is what a one-shot experiment actually is.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
