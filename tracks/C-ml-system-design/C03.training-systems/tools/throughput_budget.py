#!/usr/bin/env python3
"""C03 - Time-to-train and dollar budget, with an input-pipeline bottleneck.

An ANALYTIC MODEL, not a measurement. It exists to make one number visible:
the dollar cost of a GPU that is waiting for data.

Model
-----
    compute_capacity = gpus * per_gpu_samples_per_s * scaling_efficiency
    loader_capacity  = aggregate samples/s the input pipeline can deliver

With prefetch (the default), loading of step i+1 overlaps compute of step i, so

    achieved = min(compute_capacity, loader_capacity)

Without prefetch (--no-overlap), the step is serial and the rates add as
reciprocals - the classic reason a single-worker DataLoader halves throughput:

    achieved = 1 / (1/compute_capacity + 1/loader_capacity)

GPU utilization is then achieved / compute_capacity, and every point below
100% is money spent on idle accelerators.

Stdlib only. Run `python throughput_budget.py --help`.
"""

from __future__ import annotations

import argparse
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


def achieved_rate(compute_capacity, loader_capacity, overlap):
    if loader_capacity <= 0:
        die("loader capacity resolved to zero samples/s - nothing would ever train.")
    if overlap:
        return min(compute_capacity, loader_capacity)
    return 1.0 / (1.0 / compute_capacity + 1.0 / loader_capacity)


def run_cost(args, achieved, compute_capacity):
    total_samples = args.dataset_size * args.epochs
    seconds = total_samples / achieved
    hours = seconds / HOUR
    gpu_hours = hours * args.gpus
    dollars = gpu_hours * args.gpu_hour_cost
    utilization = achieved / compute_capacity
    return {
        "samples": total_samples,
        "hours": hours,
        "gpu_hours": gpu_hours,
        "dollars": dollars,
        "utilization": utilization,
        "achieved": achieved,
    }


def print_scenario(label, result):
    print(f"* {label}")
    print(f"    achieved throughput : {result['achieved']:,.0f} samples/s "
          f"({result['utilization'] * 100:.1f}% GPU utilization)")
    print(f"    time to train       : {result['hours']:.2f} h "
          f"({result['hours'] / 24:.2f} days)")
    print(f"    cost                : ${result['dollars']:,.0f} "
          f"({result['gpu_hours']:,.0f} GPU-hours)")
    print()


def sweep_loader(args, compute_capacity, ideal):
    print("  Loader capacity as a multiple of what the GPUs can consume:")
    print("    loader/compute   samples/s      util     hours      cost    tax vs ideal")
    print("    -------------- ----------- --------- --------- --------- ---------------")
    for ratio in (0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 2.0):
        loader = compute_capacity * ratio
        achieved = achieved_rate(compute_capacity, loader, args.overlap)
        r = run_cost(args, achieved, compute_capacity)
        tax = r["dollars"] - ideal["dollars"]
        print(f"    {ratio:14.2f} {r['achieved']:11,.0f} "
              f"{r['utilization'] * 100:8.1f}% {r['hours']:9.1f} "
              f"${r['dollars']:8,.0f} ${tax:14,.0f}")
    print()
    if args.overlap:
        print("  Note the shape: over-provisioning the loader buys nothing (the GPUs cap")
        print("  the rate), while under-provisioning costs proportionally. Input pipelines")
        print("  should be sized with headroom, because the penalty is one-sided.")
    else:
        print("  Without prefetch there is no flat region: utilization keeps improving")
        print("  as the loader gets faster and never reaches 100%, because loading and")
        print("  compute take turns. Overlap is the fix; a faster loader is only a patch.")
    print()


def build_parser():
    p = argparse.ArgumentParser(
        prog="throughput_budget.py",
        description="Time-to-train and dollar cost, including the data-starvation tax.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="All output is derived from these inputs - it is a model, not a benchmark.",
    )
    p.add_argument("--gpus", type=int, default=64)
    p.add_argument("--samples-per-second-per-gpu", type=float, default=120.0,
                   help="throughput of one GPU when it never waits for data")
    p.add_argument("--scaling-efficiency", type=float, default=0.90,
                   help="fraction of linear scaling kept after collectives "
                        "(see B07 for where this number comes from)")
    p.add_argument("--dataset-size", type=float, default=2.0e7,
                   help="samples in one epoch")
    p.add_argument("--epochs", type=float, default=8.0)
    p.add_argument("--gpu-hour-cost", type=float, default=2.50,
                   help="all-in $ per GPU-hour")
    p.add_argument("--loader-samples-per-second", type=float, default=5000.0,
                   help="aggregate samples/s the input pipeline delivers; "
                        "use 0 for an unconstrained loader")
    p.add_argument("--no-overlap", dest="overlap", action="store_false",
                   help="model a loader with no prefetch: load and compute serialise")
    p.set_defaults(overlap=True)
    return p


def validate(args):
    if args.gpus < 1:
        die(f"--gpus must be at least 1 (got {args.gpus}).")
    require_positive("samples-per-second-per-gpu", args.samples_per_second_per_gpu)
    require_positive("dataset-size", args.dataset_size)
    require_positive("epochs", args.epochs)
    require_positive("gpu-hour-cost", args.gpu_hour_cost, allow_zero=True)
    require_positive("loader-samples-per-second", args.loader_samples_per_second,
                     allow_zero=True)
    if not 0.0 < args.scaling_efficiency <= 1.0:
        die("--scaling-efficiency must be in (0, 1] "
            f"(got {args.scaling_efficiency}).")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate(args)

    compute_capacity = (args.gpus * args.samples_per_second_per_gpu
                        * args.scaling_efficiency)
    loader = args.loader_samples_per_second or float("inf")

    print("Training throughput and cost budget - analytic model")
    print("-" * 78)
    print(f"  GPUs                       : {args.gpus}")
    print(f"  per-GPU rate               : {args.samples_per_second_per_gpu:,.0f} samples/s")
    print(f"  scaling efficiency         : {args.scaling_efficiency:.0%}")
    print(f"  compute capacity           : {compute_capacity:,.0f} samples/s")
    loader_text = "unconstrained" if loader == float("inf") else f"{loader:,.0f} samples/s"
    print(f"  input pipeline capacity    : {loader_text}")
    print(f"  workload                   : {args.dataset_size:,.0f} samples x "
          f"{args.epochs:g} epochs")
    print(f"  overlap (prefetch)         : {'on' if args.overlap else 'OFF'}")
    print()

    ideal = run_cost(args, compute_capacity, compute_capacity)
    print("1. Scenarios")
    print_scenario("GPU-bound ceiling (loader never starves the GPUs)", ideal)

    actual_rate = achieved_rate(compute_capacity, loader, args.overlap)
    actual = run_cost(args, actual_rate, compute_capacity)
    print_scenario("Your configuration", actual)

    tax_hours = actual["hours"] - ideal["hours"]
    tax_dollars = actual["dollars"] - ideal["dollars"]
    if tax_dollars < 1.0:
        print("  The input pipeline is not the bottleneck here. Spend your effort on")
        print("  the step itself (kernel efficiency, precision, parallelism strategy).")
    else:
        print(f"  Starvation tax: {tax_hours:.1f} extra hours, ${tax_dollars:,.0f} extra,")
        print(f"  for the same finished model. That is what "
              f"{(1 - actual['utilization']) * 100:.0f}% idle GPU costs.")
    print()

    print("2. Sensitivity to input-pipeline capacity")
    sweep_loader(args, compute_capacity, ideal)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130)
