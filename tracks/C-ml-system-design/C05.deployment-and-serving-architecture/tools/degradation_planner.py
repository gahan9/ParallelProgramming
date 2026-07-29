#!/usr/bin/env python3
"""End-to-end availability with and without graceful degradation.

Standard library only (json, math, argparse, sys). No numpy.

A serving stack is a chain of things that must all work (gateway, router,
feature lookup, model) followed by a set of things where only ONE has to work
(primary model, cached prediction, small model, static heuristic). Serial
components multiply; alternatives combine as one-minus-the-product-of-failures.
This tool does that arithmetic, converts it into downtime per year, and prices
the answer in "how much did the fallback chain buy me".

It also does two things a naive availability calculation refuses to do:

  * Quality-weighted availability. A cached prediction that is four hours old
    is not a full-quality response. Reporting 99.99% while silently serving
    stale answers is how teams lose trust in their own SLOs.
  * Shared-fate correlation. Independence is the assumption that makes the
    arithmetic pretty and the estimate wrong. If the primary and the fallback
    both live behind the same load balancer, in the same region, or depend on
    the same config service, they fail together. The --shared-fate parameter
    makes that cost visible.

Input is a JSON file (see --print-example) or the built-in demo stack.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

MINUTES_PER_YEAR = 365.25 * 24 * 60

EXAMPLE = {
    "name": "LLM assistant serving stack",
    "serial": [
        {"name": "edge / CDN", "availability": 0.99999},
        {"name": "API gateway + authn", "availability": 0.9999},
        {"name": "model router", "availability": 0.9995}
    ],
    "fallbacks": [
        {"name": "primary GPU model (70B)", "availability": 0.995, "quality": 1.00},
        {"name": "response cache (semantic, <1h old)", "availability": 0.98, "quality": 0.72},
        {"name": "small model (7B) on spare capacity", "availability": 0.99, "quality": 0.65},
        {"name": "static templated answer + retry hint", "availability": 0.9999, "quality": 0.20}
    ]
}


# --------------------------------------------------------------------------
# Core arithmetic
# --------------------------------------------------------------------------


def serial_availability(components):
    """All of these must work: probabilities multiply."""
    product = 1.0
    for c in components:
        product *= c["availability"]
    return product


def any_of_availability(alternatives, shared_fate=0.0):
    """At least one of these must work.

    With shared_fate = f, a fraction f of the failure probability mass is a
    common-mode event that takes every alternative down at once. The remaining
    (1 - f) behaves independently. f = 0 is the textbook optimistic answer;
    f = 1 says the fallbacks are decorative.
    """
    independent = 1.0
    for a in alternatives:
        independent *= (1.0 - a["availability"])
    independent_success = 1.0 - independent
    # Common mode: the whole chain is down together with probability equal to
    # the shared-fate fraction of the primary's failure probability.
    common_mode_failure = shared_fate * (1.0 - alternatives[0]["availability"])
    return (1.0 - common_mode_failure) * independent_success


def quality_weighted(alternatives, shared_fate=0.0):
    """Expected quality delivered per request, 0-1.

    The chain is tried in order, so alternative i is reached only when every
    earlier one has failed.
    """
    total = 0.0
    reach = 1.0
    for a in alternatives:
        total += reach * a["availability"] * a.get("quality", 1.0)
        reach *= (1.0 - a["availability"])
    common_mode_failure = shared_fate * (1.0 - alternatives[0]["availability"])
    return (1.0 - common_mode_failure) * total


def downtime_minutes_per_year(availability):
    return (1.0 - availability) * MINUTES_PER_YEAR


def nines(availability):
    """How many nines, as a continuous number. 0.999 -> 3.0."""
    if availability >= 1.0:
        return float("inf")
    if availability <= 0.0:
        return 0.0
    return -math.log10(1.0 - availability)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def fmt_downtime(minutes):
    if minutes >= 1440:
        return "%.1f days" % (minutes / 1440.0)
    if minutes >= 60:
        return "%.1f hours" % (minutes / 60.0)
    return "%.1f min" % minutes


def validate(stack, shared_fate):
    if not 0.0 <= shared_fate <= 1.0:
        raise ValueError("--shared-fate must be between 0 and 1, got %r" % shared_fate)
    if not isinstance(stack, dict):
        raise ValueError("the config must be a JSON object")
    for key in ("serial", "fallbacks"):
        if key not in stack or not isinstance(stack[key], list) or not stack[key]:
            raise ValueError("the config needs a non-empty '%s' list "
                             "(run with --print-example to see the shape)" % key)
    for group in ("serial", "fallbacks"):
        for entry in stack[group]:
            if "name" not in entry or "availability" not in entry:
                raise ValueError("every %s entry needs 'name' and 'availability'" % group)
            a = entry["availability"]
            if not isinstance(a, (int, float)) or not 0.0 < a <= 1.0:
                raise ValueError("availability for %r must be in (0, 1], got %r"
                                 % (entry["name"], a))
            q = entry.get("quality", 1.0)
            if not 0.0 <= q <= 1.0:
                raise ValueError("quality for %r must be in [0, 1], got %r" % (entry["name"], q))


def report(stack, shared_fate):
    validate(stack, shared_fate)
    serial = stack["serial"]
    fallbacks = stack["fallbacks"]

    print("=" * 88)
    print("AVAILABILITY WITH AND WITHOUT GRACEFUL DEGRADATION")
    print("  stack: %s" % stack.get("name", "(unnamed)"))
    print("=" * 88)

    print()
    print("Serial path -- every one of these must work")
    print("%-46s %12s %14s" % ("component", "availability", "downtime/yr"))
    print("%-46s %12s %14s" % ("-" * 46, "-" * 12, "-" * 14))
    for c in serial:
        print("%-46s %11.5f%% %14s"
              % (c["name"], 100 * c["availability"],
                 fmt_downtime(downtime_minutes_per_year(c["availability"]))))
    s_avail = serial_availability(serial)
    print("%-46s %11.5f%% %14s"
          % ("SERIAL SUBTOTAL (product)", 100 * s_avail,
             fmt_downtime(downtime_minutes_per_year(s_avail))))

    print()
    print("Fallback chain -- tried in order, only one has to work")
    print("%-46s %12s %8s %11s" % ("alternative", "availability", "quality", "reached"))
    print("%-46s %12s %8s %11s" % ("-" * 46, "-" * 12, "-" * 8, "-" * 11))
    reach = 1.0
    for a in fallbacks:
        print("%-46s %11.5f%% %7.2f %10.4f%%"
              % (a["name"], 100 * a["availability"], a.get("quality", 1.0), 100 * reach))
        reach *= (1.0 - a["availability"])

    primary = fallbacks[0]
    no_degrade = s_avail * primary["availability"]
    with_degrade = s_avail * any_of_availability(fallbacks, shared_fate)
    quality = s_avail * quality_weighted(fallbacks, shared_fate)

    print()
    print("Result (shared-fate correlation = %.2f)" % shared_fate)
    print("%-46s %13s %8s %16s" % ("scenario", "availability", "nines", "downtime/yr"))
    print("%-46s %13s %8s %16s" % ("-" * 46, "-" * 13, "-" * 8, "-" * 16))
    for label, value in (("primary only (500 on any failure)", no_degrade),
                         ("with fallback chain (any response)", with_degrade),
                         ("quality-weighted (honest number)", quality)):
        print("%-46s %12.5f%% %8.2f %16s"
              % (label, 100 * value, nines(value),
                 fmt_downtime(downtime_minutes_per_year(value))))

    saved = downtime_minutes_per_year(no_degrade) - downtime_minutes_per_year(with_degrade)
    print()
    print("  The fallback chain buys %s of availability per year"
          % fmt_downtime(saved))
    print("  (%.2f nines -> %.2f nines). But %s of that is served at reduced"
          % (nines(no_degrade), nines(with_degrade),
             fmt_downtime(downtime_minutes_per_year(quality)
                          - downtime_minutes_per_year(with_degrade))))
    print("  quality, which is why the third row exists. Publish the third row.")

    # Sensitivity: which single improvement buys the most?
    print()
    print("Sensitivity -- effect of taking each component to one extra nine")
    print("%-46s %18s" % ("improve this", "downtime saved/yr"))
    print("%-46s %18s" % ("-" * 46, "-" * 18))
    deltas = []
    base = with_degrade
    for group_name, group in (("serial", serial), ("fallback", fallbacks)):
        for index, entry in enumerate(group):
            improved = dict(entry)
            improved["availability"] = 1.0 - (1.0 - entry["availability"]) / 10.0
            new_serial = list(serial)
            new_fallbacks = list(fallbacks)
            if group_name == "serial":
                new_serial[index] = improved
            else:
                new_fallbacks[index] = improved
            value = serial_availability(new_serial) * any_of_availability(new_fallbacks, shared_fate)
            deltas.append((downtime_minutes_per_year(base) - downtime_minutes_per_year(value),
                           entry["name"]))
    for saved_minutes, name in sorted(deltas, reverse=True):
        print("%-46s %18s" % (name, fmt_downtime(max(0.0, saved_minutes))))
    print()
    print("  Spend effort at the top of this list. Anything on the serial path is")
    print("  usually worth more than the fourth fallback, because the serial path")
    print("  has no alternative by definition.")

    # Correlation sweep.
    print()
    print("Shared-fate sensitivity -- what independence is worth")
    print("%-20s %14s %10s %16s" % ("shared fate", "availability", "nines", "downtime/yr"))
    print("%-20s %14s %10s %16s" % ("-" * 20, "-" * 14, "-" * 10, "-" * 16))
    for f in (0.0, 0.1, 0.25, 0.5, 1.0):
        value = s_avail * any_of_availability(fallbacks, f)
        print("%-20s %13.5f%% %10.2f %16s"
              % ("%.0f%%" % (100 * f), 100 * value, nines(value),
                 fmt_downtime(downtime_minutes_per_year(value))))
    print()
    print("  Read the last row as the answer to 'what if the fallback shares a")
    print("  dependency with the primary?'. A cache in the same cluster, a small")
    print("  model behind the same router, or a heuristic that reads the same")
    print("  feature store are all closer to the bottom row than the top one.")
    print("  Test this by actually failing the primary in a game day, not by")
    print("  believing the spreadsheet.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_stack(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ValueError("no such config file: %s" % path)
    except json.JSONDecodeError as exc:
        raise ValueError("%s is not valid JSON: %s" % (path, exc))


def main(argv=None):
    if sys.version_info < (3, 8):
        sys.stderr.write("ERROR: this tool needs Python 3.8 or newer.\n")
        return 2

    parser = argparse.ArgumentParser(
        description="End-to-end availability with and without graceful degradation.")
    parser.add_argument("--config", help="JSON file describing the stack (default: built-in demo)")
    parser.add_argument("--shared-fate", type=float, default=0.0,
                        help="0-1 correlation between the primary and its fallbacks")
    parser.add_argument("--print-example", action="store_true",
                        help="print the built-in demo config as JSON and exit")
    args = parser.parse_args(argv)

    try:
        if args.print_example:
            print(json.dumps(EXAMPLE, indent=2))
            return 0
        stack = load_stack(args.config) if args.config else EXAMPLE
        report(stack, args.shared_fate)
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
