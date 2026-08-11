#!/usr/bin/env python3
"""Drill 4 starter: fit alpha and beta from the measured CSV that lab 2 emits.

This file RUNS as given and reports that the fit is not implemented. It does not crash,
and it must never crash: on bad input it exits non-zero with ONE line of explanation,
never a traceback. Keeping that property while you add the fit is part of the drill - a
tool that dumps a stack trace at a colleague has failed at its actual job, which is to
tell them what is wrong with their file.

Standard library only. No numpy, no third-party anything, so this must run under:
  python -S -I fit_alpha_beta.py --input ../pingpong.csv

Reference implementation with commentary: ../solutions/fit_alpha_beta.py
The module's own version, which you can diff against: ../tools/fit_alpha_beta.py
"""

from __future__ import annotations

import argparse
import csv
import sys

PROGRAM = "fit_alpha_beta"


def die(message):
    """One line on stderr, non-zero exit, no traceback. Use this for every failure."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def load(path):
    """Read the lab-2 CSV into a list of (bytes, seconds), sorted by size.

    Given to you, apart from the three validations below. The broad except clauses are
    the safety net that keeps the promise in the docstring while the TODOs are still
    open; as you implement each check, its specific message should fire first and the
    net should stop being what the user sees.
    """
    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.reader(handle) if r and any(f.strip() for f in r)]
    except OSError as exc:
        die("cannot read %s: %s" % (path, exc.strerror or exc))
    except UnicodeDecodeError:
        die("%s is not valid UTF-8 text; is it really the CSV from lab 2?" % path)

    if not rows:
        die("%s is empty" % path)

    header = [(h or "").strip().lower() for h in rows[0]]
    # Lab 2 emits: bytes,iters,half_roundtrip_us
    try:
        size_idx = next(i for i, name in enumerate(header) if "byte" in name)
        time_idx = next(i for i, name in enumerate(header) if name.endswith("_us"))
    except StopIteration:
        die("%s: expected a header with a 'bytes' column and a '*_us' time column, "
            "as lab 2 writes" % path)

    data = []
    for lineno, row in enumerate(rows[1:], start=2):
        # TODO 1a: reject a MALFORMED row with a one-line message naming the line
        #   number. Malformed covers: too few columns, and a value in the size or time
        #   column that is not a number. Right now the try/except below swallows both
        #   into one generic message, which is honest but unhelpful.
        try:
            nbytes = float(row[size_idx])
            seconds = float(row[time_idx]) * 1e-6      # the column is microseconds
        except (ValueError, IndexError):
            die("%s line %d: malformed row" % (path, lineno))
        data.append((nbytes, seconds))

    # TODO 1b: reject a file with FEWER THAN THREE usable rows. Two rows can be joined
    #   by a straight line with zero residual, so they cannot distinguish a latency
    #   floor from a bandwidth slope - the fit would be arithmetic without evidence.
    #
    # TODO 1c: reject a NON-MONOTONIC size column, naming the two sizes involved. A
    #   sweep that goes 1024 then 512 is either a concatenation of two runs or a sort
    #   that never happened; either way the regime split below is meaningless on it.

    return data


def fit(data):
    """Return a dict with alpha_s, beta_s_per_byte, bandwidth and crossover_bytes.

    TODO 2: implement the piecewise fit of T(m) = alpha + m*beta.

      alpha: the small-m rows form a flat floor, because there the payload term is
        negligible against the startup term. Identify that leading run - a row whose
        time is within a few percent of the fastest row is on the floor - and take a
        representative value of it. Prefer the median to the mean: one descheduled
        iteration produces an outlier, and the median ignores it.

        beta: the large-m rows form a straight line whose SLOPE is beta. Fit it with
        ordinary least squares (about ten lines with no numpy: you need the means, the
        two sums sxx and sxy, then slope = sxy/sxx). You will want `import math` for the
        residual; that is the only import this drill adds.

        Two traps worth designing against:
          * Do not let the flat rows into the slope window. They are alpha, not beta,
            and including them drags the fitted bandwidth toward the latency regime.
          * Judge the fit by RELATIVE residuals, not absolute ones. The largest message
            is thousands of times slower than the smallest, so an absolute residual is
            dominated by the tail and will happily accept a window containing a protocol
            step.

      crossover: m* = alpha / beta, the size at which the two terms are equal.

    Return None while unimplemented so main() can report that cleanly.
    """
    del data                    # delete this line when you implement the fit
    return None


def classify(nbytes, crossover):
    """Label one message size against the crossover.

    TODO 3: return "latency-bound" below m*/2, "bandwidth-bound" above 2*m*, and
      "balanced" in between. The 2x band matters: within it neither term dominates, so
      both fewer messages and fewer bytes help, and a claim that a message is
      "latency-bound" is not supportable there.

    Guard the degenerate case: a crossover of zero, or one that is not finite, means the
    fit found no usable beta, and the honest label for a size then is "unknown".
    """
    del nbytes, crossover       # delete this line when you implement it
    return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fit alpha and beta from the measured CSV lab 2 emits (drill 4).")
    ap.add_argument("--input", "-i", required=True, help="CSV from ./ping_pong")
    ap.add_argument("--classify", default="",
                    help="comma-separated sizes in bytes to label against m*")
    args = ap.parse_args(argv)

    try:
        sizes = [float(s) for s in args.classify.split(",") if s.strip()]
    except ValueError:
        ap.error("--classify takes comma-separated numbers of bytes")

    data = load(args.input)
    result = fit(data)
    if result is None:
        die("fit() is not implemented yet (TODO 2). Read %d rows from %s, so the input "
            "path works - now do the arithmetic." % (len(data), args.input))

    print("fitted from: %s   (%d rows)" % (args.input, len(data)))
    print("alpha              %10.3f us" % (result["alpha_s"] * 1e6))
    print("beta               %10.4f ns/byte" % (result["beta_s_per_byte"] * 1e9))
    print("implied bandwidth  %10.2f GB/s" % (result["bandwidth_bytes_per_s"] / 1e9))
    print("crossover m*       %10.0f bytes" % result["crossover_bytes"])
    for nbytes in sizes:
        print("%12.0f bytes  %s" % (nbytes, classify(nbytes, result["crossover_bytes"])))

    # A closing thought to put in your own output: alpha and beta belong to the link
    # class you measured. A fit of one node's shared memory is not a fit of a network,
    # and labelling the numbers with their provenance is the difference between a
    # measurement and a rumour.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
