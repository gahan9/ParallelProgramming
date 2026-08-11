#!/usr/bin/env python3
"""Drill 4 worked solution: fit alpha and beta from lab 2's measured CSV.

Run (from this directory):
  python -S -I fit_alpha_beta.py --input ../pingpong.csv
  python -S -I fit_alpha_beta.py --input ../pingpong.csv --classify 1024,65536,16777216

WHY THE FIT IS PIECEWISE

The model is Hockney's T(m) = alpha + m*beta, which is a straight line - so the obvious
move is one least-squares fit over every row. That is wrong here, and understanding why is
most of the drill.

A ping-pong sweep spans four or five orders of magnitude in m, and over that range the
data is NOT one line. At small m the payload term is negligible and the curve is flat at
alpha. At large m the startup term is negligible and the curve is a line of slope beta.
Between them sits a step where the implementation switches from its eager protocol to
rendezvous, which changes the effective beta. Fitting one line through all of that gives a
slope that is a weighted average of two different regimes and an intercept that is not
alpha - a number with no referent, that will still look plausible.

So: estimate alpha from the flat region and beta from the linear region, and pick the
regions from the data rather than by hardcoding a size.

WHY THE MEDIAN FOR ALPHA

The flat rows all measure the same quantity, so any average would do statistically - but
the failure mode is a single row inflated by a descheduled iteration or an interrupt, and
the mean chases an outlier while the median ignores it. Latency data is where outliers
live, because everything that can go wrong makes it slower and nothing makes it faster.
For the same reason the floor is identified relative to the FASTEST row, not the mean.

WHY RELATIVE RESIDUALS

The largest message in the sweep is thousands of times slower than the smallest. An
absolute residual is therefore dominated entirely by the last few rows, so a window that
swallows the whole flat region - or a protocol step - still scores beautifully while being
wrong. Charging every point its relative error is what lets the window search notice that
a step is inside the window.

WHY THE LONGEST ACCEPTABLE WINDOW RATHER THAN THE BEST-FITTING ONE

A two-point window fits a line perfectly and tells you nothing. Minimising the residual
would reliably select the least informative answer available. The longest window whose fit
is still good is the strongest claim the data actually supports, so that is what to report
- and reporting how many rows it covers lets the reader judge it.

WHY THE ERROR HANDLING IS PART OF THE EXERCISE

This tool's job is to tell a colleague what is wrong with their file. A traceback fails at
that job: it names a line of MY code rather than a line of THEIR data. Every rejection
here is one line, names the file and the line number, says what was expected, and exits
non-zero so a script calling this can tell.

Standard library only, so it runs under `python -S -I`.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

PROGRAM = "fit_alpha_beta"


def die(message):
    """One line on stderr, non-zero exit, never a traceback."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def load(path):
    """Read lab 2's CSV into a validated list of (bytes, seconds)."""
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
    size_idx = next((i for i, name in enumerate(header) if "byte" in name), None)
    time_idx = next((i for i, name in enumerate(header) if name.endswith("_us")), None)
    if size_idx is None or time_idx is None:
        die("%s: expected a header with a 'bytes' column and a '*_us' time column, "
            "as lab 2 writes" % path)

    data = []
    for lineno, row in enumerate(rows[1:], start=2):
        # TODO 1a, done: distinguish the two ways a row can be malformed, because they
        # have different causes. Too few columns usually means the file was edited or
        # truncated; a non-numeric field usually means a stray header from a concatenated
        # run, or mpirun's output interleaved into the redirect.
        if max(size_idx, time_idx) >= len(row):
            die("%s line %d: expected at least %d columns, found %d"
                % (path, lineno, max(size_idx, time_idx) + 1, len(row)))
        try:
            nbytes = float(row[size_idx])
            seconds = float(row[time_idx]) * 1e-6
        except ValueError:
            die("%s line %d: non-numeric value in the size or time column" % (path, lineno))
        if not (math.isfinite(nbytes) and math.isfinite(seconds)):
            die("%s line %d: size and time must be finite" % (path, lineno))
        if nbytes < 0:
            die("%s line %d: negative message size %g" % (path, lineno, nbytes))
        if seconds <= 0:
            die("%s line %d: non-positive time %g; no measurement takes zero time"
                % (path, lineno, seconds))
        data.append((nbytes, seconds))

    # TODO 1b, done. Three rows is the minimum that can distinguish two regimes: two
    # points define a line with zero residual, so a two-row file would produce a
    # confident fit backed by no evidence at all.
    if len(data) < 3:
        die("%s has %d usable row(s); need at least 3 to separate the latency floor from "
            "the bandwidth slope" % (path, len(data)))

    # TODO 1c, done. Naming both sizes matters: it tells the user WHERE to look, which is
    # the difference between a message they can act on and one they have to investigate.
    for i in range(1, len(data)):
        if data[i][0] <= data[i - 1][0]:
            die("%s: size column is not strictly increasing (%g then %g); sort the file "
                "or remove duplicate sizes" % (path, data[i - 1][0], data[i][0]))

    return data


def ols(points):
    """Fit y = intercept + slope*x. Returns (intercept, slope, relative_rms_residual)."""
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    if sxx <= 0:
        raise ValueError("every size in the window is identical")
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    rel_sq = sum(((y - (intercept + slope * x)) / y) ** 2 for x, y in points)
    return intercept, slope, math.sqrt(rel_sq / n)


def fit(data, flat_tolerance=0.10, residual_target=0.05):
    """Piecewise fit of T(m) = alpha + m*beta. See the module docstring for the why."""
    t_floor = min(t for _, t in data)

    # The floor is the LEADING run within tolerance of the fastest row. Leading, not
    # "every row within tolerance": a late row that happens to fall back inside the band
    # is a measurement artefact, not evidence that the payload became free again.
    flat = []
    for nbytes, seconds in data:
        if seconds > t_floor * (1.0 + flat_tolerance):
            break
        flat.append((nbytes, seconds))
    if not flat:
        flat = [data[0]]
    flat_times = sorted(t for _, t in flat)
    mid = len(flat_times) // 2
    alpha = (flat_times[mid] if len(flat_times) % 2
             else 0.5 * (flat_times[mid - 1] + flat_times[mid]))

    # Longest trailing window that starts above the floor and still fits.
    best = None
    for start in range(max(min(len(flat), len(data) - 3), 0), len(data) - 2):
        window = data[start:]
        try:
            intercept, slope, rel = ols(window)
        except ValueError:
            continue
        if slope <= 0:
            continue
        candidate = (start, intercept, slope, rel, len(window))
        if rel <= residual_target:
            best = candidate
            break
        if best is None or rel < best[3]:
            best = candidate
    if best is None:
        die("could not fit a positive bandwidth slope: the time column does not grow with "
            "message size, so this data cannot yield a beta")
    start, intercept, beta, rel, window_len = best

    return {
        "alpha_s": alpha,
        "beta_s_per_byte": beta,
        "bandwidth_bytes_per_s": 1.0 / beta,
        "crossover_bytes": alpha / beta,
        "flat_rows": len(flat),
        "flat_max_bytes": flat[-1][0],
        "fit_first_bytes": data[start][0],
        "fit_rows": window_len,
        "fit_intercept_s": intercept,
        "fit_relative_rms": rel,
        "rows": len(data),
    }


def classify(nbytes, crossover):
    """latency-bound below m*/2, bandwidth-bound above 2*m*, balanced in the 2x band."""
    if not math.isfinite(crossover) or crossover <= 0:
        return "unknown"
    if nbytes < crossover / 2.0:
        return "latency-bound"
    if nbytes > crossover * 2.0:
        return "bandwidth-bound"
    return "balanced"


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

    # Name the input. These two numbers describe one transport on one machine, and a fit
    # of intra-node shared memory is not a fit of a network; detached from its provenance
    # the pair is worse than useless, because it looks authoritative.
    print("fitted from: %s   (%d rows, %.0f to %.0f bytes)"
          % (args.input, result["rows"], data[0][0], data[-1][0]))
    print("alpha              %10.3f us        median of the %d-row floor (to %.0f bytes)"
          % (result["alpha_s"] * 1e6, result["flat_rows"], result["flat_max_bytes"]))
    print("beta               %10.4f ns/byte   slope of the %d rows from %.0f bytes up"
          % (result["beta_s_per_byte"] * 1e9, result["fit_rows"], result["fit_first_bytes"]))
    print("implied bandwidth  %10.2f GB/s      = 1/beta, 1 GB = 1e9 bytes"
          % (result["bandwidth_bytes_per_s"] / 1e9))
    print("crossover m*       %10.0f bytes     = alpha/beta"
          % result["crossover_bytes"])
    print("fit residual       %10.2f %%         relative RMS over the fitted window"
          % (result["fit_relative_rms"] * 100.0))
    if sizes:
        print("")
        for nbytes in sizes:
            print("%12.0f bytes  %s" % (nbytes, classify(nbytes, result["crossover_bytes"])))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
