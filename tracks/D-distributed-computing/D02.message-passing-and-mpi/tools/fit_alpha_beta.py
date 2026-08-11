#!/usr/bin/env python3
"""Fit alpha and beta from MEASURED ping-pong data (D02 lab 2).

This tool is the one exception in Track D: it does not print a model disclaimer,
because it does not model anything. It consumes a CSV that `mpi/ping_pong.c` wrote on
a real machine and reports what that data says. It names the file it fitted so the
numbers stay attached to their provenance -- alpha and beta are properties of a
transport, and a fit of intra-node shared memory is not a fit of a network.

Input: the CSV lab 2 emits, a header plus one row per message size.

    bytes,iters,half_roundtrip_us
    0,1000,0.412
    1,1000,0.418
    ...

Model being fitted, from Hockney: T(m) = alpha + m*beta. The fit is piecewise because
the data is: a flat latency floor at small m, a straight bandwidth line at large m,
and frequently a step between them where the implementation switches from its eager
protocol to rendezvous.

Standard library only, so this runs anywhere:
  python -S -I tools/fit_alpha_beta.py --input pingpong.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys

PROGRAM = "fit_alpha_beta"

# Time-column names understood, mapped to their multiplier into seconds. Anything
# else is rejected with a message rather than silently assumed to be seconds.
TIME_UNITS = (
    ("_ns", 1e-9), ("_us", 1e-6), ("_ms", 1e-3), ("_s", 1.0),
    ("ns", 1e-9), ("us", 1e-6), ("ms", 1e-3), ("seconds", 1.0), ("sec", 1.0),
)


def die(message: str) -> None:
    """One line on stderr, non-zero exit, never a traceback."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def find_columns(header):
    """Return (bytes_index, time_index, seconds_multiplier) or raise ValueError."""
    lowered = [(h or "").strip().lower() for h in header]
    size_idx = None
    for i, name in enumerate(lowered):
        if "byte" in name or name in ("size", "m", "bytes"):
            size_idx = i
            break
    if size_idx is None:
        raise ValueError("no size column found; expected a header containing 'bytes'")

    for i, name in enumerate(lowered):
        if i == size_idx or "byte" in name or "iter" in name:
            continue
        for suffix, mult in TIME_UNITS:
            if name.endswith(suffix):
                return size_idx, i, mult
    raise ValueError(
        "no time column found; expected a name ending in _ns/_us/_ms/_s, "
        "as in 'half_roundtrip_us'")


def read_rows(path):
    """Parse the CSV into a sorted list of (bytes, seconds). Exits on bad input."""
    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = [r for r in reader if r and any(field.strip() for field in r)]
    except OSError as exc:
        die("cannot read %s: %s" % (path, exc.strerror or exc))
    except UnicodeDecodeError:
        die("%s is not valid UTF-8 text; is it really the CSV from lab 2?" % path)

    if len(rows) < 2:
        die("%s has no data rows (expected a header plus at least 3 rows)" % path)

    try:
        size_idx, time_idx, mult = find_columns(rows[0])
    except ValueError as exc:
        die("%s: %s" % (path, exc))

    data = []
    for lineno, row in enumerate(rows[1:], start=2):
        if max(size_idx, time_idx) >= len(row):
            die("%s line %d: expected at least %d columns, found %d"
                % (path, lineno, max(size_idx, time_idx) + 1, len(row)))
        try:
            nbytes = float(row[size_idx])
            seconds = float(row[time_idx]) * mult
        except ValueError:
            die("%s line %d: non-numeric value in the size or time column" % (path, lineno))
        if not (math.isfinite(nbytes) and math.isfinite(seconds)):
            die("%s line %d: size and time must be finite" % (path, lineno))
        if nbytes < 0:
            die("%s line %d: negative message size %g" % (path, lineno, nbytes))
        if seconds <= 0:
            die("%s line %d: non-positive time %g; a measurement cannot take zero time"
                % (path, lineno, seconds))
        data.append((nbytes, seconds))

    if len(data) < 3:
        die("%s has %d usable row(s); need at least 3 to separate two regimes"
            % (path, len(data)))
    for i in range(1, len(data)):
        if data[i][0] <= data[i - 1][0]:
            die("%s: size column is not strictly increasing (%g then %g); "
                "sort the file or remove duplicates"
                % (path, data[i - 1][0], data[i][0]))
    return data


def ols(points):
    """Least-squares fit y = intercept + slope*x. Returns (intercept, slope, rel_rms).

    The residual is reported as the RMS of the PER-POINT relative error, not of the
    absolute error. Over a size sweep the largest message is thousands of times slower
    than the smallest, so an absolute residual is dominated by the tail and a window
    that swallows the latency floor or a protocol step still scores well. The relative
    form charges every point equally, which is what makes the window search below able
    to notice a step.
    """
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    if sxx <= 0:
        raise ValueError("all sizes identical in the fitted window")
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    rel_sq = 0.0
    for x, y in points:
        if y <= 0:
            raise ValueError("non-positive time in the fitted window")
        rel_sq += ((y - (intercept + slope * x)) / y) ** 2
    return intercept, slope, math.sqrt(rel_sq / n)


def fit(data, flat_tolerance=0.10, residual_target=0.05):
    """Piecewise fit. Returns a dict of fitted parameters and the windows used.

    The latency floor is the leading run of sizes whose time stays within
    `flat_tolerance` of the fastest row: those rows are paying startup only, so their
    times estimate alpha with no bandwidth contamination.

    The bandwidth line is chosen by trying every trailing window that starts STRICTLY
    ABOVE the latency floor and holds at least three rows, then keeping the LONGEST one
    whose relative RMS residual is under `residual_target`. Two deliberate choices:

    * Longest-acceptable, not best-fitting. A two-point window fits perfectly and says
      nothing, so minimising the residual would reliably pick the least informative
      answer. The longest window that still fits is the strongest claim the data
      supports.
    * Starting above the floor, because the flat rows are alpha, not beta. Letting them
      into the slope window drags the fitted bandwidth toward the latency regime while
      still scoring well on residuals.
    """
    times = [t for _, t in data]
    t_floor = min(times)

    flat = []
    for nbytes, seconds in data:
        if seconds <= t_floor * (1.0 + flat_tolerance):
            flat.append((nbytes, seconds))
        else:
            break
    if not flat:
        flat = [data[0]]
    flat_times = sorted(t for _, t in flat)
    mid = len(flat_times) // 2
    alpha = (flat_times[mid] if len(flat_times) % 2
             else 0.5 * (flat_times[mid - 1] + flat_times[mid]))

    best = None
    first_start = min(len(flat), len(data) - 3)
    for start in range(max(first_start, 0), len(data) - 2):
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
            break                      # first (longest) acceptable window wins
        if best is None or rel < best[3]:
            best = candidate
    if best is None:
        die("could not fit a positive bandwidth slope; the time column does not grow "
            "with message size, so this data cannot yield a beta")
    start, intercept, beta, rel, window_len = best

    crossover = alpha / beta if beta > 0 else float("inf")
    return {
        "alpha_s": alpha,
        "beta_s_per_byte": beta,
        "bandwidth_bytes_per_s": 1.0 / beta,
        "crossover_bytes": crossover,
        "flat_rows": len(flat),
        "flat_max_bytes": flat[-1][0],
        "fit_first_bytes": data[start][0],
        "fit_rows": window_len,
        "fit_intercept_s": intercept,
        "fit_relative_rms": rel,
        "rows": len(data),
        "min_time_s": t_floor,
    }


def find_protocol_step(data, alpha, beta, threshold=1.15):
    """Largest jump in excess of what alpha + m*beta predicts, i.e. a protocol switch.

    Returns None when nothing stands out. The eager-to-rendezvous transition is a
    tuning knob, not a constant, so this reports a candidate and refuses to name it a
    threshold.
    """
    worst = None
    for i in range(1, len(data)):
        (s0, t0), (s1, t1) = data[i - 1], data[i]
        predicted = (alpha + s1 * beta) / (alpha + s0 * beta)
        if predicted <= 0 or t0 <= 0:
            continue
        excess = (t1 / t0) / predicted
        if excess >= threshold and (worst is None or excess > worst["excess"]):
            worst = {"between_bytes": s0, "and_bytes": s1, "excess": excess,
                     "time_ratio": t1 / t0, "modelled_ratio": predicted}
    return worst


def classify(nbytes, crossover):
    """Label a message size against the crossover m*, with 'balanced' spanning 2x."""
    if crossover <= 0 or not math.isfinite(crossover):
        return "unknown"
    if nbytes < crossover / 2.0:
        return "latency-bound"
    if nbytes > crossover * 2.0:
        return "bandwidth-bound"
    return "balanced"


def human_bytes(nbytes):
    if nbytes < 1024:
        return "%.0f B" % nbytes
    for unit, scale in (("KiB", 1024.0), ("MiB", 1024.0**2), ("GiB", 1024.0**3)):
        if nbytes < scale * 1024 or unit == "GiB":
            return "%.3g %s" % (nbytes / scale, unit)
    return "%.0f B" % nbytes


SUFFIXES = {
    "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9,
    "kib": 1024, "mib": 1024**2, "gib": 1024**3,
}


def parse_size(text):
    s = text.strip().lower()
    scale = 1.0
    for suffix in ("kib", "mib", "gib", "kb", "mb", "gb", "b"):
        if s.endswith(suffix):
            s, scale = s[: -len(suffix)].strip(), float(SUFFIXES[suffix])
            break
    try:
        value = float(s) * scale
    except ValueError:
        raise ValueError("cannot parse %r as a size; write a number with an optional "
                         "B/KB/MB/GB/KiB/MiB/GiB suffix" % text)
    if value < 0 or not math.isfinite(value):
        raise ValueError("size must be finite and non-negative: %s" % text)
    return value


def build_parser():
    ap = argparse.ArgumentParser(
        description="Fit alpha and beta from the measured CSV that lab 2 emits.",
        epilog="This tool reports MEASURED data, so it names its input file rather "
               "than printing a model disclaimer.",
    )
    ap.add_argument("--input", "-i", required=True,
                    help="CSV from `mpirun -np 2 ./ping_pong` (bytes,iters,time)")
    ap.add_argument("--classify", default="",
                    help="comma-separated sizes to label against the fitted crossover")
    ap.add_argument("--flat-tolerance", type=float, default=0.10,
                    help="a row within this fraction of the fastest row is latency-only "
                         "(default 0.10)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not (0.0 < args.flat_tolerance < 1.0):
        ap.error("--flat-tolerance must be between 0 and 1")

    try:
        sizes = [parse_size(s) for s in args.classify.split(",") if s.strip()]
    except ValueError as exc:
        ap.error("bad --classify: %s" % exc)

    data = read_rows(args.input)
    result = fit(data, flat_tolerance=args.flat_tolerance)
    step = find_protocol_step(data, result["alpha_s"], result["beta_s_per_byte"])
    result["protocol_step"] = step
    result["source_file"] = args.input
    result["classified"] = [
        {"bytes": m, "regime": classify(m, result["crossover_bytes"])} for m in sizes
    ]

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 74)
    print("MEASURED data fitted from: %s" % args.input)
    print("%d rows, sizes %s to %s"
          % (result["rows"], human_bytes(data[0][0]), human_bytes(data[-1][0])))
    print("=" * 74)
    print("alpha (latency)      %10.3f us      median of the %d-row latency floor "
          "(up to %s)"
          % (result["alpha_s"] * 1e6, result["flat_rows"],
             human_bytes(result["flat_max_bytes"])))
    print("beta  (1/bandwidth)  %10.4f ns/byte slope over the %d rows from %s up"
          % (result["beta_s_per_byte"] * 1e9, result["fit_rows"],
             human_bytes(result["fit_first_bytes"])))
    print("implied bandwidth    %10.2f GB/s    = 1/beta, 1 GB = 1e9 bytes"
          % (result["bandwidth_bytes_per_s"] / 1e9))
    print("crossover m* = a/b   %10s      below this a message is latency-bound"
          % human_bytes(result["crossover_bytes"]))
    print("fit quality          %10.2f %%       relative RMS residual of the linear window"
          % (result["fit_relative_rms"] * 100.0))
    print("")

    if step is not None:
        print("Candidate eager-to-rendezvous step: between %s and %s, the time grew %.2fx"
              % (human_bytes(step["between_bytes"]), human_bytes(step["and_bytes"]),
                 step["time_ratio"]))
        print("  where alpha + m*beta predicts only %.2fx. That excess is the protocol"
              % step["modelled_ratio"])
        print("  switch, which is a tunable, not a constant: do not quote it as one.")
    else:
        print("No clear protocol step in this data. Either the sweep is too coarse to")
        print("resolve it, or this transport does not switch protocols in this range.")
    print("")

    if sizes:
        print("%14s  %s" % ("size", "regime"))
        print("-" * 40)
        for entry in result["classified"]:
            print("%14s  %s" % (human_bytes(entry["bytes"]), entry["regime"]))
        print("-" * 40)
        print("'balanced' spans m*/2 to 2*m*: near the crossover neither term dominates,")
        print("so both fewer messages and fewer bytes help.")
        print("")

    print("Honest limits of this fit, all inherited from how lab 2 measures:")
    print("  * half a round trip, so it assumes the two directions are symmetric")
    print("  * two ranks only, so there is no contention; this is a best case")
    print("  * on one node you measured a shared-memory transport, not a network")
    print("  * alpha and beta belong to THIS link class; label them with where they came from")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
