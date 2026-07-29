#!/usr/bin/env python3
"""Summarize rocprofv3 CSV traces into a phase/kernel breakdown with percentages.

rocprofv3's own ``--summary`` table is inclusive: a HIP call's duration contains
the HSA calls and copies it triggers, so the percentages overlap and do not sum
to 100. This script instead reports the two views you actually make decisions on:

  1. GPU device activity - kernel dispatches and memory copies, which do not
     nest, so their percentages are a real breakdown of device-busy time.
  2. Host API cost - the HIP calls that dominate wall clock, kept separate and
     labelled inclusive so the two views are never accidentally added together.

Standard library only, so it runs anywhere the profiler ran.

Usage:
    python3 tools/parse_rocprof.py [path] [--top N]

``path`` is a directory produced by ``rocprofv3 -d <dir>`` (searched
recursively) or a single ``*_trace.csv`` file. Defaults to ``profiling_output``.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

# rocprofv3 writes one CSV per domain, prefixed with the process id and nested
# under <dir>/<hostname>/. Match on the suffix so the prefix does not matter.
DOMAINS = (
    ("kernel_trace.csv", "Kernel_Name", "KERNEL_DISPATCH"),
    ("memory_copy_trace.csv", "Direction", "MEMORY_COPY"),
    ("hip_api_trace.csv", "Function", "HIP_API"),
)

NS_PER_MS = 1.0e6


class TraceError(Exception):
    """A problem the user can fix, reported as a message instead of a traceback."""


def find_csvs(path):
    """Return {suffix: [file, ...]} for every recognized trace CSV under path."""
    if not os.path.exists(path):
        raise TraceError(
            "no such file or directory: {}\n"
            "Generate traces first:  make profile-hip".format(path)
        )

    if os.path.isfile(path):
        for suffix, _, _ in DOMAINS:
            if path.endswith(suffix):
                return {suffix: [path]}
        raise TraceError(
            "{} is not a recognized rocprofv3 trace.\n"
            "Expected a file ending in one of: {}".format(
                path, ", ".join(s for s, _, _ in DOMAINS)
            )
        )

    found = defaultdict(list)
    for root, _dirs, files in os.walk(path):
        for name in files:
            for suffix, _, _ in DOMAINS:
                if name.endswith(suffix):
                    found[suffix].append(os.path.join(root, name))

    if not found:
        raise TraceError(
            "no rocprofv3 trace CSVs under {}.\n"
            "Looked for files ending in: {}\n"
            "Two things commonly cause this:\n"
            "  - the profiler ran without --output-format csv (rocprofv3 defaults\n"
            "    to the rocpd SQLite3 format as of ROCm 7.x), or\n"
            "  - the run wrote to a different -d directory.\n"
            "Generate traces with:  make profile-hip".format(
                path, ", ".join(s for s, _, _ in DOMAINS)
            )
        )
    return found


def read_rows(files, name_column):
    """Aggregate {name: (calls, total_ns)} plus the overall timestamp span."""
    totals = defaultdict(lambda: [0, 0])
    first_ts, last_ts = None, None
    skipped = 0

    for path in files:
        try:
            handle = open(path, newline="", encoding="utf-8-sig", errors="replace")
        except OSError as exc:
            raise TraceError("cannot read {}: {}".format(path, exc.strerror))

        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue  # zero-byte file: the domain produced no events
            missing = {name_column, "Start_Timestamp", "End_Timestamp"} - set(reader.fieldnames)
            if missing:
                raise TraceError(
                    "{} is missing expected column(s): {}\n"
                    "rocprofv3's CSV schema changes between releases; this parser "
                    "targets the ROCm 7.x layout.".format(path, ", ".join(sorted(missing)))
                )

            for row in reader:
                try:
                    start = int(row["Start_Timestamp"])
                    end = int(row["End_Timestamp"])
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                entry = totals[row[name_column] or "(unnamed)"]
                entry[0] += 1
                entry[1] += end - start
                first_ts = start if first_ts is None else min(first_ts, start)
                last_ts = end if last_ts is None else max(last_ts, end)

    return totals, first_ts, last_ts, skipped


def shorten(name, width):
    return name if len(name) <= width else name[: width - 3] + "..."


def print_table(title, note, totals, denominator, top):
    print()
    print(title)
    print("-" * len(title))
    if not totals:
        print("  (no events recorded in this domain)")
        return
    if note:
        print(note)

    rows = sorted(totals.items(), key=lambda kv: kv[1][1], reverse=True)
    print("  {:<46} {:>6} {:>12} {:>12} {:>8}".format(
        "NAME", "CALLS", "TOTAL (ms)", "AVG (ms)", "PERCENT"))
    for name, (calls, total_ns) in rows[:top]:
        pct = 100.0 * total_ns / denominator if denominator else 0.0
        print("  {:<46} {:>6} {:>12.4f} {:>12.4f} {:>7.2f}%".format(
            shorten(name, 46), calls, total_ns / NS_PER_MS,
            total_ns / calls / NS_PER_MS, pct))
    if len(rows) > top:
        print("  ... {} more (raise --top to see them)".format(len(rows) - top))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase/kernel breakdown from rocprofv3 CSV traces.")
    parser.add_argument("path", nargs="?", default="profiling_output",
                        help="rocprofv3 output directory or a single *_trace.csv")
    parser.add_argument("--top", type=int, default=12,
                        help="rows to show per table (default: 12)")
    args = parser.parse_args(argv)

    if args.top < 1:
        print("error: --top must be at least 1", file=sys.stderr)
        return 2

    try:
        found = find_csvs(args.path)
        collected = {}
        span_start, span_end = None, None
        skipped_total = 0

        for suffix, column, label in DOMAINS:
            files = found.get(suffix)
            if not files:
                continue
            totals, first_ts, last_ts, skipped = read_rows(files, column)
            collected[label] = totals
            skipped_total += skipped
            if first_ts is not None:
                span_start = first_ts if span_start is None else min(span_start, first_ts)
                span_end = last_ts if span_end is None else max(span_end, last_ts)
    except TraceError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    kernels = collected.get("KERNEL_DISPATCH", {})
    copies = collected.get("MEMORY_COPY", {})
    api = collected.get("HIP_API", {})

    if not any((kernels, copies, api)):
        print("error: found trace CSVs under {} but every one of them is empty.\n"
              "The application may have exited before dispatching any GPU work."
              .format(args.path), file=sys.stderr)
        return 1

    device_ns = (sum(v[1] for v in kernels.values())
                 + sum(v[1] for v in copies.values()))

    print("rocprofv3 breakdown for: {}".format(args.path))
    if span_start is not None:
        span_ms = (span_end - span_start) / NS_PER_MS
        print("traced span      : {:.3f} ms (first event start -> last event end)".format(span_ms))
        if span_ms > 0:
            print("GPU device busy  : {:.3f} ms ({:.2f}% of the traced span)".format(
                device_ns / NS_PER_MS, 100.0 * device_ns / (span_ms * NS_PER_MS)))
    if skipped_total:
        print("skipped rows     : {} (unparseable timestamps)".format(skipped_total))

    device = {}
    device.update({"[kernel] " + k: v for k, v in kernels.items()})
    device.update({"[copy]   " + k: v for k, v in copies.items()})
    print_table(
        "GPU device activity (kernels + copies; these do not overlap, so this is a true breakdown)",
        None, device, device_ns, args.top)

    api_ns = sum(v[1] for v in api.values())
    print_table(
        "Host HIP API cost (INCLUSIVE - a call contains the work it triggers)",
        "  Percentages are of total HIP API time and intentionally overlap the table above.\n"
        "  Do not add the two tables together.",
        api, api_ns, args.top)

    print()
    print("Read it in this order: is device-busy a large share of the traced span?")
    print("If not, the bottleneck is host-side (allocation, copies, launch gaps),")
    print("and no kernel optimization will move end-to-end time. See README section 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
