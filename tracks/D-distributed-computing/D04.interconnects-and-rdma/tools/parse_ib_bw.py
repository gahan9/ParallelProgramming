#!/usr/bin/env python3
"""Read a captured `ib_write_bw` log and report it in both units, without guessing.

This is lab 3's second half. It PARSES OUTPUT YOU CAPTURED on a real fabric:

    ib_write_bw -d mlx5_0 --report_gbits > ib_write_bw.log      # on the client host
    python -S -I tools/parse_ib_bw.py --input ib_write_bw.log --report gbits

It measures nothing and models nothing, so unlike tools/staging_cost.py and
tools/injection_model.py it prints no model banner. It reports what it read and names
the file it read it from, because a bandwidth number without its provenance is not a
result.

The tool exists for one reason above all others: **the units**. perftest reports
gigaBITS per second with `--report_gbits` and megaBYTES per second without it, this
curriculum works in gigaBYTES per second, and the factor of eight between them is the
most common arithmetic error in fabric discussions (README section 5.2). So every row is
printed in both units, every column is labelled, and where the log's own unit is
ambiguous the tool says so instead of picking one.

Stdlib only:
    python -S -I tools/parse_ib_bw.py --input ib_write_bw.log --report gbits
"""

from __future__ import annotations

import argparse
import json
import re
import sys

PROGRAM = "parse_ib_bw"

# perftest prints the unit inside the column header, which is the only trustworthy place
# to read it from: the flag that produced the log is not in the log.
UNIT_RE = re.compile(r"BW\s+(?:peak|average)\s*\[\s*([A-Za-z/]+)\s*\]", re.IGNORECASE)

# `Key : value` metadata from perftest's preamble. Only fields that change how a number
# should be read are kept -- a device name and a link type qualify, a PSN does not.
META_KEYS = (
    "Device", "Transport type", "Link type", "Connection type", "Mtu",
    "Number of qps", "TX depth", "Max inline data", "ibv_wr* API", "Test",
)
META_RE = re.compile(r"([A-Za-z][A-Za-z0-9 _*.()-]*?)\s*:\s*(\S.*?)\s*$")

GB = 1e9


def die(message):
    """One line on stderr and a non-zero exit. Never a traceback."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def unit_to_bytes_per_s(unit, mb_bytes):
    """Return (multiplier into bytes/s, human description) for a perftest unit string.

    Unrecognized units are an error, not an assumption. Reporting a number in the wrong
    unit is worse than reporting no number, because it is quotable.
    """
    key = unit.strip().lower().replace("sec", "s")
    if key in ("gb/s", "gbit/s", "gbits/s", "gbps"):
        return 1e9 / 8.0, "gigabits per second (--report_gbits)"
    if key in ("mb/s", "mbyte/s", "mbytes/s"):
        return float(mb_bytes), "megabytes per second, MB taken as %d bytes" % mb_bytes
    if key in ("gib/s",):
        return float(1024 ** 3), "gibibytes per second"
    if key in ("mib/s",):
        return float(1024 ** 2), "mebibytes per second"
    return None, None


def read_log(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError as exc:
        die("cannot read %s: %s" % (path, exc.strerror or exc))
    return ""


def parse_metadata(lines):
    """Pull the preamble's `Key : value` fields out, two columns per line and all.

    perftest prints two fields per line separated by a tab, so each line is split before
    matching; the value is then truncated at the first run of two or more spaces, which
    catches builds that align the second column with spaces instead.
    """
    meta = {}
    for line in lines:
        stripped = line.strip().lstrip("#").strip()
        for chunk in stripped.split("\t"):
            match = META_RE.match(chunk.strip())
            if not match:
                continue
            key = match.group(1).strip()
            value = re.split(r"\s{2,}", match.group(2).strip())[0]
            for wanted in META_KEYS:
                if key.lower() == wanted.lower():
                    meta[wanted] = value
    return meta


def declares_synthetic(text):
    """True when the file says of itself that it is not a capture.

    The samples/ directory labels its files this way. Echoing the label is cheap
    insurance against a generated example being quoted as a measurement, which is the
    one failure this curriculum treats as unrecoverable.
    """
    return "SYNTHETIC" in "\n".join(text.splitlines()[:25]).upper()


def parse_rows(lines, path):
    """Return a list of dicts with the raw numeric columns, in file order."""
    rows = []
    for lineno, line in enumerate(lines, start=1):
        tokens = line.replace(",", " ").split()
        if len(tokens) < 4:
            continue
        if not tokens[0].isdigit():
            continue
        try:
            nbytes = int(tokens[0])
            iters = int(float(tokens[1]))
            peak = float(tokens[2])
            average = float(tokens[3])
        except ValueError:
            continue
        msg_rate = None
        if len(tokens) >= 5:
            try:
                msg_rate = float(tokens[4])
            except ValueError:
                msg_rate = None
        if nbytes <= 0:
            die("%s line %d: message size must be positive, got %d"
                % (path, lineno, nbytes))
        if average < 0 or peak < 0:
            die("%s line %d: negative bandwidth (%g peak, %g average)"
                % (path, lineno, peak, average))
        rows.append({"bytes": nbytes, "iterations": iters, "peak": peak,
                     "average": average, "msg_rate_mpps": msg_rate, "line": lineno})
    if not rows:
        die("%s has no data rows. Expected perftest's table, one row per message size, "
            "starting with the byte count" % path)
    return rows


def knee_bytes(rows, fraction):
    """Smallest message size whose average reaches `fraction` of the best row.

    This is read straight out of the data, not fitted: it is the size at which the
    fabric stops being dominated by per-message cost, which is the same question D01
    asks as `m*` and which you should compare against.
    """
    best = max(row["average"] for row in rows)
    if best <= 0:
        return None
    for row in sorted(rows, key=lambda r: r["bytes"]):
        if row["average"] >= fraction * best:
            return row["bytes"]
    return None


def human_bytes(nbytes):
    if nbytes < 1024:
        return "%d B" % nbytes
    for unit, scale in (("KiB", 1024.0), ("MiB", 1024.0 ** 2), ("GiB", 1024.0 ** 3)):
        if nbytes < scale * 1024 or unit == "GiB":
            return "%.4g %s" % (nbytes / scale, unit)
    return "%d B" % nbytes


def seconds(value):
    if value >= 1e-3:
        return "%.3f ms" % (value * 1e3)
    return "%.3f us" % (value * 1e6)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Parse a captured ib_write_bw log and report every row in both "
                    "gigabits and gigabytes per second.",
        epilog="This tool reports MEASURED data, so it names its input file rather than "
               "printing a model disclaimer. It refuses to guess a unit.")
    parser.add_argument("--input", "-i", required=True,
                        help="file holding the output of ib_write_bw (or ib_read_bw, "
                             "ib_send_bw: the table layout is the same)")
    parser.add_argument("--report", choices=("gbits", "gbytes", "both"), default="both",
                        help="which unit the summary lines use; the table always shows "
                             "both (default both)")
    parser.add_argument("--knee-fraction", type=float, default=0.9,
                        help="fraction of the best row used to locate the size at which "
                             "bandwidth saturates (default 0.9)")
    parser.add_argument("--mb-bytes", type=int, default=1048576,
                        help="bytes per MB when the log's header says MB/sec; perftest "
                             "builds differ, so this is explicit (default 1048576)")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.knee_fraction <= 1.0:
        parser.error("--knee-fraction must be in (0, 1]")
    if args.mb_bytes not in (1000000, 1048576):
        parser.error("--mb-bytes must be 1000000 or 1048576")

    text = read_log(args.input)
    if not text.strip():
        die("%s is empty; capture the test output first, e.g. "
            "`ib_write_bw -d mlx5_0 --report_gbits | tee %s`" % (args.input, args.input))
    lines = text.splitlines()

    unit_match = UNIT_RE.search(text)
    if not unit_match:
        die("%s has no 'BW average[...]' column header, so the unit is unknown. This "
            "tool will not assume one: re-capture including perftest's header line, or "
            "re-run with --report_gbits" % args.input)
    unit_text = unit_match.group(1)
    multiplier, unit_description = unit_to_bytes_per_s(unit_text, args.mb_bytes)
    if multiplier is None:
        die("%s: unrecognized bandwidth unit '%s' in the column header. Recognized: "
            "Gb/sec, MB/sec, GiB/sec, MiB/sec" % (args.input, unit_text))

    meta = parse_metadata(lines)
    rows = parse_rows(lines, args.input)
    synthetic = declares_synthetic(text)

    for row in rows:
        row["peak_bytes_per_s"] = row["peak"] * multiplier
        row["average_bytes_per_s"] = row["average"] * multiplier
        row["peak_gbytes_per_s"] = row["peak_bytes_per_s"] / GB
        row["average_gbytes_per_s"] = row["average_bytes_per_s"] / GB
        row["peak_gbits_per_s"] = row["peak_bytes_per_s"] * 8.0 / GB
        row["average_gbits_per_s"] = row["average_bytes_per_s"] * 8.0 / GB
        row["seconds_per_message"] = (row["bytes"] / row["average_bytes_per_s"]
                                      if row["average_bytes_per_s"] > 0 else None)

    ordered = sorted(rows, key=lambda r: r["bytes"])
    smallest, largest = ordered[0], ordered[-1]
    best = max(rows, key=lambda r: r["average_bytes_per_s"])
    knee = knee_bytes(rows, args.knee_fraction)

    result = {
        "source_file": args.input,
        "unit_in_log": unit_text,
        "unit_interpretation": unit_description,
        "metadata": meta,
        "row_count": len(rows),
        "rows": rows,
        "best_average_gbytes_per_s": best["average_gbytes_per_s"],
        "best_average_gbits_per_s": best["average_gbits_per_s"],
        "best_at_bytes": best["bytes"],
        "saturation_bytes": knee,
        "saturation_fraction": args.knee_fraction,
        "source_declares_synthetic": synthetic,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("=" * 86)
    print("PARSED, not modelled: %s" % args.input)
    print("%d row(s), %s to %s, unit read from the header as %s"
          % (len(rows), human_bytes(smallest["bytes"]), human_bytes(largest["bytes"]),
             unit_description))
    print("Whatever produced that file is where these numbers come from; this tool only")
    print("converted the units and sorted the rows.")
    if synthetic:
        print("")
        print("!! This file DECLARES ITSELF SYNTHETIC in its own header. It is an example")
        print("!! of the format, not a measurement. Do not quote a number from it.")
    print("=" * 86)
    if meta:
        for key in META_KEYS:
            if key in meta:
                print("  %-18s %s" % (key, meta[key]))
    else:
        print("  (no perftest preamble in this capture, so device, transport, MTU and")
        print("   queue depth are unknown -- capture the whole output, not just the table)")
    print("")

    header = ("%12s %10s %12s %12s %12s %12s %12s"
              % ("bytes", "iters", "peak Gb/s", "avg Gb/s", "peak GB/s", "avg GB/s",
                 "s/message"))
    print(header)
    print("-" * len(header))
    for row in ordered:
        print("%12d %10d %12.3f %12.3f %12.4f %12.4f %12s"
              % (row["bytes"], row["iterations"], row["peak_gbits_per_s"],
                 row["average_gbits_per_s"], row["peak_gbytes_per_s"],
                 row["average_gbytes_per_s"],
                 "n/a" if row["seconds_per_message"] is None
                 else seconds(row["seconds_per_message"])))
    print("-" * len(header))
    print("")

    if args.report in ("gbits", "both"):
        print("best average: %.3f Gb/s at %s"
              % (best["average_gbits_per_s"], human_bytes(best["bytes"])))
    if args.report in ("gbytes", "both"):
        print("best average: %.4f GB/s at %s   (the same number; the factor is 8)"
              % (best["average_gbytes_per_s"], human_bytes(best["bytes"])))
    print("")

    if knee is not None and len(ordered) >= 3:
        print("bandwidth reaches %.0f%% of its best by %s."
              % (args.knee_fraction * 100.0, human_bytes(knee)))
        print("Below that size the per-message cost dominates -- that end of the sweep is")
        print("your alpha; the flat top is your beta. That is D01's crossover m*,")
        print("measured on your own fabric instead of assumed.")
    else:
        print("Too few rows to say where bandwidth saturates. Run the size sweep")
        print("(`ib_write_bw -a`) rather than a single size: one bandwidth number")
        print("without its message size is uninterpretable.")
    print("")

    print("Before quoting any row above, answer README section 5.2's seven questions.")
    print("Three of them are already visible here and three are not:")
    print("  * unit: read from this log's header, printed both ways above;")
    print("  * message size: every row carries its own, which is why there is no")
    print("    single 'the bandwidth' in this file;")
    print("  * warm-up: perftest discards its own, but a first-run registration cost")
    print("    would show as an outlier row -- look for one;")
    print("  * NOT visible here: whether the buffers were host or device memory,")
    print("    whether anything else was using the fabric, and how many operations were")
    print("    outstanding. Record those next to the file or the file is not")
    print("    reproducible.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
