#!/usr/bin/env python3
"""Drill 4 - the benchmark-report auditor (D03 sections 2.4 and 5.1).

Given a report from rccl-tests or nccl-tests, recompute what the tool should have
printed and flag every row where the report disagrees with its own raw time.

This file RUNS as given and FAILS its own self-test, because the three audit
decisions are marked TODOs. A FAIL on the first run means the harness works.

  python -S -I ex4_report_auditor.py                       # self-test, no GPU
  python -S -I ex4_report_auditor.py --input allreduce.log --collective allreduce \
      --ranks 4                                            # audit a real report

  TODO 1  traffic_factor(): f(N) for each collective, from README section 2.4.
          Decide what N == 1 means and say why in a comment.
  TODO 2  audit_row(): recompute algbw = size/time and busbw = algbw*f(N), and
          return a list of one-line findings for
            - a reported algbw or busbw that disagrees with the raw time,
            - a nonzero #wrong, which is a defect and not a result.
          The trap: both suites print bandwidths to two decimals, so a reported
          0.00 differs from a recomputed 0.004 by 100% and is not wrong. Use the
          printed precision, which the parser records as <name>_quantum.
  TODO 3  infer_bandwidth_column(): handle a report with ONE bandwidth column and
          no statement of which it is. Return 'algbw' or 'busbw' - and note in a
          comment when the inference is not trustworthy.

Standard library only. The self-test needs no report file: it generates SYNTHETIC
fixtures from an alpha-beta model, because a committed performance file would be a
number this repository cannot reproduce.
"""

from __future__ import annotations

import argparse
import math
import sys

PROGRAM = "ex4_report_auditor"

COLLECTIVES = ("allreduce", "allgather", "reducescatter", "alltoall",
               "broadcast", "reduce")


def traffic_factor(collective, ranks):
    """f(N) for one collective. TODO 1.

    Return a float. Raise ValueError with one line for an unknown collective.
    """
    return None


def audit_row(row, collective, ranks):
    """Return a list of one-line findings for this row. Empty list means clean.

    TODO 2. `row` has 'bytes', 'seconds', and whichever of 'algbw', 'busbw', 'bw'
    and 'wrong' the report carried, plus '<name>_quantum' giving half of the last
    printed digit for each numeric column.
    """
    return None


def infer_bandwidth_column(row, collective, ranks):
    """Return 'algbw' or 'busbw' for a report with one unlabelled column. TODO 3."""
    return None


# ---------------------------------------------------------------------------
# Parser and harness below this line. Plumbing, not the drill: you should not
# need to change anything past here.
# ---------------------------------------------------------------------------

ROLE = {
    "size": "size", "count": "count", "type": "type", "redop": "redop",
    "root": "root", "time": "time", "algbw": "algbw", "busbw": "busbw",
    "bw": "bw", "bandwidth": "bw", "#wrong": "wrong", "wrong": "wrong",
}


def die(message):
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def parse_report(text, time_scale=1e-6):
    """Rows from a report, driven by the '#' column header the suites print."""
    roles = None
    for line in text.splitlines():
        if not line.lstrip().startswith("#"):
            continue
        tokens = [t.lower() for t in line.lstrip().lstrip("#").split()]
        if "size" in tokens and "time" in tokens:
            roles, seen = [], {}
            for token in tokens:
                role = ROLE.get(token, "ignored")
                occurrence = seen.get(role, 0)
                seen[role] = occurrence + 1
                roles.append((role, occurrence))
            break
    if roles is None:
        die("no column header found; keep the '#' lines the benchmark prints")

    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) != len(roles):
            continue
        shared, groups = {}, {}
        for (role, occurrence), token in zip(roles, tokens):
            if role in ("ignored", "count", "type", "redop", "root"):
                continue
            if role == "size":
                shared["size"] = token
                continue
            groups.setdefault(occurrence, {})[role] = token
        try:
            nbytes = float(shared["size"])
        except (KeyError, ValueError):
            continue
        for occurrence in sorted(groups):
            row = {"line": lineno, "bytes": nbytes,
                   "placement": "out-of-place" if occurrence == 0 else "in-place"}
            ok = True
            for key, token in groups[occurrence].items():
                try:
                    row[key] = float(token)
                except ValueError:
                    ok = False
                    continue
                decimals = len(token.split(".")[1]) if "." in token else 0
                row[key + "_quantum"] = 0.5 * (10.0 ** -decimals)
            if ok and "time" in row:
                row["seconds"] = row["time"] * time_scale
                rows.append(row)
    if not rows:
        die("no parseable data rows")
    return rows


def synthetic_report(ranks, factor, alpha_us=8.0, bandwidth_gbs=50.0, steps=None,
                     bad_row=None, one_column=None):
    """A SYNTHETIC report built from alpha-beta arithmetic, never a measurement."""
    steps = steps if steps is not None else 2 * (ranks - 1)
    alpha, beta = alpha_us * 1e-6, 1.0 / (bandwidth_gbs * 1e9)
    sizes = [1024 * (4 ** i) for i in range(6)]
    lines = ["# SYNTHETIC EXAMPLE: alpha-beta arithmetic, not a measurement"]
    if one_column:
        lines.append("#   size   count   type  redop  root   time     bw #wrong")
    else:
        lines.append("#   size   count   type  redop  root   time  algbw  busbw "
                     "#wrong   time  algbw  busbw #wrong")
    for index, nbytes in enumerate(sizes):
        seconds = steps * alpha + factor * nbytes * beta
        algbw = nbytes / seconds / 1e9
        busbw = algbw * factor
        wrong = 0
        busbw_oop = busbw
        # Only the out-of-place group is corrupted, so the self-test can assert
        # that exactly one parsed row is flagged.
        if bad_row is not None and index == bad_row:
            busbw_oop = busbw * 1.4
            wrong = 7
        if one_column:
            shown = busbw if one_column == "busbw" else algbw
            lines.append("%8d %7d  float    sum    -1 %6.2f %6.2f %6d"
                         % (nbytes, nbytes // 4, seconds * 1e6, shown, wrong))
        else:
            lines.append("%8d %7d  float    sum    -1 %6.2f %6.2f %6.2f %6d "
                         "%6.2f %6.2f %6.2f %6d"
                         % (nbytes, nbytes // 4, seconds * 1e6, algbw, busbw_oop,
                            wrong, seconds * 1e6, algbw, busbw, 0))
    return "\n".join(lines) + "\n"


def self_test():
    failures = []

    def record(label, ok, detail=""):
        print("  %-38s %s" % (label, "ok" if ok else "FAIL" + (": " + detail if detail else "")))
        if not ok:
            failures.append(label)

    for collective, ranks, expected in (("allreduce", 8, 1.75),
                                        ("allreduce", 4, 1.5),
                                        ("allgather", 8, 0.875),
                                        ("reducescatter", 16, 0.9375),
                                        ("broadcast", 64, 1.0),
                                        ("reduce", 64, 1.0)):
        try:
            got = traffic_factor(collective, ranks)
        except ValueError as exc:
            record("f(N) %s N=%d" % (collective, ranks), False,
                   "raised ValueError(%s)" % exc)
            continue
        ok = got is not None and abs(float(got) - expected) < 1e-9
        record("f(N) %s N=%d" % (collective, ranks), ok,
               "got %r, expected %g" % (got, expected))

    try:
        traffic_factor("nosuchcollective", 8)
        record("f(N) refuses an unknown collective", False, "no ValueError")
    except ValueError:
        record("f(N) refuses an unknown collective", True)
    except Exception as exc:  # noqa: BLE001
        record("f(N) refuses an unknown collective", False,
               "raised %s, not ValueError" % type(exc).__name__)

    clean = parse_report(synthetic_report(4, 1.5))
    findings = [audit_row(r, "allreduce", 4) for r in clean]
    ok = all(isinstance(f, list) and not f for f in findings)
    record("a consistent report raises nothing", ok,
           "got %r" % (findings[:2],))

    dirty = parse_report(synthetic_report(4, 1.5, bad_row=4))
    per_row = [audit_row(r, "allreduce", 4) for r in dirty]
    flagged = [i for i, f in enumerate(per_row) if isinstance(f, list) and f]
    # Row index 4 out-of-place is rows[8] once in-place rows are counted.
    ok = flagged == [8] and len(per_row[8]) >= 2
    record("the corrupted row is the only one flagged", ok,
           "flagged rows %r" % (flagged,))

    for actual in ("busbw", "algbw"):
        rows = parse_report(synthetic_report(4, 1.5, one_column=actual))
        votes = [infer_bandwidth_column(r, "allreduce", 4) for r in rows]
        ok = votes and all(v == actual for v in votes)
        record("one unlabelled column inferred as %s" % actual, ok,
               "got %r" % (votes,))

    print("")
    if failures:
        print("FAIL: %d checks (expected until the TODOs are done)" % len(failures))
        return 1
    print("PASS: every check")
    print("Now run it against a real report: --input <log> --collective allreduce "
          "--ranks <N>")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Drill 4: audit an rccl-tests / nccl-tests report.",
        epilog="With no --input it runs the self-test against synthetic fixtures.",
    )
    ap.add_argument("--input", "-i", default="")
    ap.add_argument("--collective", "-c", default="allreduce")
    ap.add_argument("--ranks", "-N", type=int, default=0)
    ap.add_argument("--time-unit", default="us", choices=("us", "ms", "s"))
    args = ap.parse_args(argv)

    if not args.input:
        print("self-test for %s" % PROGRAM)
        return self_test()
    if args.ranks < 1:
        ap.error("--ranks is required with --input: f(N) moves with N")

    try:
        with open(args.input, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        die("cannot read %s: %s" % (args.input, exc.strerror or exc))

    scale = {"us": 1e-6, "ms": 1e-3, "s": 1.0}[args.time_unit]
    rows = parse_report(text, time_scale=scale)
    print("parsed %d rows from %s" % (len(rows), args.input))
    total = 0
    for row in rows:
        findings = audit_row(row, args.collective, args.ranks)
        if findings is None:
            die("audit_row() is still the unfinished stub (TODO 2)")
        for finding in findings:
            total += 1
            print("  line %d (%.0f bytes, %s): %s"
                  % (row["line"], row["bytes"], row["placement"], finding))
    print("%d finding(s)" % total)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
