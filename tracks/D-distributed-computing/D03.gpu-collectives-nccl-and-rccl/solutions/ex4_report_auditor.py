#!/usr/bin/env python3
"""Drill 4 solution - the benchmark-report auditor (D03 sections 2.4 and 5.1).

Three decisions carry this drill, and none of them is the arithmetic.

DECISION 1 - the audit is against the RAW TIME, not against the other bandwidth
column. `time` is the only measurement in the report; algbw and busbw are both
derived from it. Checking busbw against algbw only proves the tool can multiply.
Recomputing both from size and time is what catches a report generated with the
wrong rank count, hand-edited, or pasted together from two runs.

DECISION 2 - rounding is not disagreement. Both suites print bandwidths with two
decimals, so a reported 0.00 GB/s means "below 0.005" and a recomputed 0.004 is in
perfect agreement with it while being 100% away in relative terms. An auditor that
ignores this flags every small-message row in every healthy report, gets a
reputation for crying wolf, and is then ignored on the row that mattered. So a
finding must survive BOTH a relative tolerance and the printed precision. Same
argument for the busbw/algbw ratio: the ratio of two two-decimal numbers is only
as sharp as the rounding, so the ratio check is bounded by the quanta before it is
allowed to accuse anyone.

DECISION 3 - an inference must announce itself. Given one unlabelled bandwidth
column, comparing it against both candidates and keeping the closer one is right,
but the answer is only trustworthy when the two candidates are far apart. f(N) is
1.75 for an 8-rank all-reduce, so there the inference is solid; it is (N-1)/N for
all-gather, which is 0.98 at N=64, so there the two candidates differ by 2% and the
inference is worthless. The tool must say which situation it is in. A silent guess
in a tool built to catch a misread column would be the joke writing itself.

N == 1 returns a factor of 0, for the reason argued in ex1_busbw_convert.py: no
byte crosses a wire at one rank, so there is no bus bandwidth to report.

  python -S -I ex4_report_auditor.py
  python -S -I ex4_report_auditor.py --input allreduce.log --collective allreduce \
      --ranks 4

Standard library only.
"""

from __future__ import annotations

import argparse
import math
import sys

PROGRAM = "ex4_report_auditor"

FACTORS = {
    "allreduce": lambda n: 2.0 * (n - 1.0) / n,
    "allgather": lambda n: (n - 1.0) / n,
    "reducescatter": lambda n: (n - 1.0) / n,
    "alltoall": lambda n: (n - 1.0) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}

# Relative disagreement tolerated before a finding is raised. 2% is generous
# enough to absorb the suites' own rounding at large sizes and tight enough to
# catch a factor applied at the wrong rank count, which is never a 2% error.
TOLERANCE = 0.02


def traffic_factor(collective, ranks):
    """f(N) for one collective (README 2.4). Raises ValueError on a bad name."""
    key = str(collective).strip().lower().replace("_", "").replace("-", "")
    if key not in FACTORS:
        raise ValueError("unknown collective %r; expected one of %s"
                         % (collective, ", ".join(sorted(FACTORS))))
    if not isinstance(ranks, int) or ranks < 1:
        raise ValueError("ranks must be an int of at least 1, got %r" % (ranks,))
    if ranks == 1:
        return 0.0
    return FACTORS[key](float(ranks))


def _disagrees(reported, recomputed, quantum):
    """True only when the gap survives the tolerance AND the printed precision."""
    scale = max(abs(reported), abs(recomputed), 1e-12)
    relative = abs(reported - recomputed) / scale
    return relative > TOLERANCE and abs(reported - recomputed) > quantum


def audit_row(row, collective, ranks):
    """One-line findings for this row; an empty list means the row is clean."""
    factor = traffic_factor(collective, ranks)
    seconds = row.get("seconds", 0.0)
    findings = []

    if seconds <= 0 or not math.isfinite(seconds):
        return ["time is %r: a measurement cannot take zero or negative time"
                % (row.get("time"),)]

    algbw = row["bytes"] / seconds / 1e9
    busbw = algbw * factor

    wrong = row.get("wrong")
    if wrong is not None and wrong > 0:
        findings.append("#wrong = %g: a correctness defect, not a result" % wrong)

    for name, recomputed in (("algbw", algbw), ("busbw", busbw)):
        if name not in row:
            continue
        if _disagrees(row[name], recomputed, row.get(name + "_quantum", 0.0)):
            findings.append(
                "%s reported %.3f GB/s but size/time gives %.3f GB/s"
                % (name, row[name], recomputed))

    # The ratio check is a different question from the two above: it asks whether
    # the report was generated with the rank count you are auditing it against.
    if "algbw" in row and "busbw" in row and factor > 0 and row["algbw"] > 0:
        q_alg = row.get("algbw_quantum", 0.0)
        q_bus = row.get("busbw_quantum", 0.0)
        hi_den = row["algbw"] - q_alg
        ratio_lo = (row["busbw"] - q_bus) / (row["algbw"] + q_alg)
        ratio_hi = ((row["busbw"] + q_bus) / hi_den) if hi_den > 0 else float("inf")
        if not (ratio_lo * (1.0 - TOLERANCE) <= factor <= ratio_hi * (1.0 + TOLERANCE)):
            findings.append(
                "busbw/algbw = %.4f (%.3f..%.3f allowing for rounding) but f(N) "
                "for %s at N=%d is %.4f: the rank count or the collective is not "
                "what was assumed"
                % (row["busbw"] / row["algbw"], ratio_lo, ratio_hi, collective,
                   ranks, factor))
    return findings


def infer_bandwidth_column(row, collective, ranks):
    """'algbw' or 'busbw' for a report with one unlabelled bandwidth column.

    Reliable only when f(N) is far from 1. At f(N) = 0.98 - an all-gather over 64
    ranks - the two candidates differ by 2%, which is inside the suites' own
    rounding, and any answer here is a coin toss dressed as analysis. The caller
    gets that warning from inference_is_reliable() below rather than being left to
    discover it.
    """
    if "bw" not in row:
        return None
    factor = traffic_factor(collective, ranks)
    seconds = row.get("seconds", 0.0)
    if seconds <= 0:
        return None
    algbw = row["bytes"] / seconds / 1e9
    busbw = algbw * factor
    return "algbw" if abs(row["bw"] - algbw) <= abs(row["bw"] - busbw) else "busbw"


def inference_is_reliable(collective, ranks, margin=0.10):
    """Can the two candidates be told apart at all at this rank count?"""
    factor = traffic_factor(collective, ranks)
    return abs(factor - 1.0) >= margin


# ---------------------------------------------------------------------------
# Parser and harness, identical to ../exercises/ex4_report_auditor.py.
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
        got = traffic_factor(collective, ranks)
        record("f(N) %s N=%d" % (collective, ranks),
               abs(float(got) - expected) < 1e-9, "got %r" % (got,))

    try:
        traffic_factor("nosuchcollective", 8)
        record("f(N) refuses an unknown collective", False, "no ValueError")
    except ValueError:
        record("f(N) refuses an unknown collective", True)

    clean = parse_report(synthetic_report(4, 1.5))
    findings = [audit_row(r, "allreduce", 4) for r in clean]
    record("a consistent report raises nothing",
           all(isinstance(f, list) and not f for f in findings),
           "got %r" % (findings[:2],))

    dirty = parse_report(synthetic_report(4, 1.5, bad_row=4))
    per_row = [audit_row(r, "allreduce", 4) for r in dirty]
    flagged = [i for i, f in enumerate(per_row) if f]
    record("the corrupted row is the only one flagged",
           flagged == [8] and len(per_row[8]) >= 2, "flagged %r" % (flagged,))

    for actual in ("busbw", "algbw"):
        rows = parse_report(synthetic_report(4, 1.5, one_column=actual))
        votes = [infer_bandwidth_column(r, "allreduce", 4) for r in rows]
        record("one unlabelled column inferred as %s" % actual,
               bool(votes) and all(v == actual for v in votes), "got %r" % (votes,))

    record("inference declared unreliable at f(N) ~ 1",
           inference_is_reliable("allreduce", 8) and
           not inference_is_reliable("allgather", 64))

    print("")
    if failures:
        print("FAIL: %d checks" % len(failures))
        return 1
    print("PASS: every check")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Drill 4 solution: audit an rccl-tests / nccl-tests report.",
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
    try:
        factor = traffic_factor(args.collective, args.ranks)
    except ValueError as exc:
        die(str(exc))
    print("parsed %d rows from %s; %s at N=%d has f(N) = %.4f"
          % (len(rows), args.input, args.collective, args.ranks, factor))

    unlabelled = [r for r in rows if "bw" in r and "algbw" not in r]
    if unlabelled:
        votes = {}
        for row in unlabelled:
            choice = infer_bandwidth_column(row, args.collective, args.ranks)
            votes[choice] = votes.get(choice, 0) + 1
        choice = max(votes, key=votes.get)
        print("one unlabelled bandwidth column: INFERRING %s (%d of %d rows)"
              % (choice, votes[choice], len(unlabelled)))
        if not inference_is_reliable(args.collective, args.ranks):
            print("  ...but f(N) = %.4f is within 10%% of 1 here, so the two "
                  "candidates are" % factor)
            print("  closer than the printed precision. Treat this as unknown, "
                  "not as inferred.")

    total = 0
    for row in rows:
        for finding in audit_row(row, args.collective, args.ranks):
            total += 1
            print("  line %d (%.0f bytes, %s): %s"
                  % (row["line"], row["bytes"], row["placement"], finding))
    print("%d finding(s)" % total)
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
