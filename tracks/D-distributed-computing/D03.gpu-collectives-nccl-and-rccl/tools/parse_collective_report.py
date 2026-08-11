#!/usr/bin/env python3
"""Audit a MEASURED rccl-tests / nccl-tests report (D03 sections 2.4 and 5.1).

This tool consumes a report that a benchmark wrote on a real machine and checks
it. It models nothing, so it prints no model disclaimer: it names the file it
read, says how it read it, and reports what the data does or does not support.

What it checks, per row:

  * recomputes algbw = size / time and busbw = algbw * f(N) from the raw time
    column, and flags any row where the report's own bandwidth columns disagree
    beyond a rounding tolerance;
  * flags any nonzero #wrong, which is a correctness defect and not a result;
  * confirms busbw / algbw equals the collective's f(N) for the rank count.

And the case that motivates the tool: a report with only ONE bandwidth column
that does not say which one it is. The tool computes both candidates and says
which it is inferring and why, rather than picking silently.

There is deliberately no sample report committed to this repository: a checked-in
performance file would be a number the repo cannot reproduce. To see the tool
work without a GPU, generate a clearly-labelled synthetic report from an
alpha-beta model and pipe it back in:

  python -S -I tools/parse_collective_report.py --emit-example > example.log
  python -S -I tools/parse_collective_report.py --input example.log \\
      --collective allreduce --ranks 4

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

PROGRAM = "parse_collective_report"

SYNTHETIC_MARKER = "SYNTHETIC EXAMPLE"

FACTORS = {
    "allreduce": lambda n: 2.0 * (n - 1) / n,
    "reducescatter": lambda n: (n - 1) / n,
    "allgather": lambda n: (n - 1) / n,
    "alltoall": lambda n: (n - 1) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}

TIME_SCALE = {"us": 1e-6, "ms": 1e-3, "s": 1.0}

# Column names the benchmarks use, mapped to the role this tool needs.
ROLE = {
    "size": "size",
    "count": "count",
    "type": "type",
    "redop": "redop",
    "root": "root",
    "time": "time",
    "algbw": "algbw",
    "busbw": "busbw",
    "bw": "bw",
    "bandwidth": "bw",
    "#wrong": "wrong",
    "wrong": "wrong",
    "error": "wrong",
    "#error": "wrong",
}


def die(message: str) -> None:
    """One line on stderr, non-zero exit, never a traceback."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


def normalize_collective(name: str) -> str:
    key = name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key not in FACTORS:
        die("unknown collective %r; expected one of %s"
            % (name, ", ".join(sorted(FACTORS))))
    return key


def human_bytes(nbytes: float) -> str:
    if nbytes < 1024:
        return "%.0f B" % nbytes
    for unit, scale in (("KiB", 1024.0), ("MiB", 1024.0**2), ("GiB", 1024.0**3)):
        if nbytes < scale * 1024 or unit == "GiB":
            return "%.3g %s" % (nbytes / scale, unit)
    return "%.0f B" % nbytes


def read_text(path: str) -> str:
    """Read a report as text, tolerating the encodings shells actually produce.

    PowerShell's `>` writes UTF-16LE, so a log captured on Windows is not UTF-8.
    Decoding it as UTF-8 yields a file full of replacement characters and a
    confusing "no data rows" complaint about a file that clearly has rows, so the
    BOM is checked instead of assumed away.
    """
    if path == "-":
        return sys.stdin.read()
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        die("cannot read %s: %s" % (path, exc.strerror or exc))
        return ""  # unreachable; die() raises
    for bom, encoding in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"),
                          (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                die("%s starts with a %s byte-order mark but does not decode as "
                    "%s" % (path, encoding, encoding))
    if b"\x00" in raw[:4096]:
        die("%s looks like UTF-16 without a byte-order mark; re-save it as UTF-8 "
            "(PowerShell: use Out-File -Encoding utf8, not '>')" % path)
    return raw.decode("utf-8", errors="replace")


def find_header(lines):
    """Return (column roles, header line) from the benchmark's own header.

    Both suites print a '#'-prefixed header naming every column, which is the
    authoritative schema and cheaper to trust than a heuristic. The units line
    underneath it has fewer tokens and is skipped by requiring both 'size' and
    'time' on the line.
    """
    for line in lines:
        if not line.lstrip().startswith("#"):
            continue
        tokens = line.lstrip().lstrip("#").split()
        lowered = [t.lower() for t in tokens]
        if "size" in lowered and "time" in lowered:
            roles, seen = [], {}
            for token in lowered:
                role = ROLE.get(token)
                if role is None:
                    roles.append(("ignored:" + token, 0))
                    continue
                occurrence = seen.get(role, 0)
                seen[role] = occurrence + 1
                roles.append((role, occurrence))
            return roles, line.rstrip()
    return None, None


def fallback_roles(nfields: int):
    """Schema guess for a report with no usable header, by field count.

    Stated in the output rather than applied silently, because a guess about
    which column is which is exactly the mistake this tool exists to catch.
    """
    layouts = {
        13: ["size", "count", "type", "redop", "root",
             "time", "algbw", "busbw", "wrong", "time", "algbw", "busbw", "wrong"],
        12: ["size", "count", "type", "redop",
             "time", "algbw", "busbw", "wrong", "time", "algbw", "busbw", "wrong"],
        9: ["size", "count", "type", "redop", "root", "time", "algbw", "busbw",
            "wrong"],
        8: ["size", "count", "type", "redop", "time", "algbw", "busbw", "wrong"],
    }
    names = layouts.get(nfields)
    if names is None:
        return None
    roles, seen = [], {}
    for name in names:
        occurrence = seen.get(name, 0)
        seen[name] = occurrence + 1
        roles.append((name, occurrence))
    return roles


def parse_rows(text: str, path: str):
    """Return (rows, meta). Each row is one measurement group from one size."""
    lines = text.splitlines()
    roles, header = find_header(lines)
    meta = {
        "source": path,
        "header": header,
        "schema": "from the report's own header" if roles else None,
        "synthetic": SYNTHETIC_MARKER in text,
        "header_ranks": None,
        "skipped_lines": 0,
    }

    # nccl-tests and rccl-tests both open with a settings line naming the rank
    # geometry. Total ranks is nThread * nGpus * processes, so a report from a
    # multi-process launch cannot be checked from this line alone.
    for line in lines:
        low = line.lower()
        if "ngpus" in low and low.lstrip().startswith("#"):
            tokens = line.replace("#", " ").split()
            geometry = {}
            for i, token in enumerate(tokens[:-1]):
                if token.lower() in ("nthread", "ngpus", "nproc", "nprocs"):
                    try:
                        geometry[token.lower()] = int(tokens[i + 1])
                    except ValueError:
                        pass
            if "ngpus" in geometry:
                ranks = geometry["ngpus"] * geometry.get("nthread", 1)
                ranks *= geometry.get("nprocs", geometry.get("nproc", 1))
                meta["header_ranks"] = ranks
                meta["geometry"] = geometry
            break

    data_lines = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        try:
            int(tokens[0])
        except (ValueError, IndexError):
            meta["skipped_lines"] += 1
            continue
        data_lines.append((lineno, tokens))

    if not data_lines:
        die("%s contains no data rows; the first token of a data row is the size "
            "in bytes. Is this the output of all_reduce_perf?" % path)

    if roles is None:
        roles = fallback_roles(len(data_lines[0][1]))
        if roles is None:
            die("%s has no column header and %d fields per row, which matches no "
                "known layout; rerun the benchmark keeping its '#' header lines"
                % (path, len(data_lines[0][1])))
        meta["schema"] = "INFERRED from a %d-field row (no header in the file)" % len(
            data_lines[0][1])

    rows = []
    for lineno, tokens in data_lines:
        if len(tokens) != len(roles):
            meta["skipped_lines"] += 1
            continue
        shared, groups = {}, {}
        for (role, occurrence), token in zip(roles, tokens):
            if role.startswith("ignored:"):
                continue
            if role in ("size", "count", "type", "redop", "root"):
                shared[role] = token
                continue
            groups.setdefault(occurrence, {})[role] = token
        try:
            size = float(shared["size"])
        except (KeyError, ValueError):
            meta["skipped_lines"] += 1
            continue
        for occurrence in sorted(groups):
            entry = {
                "line": lineno,
                "bytes": size,
                "count": shared.get("count"),
                "type": shared.get("type"),
                "redop": shared.get("redop"),
                "placement": "out-of-place" if occurrence == 0 else "in-place",
            }
            bad = False
            for key, token in groups[occurrence].items():
                try:
                    entry[key] = float(token)
                except ValueError:
                    bad = True
                    continue
                # Both suites print bandwidths with two decimals, so a reported
                # 0.00 GB/s means "somewhere below 0.005". Recording the printed
                # precision is what stops the audit flagging rounding as a defect
                # on every small-message row.
                decimals = len(token.split(".")[1]) if "." in token else 0
                entry[key + "_quantum"] = 0.5 * (10.0 ** -decimals)
            if bad or "time" not in entry:
                meta["skipped_lines"] += 1
                continue
            rows.append(entry)

    if not rows:
        die("%s: found data lines but none had a parseable time column" % path)
    return rows, meta


def traffic_factor(collective: str, ranks: int) -> float:
    """f(N), with the same N == 1 rule as tools/busbw.py: nothing on the wire."""
    if ranks <= 1:
        return 0.0
    return FACTORS[collective](float(ranks))


def audit(rows, meta, collective, ranks, time_unit, tolerance):
    """Recompute every derived column and label the disagreements."""
    factor = traffic_factor(collective, ranks)
    scale = TIME_SCALE[time_unit]
    findings = []
    for row in rows:
        seconds = row["time"] * scale
        row["seconds"] = seconds
        row["algbw_recomputed"] = (row["bytes"] / seconds / 1e9) if seconds > 0 else float("inf")
        row["busbw_recomputed"] = row["algbw_recomputed"] * factor
        row["flags"] = []

        wrong = row.get("wrong")
        if wrong is not None and wrong > 0:
            row["flags"].append("#wrong=%g: a correctness defect, not a result" % wrong)

        if "bw" in row and "algbw" not in row and "busbw" not in row:
            # The motivating case: one unlabelled bandwidth column.
            d_alg = abs(row["bw"] - row["algbw_recomputed"])
            d_bus = abs(row["bw"] - row["busbw_recomputed"])
            row["inferred_column"] = "algbw" if d_alg <= d_bus else "busbw"
            row["inference_error"] = min(d_alg, d_bus)
            continue

        for name in ("algbw", "busbw"):
            if name not in row:
                continue
            recomputed = row[name + "_recomputed"]
            reported = row[name]
            if recomputed <= 0 and reported <= 0:
                continue
            scale_ref = max(abs(recomputed), abs(reported), 1e-12)
            rel = abs(reported - recomputed) / scale_ref
            row[name + "_relative_error"] = rel
            # A disagreement counts only if it survives BOTH the relative
            # tolerance and the precision the column was printed with.
            quantum = row.get(name + "_quantum", 0.0)
            if rel > tolerance and abs(reported - recomputed) > quantum:
                row["flags"].append(
                    "%s reported %.3f but size/time gives %.3f (%.1f%% off)"
                    % (name, reported, recomputed, rel * 100.0))

        if "algbw" in row and "busbw" in row and row["algbw"] > 0 and factor > 0:
            row["reported_ratio"] = row["busbw"] / row["algbw"]
            # The ratio of two rounded numbers is only as sharp as the rounding.
            # Bound it by the printed precision before calling it a mismatch,
            # otherwise every small-message row where algbw prints as 0.01 looks
            # like a rank-count error.
            q_alg = row.get("algbw_quantum", 0.0)
            q_bus = row.get("busbw_quantum", 0.0)
            lo_den, hi_den = row["algbw"] + q_alg, row["algbw"] - q_alg
            ratio_lo = (row["busbw"] - q_bus) / lo_den if lo_den > 0 else 0.0
            ratio_hi = ((row["busbw"] + q_bus) / hi_den if hi_den > 0
                        else float("inf"))
            row["ratio_bounds"] = [ratio_lo, ratio_hi]
            if not (ratio_lo * (1.0 - tolerance) <= factor <= ratio_hi * (1.0 + tolerance)):
                row["flags"].append(
                    "busbw/algbw = %.4f (%.3f..%.3f allowing for rounding) but "
                    "f(N) for %s at N=%d is %.4f: the rank count or the "
                    "collective is not what was assumed"
                    % (row["reported_ratio"], ratio_lo, ratio_hi, collective,
                       ranks, factor))
        if row["flags"]:
            findings.append(row)
    return factor, findings


def emit_example(ranks, collective, alpha_us, bandwidth_gbs, sizes, inject_error,
                 one_column=None):
    """Print a SYNTHETIC report so the parser can be demonstrated GPU-free.

    Every time here comes from the alpha-beta model printed in the header, not
    from a measurement, and the file says so on its first line. It exists so the
    parser has something to chew on; quoting a number out of it would be quoting
    arithmetic as hardware.

    `one_column` reproduces the layout that motivates the whole tool: a report
    that prints a single bandwidth column and does not say which of the two it
    is. The generator knows the answer, so it is written into the header comment
    where a person can read it - but not into the column label, which is what the
    parser sees. That is the only way to test an inference: hold the truth
    somewhere the code under test cannot reach.
    """
    factor = FACTORS[collective](float(ranks)) if ranks > 1 else 1.0  # fixture only
    alpha = alpha_us * 1e-6
    beta = 1.0 / (bandwidth_gbs * 1e9)
    steps = 2 * (ranks - 1) if collective == "allreduce" else max(1, ranks - 1)

    out = []
    out.append("# %s -- generated by %s, NOT a measurement." % (SYNTHETIC_MARKER, PROGRAM))
    out.append("# Every time below is T = steps*alpha + f(N)*m*beta with steps = %d,"
               % steps)
    out.append("# alpha = %g us, bandwidth = %g GB/s, f(N) = %.4f at N = %d."
               % (alpha_us, bandwidth_gbs, factor, ranks))
    out.append("# No hardware was involved and no number here is a measurement.")
    out.append("# nThread 1 nGpus %d minBytes %d maxBytes %d step: 2(factor) "
               "warmup iters: 5 iters: 20 validation: 1" % (ranks, sizes[0], sizes[-1]))
    out.append("#")
    if one_column:
        out.append("# This variant prints ONE bandwidth column labelled only 'bw', as "
                   "some")
        out.append("# in-house harnesses do. For the record, and deliberately not in "
                   "the column")
        out.append("# label the parser reads, it holds %s." % one_column)
        out.append("#       size         count      type   redop    root     time   "
                   "   bw #wrong")
        out.append("#        (B)    (elements)                                     "
                   "(us)  (GB/s)")
    else:
        out.append("#                                                              "
                   "out-of-place                       in-place")
        out.append("#       size         count      type   redop    root     time   "
                   "algbw   busbw #wrong     time   algbw   busbw #wrong")
        out.append("#        (B)    (elements)                                     "
                   "(us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)")
    for index, nbytes in enumerate(sizes):
        seconds = steps * alpha + factor * nbytes * beta
        algbw = nbytes / seconds / 1e9
        busbw = algbw * factor
        # The in-place column is given the same arithmetic: this file is a parser
        # fixture, and inventing a plausible difference between the two would be
        # inventing a measurement.
        bad = inject_error and index == len(sizes) // 2
        shown_busbw = busbw * 1.35 if bad else busbw
        wrong = 3 if bad else 0
        if one_column:
            shown = busbw if one_column == "busbw" else algbw
            out.append("%12d %13d     float     sum      -1 %8.2f %7.2f %6d"
                       % (nbytes, nbytes // 4, seconds * 1e6,
                          shown * (1.35 if bad else 1.0), wrong))
            continue
        out.append("%12d %13d     float     sum      -1 %8.2f %7.2f %7.2f %6d "
                   "%8.2f %7.2f %7.2f %6d"
                   % (nbytes, nbytes // 4, seconds * 1e6, algbw, shown_busbw, wrong,
                      seconds * 1e6, algbw, busbw, 0))
    out.append("# Out of bounds values : %d %s" % (3 if inject_error else 0,
                                                   "FAILED" if inject_error else "OK"))
    out.append("# %s: every row above is arithmetic, not hardware." % SYNTHETIC_MARKER)
    print("\n".join(out))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Audit a measured rccl-tests / nccl-tests report.",
        epilog="Measured input, measured output: this tool models nothing. It "
               "names the file it read and what it had to infer.",
    )
    ap.add_argument("--input", "-i", default=None,
                    help="report file, or '-' for stdin")
    ap.add_argument("--collective", "-c", default="allreduce",
                    help="allreduce, reducescatter, allgather, alltoall, "
                         "broadcast or reduce (default allreduce)")
    ap.add_argument("--ranks", "-N", type=int, default=0,
                    help="ranks in the collective; defaults to the rank count in "
                         "the report header when it has one")
    ap.add_argument("--time-unit", default="us", choices=sorted(TIME_SCALE),
                    help="unit of the time column (default us, what both suites print)")
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="relative disagreement tolerated before a row is flagged "
                         "(default 0.02, i.e. rounding in the printed columns)")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when any row is flagged (for CI)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--emit-example", action="store_true",
                    help="print a SYNTHETIC report, derived from an alpha-beta "
                         "model, so this parser can be exercised without a GPU")
    ap.add_argument("--example-error", action="store_true",
                    help="with --emit-example, corrupt one row so the audit has "
                         "something to catch")
    ap.add_argument("--example-one-column", choices=("algbw", "busbw"), default=None,
                    help="with --emit-example, print a single bandwidth column "
                         "labelled only 'bw' holding this quantity, to exercise "
                         "the inference path")
    ap.add_argument("--example-alpha-us", type=float, default=8.0,
                    help="with --emit-example: per-step latency in the model")
    ap.add_argument("--example-bandwidth-gbs", type=float, default=50.0,
                    help="with --emit-example: per-link bandwidth in the model")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    collective = normalize_collective(args.collective)

    # 0 is the sentinel for "not supplied, take it from the report header". A negative
    # value is a typo, not a sentinel, and treating it as absent would silently audit
    # against a rank count the user did not ask for.
    if args.ranks < 0:
        ap.error("--ranks must be at least 1")

    if args.emit_example:
        ranks = args.ranks if args.ranks > 0 else 4
        if args.example_alpha_us <= 0 or args.example_bandwidth_gbs <= 0:
            ap.error("--example-alpha-us and --example-bandwidth-gbs must be positive")
        sizes = [8 * (4 ** i) for i in range(12)]
        emit_example(ranks, collective, args.example_alpha_us,
                     args.example_bandwidth_gbs, sizes, args.example_error,
                     args.example_one_column)
        return 0

    if not args.input:
        ap.error("--input is required (or --emit-example to generate a fixture)")
    if not (0.0 < args.tolerance < 1.0):
        ap.error("--tolerance must be between 0 and 1")

    text = read_text(args.input)
    rows, meta = parse_rows(text, args.input)

    ranks = args.ranks
    if ranks <= 0:
        if meta["header_ranks"]:
            ranks = meta["header_ranks"]
            meta["ranks_source"] = "the report header"
        else:
            die("--ranks is required: this report's header does not state a rank "
                "count, and f(N) cannot be applied without one")
    else:
        meta["ranks_source"] = "--ranks"
        if meta["header_ranks"] and meta["header_ranks"] != ranks:
            meta["ranks_conflict"] = meta["header_ranks"]

    factor, findings = audit(rows, meta, collective, ranks, args.time_unit,
                             args.tolerance)

    result = {
        "source": meta["source"],
        "schema": meta["schema"],
        "collective": collective,
        "ranks": ranks,
        "factor": factor,
        "rows": rows,
        "flagged": len(findings),
        "synthetic_input": meta["synthetic"],
    }
    if args.json:
        print(json.dumps(result, indent=2))
        return 1 if (args.strict and findings) else 0

    print("=" * 78)
    print("PARSED %d measurement rows from: %s" % (len(rows), meta["source"]))
    print("schema: %s" % meta["schema"])
    print("collective %s, N = %d (from %s), f(N) = %.4f, time column in %s"
          % (collective, ranks, meta.get("ranks_source", "--ranks"), factor,
             args.time_unit))
    print("=" * 78)
    if meta["synthetic"]:
        print("WARNING: this input is labelled '%s'. It is arithmetic, not"
              % SYNTHETIC_MARKER)
        print("         hardware. Do not quote a single number out of it.")
        print("")
    if "ranks_conflict" in meta:
        print("WARNING: --ranks says %d but the report header implies %d. f(N) "
              "moves with N," % (ranks, meta["ranks_conflict"]))
        print("         so one of the two makes every bandwidth below wrong. "
              "Resolve it before reading on.")
        print("")

    inferred = [r for r in rows if "inferred_column" in r]
    if inferred:
        votes = {}
        for row in inferred:
            votes[row["inferred_column"]] = votes.get(row["inferred_column"], 0) + 1
        choice = max(votes, key=votes.get)
        # At f(N) close to 1 the two candidates are the same number to within the
        # printed precision, so there is nothing to infer. Reporting a winner there
        # would be reporting the tie-break rule, not a property of the file.
        if abs(factor - 1.0) <= 0.10:
            print("This report has ONE unlabelled bandwidth column, and for %s at"
                  % collective)
            print("N=%d the factor is %.4f. algbw and busbw are therefore the same"
                  % (ranks, factor))
            print("number to within the printed precision, so WHICH ONE IT IS CANNOT")
            print("BE INFERRED from the numbers. Ask whoever ran it, or re-run with a")
            print("collective whose f(N) is not near 1. Both recomputed columns below")
            print("are still valid; only the label is unknown.")
        else:
            print("This report has ONE unlabelled bandwidth column. INFERRING that it")
            print("is %s: on %d of %d rows it matches %s recomputed from size/time"
                  % (choice, votes.get(choice, 0), len(inferred), choice))
            print("more closely than the alternative. That is an inference, not a")
            print("fact about the file. It is only possible at all because %s at N=%d"
                  % (collective, ranks))
            print("has f(N) = %.4f, so the two candidates differ by more than the"
                  % factor)
            print("printed precision.")
        print("")

    # When the report gave one unlabelled bandwidth column, that column is NOT an
    # algbw and must not be printed under an 'algbw' heading: the whole point of the
    # tool here is that nobody knows which quantity it is.
    reported_label = "bw unlabld" if inferred else "algbw rep"
    header = ("%12s %13s %6s %10s %10s %10s %10s %7s" %
              ("bytes", "placement", "time", reported_label, "algbw calc",
               "busbw rep", "busbw calc", "#wrong"))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("%12s %13s %6.1f %10s %10.2f %10s %10.2f %7s"
              % (human_bytes(row["bytes"]), row["placement"], row["time"],
                 ("%.2f" % row["algbw"]) if "algbw" in row else
                 (("%.2f?" % row["bw"]) if "bw" in row else "-"),
                 row["algbw_recomputed"],
                 ("%.2f" % row["busbw"]) if "busbw" in row else "-",
                 row["busbw_recomputed"],
                 ("%g" % row["wrong"]) if row.get("wrong") is not None else "-"))
    print("-" * len(header))
    if meta["skipped_lines"]:
        print("%d line(s) did not match the schema and were skipped."
              % meta["skipped_lines"])

    if findings:
        print("")
        print("%d row(s) flagged:" % len(findings))
        for row in findings:
            print("  line %d, %s %s:" % (row["line"], human_bytes(row["bytes"]),
                                         row["placement"]))
            for flag in row["flags"]:
                print("    - %s" % flag)
    else:
        print("")
        print("No row disagrees with its own raw time by more than %.1f%%, and "
              "#wrong is 0" % (args.tolerance * 100.0))
        print("everywhere it is reported. The report is internally consistent.")

    print("")
    print("What this audit cannot tell you, stated because the table does not:")
    print("  * whether the rank count is the one the job actually ran with; f(N)")
    print("    is applied from %s and nothing in the numbers can confirm it."
          % meta.get("ranks_source", "--ranks"))
    print("  * whether the plateau is good. That needs the link rate for this")
    print("    path, which is a hardware fact this tool does not have.")
    print("  * whether the smallest row is the latency floor of the fabric or of")
    print("    the launch path. Both live in that number.")
    return 1 if (args.strict and findings) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
