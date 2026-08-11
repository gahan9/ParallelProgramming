#!/usr/bin/env python3
"""Drill 1 - the bus bandwidth converter (D03 section 2.4).

Write busbw(collective, ranks, nbytes, seconds) so it returns BOTH bandwidths for
allreduce, allgather, reducescatter, alltoall, broadcast and reduce.

This file RUNS as given and FAILS its own self-test, because the conversion is a
marked TODO. A FAIL on the first run means the harness works.

  python -S -I ex1_busbw_convert.py                     # run the self-test
  python -S -I ex1_busbw_convert.py --collective allreduce --ranks 8 \
      --bytes 1073741824 --time-ms 40                   # convert one measurement

Requirements, all of which the self-test checks:

  TODO 1  return a dict with 'algbw' and 'busbw' in bytes per second, using the
          traffic factors f(N) from the README section 2.4 table.
  TODO 2  raise ValueError with a ONE-LINE message for an unknown collective, for
          ranks < 1, and for a non-positive or non-finite time. A traceback names
          your code; a ValueError with a sentence names the caller's mistake.
  TODO 3  decide what ranks == 1 means. The all-reduce factor 2(N-1)/N is 0 there.
          Is that an error or the correct answer? Pick one, make the code do it,
          and defend it in a comment - the defence is the graded part.

Standard library only: no imports beyond argparse, math and sys.
"""

from __future__ import annotations

import argparse
import math
import sys

PROGRAM = "ex1_busbw_convert"

# The collectives this converter must handle. The factors are NOT given: deriving
# them from "how many bytes does each rank move in the ring form" is the drill.
# README section 2.2 has the byte counts and 2.4 has the table to check against.
COLLECTIVES = ("allreduce", "allgather", "reducescatter", "alltoall",
               "broadcast", "reduce")


def busbw(collective, ranks, nbytes, seconds):
    """Return {'algbw': ..., 'busbw': ..., 'factor': ...} in bytes per second.

    algbw is the application-visible rate, m / t. busbw is what crossed the wire,
    algbw * f(N), and it is the only one of the two comparable to a link rate.

    TODO 1: implement this.
    TODO 2: validate the arguments and raise ValueError with one line of text.
    TODO 3: decide and document what ranks == 1 means.
    """
    # Delete this line once the TODOs are done; it exists so the unfinished
    # starter runs and reports FAIL instead of raising.
    return None


# ---------------------------------------------------------------------------
# Harness below this line. You should not need to change it.
# ---------------------------------------------------------------------------

# Round invented inputs, so every expectation below is arithmetic you can check
# on paper. None of these is a measurement and none should ever be quoted as one.
CASES = (
    # collective,       ranks, bytes,      seconds, expected f(N)
    ("allreduce", 8, 1073741824, 0.040, 2.0 * 7 / 8),
    ("allreduce", 4, 1048576, 0.001, 2.0 * 3 / 4),
    ("allgather", 8, 268435456, 0.010, 7.0 / 8),
    ("reducescatter", 16, 268435456, 0.010, 15.0 / 16),
    ("alltoall", 2, 65536, 0.0001, 1.0 / 2),
    ("broadcast", 64, 4194304, 0.002, 1.0),
    ("reduce", 64, 4194304, 0.002, 1.0),
)

BAD_CASES = (
    ("nosuchcollective", 8, 1024, 0.001, "unknown collective"),
    ("allreduce", 0, 1024, 0.001, "ranks below 1"),
    ("allreduce", -3, 1024, 0.001, "negative ranks"),
    ("allreduce", 8, 1024, 0.0, "zero time"),
    ("allreduce", 8, 1024, -0.001, "negative time"),
    ("allreduce", 8, 1024, float("nan"), "non-finite time"),
)


def check_case(collective, ranks, nbytes, seconds, factor):
    """Return None when the case passes, or a one-line reason when it fails."""
    try:
        got = busbw(collective, ranks, nbytes, seconds)
    except ValueError as exc:
        return "raised ValueError(%s) on a valid input" % exc
    if got is None:
        return "returned None (TODO 1 not implemented)"
    if not isinstance(got, dict):
        return "returned %s, expected a dict" % type(got).__name__
    for key in ("algbw", "busbw"):
        if key not in got:
            return "result has no '%s' key" % key
    want_alg = nbytes / seconds
    want_bus = want_alg * factor
    for key, want in (("algbw", want_alg), ("busbw", want_bus)):
        have = float(got[key])
        if abs(have - want) > 1e-6 * max(abs(want), 1.0):
            return "%s = %.6g, expected %.6g (f(N) should be %.6g)" % (
                key, have, want, factor)
    return None


def check_bad_case(collective, ranks, nbytes, seconds, what):
    try:
        got = busbw(collective, ranks, nbytes, seconds)
    except ValueError as exc:
        message = str(exc)
        if not message.strip():
            return "raised ValueError with an empty message on %s" % what
        if "\n" in message:
            return "ValueError message for %s spans several lines" % what
        return None
    except Exception as exc:  # noqa: BLE001 - the drill is that this must not happen
        return "raised %s on %s; the contract is ValueError" % (
            type(exc).__name__, what)
    return "returned %r on %s; expected ValueError" % (got, what)


def self_test():
    failures = []
    for case in CASES:
        reason = check_case(*case)
        label = "%s N=%d" % (case[0], case[1])
        print("  %-24s %s" % (label, "ok" if reason is None else "FAIL: " + reason))
        if reason is not None:
            failures.append(label)
    print("")
    print("  refusing bad input:")
    for case in BAD_CASES:
        reason = check_bad_case(*case)
        label = case[4]
        print("  %-24s %s" % (label, "ok" if reason is None else "FAIL: " + reason))
        if reason is not None:
            failures.append(label)
    print("")
    if failures:
        print("FAIL: %d of %d checks (expected until the TODOs are done)"
              % (len(failures), len(CASES) + len(BAD_CASES)))
        print("Once it passes, answer TODO 3 in a comment: is N == 1 an error?")
        return 1
    print("PASS: all %d checks" % (len(CASES) + len(BAD_CASES)))
    print("Now compare your factors against ../tools/busbw.py --table --ranks 8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Drill 1: convert a measured collective timing to algbw and busbw.",
        epilog="With no arguments it runs the self-test.",
    )
    ap.add_argument("--collective", "-c", default="")
    ap.add_argument("--ranks", "-N", type=int, default=0)
    ap.add_argument("--bytes", "-b", type=float, default=0.0)
    ap.add_argument("--time-ms", type=float, default=0.0)
    args = ap.parse_args(argv)

    if not args.collective:
        print("self-test for %s" % PROGRAM)
        return self_test()

    if args.bytes <= 0 or not math.isfinite(args.bytes):
        ap.error("--bytes must be positive and finite")
    if args.time_ms <= 0 or not math.isfinite(args.time_ms):
        ap.error("--time-ms must be positive and finite")

    try:
        result = busbw(args.collective, args.ranks, args.bytes, args.time_ms * 1e-3)
    except ValueError as exc:
        sys.stderr.write("%s: %s\n" % (PROGRAM, exc))
        return 2
    if result is None:
        sys.stderr.write("%s: busbw() is still the unfinished stub (TODO 1)\n"
                         % PROGRAM)
        return 1
    print("algbw %.3f GB/s   busbw %.3f GB/s   (%s over %d ranks)"
          % (result["algbw"] / 1e9, result["busbw"] / 1e9, args.collective,
             args.ranks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
