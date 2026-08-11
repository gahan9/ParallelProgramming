#!/usr/bin/env python3
"""Drill 1 solution - the bus bandwidth converter (D03 section 2.4).

The arithmetic is four lines. Everything worth marking is in the three decisions
around it, so they are argued here rather than left implicit in the code.

DECISION 1 - where the factors come from, and why they are not a lookup table of
magic numbers. Each f(N) is "bytes this rank moves in the ring form, divided by
the buffer size", derived in README section 2.2:

  all-reduce      reduce-scatter then all-gather, 2(N-1) steps of m/N   ->  2(N-1)/N
  reduce-scatter  half of that                                          ->   (N-1)/N
  all-gather      the other half                                        ->   (N-1)/N
  all-to-all      each rank sends (N-1)/N of its buffer away            ->   (N-1)/N
  broadcast       every byte crosses once, whatever N is                ->        1
  reduce          the mirror image of broadcast                         ->        1

Deriving them beats memorising them because the derivation is what tells you the
factor for a collective not in the table, and because the shape - all-reduce > 1,
reduce-scatter < 1, broadcast == 1 - is the part that stops you misreading a
benchmark. Broadcast being 1 is why a broadcast benchmark is the one place the
algbw/busbw distinction is invisible, and therefore never learned.

DECISION 2 - ValueError, not a print or an assert. This function is a library
routine: it cannot know whether its caller is a CLI, a notebook or a CI job, so it
raises and lets the caller decide how to report. The message is one line and names
the offending value, because it will be read by someone who did not write this
code. assert is wrong here for a second reason: it vanishes under python -O, and
"validation that disappears in production" is worse than none.

DECISION 3 - N == 1 returns a factor of 0, and that is the answer rather than an
error. The tabulated factors are derived for N >= 2, and at N == 1 no byte crosses
any wire for ANY collective - including broadcast, whose factor is otherwise the
constant 1. Two candidate behaviours, and why this one:

  * raise ValueError. Tempting, but a single-rank run is a legitimate baseline: it
    is exactly how you measure the launch overhead with the fabric taken out. A
    converter that refuses to process it forces the caller to special-case it, and
    that special case is where a wrong number gets invented.
  * return busbw = algbw (factor 1). This is the dangerous one. It reports local
    memory bandwidth as interconnect bandwidth, which is the precise error the
    algbw/busbw distinction exists to prevent, and it does it silently.

So: 0, uniformly, with the CLI printing why. The rule to state in review is that a
bus bandwidth of 0 at N == 1 is not a bug, it is the fabric not being involved.

  python -S -I ex1_busbw_convert.py
  python -S -I ex1_busbw_convert.py --collective allreduce --ranks 8 \
      --bytes 1073741824 --time-ms 40

Standard library only.
"""

from __future__ import annotations

import argparse
import math
import sys

PROGRAM = "ex1_busbw_convert"

# f(N) as callables, so the derivation stays visible in the source instead of
# being flattened into decimals. Sorted keys give a deterministic error message.
FACTORS = {
    "allreduce": lambda n: 2.0 * (n - 1.0) / n,
    "allgather": lambda n: (n - 1.0) / n,
    "reducescatter": lambda n: (n - 1.0) / n,
    "alltoall": lambda n: (n - 1.0) / n,
    "broadcast": lambda n: 1.0,
    "reduce": lambda n: 1.0,
}


def normalize(collective):
    """Fold 'all_reduce', 'All-Reduce' and 'allreduce' onto one key.

    Callers write the name three different ways and a converter that only accepts
    one of them gets wrapped in a normalizing helper by every caller instead.
    """
    if not isinstance(collective, str):
        raise ValueError("collective must be a string, got %s"
                         % type(collective).__name__)
    return collective.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def busbw(collective, ranks, nbytes, seconds):
    """Return {'algbw', 'busbw', 'factor'} in bytes per second.

    algbw = m / t is application-visible throughput. busbw = algbw * f(N) is what
    crossed the wire and is the only one of the two comparable to a link rate.
    """
    key = normalize(collective)
    if key not in FACTORS:
        raise ValueError("unknown collective %r; expected one of %s"
                         % (collective, ", ".join(sorted(FACTORS))))
    if not isinstance(ranks, int) or isinstance(ranks, bool):
        raise ValueError("ranks must be an int, got %s" % type(ranks).__name__)
    if ranks < 1:
        raise ValueError("ranks must be at least 1, got %d" % ranks)
    if not math.isfinite(nbytes) or nbytes <= 0:
        raise ValueError("bytes must be finite and positive, got %r" % (nbytes,))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive, got %r; a "
                         "collective cannot take zero time" % (seconds,))

    # DECISION 3, in one line. See the module docstring for the defence.
    factor = 0.0 if ranks == 1 else FACTORS[key](float(ranks))
    algbw = nbytes / seconds
    return {"algbw": algbw, "busbw": algbw * factor, "factor": factor}


# ---------------------------------------------------------------------------
# Harness, identical to the one in ../exercises/ex1_busbw_convert.py.
# ---------------------------------------------------------------------------

# Round invented inputs, so every expectation below is arithmetic you can check
# on paper. None of these is a measurement and none should ever be quoted as one.
CASES = (
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
    try:
        got = busbw(collective, ranks, nbytes, seconds)
    except ValueError as exc:
        return "raised ValueError(%s) on a valid input" % exc
    if got is None:
        return "returned None"
    want_alg = nbytes / seconds
    want_bus = want_alg * factor
    for key, want in (("algbw", want_alg), ("busbw", want_bus)):
        if key not in got:
            return "result has no '%s' key" % key
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
    except Exception as exc:  # noqa: BLE001
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
        print("  %-24s %s" % (case[4], "ok" if reason is None else "FAIL: " + reason))
        if reason is not None:
            failures.append(case[4])
    print("")
    # The N == 1 stance, asserted rather than described, so a later edit that
    # changes it fails here instead of quietly changing what the tool claims.
    one = busbw("allreduce", 1, 1024, 0.001)
    label = "N == 1 gives busbw 0"
    ok = one["busbw"] == 0.0 and one["algbw"] > 0.0
    print("  %-24s %s" % (label, "ok" if ok else "FAIL"))
    if not ok:
        failures.append(label)
    print("")
    if failures:
        print("FAIL: %d checks" % len(failures))
        return 1
    print("PASS: all %d checks" % (len(CASES) + len(BAD_CASES) + 1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Drill 1 solution: measured collective timing to algbw and busbw.",
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

    try:
        result = busbw(args.collective, args.ranks, args.bytes, args.time_ms * 1e-3)
    except ValueError as exc:
        sys.stderr.write("%s: %s\n" % (PROGRAM, exc))
        return 2
    print("algbw %.3f GB/s   busbw %.3f GB/s   f(N) = %.4f   (%s over %d ranks)"
          % (result["algbw"] / 1e9, result["busbw"] / 1e9, result["factor"],
             args.collective, args.ranks))
    if args.ranks == 1:
        print("N == 1: f(N) is 0 because nothing crossed a wire. That is the "
              "answer, not a failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
