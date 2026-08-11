"""Exercise 4 - variance amplification: why jitter is a scaling term, not noise.

A collective is a barrier, so every rank leaves it at the speed of the slowest arrival.
The step time is therefore not the MEAN per-rank time but the MAXIMUM over N samples --
and the maximum of N draws grows with N even when the distribution never changes. This is
the fourth scaling mechanism, after serial fraction, latency and bandwidth, and it is the
one a faster fabric cannot fix.

This drill reproduces the table in section 3.4 of the module README from a Gaussian
model, so that the figures there have a producer in the repository rather than being
asserted.

Your job:
  TODO 1  implement barrier_step_time
  TODO 2  implement waiting_fraction
  TODO 3  implement sample_step_times
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
Worked solution with commentary: ../solutions/README.md

Run: python 04_straggler_simulation.py
"""

import random

BANNER = ("Gaussian jitter model, not measured: illustrative distribution, "
          "not any real cluster")

# One rank's step time, normalized so the mean is 1.0. A 2% standard deviation is an
# unremarkable amount of jitter -- clock throttling, ECC scrubbing, an OS timer tick, a
# neighbour's traffic. The point of the drill is that unremarkable is not harmless.
MEAN_STEP = 1.0
SIGMA = 0.02

RANK_COUNTS = (8, 64, 1024, 16384)
TRIALS = 100
SEED = 20260811


def barrier_step_time(step_times):
    """TODO 1: wall-clock time of one barrier-synchronized step.

    Every rank waits for the last arrival, so this is the maximum, not the mean. Raise
    ValueError with a one-line message on an empty sequence -- a step with no ranks has
    no duration, and returning 0.0 would quietly report a perfect result.
    """
    return None


def waiting_fraction(step_times):
    """TODO 2: share of aggregate rank-time spent waiting at the barrier.

    Aggregate rank-time for the step is len(step_times) * barrier_step_time, of which
    sum(step_times) was useful work. Return the idle share of that total, so 0.0 means
    every rank finished at the same instant. Same empty-sequence rejection as TODO 1.
    """
    return None


def sample_step_times(ranks, sigma, rng):
    """TODO 3: draw one step's per-rank times from a normal distribution.

    Use rng.gauss(MEAN_STEP, sigma) so the run is reproducible from SEED -- an
    unseeded simulation whose numbers move between runs cannot be cited by the README.
    Reject ranks < 1 and sigma < 0 with a one-line ValueError.
    """
    return None


def simulate(ranks, sigma, trials, seed):
    """Average the barrier step time and the idle share over `trials` independent steps."""
    rng = random.Random(seed)
    slowest_total = 0.0
    idle_total = 0.0
    for _ in range(trials):
        times = sample_step_times(ranks, sigma, rng)
        if times is None:
            return None
        slowest = barrier_step_time(times)
        idle = waiting_fraction(times)
        if slowest is None or idle is None:
            return None
        slowest_total += slowest
        idle_total += idle
    return slowest_total / trials, idle_total / trials


def main():
    print(BANNER)
    print(f"per-rank step time ~ Normal(mean {MEAN_STEP:g}, sd {SIGMA:g}), "
          f"{TRIALS} steps per rank count, seed {SEED}\n")

    header = (f"{'ranks':>8}{'slowest rank':>15}{'throughput lost':>18}"
              f"{'rank-time idle':>17}")
    print(header)
    print("-" * len(header))
    for ranks in RANK_COUNTS:
        result = simulate(ranks, SIGMA, TRIALS, SEED)
        if result is None:
            print(f"{ranks:>8}  not implemented (TODO 1-3)")
            continue
        slowest, idle = result
        lost = 1.0 - MEAN_STEP / slowest
        print(f"{ranks:>8}{slowest:>14.3f}x{lost * 100:>17.1f}%{idle * 100:>16.1f}%")
    print("-" * len(header))

    print("\nrejection checks (each must raise ValueError):")
    for label, function, args in (("empty step list", barrier_step_time, ([],)),
                                  ("empty step list", waiting_fraction, ([],)),
                                  ("ranks < 1", sample_step_times,
                                   (0, SIGMA, random.Random(SEED)))):
        try:
            value = function(*args)
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        else:
            outcome = ("no ValueError raised" if value is None
                       else f"returned {value}")
        print(f"  {function.__name__}({label}) -> {outcome}")

    print("\nTODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) The distribution is IDENTICAL at every rank count. Explain why the")
    print("     slowest-rank column still rises, in one sentence a manager would accept.")
    print("  b) Compare the growth from 8 to 1024 ranks with the growth from 1024 to")
    print("     16384. Is this loss bounded? What function of N does it follow?")
    print("  c) A faster fabric does not move this column. Name two interventions that")
    print("     do, and say which one a cluster operator can apply today.")
    print("  d) Re-run with SIGMA = 0.10 -- one sick node's worth of jitter. State the")
    print("     loss at 1024 ranks and whether you would drain that node.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
# d)
