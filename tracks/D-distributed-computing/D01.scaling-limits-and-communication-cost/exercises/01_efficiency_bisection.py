"""Exercise 1 - efficiency, not speedup, and the rank count where it stops paying.

Speedup is the number people quote. Efficiency is the number that gets budgeted, and it
collapses long before the speedup curve looks troubled: at 1024 ranks a 98%-parallel
code is still reporting a respectable speedup while wasting most of the machine. This
drill makes you compute both, then find the largest rank count that still holds a target
efficiency.

Your job:
  TODO 1  implement amdahl_speedup
  TODO 2  implement parallel_efficiency, rejecting inputs the model cannot mean
  TODO 3  implement largest_ranks_at_efficiency by BISECTION, not by the closed form
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
That gap is the exercise. Worked solution with commentary: ../solutions/README.md

Run: python 01_efficiency_bisection.py
"""

BANNER = "analytic scaling model, not measured: Amdahl assumes communication is free"

SERIAL_FRACTION = 0.02
MAX_RANKS = 1024
TARGET = 0.5

# An exact tie (s=0.02 against a 50% target ties at N=51) can land a few ulps on the
# wrong side of the comparison, so compare with slack rather than with ==.
TOL = 1e-12


def amdahl_speedup(serial_fraction, ranks):
    """TODO 1: fixed-problem speedup with free communication.

    The serial part does not divide and the parallel part does:
        speedup = 1 / (s + (1 - s) / N)
    Return a float. No validation here; TODO 2 owns that.
    """
    return None


def parallel_efficiency(serial_fraction, ranks):
    """TODO 2: return (speedup, efficiency) where efficiency = speedup / ranks.

    Validate first, and raise ValueError with a ONE-LINE message that names the
    offending value:
      * ranks < 1 is not a machine
      * serial_fraction outside [0, 1] is not a fraction
    A tool built on this must be able to turn that exception into a one-line error and
    an exit code, so the message has to be readable on its own.
    """
    return None


def largest_ranks_at_efficiency(serial_fraction, target, ceiling=1 << 40):
    """TODO 3: largest rank count whose efficiency is still at or above `target`.

    Efficiency is 1/(s*N + (1 - s)), strictly decreasing in N whenever s > 0, so
    bisect on the integer N:
      * if efficiency at `ceiling` still meets the target, return None -- s = 0 is the
        honest case, where every rank is fully used at every scale
      * otherwise keep the invariant "low meets the target, high does not" and halve the
        interval until they are adjacent; return low
      * use TOL when comparing against the target

    A closed form exists: N <= (1/target - 1 + s)/s. Use it to CHECK your bisection,
    not to replace it -- the point of bisection is that it still works once the
    communication term goes in and there is no closed form left.
    """
    return None


def main():
    print(BANNER)
    print(f"s = {SERIAL_FRACTION:g}   target efficiency = {TARGET * 100:g}%\n")

    header = f"{'ranks':>8}{'speedup':>12}{'efficiency':>13}"
    print(header)
    print("-" * len(header))
    ranks = 1
    while ranks <= MAX_RANKS:
        result = parallel_efficiency(SERIAL_FRACTION, ranks)
        if result is None:
            print(f"{ranks:>8}  not implemented (TODO 1, 2)")
        else:
            speedup, efficiency = result
            print(f"{ranks:>8}{speedup:>11.2f}x{efficiency * 100:>12.1f}%")
        ranks *= 2
    print("-" * len(header))

    knee = largest_ranks_at_efficiency(SERIAL_FRACTION, TARGET)
    if knee is None:
        print("largest rank count at target: not implemented (TODO 3), or no wall exists")
    else:
        closed_form = int((1.0 / TARGET - 1.0 + SERIAL_FRACTION) / SERIAL_FRACTION)
        agree = "agrees" if knee == closed_form else f"DISAGREES with {closed_form}"
        print(f"largest rank count holding {TARGET * 100:g}% efficiency: {knee} "
              f"(closed form {agree})")

    print("\nrejection checks (each must raise ValueError, not return a number):")
    for serial, ranks in ((0.02, 0), (0.02, -8), (-0.1, 8), (1.5, 8)):
        try:
            outcome = parallel_efficiency(serial, ranks)
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        else:
            outcome = "no ValueError raised" if outcome is None else f"returned {outcome}"
        print(f"  s={serial:<6} ranks={ranks:<4} -> {outcome}")

    print("\nTODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) At s=0.02 the speedup at 1024 ranks is close to the ceiling 1/s. State")
    print("     the efficiency there, and what you would say to someone asking for 1024")
    print("     ranks for this workload.")
    print("  b) Your bisection and the closed form should agree exactly. Construct an s")
    print("     and a target where a naive float comparison makes them disagree, and say")
    print("     which one you would trust.")
    print("  c) Amdahl says nothing about communication. Sketch what happens to the")
    print("     efficiency column once a fixed per-step all-reduce cost is added, and")
    print("     say whether the wall moves in or out.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
