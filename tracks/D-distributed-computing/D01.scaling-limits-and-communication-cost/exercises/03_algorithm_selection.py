"""Exercise 3 - reproduce the choice a collective library makes on your behalf.

A production collective library does not have "an all-reduce implementation"; it has a
decision tree, evaluated per call from the message size and the rank count. This drill
builds the two-algorithm version of that tree: recursive doubling, which minimizes steps
and pays in bytes, against a ring, which minimizes bytes and pays in steps.

Your job:
  TODO 1  implement recursive_doubling_cost
  TODO 2  implement ring_cost
  TODO 3  implement choose, returning the cheaper of the two
  TODO 4  implement tie_bytes by BISECTION on the message size
  TODO 5  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
Worked solution with commentary: ../solutions/README.md

Run: python 03_algorithm_selection.py
"""

BANNER = ("analytic collective model, not measured: alpha-beta costs, no contention "
          "or topology")

US = 1e-6
GB = 1e9
KIB = 1024.0

ALPHA_S = 2.5 * US
BETA_S_PER_BYTE = 1.0 / (25.0 * GB)

RANK_COUNTS = (1, 2, 8, 64, 512)
SIZES_BYTES = (1024.0, 65536.0, 1048576.0, 16777216.0, 1073741824.0)


def recursive_doubling_cost(ranks, nbytes, alpha_s, beta_s_per_byte):
    """TODO 1: ceil(log2(N)) rounds, each exchanging the whole buffer with one partner.

    Cost = steps * alpha + steps * nbytes * beta.

    Two things to get right:
      * ranks == 1 must cost exactly 0.0, not a negative number and not one step. An
        all-reduce over one rank is a no-op, and a cost model that returns a negative
        time here will silently win every comparison.
      * ceil(log2(N)) in floats is a trap at exact powers of two. (ranks - 1).bit_length()
        is exact for every ranks >= 1.
    Reject ranks < 1 and nbytes < 0 with a one-line ValueError.
    """
    return None


def ring_cost(ranks, nbytes, alpha_s, beta_s_per_byte):
    """TODO 2: reduce-scatter then all-gather around a ring.

    2*(N-1) steps, and 2*nbytes*(N-1)/N bytes leaving each rank -- which Patarasuk and
    Yuan proved is a lower bound, so no algorithm moves fewer bytes than this. Same two
    edge cases as TODO 1.

    Notice, once it runs, that the byte term is essentially 2*nbytes for any N above a
    handful. That is why a healthy ring all-reduce takes nearly the same time at 64 ranks
    as at 512, and why "our all-reduce did not get slower when we doubled the cluster" is
    not evidence of anything.
    """
    return None


def choose(ranks, nbytes, alpha_s, beta_s_per_byte):
    """TODO 3: return (name, seconds) for the cheaper algorithm.

    Break an exact tie deterministically -- an unstable comparison here would make a
    library's algorithm choice depend on floating-point luck.
    """
    return None


def tie_bytes(ranks, alpha_s, beta_s_per_byte, high=1 << 40):
    """TODO 4: message size at which the two algorithms cost the same, by bisection.

    The difference ring_cost - recursive_doubling_cost is monotonic in nbytes (the ring
    starts behind on latency and catches up on bytes), so bisect on the size:
      * at ranks <= 2 there is no crossover at all -- recursive doubling moves the same
        bytes in fewer steps, so it wins at every size. Return None
      * otherwise bisect [0, high] until the interval is smaller than a byte

    A closed form exists here too; derive it afterwards and check your bisection against
    it. The two must agree, and if they do not, the closed form is easier to debug.
    """
    return None


def human_bytes(nbytes):
    if nbytes < KIB:
        return f"{nbytes:.0f} B"
    for limit, unit in ((KIB ** 2, "KiB"), (KIB ** 3, "MiB")):
        if nbytes < limit:
            return f"{nbytes / (limit / KIB):.1f} {unit}"
    return f"{nbytes / KIB ** 3:.2f} GiB"


def main():
    print(BANNER)
    print(f"alpha = {ALPHA_S / US:g} us   bandwidth = {1.0 / BETA_S_PER_BYTE / GB:g} GB/s\n")

    print("winner per cell, and how much cheaper it is than the loser:")
    header = f"{'size':>10}" + "".join(f"{f'N={n}':>18}" for n in RANK_COUNTS)
    print(header)
    print("-" * len(header))
    for nbytes in SIZES_BYTES:
        cells = []
        for ranks in RANK_COUNTS:
            doubling = recursive_doubling_cost(ranks, nbytes, ALPHA_S, BETA_S_PER_BYTE)
            ring = ring_cost(ranks, nbytes, ALPHA_S, BETA_S_PER_BYTE)
            picked = choose(ranks, nbytes, ALPHA_S, BETA_S_PER_BYTE)
            if doubling is None or ring is None or picked is None:
                cells.append(f"{'not implemented':>18}")
                continue
            name, best = picked
            loser = doubling if name == "ring" else ring
            tag = "ring" if name == "ring" else "doubling"
            margin = f"{loser / best:.1f}x" if best > 0 else "--"
            cells.append(f"{tag:>10}{margin:>8}")
        print(f"{human_bytes(nbytes):>10}" + "".join(cells))
    print("-" * len(header))

    print("\ntie point, and how it moves with rank count:")
    for ranks in RANK_COUNTS:
        tie = tie_bytes(ranks, ALPHA_S, BETA_S_PER_BYTE)
        if tie is None:
            print(f"  N={ranks:<5} no crossover, or not implemented (TODO 4)")
        else:
            print(f"  N={ranks:<5} tie at {tie:>12.0f} bytes ({human_bytes(tie)})")

    print("\nedge case: ranks == 1 must cost exactly zero for both algorithms")
    for name, function in (("recursive doubling", recursive_doubling_cost),
                           ("ring", ring_cost)):
        value = function(1, 1048576.0, ALPHA_S, BETA_S_PER_BYTE)
        verdict = ("not implemented" if value is None
                   else "ok" if value == 0.0 else f"WRONG: {value}")
        print(f"  {name:<20} -> {verdict}")

    print("\nTODO 5 - answer these in a comment at the bottom of this file:")
    print("  a) Read one column of the table top to bottom. At which size does the")
    print("     winner flip, and which term caused the flip?")
    print("  b) Read one row left to right. The tie point moves with rank count. Say")
    print("     which direction and why, and what that means for a tuning constant")
    print("     baked into a build.")
    print("  c) At N=2 there is no crossover. Explain in one sentence why not.")
    print("  d) This model ignores contention. Name a fabric on which the ring's cost")
    print("     would be badly underestimated, and say what you would measure to catch")
    print("     it.")


if __name__ == "__main__":
    main()

# TODO 5 answers:
# a)
# b)
# c)
# d)
