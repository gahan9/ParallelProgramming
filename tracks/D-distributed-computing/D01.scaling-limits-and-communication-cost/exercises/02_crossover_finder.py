"""Exercise 2 - the crossover m*, and classifying a message by which term owns it.

Under the alpha-beta model a message costs alpha + m*beta. The single most useful number
that falls out is not a time, it is the size where the two terms are equal. Below it,
bytes are nearly free and the only lever is sending fewer messages. Above it, latency is
noise and the only lever is bandwidth. Optimizing for the wrong side of that line is the
classic expensive mistake.

Your job:
  TODO 1  implement crossover_bytes
  TODO 2  implement classify, using a factor-of-two band around the crossover
  TODO 3  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
Worked solution with commentary: ../solutions/README.md

Run: python 02_crossover_finder.py
"""

BANNER = "analytic alpha-beta model, not measured: one message, idle link, no contention"

US = 1e-6
GB = 1e9
KIB = 1024.0

# Order-of-magnitude link classes, illustrative rather than any specific part:
# (label, alpha in microseconds, bandwidth in GB/s)
LINKS = (
    ("intra-node GPU link", 1.5, 200.0),
    ("inter-node RDMA", 2.5, 25.0),
    ("datacenter Ethernet, TCP", 100.0, 10.0),
)

SIZES_BYTES = (256.0, 4096.0, 65536.0, 1048576.0, 67108864.0)

# Within a factor of two of the crossover neither term can be ignored, so the honest
# answer is "balanced" rather than a side.
BALANCED_BAND = 2.0


def crossover_bytes(alpha_s, beta_s_per_byte):
    """TODO 1: message size at which the latency and transfer terms are equal.

    Set alpha = m * beta and solve for m. Note what the answer depends on: only the
    link, never the workload -- which is why a fused-message threshold tuned on one
    cluster is wrong on the next.

    Guard beta <= 0 by raising ValueError with a one-line message; an infinite-bandwidth
    link has no crossover and silently returning inf would hide a bad input.
    """
    return None


def classify(nbytes, alpha_s, beta_s_per_byte):
    """TODO 2: return "latency-bound", "balanced", or "bandwidth-bound".

    Use the crossover from TODO 1 and a factor of BALANCED_BAND on either side:
    below crossover/2 is latency-bound, above crossover*2 is bandwidth-bound, and the
    band between them is balanced. Reject nbytes < 0 with a ValueError.
    """
    return None


def attempt(function, *args):
    """Report what `function` did with bad input: raised, returned, or not written yet."""
    try:
        value = function(*args)
    except ValueError as exc:
        return f"ValueError: {exc}"
    return "no ValueError raised" if value is None else f"returned {value}"


def human_bytes(nbytes):
    if nbytes < KIB:
        return f"{nbytes:.0f} B"
    for limit, unit in ((KIB ** 2, "KiB"), (KIB ** 3, "MiB")):
        if nbytes < limit:
            return f"{nbytes / (limit / KIB):.1f} {unit}"
    return f"{nbytes / KIB ** 3:.2f} GiB"


def main():
    print(BANNER)
    print("m* is a property of the link, so it is computed once per link class.\n")

    header = f"{'link':<26}{'alpha us':>10}{'GB/s':>8}{'m* bytes':>12}{'m*':>12}"
    print(header)
    print("-" * len(header))
    crossovers = {}
    for label, alpha_us, bw_gbs in LINKS:
        star = crossover_bytes(alpha_us * US, 1.0 / (bw_gbs * GB))
        crossovers[label] = star
        if star is None:
            print(f"{label:<26}{alpha_us:>10.1f}{bw_gbs:>8.2f}"
                  f"  not implemented (TODO 1)")
        else:
            print(f"{label:<26}{alpha_us:>10.1f}{bw_gbs:>8.2f}{star:>12.0f}"
                  f"{human_bytes(star):>12}")
    print("-" * len(header))

    print("\nregime of a given message size on each link:")
    header = f"{'size':>12}" + "".join(f"{label:>26}" for label, _, _ in LINKS)
    print(header)
    print("-" * len(header))
    for nbytes in SIZES_BYTES:
        cells = []
        for _label, alpha_us, bw_gbs in LINKS:
            verdict = classify(nbytes, alpha_us * US, 1.0 / (bw_gbs * GB))
            cells.append(f"{verdict if verdict else 'not implemented':>26}")
        print(f"{human_bytes(nbytes):>12}" + "".join(cells))
    print("-" * len(header))

    print("\nrejection checks (each must raise ValueError):")
    print(f"  beta <= 0      -> {attempt(crossover_bytes, 1.0, 0.0)}")
    print(f"  negative size  -> {attempt(classify, -1.0, 2.5 * US, 4e-11)}")

    print("\nTODO 3 - answer these in a comment at the bottom of this file:")
    print("  a) A 4 KiB message: which links call it latency-bound, and what is the")
    print("     one optimization that helps on all of them?")
    print("  b) The three links disagree about m*. Explain why a hard-coded 'fuse")
    print("     messages under 1 MB' rule is a bug rather than a tuning value.")
    print("  c) The Ethernet link has the LARGEST m* of the three despite having far")
    print("     less bandwidth than the GPU link. Which parameter put it there, and")
    print("     what does that tell you about which number to ask a vendor for?")
    print("  d) Compressing a message that sits below m* saves bytes. What does it save")
    print("     in time, and what does it cost?")


if __name__ == "__main__":
    main()

# TODO 3 answers:
# a)
# b)
# c)
# d)
