"""Exercise 2 - what staging through host memory actually costs, and when.

The confident claim "GPUDirect gives you 2x" is usually wrong, and the reason is
arithmetic rather than opinion. This drill makes you derive it.

Two paths for m bytes from one GPU's memory to a remote GPU's memory (README 2.3):

    staged   GPU -> host, host -> pinned/NIC, wire, NIC -> host, host -> GPU
    direct   NIC reads GPU memory, wire, remote NIC writes GPU memory

Two bounds, because they answer different questions. The SERIAL bound assumes each
segment waits for the one before it, which is what an un-chunked transfer pays. The
PIPELINED bound assumes the transfer is chunked and the segments overlap, so the cost is
whichever single resource is busiest -- and the staged path puts twice the traffic
through the local PCIe complex on each side.

Your job:
  TODO 1  implement staged_cost
  TODO 2  implement direct_cost
  TODO 3  implement crossover_pcie_bps
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
Reference implementation: ../tools/staging_cost.py
Worked solution with commentary: ../solutions/README.md

Run: python 02_staging_cost.py
"""

BANNER = ("analytic bandwidth model, not measured: no latency term, no host-memory "
          "ceiling,\nno CPU cost, no contention - so it UNDERSTATES the direct path")

GB = 1e9

# README section 2.3's worked example, to check your implementation against. Note that
# 1 GB here is 1e9 bytes, not a gibibyte: the section's 0.30 s and 0.20 s come out only
# with the decimal reading, and lab 4 passes 1073741824 instead, which is a gibibyte and
# gives 0.322 s and 0.215 s. Getting that wrong by 7% is the smallest version of the
# unit mistake README section 5.2 warns about.
REFERENCE_BYTES = 1.0e9
REFERENCE_PCIE_GBS = 20.0
REFERENCE_NET_GBS = 10.0
REFERENCE_STAGED_SERIAL_S = 0.30
REFERENCE_DIRECT_SERIAL_S = 0.20

# (label, PCIe GB/s, network GB/s). Illustrative orders of magnitude, not a datasheet
# quote for any part -- the point is the RATIO between the two columns.
CASES = (
    ("PCIe 4x the wire", 40.0, 10.0),
    ("PCIe 2x the wire", 20.0, 10.0),
    ("PCIe equal to the wire", 10.0, 10.0),
    ("PCIe narrower than the wire", 5.0, 10.0),
)

SIZES_BYTES = (65536.0, 1048576.0, 1.0e9)


def staged_cost(nbytes, pcie_bps, net_bps):
    """TODO 1: return (serial_seconds, pipelined_seconds) for the host-staged path.

    Count the segments from README 2.3's table before you write any arithmetic. How many
    times does a byte cross a PCIe link on the way from one GPU to the other? How many
    times does it cross the wire?

    For the pipelined bound, ask which single resource is busiest once the segments
    overlap. There are two candidates and the answer is a max() of them.

    Reject a non-positive size or bandwidth with ValueError and a one-line message that
    names the offending value. A tool built on this has to print that message and exit,
    not show a traceback.
    """
    return None


def direct_cost(nbytes, pcie_bps, net_bps):
    """TODO 2: return (serial_seconds, pipelined_seconds) for the direct path.

    Same shape as TODO 1 with half the PCIe traversals. Validate the same way.
    """
    return None


def crossover_pcie_bps(net_bps):
    """TODO 3: the PCIe bandwidth below which the staged path becomes bandwidth-limited.

    Set the two pipelined bounds equal and solve. The answer is a multiple of the
    network bandwidth and does not depend on the message size at all -- which is itself
    worth noticing, and is question (b).

    Reject a non-positive bandwidth with ValueError.
    """
    return None


def attempt(function, *args):
    """Report what `function` did with bad input: raised, returned, or not written yet."""
    try:
        value = function(*args)
    except ValueError as exc:
        return "ValueError: %s" % exc
    return "no ValueError raised" if value is None else "returned %r" % (value,)


def pair(value):
    """Format a (serial, pipelined) result, or say it is missing."""
    if value is None:
        return "%23s" % "not implemented"
    serial, pipelined = value
    return "%11.4f s%12.4f s" % (serial, pipelined)


def main():
    print(BANNER)
    print("")

    print("check against README section 2.3: %g bytes, PCIe %g GB/s, wire %g GB/s"
          % (REFERENCE_BYTES, REFERENCE_PCIE_GBS, REFERENCE_NET_GBS))
    staged = staged_cost(REFERENCE_BYTES, REFERENCE_PCIE_GBS * GB,
                         REFERENCE_NET_GBS * GB)
    direct = direct_cost(REFERENCE_BYTES, REFERENCE_PCIE_GBS * GB,
                         REFERENCE_NET_GBS * GB)
    for label, got, want in (("staged serial", staged, REFERENCE_STAGED_SERIAL_S),
                             ("direct serial", direct, REFERENCE_DIRECT_SERIAL_S)):
        if got is None:
            print("  %-14s not implemented (want %.2f s)" % (label, want))
        else:
            ok = "OK" if abs(got[0] - want) < 1e-9 else "MISMATCH"
            print("  %-14s %.4f s  (want %.2f s)  %s" % (label, got[0], want, ok))
    print("")

    header = ("%-30s%13s%14s%13s%14s%9s"
              % ("case", "staged ser", "staged pipe", "direct ser", "direct pipe",
                 "pipe x"))
    print(header)
    print("-" * len(header))
    for label, pcie_gbs, net_gbs in CASES:
        staged = staged_cost(1.0e9, pcie_gbs * GB, net_gbs * GB)
        direct = direct_cost(1.0e9, pcie_gbs * GB, net_gbs * GB)
        if staged is None or direct is None:
            print("%-30s%s" % (label, "  not implemented"))
            continue
        ratio = staged[1] / direct[1] if direct[1] > 0 else float("inf")
        print("%-30s%s%s%8.2fx" % (label, pair(staged), pair(direct), ratio))
    print("-" * len(header))
    print("(1e9 bytes in every row, so only the two bandwidths differ)")
    print("")

    print("crossover PCIe bandwidth, per network bandwidth:")
    for net_gbs in (5.0, 10.0, 25.0):
        knee = crossover_pcie_bps(net_gbs * GB)
        print("  wire %5.1f GB/s -> %s"
              % (net_gbs, "not implemented" if knee is None
                 else "%.1f GB/s" % (knee / GB)))
    print("")

    print("does the message size move the crossover? (it should not)")
    for nbytes in SIZES_BYTES:
        staged = staged_cost(nbytes, 20.0 * GB, 10.0 * GB)
        direct = direct_cost(nbytes, 20.0 * GB, 10.0 * GB)
        if staged is None or direct is None:
            print("  %12.0f bytes -> not implemented" % nbytes)
        else:
            print("  %12.0f bytes -> pipelined ratio %.3f" % (nbytes, staged[1] / direct[1]))
    print("")

    print("rejection checks (each must raise ValueError):")
    print("  zero bytes       -> %s" % attempt(staged_cost, 0.0, 20e9, 10e9))
    print("  negative PCIe    -> %s" % attempt(direct_cost, 1e9, -1.0, 10e9))
    print("  zero wire        -> %s" % attempt(crossover_pcie_bps, 0.0))
    print("")

    print("TODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) In the 'PCIe 2x the wire' row the serial bound says staging is 1.5x")
    print("     worse and the pipelined bound says it is identical. Both are correct.")
    print("     Which one describes a 4 KiB gradient bucket, which describes a 1 GiB")
    print("     all-reduce, and what decides?")
    print("  b) The crossover does not depend on the message size. Show why from the")
    print("     algebra, then say what that means for a rule of thumb like 'GPUDirect")
    print("     matters for large messages'.")
    print("  c) Your measurement shows the direct path is 1.05x the staged path on a")
    print("     machine where this model predicts 1.00x. Your colleague concludes")
    print("     GPUDirect is not worth enabling. Give two costs the model does not")
    print("     contain that you would measure instead, and say how you would measure")
    print("     one of them.")
    print("  d) The staged path moves roughly 4m through host memory. On a node with")
    print("     eight ranks all doing this at once, what breaks first, and which")
    print("     unrelated part of the training job notices before the collective does?")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
# d)
