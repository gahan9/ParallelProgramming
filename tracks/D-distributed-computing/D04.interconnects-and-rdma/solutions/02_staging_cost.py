"""Solution 2 - what staging through host memory actually costs, and when.

The three TODOs from ../exercises/02_staging_cost.py, filled in, with the reasoning in
../solutions/README.md. The harness is unchanged from the starter.

Run: python 02_staging_cost.py
"""

BANNER = ("analytic bandwidth model, not measured: no latency term, no host-memory "
          "ceiling,\nno CPU cost, no contention - so it UNDERSTATES the direct path")

GB = 1e9

REFERENCE_BYTES = 1.0e9
REFERENCE_PCIE_GBS = 20.0
REFERENCE_NET_GBS = 10.0
REFERENCE_STAGED_SERIAL_S = 0.30
REFERENCE_DIRECT_SERIAL_S = 0.20

CASES = (
    ("PCIe 4x the wire", 40.0, 10.0),
    ("PCIe 2x the wire", 20.0, 10.0),
    ("PCIe equal to the wire", 10.0, 10.0),
    ("PCIe narrower than the wire", 5.0, 10.0),
)

SIZES_BYTES = (65536.0, 1048576.0, 1.0e9)


def _check(nbytes, pcie_bps, net_bps):
    if nbytes <= 0:
        raise ValueError("message size must be positive, got %g bytes" % nbytes)
    if pcie_bps <= 0:
        raise ValueError("PCIe bandwidth must be positive, got %g B/s" % pcie_bps)
    if net_bps <= 0:
        raise ValueError("network bandwidth must be positive, got %g B/s" % net_bps)


def staged_cost(nbytes, pcie_bps, net_bps):
    """(serial, pipelined) seconds for the host-staged path.

    Four PCIe traversals: device->host and host->device on each side of the wire. One
    wire traversal. Serial adds them; pipelined takes the busiest resource, which is a
    PCIe complex carrying 2m on each side against a wire carrying m.
    """
    _check(nbytes, pcie_bps, net_bps)
    serial = 4.0 * nbytes / pcie_bps + nbytes / net_bps
    pipelined = max(2.0 * nbytes / pcie_bps, nbytes / net_bps)
    return (serial, pipelined)


def direct_cost(nbytes, pcie_bps, net_bps):
    """(serial, pipelined) seconds for the direct path: two PCIe traversals, no host copy."""
    _check(nbytes, pcie_bps, net_bps)
    serial = 2.0 * nbytes / pcie_bps + nbytes / net_bps
    pipelined = max(nbytes / pcie_bps, nbytes / net_bps)
    return (serial, pipelined)


def crossover_pcie_bps(net_bps):
    """PCIe bandwidth at which the two pipelined bounds meet.

        max(2m/P, m/N) == max(m/P, m/N)

    Both sides equal m/N while PCIe is wide enough that neither PCIe term dominates. The
    staged term 2m/P is the first to exceed m/N, and it does so at P = 2N. Below that the
    staged path is PCIe-limited and the direct path may still be wire-limited, so the gap
    opens; above it both paths are wire-limited and the pipelined model sees no
    difference at all. The m cancels, which is question (b).
    """
    if net_bps <= 0:
        raise ValueError("network bandwidth must be positive, got %g B/s" % net_bps)
    return 2.0 * net_bps


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
#
# a) The serial bound describes the 4 KiB bucket and the pipelined bound describes the
#    1 GiB all-reduce. What decides is whether the transfer is large enough to be split
#    into chunks that keep every segment busy at once: a 4 KiB message has nothing to
#    overlap with itself, so it pays each hop in turn, while a 1 GiB message is chunked by
#    the library and the hops run concurrently after the first chunk lands. Neither bound
#    is the truth in the middle -- the real curve rises from the serial line to the
#    pipelined line as the message crosses a few times the chunk size, and where that
#    happens is a property of the library's chunking, not of the hardware.
#
# b) Both bounds are a max() of terms that are all m over a bandwidth, so every term
#    carries exactly one factor of m: max(2m/P, m/N) == max(m/P, m/N) divides through by m
#    to max(2/P, 1/N) == max(1/P, 1/N), and the message size is gone. The crossover is
#    P = 2N regardless of size. What that means for "GPUDirect matters for large
#    messages" is that the rule of thumb is true but for the wrong reason: the size
#    dependence lives in the LATENCY term this model does not have -- the extra copies,
#    the extra completion, the extra synchronisation, all fixed costs that dominate small
#    messages. The bandwidth argument for GPUDirect is a statement about the PCIe-to-wire
#    ratio, and it holds at every size or none.
#
# c) The two the model omits and I would measure: (1) latency, the per-transfer fixed cost
#    of the two extra copies and their completions, which is what a small-message ping
#    sees and which the bandwidth model has no term for at all; and (2) CPU and host
#    memory cost, because the staged path spends host cores and roughly 4m of host memory
#    bandwidth that the rest of the job also wants. How I would measure the first:
#    `ib_write_lat` between two nodes, once with the buffers in GPU memory and once with
#    them in host memory, at the smallest size the tool supports, and compare the
#    half-round-trip -- that isolates the fixed cost from any bandwidth effect. For the
#    second, run the real training step and watch step time, not the microbenchmark: a
#    1.05x loss on a transfer that overlaps compute may cost nothing, while the host
#    bandwidth it frees may show up in the data loader.
#
# d) Host memory bandwidth breaks first. Eight ranks staging concurrently ask for roughly
#    4m of host traffic each, against a socket that has one memory controller complex and
#    is already serving everything else on the node; the pinned staging buffers also
#    compete for the same channels. The part that notices first is the input pipeline --
#    the data loader's decode and augmentation is host-memory-hungry and has no
#    scheduler protecting it, so the symptom arrives as "the loader is starving the GPUs",
#    which sends people to tune workers and prefetch depth. The collective itself is
#    protected by overlap and looks fine until the host is fully saturated.
