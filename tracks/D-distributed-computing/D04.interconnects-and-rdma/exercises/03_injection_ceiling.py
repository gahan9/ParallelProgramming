"""Exercise 3 - the injection ceiling, and the adapter count that sets it.

The arithmetic is three lines long and it decides how a cluster is bought:

    node injection bandwidth = n_nic * B_nic
    per-GPU share            = n_nic * B_nic / n_gpu
    ratio                    = per-GPU share / scale-up bandwidth per GPU

Demand for the network scales with GPU count. Supply scales with adapter count. Those
are different numbers, and a node where they differ has a ceiling that no single-node
benchmark can see, because inside the node the binding constraint is the scale-up link
instead. That is LogP's gap parameter g (D01 section 3.2) turned into a line item.

Your job:
  TODO 1  implement per_gpu_injection
  TODO 2  implement scaleup_ratio
  TODO 3  implement min_nics_for_ratio
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" until you fill the TODOs in.
Reference implementation: ../tools/injection_model.py
Worked solution with commentary: ../solutions/README.md

Run: python 03_injection_ceiling.py
"""

BANNER = ("analytic model, not measured: line-rate adapters, no PCIe limit between GPU\n"
          "and adapter, no oversubscription above the node. The real number is lower.")

GB = 1e9

# Illustrative orders of magnitude, not a specification for any part. Only the ratio
# between the two matters here, and it is stable across generations in a way that
# neither figure is.
NIC_GBS = 25.0
SCALEUP_GBS = 400.0

GPU_COUNTS = (2, 4, 8, 16)
NIC_COUNTS = (1, 2, 4, 8)


def per_gpu_injection(n_nic, nic_bps, n_gpu):
    """TODO 1: bytes per second of network bandwidth each GPU gets, shared evenly.

    Two inputs need a decision rather than a formula:

      * n_nic == 0. A node with no adapter cannot leave the node at all. That is a
        configuration answer, not a number, so make it a DEFINED error -- raise
        ValueError with a one-line message -- rather than letting it become a division
        by zero, an inf, or a silent 0.0 that reads like a measurement.
      * n_gpu < 1, and a non-positive adapter bandwidth. Reject both the same way.
    """
    return None


def scaleup_ratio(n_nic, nic_bps, n_gpu, scaleup_bps):
    """TODO 2: the per-GPU network share as a fraction of that GPU's scale-up bandwidth.

    This ratio, not either bandwidth alone, is what decides whether a flow can cross the
    node boundary without falling a tier down README section 2.1's ladder. Reject a
    non-positive scale-up bandwidth as well.
    """
    return None


def min_nics_for_ratio(target, nic_bps, n_gpu, scaleup_bps):
    """TODO 3: fewest adapters whose per-GPU share reaches `target` times the scale-up tier.

    Rearrange TODO 2 and solve for n_nic. Then think about the type of the answer: you
    cannot buy 3.2 adapters, and an answer that rounds DOWN does not meet the target.

    Reject a non-positive target.
    """
    return None


def attempt(function, *args):
    """Report what `function` did with bad input: raised, returned, or not written yet."""
    try:
        value = function(*args)
    except ValueError as exc:
        return "ValueError: %s" % exc
    except ZeroDivisionError:
        return "ZeroDivisionError - a traceback, not a message. Handle it explicitly"
    return "no ValueError raised" if value is None else "returned %r" % (value,)


def cell(value, fmt):
    return "not impl." if value is None else fmt % value


def main():
    print(BANNER)
    print("adapter %g GB/s, scale-up %g GB/s per GPU (illustrative, not a datasheet)\n"
          % (NIC_GBS, SCALEUP_GBS))

    header = "%10s" % "GPUs" + "".join("%18s" % ("%d adapter(s)" % n)
                                       for n in NIC_COUNTS)
    print("per-GPU network share, GB/s")
    print(header)
    print("-" * len(header))
    for n_gpu in GPU_COUNTS:
        row = "%10d" % n_gpu
        for n_nic in NIC_COUNTS:
            share = per_gpu_injection(n_nic, NIC_GBS * GB, n_gpu)
            row += "%18s" % cell(None if share is None else share / GB, "%.3f")
        print(row)
    print("-" * len(header))
    print("")

    print("the same table as a ratio against the scale-up tier")
    print(header)
    print("-" * len(header))
    for n_gpu in GPU_COUNTS:
        row = "%10d" % n_gpu
        for n_nic in NIC_COUNTS:
            ratio = scaleup_ratio(n_nic, NIC_GBS * GB, n_gpu, SCALEUP_GBS * GB)
            row += "%18s" % cell(ratio, "%.4f")
        print(row)
    print("-" * len(header))
    print("")

    print("adapters needed to hit a target ratio on an 8-GPU node:")
    for target in (0.0625, 0.125, 0.25, 1.0):
        needed = min_nics_for_ratio(target, NIC_GBS * GB, 8, SCALEUP_GBS * GB)
        print("  target %6.4f -> %s adapter(s)" % (target, cell(needed, "%d")))
    print("")

    print("three properties the functions must have (check them, do not assume them):")
    base = per_gpu_injection(1, NIC_GBS * GB, 8)
    doubled = per_gpu_injection(2, NIC_GBS * GB, 8)
    more_gpus = per_gpu_injection(1, NIC_GBS * GB, 16)
    if None in (base, doubled, more_gpus):
        print("  not implemented")
    else:
        print("  linear in adapters:  doubling n_nic gives %.3fx  (want exactly 2)"
              % (doubled / base))
        print("  inverse in GPUs:     doubling n_gpu gives %.3fx  (want exactly 0.5)"
              % (more_gpus / base))
        parity = [scaleup_ratio(n, NIC_GBS * GB, n, SCALEUP_GBS * GB)
                  for n in (2, 8, 64)]
        print("  n_nic == n_gpu:      ratio is %s at 2, 8 and 64 GPUs"
              % ", ".join("%.4f" % r for r in parity))
        print("                       - constant, so at parity the GPU count drops out")
    print("")

    print("rejection checks (each must raise ValueError, never divide by zero):")
    print("  zero adapters   -> %s" % attempt(per_gpu_injection, 0, NIC_GBS * GB, 8))
    print("  zero GPUs       -> %s" % attempt(per_gpu_injection, 1, NIC_GBS * GB, 0))
    print("  zero adapter BW -> %s" % attempt(per_gpu_injection, 1, 0.0, 8))
    print("  zero scale-up   -> %s" % attempt(scaleup_ratio, 1, NIC_GBS * GB, 8, 0.0))
    print("  zero target     -> %s"
          % attempt(min_nics_for_ratio, 0.0, NIC_GBS * GB, 8, SCALEUP_GBS * GB))
    print("")

    print("TODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) Along a row the ratio doubles with every doubling of adapters, with no")
    print("     diminishing returns anywhere. Name the thing that is actually limiting")
    print("     this in a real node, which this model does not contain, and say where")
    print("     it would first bend the curve.")
    print("  b) At n_nic == n_gpu the ratio no longer depends on the GPU count. Say in")
    print("     one sentence why that is the design goal, and what it costs.")
    print("  c) A vendor offers eight GPUs with one adapter, or four GPUs with four")
    print("     adapters, at the same price. Compute the per-GPU share for both, then")
    print("     name the workload each one wins on and the quantity you had to know")
    print("     about the workload to decide.")
    print("  d) Your single-node bandwidth sweep looks healthy and your two-node")
    print("     numbers are a quarter of what you expected. Explain why this model")
    print("     predicts exactly that, and give the one experiment that confirms it is")
    print("     injection rather than the fabric.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
# d)
