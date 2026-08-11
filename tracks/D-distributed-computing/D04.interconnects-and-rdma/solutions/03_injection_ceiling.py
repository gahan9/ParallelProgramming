"""Solution 3 - the injection ceiling, and the adapter count that sets it.

The three TODOs from ../exercises/03_injection_ceiling.py, filled in, with the reasoning
in ../solutions/README.md. The harness is unchanged from the starter.

Run: python 03_injection_ceiling.py
"""

import math

BANNER = ("analytic model, not measured: line-rate adapters, no PCIe limit between GPU\n"
          "and adapter, no oversubscription above the node. The real number is lower.")

GB = 1e9

NIC_GBS = 25.0
SCALEUP_GBS = 400.0

GPU_COUNTS = (2, 4, 8, 16)
NIC_COUNTS = (1, 2, 4, 8)


def per_gpu_injection(n_nic, nic_bps, n_gpu):
    """Bytes per second of network bandwidth each GPU gets, shared evenly.

    Zero adapters is a defined error, not a zero: a node with no adapter cannot leave the
    node at all, and returning 0.0 would read like a measured share of nothing rather than
    a configuration to fix.
    """
    if n_nic < 1:
        raise ValueError("a node needs at least one adapter to leave the node, got %d"
                         % n_nic)
    if n_gpu < 1:
        raise ValueError("GPU count must be at least 1, got %d" % n_gpu)
    if nic_bps <= 0:
        raise ValueError("adapter bandwidth must be positive, got %g B/s" % nic_bps)
    return n_nic * nic_bps / n_gpu


def scaleup_ratio(n_nic, nic_bps, n_gpu, scaleup_bps):
    """Per-GPU network share as a fraction of that GPU's scale-up bandwidth."""
    if scaleup_bps <= 0:
        raise ValueError("scale-up bandwidth must be positive, got %g B/s" % scaleup_bps)
    return per_gpu_injection(n_nic, nic_bps, n_gpu) / scaleup_bps


def min_nics_for_ratio(target, nic_bps, n_gpu, scaleup_bps):
    """Fewest adapters whose per-GPU share reaches `target` times the scale-up tier.

    Solving target == n_nic * B_nic / (n_gpu * B_scaleup) gives a real number, and you
    cannot buy 3.2 adapters. Round UP: rounding down returns a count that does not meet
    the target, which is the wrong direction for a number someone will put in a purchase
    order.
    """
    if target <= 0:
        raise ValueError("target ratio must be positive, got %g" % target)
    if n_gpu < 1:
        raise ValueError("GPU count must be at least 1, got %d" % n_gpu)
    if nic_bps <= 0:
        raise ValueError("adapter bandwidth must be positive, got %g B/s" % nic_bps)
    if scaleup_bps <= 0:
        raise ValueError("scale-up bandwidth must be positive, got %g B/s" % scaleup_bps)
    return int(math.ceil(target * scaleup_bps * n_gpu / nic_bps))


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
#
# a) The PCIe path between the GPU and the adapter, which this model assumes is free. An
#    adapter only reaches line rate if it sits on a link wide enough to carry it and, for
#    the direct path, on the same switch as the GPU it serves; add adapters past the point
#    where each has its own switch-local lane budget and they start sharing bridges,
#    root-complex bandwidth and, once they land across a socket, the inter-socket link.
#    The curve bends first at the adapter whose only path to its GPU crosses a host
#    bridge -- README 3.7's misplacement failure -- and it bends again, harder, above the
#    node when the fabric is oversubscribed, which is D05's subject.
#
# b) Parity is the design goal because it is the only configuration where the per-GPU
#    network share stops depending on how many GPUs share the node, which is what makes
#    a node's off-node performance a property of the node rather than of its population.
#    It costs adapters, switch ports, lanes and power in proportion to GPU count -- an
#    8-GPU node needs eight of everything -- which is why the ratio is still far below 1
#    even at parity: matching the scale-up tier is not the goal, matching per GPU is.
#
# c) Eight GPUs with one 25 GB/s adapter gives 25/8 = 3.125 GB/s per GPU. Four GPUs with
#    four adapters gives 100/4 = 25 GB/s per GPU, eight times more. The eight-GPU node
#    wins on a workload that fits inside one node and communicates over the scale-up
#    links -- a model whose parallelism is all intra-node, or inference where the batch is
#    the unit of work -- because there the adapter is nearly idle and GPUs are the thing
#    you are buying. The four-GPU node wins on anything that crosses the node boundary
#    often: pipeline or data parallelism spanning nodes, or a model too large for one
#    node. The quantity you had to know is the workload's off-node communication per unit
#    of compute -- its arithmetic intensity measured against the network rather than
#    against memory. Without that number the two options are not comparable at all.
#
# d) The model predicts it directly: inside the node a flow uses the scale-up tier, and
#    at 8 GPUs with 1 adapter the ratio table says the off-node share is 0.0078 of that
#    tier, about 1:128. A single-node sweep never touches the adapter, so it measures the
#    tier that is not the constraint; the moment the job spans two nodes every rank is
#    funnelled into one adapter and the aggregate falls to the injection ceiling. A
#    quarter of expected is the shape this predicts, though the exact factor depends on
#    how much traffic actually crosses. The experiment that confirms injection rather
#    than fabric: run the same two-node test with one rank pair, then two, then four,
#    then eight, all on the same node pair, and watch the AGGREGATE. If it is injection,
#    aggregate flattens at the node's n_nic * B_nic and per-pair bandwidth falls as 1/k;
#    if it were the fabric, one pair alone would already be slow. A second confirmation
#    is to move half the ranks to a third node: injection improves, fabric does not.
