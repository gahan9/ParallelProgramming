"""Exercise 4 - what a rank mapping costs, in links and in fabric load.

Placement is the highest-leverage software fix in this module, and the reason it is
hard to sell is that an idealized single-tenant model says it does not matter: the
flat ring's pace is set by its slowest link and that link was already inter-node. This
drill computes the number that does move -- the bytes the job pushes onto the network,
and the subset of those that must climb to the spine.

Your job:
  TODO 1  implement the three mappings, with the random one SEEDED
  TODO 2  implement ring_link_counts: cross-node and cross-rail
  TODO 3  implement fabric_load, and the ratio against the block baseline
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
That gap is the exercise. Worked solution with commentary: ../solutions/README.md

Run: python 04_rank_placement_cost.py
"""

NODES = 64
GPUS_PER_NODE = 8
RAILS = 8
MESSAGE_BYTES = 1073741824.0
SEED = 7
GIB = 1024.0**3

BANNER = ("analytic placement model, not measured: exact link counts on assumed tier "
          "bandwidths")


def placement(mapping, nodes, gpus_per_node, seed=0):
    """TODO 1: return a rank-indexed list of (node, local_index) tuples.

        block   rank r -> (r // gpus_per_node, r % gpus_per_node)
        cyclic  rank r -> (r % nodes,          r // nodes)
        random  a seeded shuffle of the same (node, local_index) slot set

    Two things matter about the random case. It must be a PERMUTATION of the slots,
    not independent draws -- every device gets exactly one rank, which is what a
    scheduler hands you when nothing has asked for locality. And it must be seeded
    through `random.Random(seed)` rather than the module-level functions, so the number
    this file prints is reproducible and can be cited.

    Raise ValueError with a one-line message for an unknown mapping name, or for a
    node or GPU count below 1.
    """
    return None


def ring_link_counts(slots, rails):
    """TODO 2: return (cross_node, cross_rail) for a flat ring over these slots.

    The ring's links are (r, (r + 1) % len(slots)) for every r -- including the wrap,
    which is a real link and is where an off-by-one hides.

        cross_node   the two ranks are on different nodes, so the link is on the
                     network rather than the intra-node fast link
        cross_rail   the link is cross-node AND the two local indices land on
                     different rails, `local % rails`, so it cannot stay inside one
                     leaf switch and must climb to the spine

    Count cross-rail only among cross-node links. Two GPUs in one chassis talk over
    the fast link and never reach a leaf switch at all, so asking which rail they are
    on is asking about a switch neither of them used.
    """
    return None


def fabric_load(cross_node, nbytes, ranks):
    """TODO 3: total cross-node bytes the job pushes into the fabric.

    Each ring link carries 2 * nbytes * (ranks - 1) / ranks bytes over a full
    all-reduce -- about 2m -- so the fabric load is that times the cross-node link
    count. Return a float, and return 0.0 when ranks <= 1.
    """
    return None


def main():
    print(BANNER)
    ranks = NODES * GPUS_PER_NODE
    print(f"{NODES} nodes x {GPUS_PER_NODE} GPUs = {ranks} ranks, {RAILS} rails, "
          f"m = {MESSAGE_BYTES / GIB:g} GiB, random seed {SEED}\n")

    header = (f"{'mapping':<10}{'cross-node':>12}{'cross-rail':>12}"
              f"{'fabric GiB':>13}{'spine GiB':>12}{'vs block':>10}")
    print(header)
    print("-" * len(header))
    baseline = None
    for mapping in ("block", "cyclic", "random"):
        slots = placement(mapping, NODES, GPUS_PER_NODE, SEED)
        counts = None if slots is None else ring_link_counts(slots, RAILS)
        if counts is None:
            print(f"{mapping:<10}  not implemented (TODO 1, 2)")
            continue
        cross_node, cross_rail = counts
        load = fabric_load(cross_node, MESSAGE_BYTES, ranks)
        spine = fabric_load(cross_rail, MESSAGE_BYTES, ranks)
        if load is None:
            print(f"{mapping:<10}{cross_node:>12}{cross_rail:>12}"
                  f"  not implemented (TODO 3)")
            continue
        if baseline is None:
            baseline = load
        ratio = load / baseline if baseline else float("inf")
        print(f"{mapping:<10}{cross_node:>12}{cross_rail:>12}{load / GIB:>13.2f}"
              f"{spine / GIB:>12.2f}{ratio:>9.2f}x")
    print("-" * len(header))

    print("\nassertions the derivation requires (N > 1 and G > 1):")
    checks = (
        ("block gives exactly N cross-node links",
         lambda: ring_link_counts(placement("block", NODES, GPUS_PER_NODE), RAILS),
         lambda v: v[0] == NODES),
        ("and every one of them is cross-rail: local G-1 joins local 0",
         lambda: ring_link_counts(placement("block", NODES, GPUS_PER_NODE), RAILS),
         lambda v: v[1] == NODES),
        ("cyclic gives N*G cross-node links -- every link in the ring",
         lambda: ring_link_counts(placement("cyclic", NODES, GPUS_PER_NODE), RAILS),
         lambda v: v[0] == NODES * GPUS_PER_NODE),
        ("but only G of them are cross-rail: consecutive ranks share a local index",
         lambda: ring_link_counts(placement("cyclic", NODES, GPUS_PER_NODE), RAILS),
         lambda v: v[1] == GPUS_PER_NODE),
        ("cyclic fabric load is exactly G times block's",
         lambda: (ring_link_counts(placement("block", NODES, GPUS_PER_NODE), RAILS),
                  ring_link_counts(placement("cyclic", NODES, GPUS_PER_NODE), RAILS)),
         lambda v: v[1][0] == GPUS_PER_NODE * v[0][0]),
        ("random approaches cyclic on cross-node: within 5% of every link",
         lambda: ring_link_counts(placement("random", NODES, GPUS_PER_NODE, SEED),
                                  RAILS),
         lambda v: v[0] >= 0.95 * NODES * GPUS_PER_NODE),
        ("random is far worse than either on cross-rail: over half the ring",
         lambda: ring_link_counts(placement("random", NODES, GPUS_PER_NODE, SEED),
                                  RAILS),
         lambda v: v[1] > 0.5 * NODES * GPUS_PER_NODE),
        ("the random mapping is reproducible: same seed, same counts",
         lambda: (ring_link_counts(placement("random", NODES, GPUS_PER_NODE, SEED),
                                   RAILS),
                  ring_link_counts(placement("random", NODES, GPUS_PER_NODE, SEED),
                                   RAILS)),
         lambda v: v[0] == v[1]),
    )
    for label, produce, holds in checks:
        try:
            value = produce()
        except ValueError as exc:
            print(f"  {label:<66} ValueError: {exc}")
            continue
        if value is None or (isinstance(value, tuple) and value[0] is None):
            print(f"  {label:<66} not implemented")
        else:
            print(f"  {label:<66} {'OK' if holds(value) else 'FAILS'}")

    print("\nrejection checks (each must raise ValueError, not return a number):")
    cases = (
        ("mapping 'scattered'", lambda: placement("scattered", 4, 8)),
        ("nodes = 0", lambda: placement("block", 0, 8)),
        ("gpus_per_node = -8", lambda: placement("block", 4, -8)),
    )
    for label, produce in cases:
        try:
            outcome = produce()
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        else:
            outcome = "no ValueError raised" if outcome is None else "returned a mapping"
        print(f"  {label:<24} -> {outcome}")

    print("\nTODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) Cyclic mapping multiplies the fabric load by G and yet an idealized")
    print("     model shows almost no change in the ring's pace. Reconcile that, and")
    print("     say which number belongs in the bug report.")
    print("  b) Cyclic has G times block's cross-node links but FEWER cross-rail ones,")
    print("     while random is worse than both. Explain both facts from the mappings,")
    print("     and say which mapping you would rather be handed on a rail-optimized")
    print("     fabric and why the answer depends on the collective.")
    print("  c) Block mapping's cross-node links are all cross-rail here. Describe a")
    print("     ring ORDERING that drives the cross-rail count to zero without moving")
    print("     a single rank between devices.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
