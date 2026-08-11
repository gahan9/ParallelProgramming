"""Solution 4 - what a rank mapping costs, in links and in fabric load.

Same harness as ../exercises/04_rank_placement_cost.py with the four TODOs filled in.
Commentary, and the answers in full, are in README.md next to this file.

Run: python 04_rank_placement_cost.py
"""

import random

NODES = 64
GPUS_PER_NODE = 8
RAILS = 8
MESSAGE_BYTES = 1073741824.0
SEED = 7
GIB = 1024.0**3

BANNER = ("analytic placement model, not measured: exact link counts on assumed tier "
          "bandwidths")

MAPPINGS = ("block", "cyclic", "random")


def placement(mapping, nodes, gpus_per_node, seed=0):
    """Rank-indexed list of (node, local_index) for one mapping."""
    if nodes < 1:
        raise ValueError(f"nodes must be at least 1, got {nodes}")
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be at least 1, got {gpus_per_node}")
    ranks = nodes * gpus_per_node
    if mapping == "block":
        return [(r // gpus_per_node, r % gpus_per_node) for r in range(ranks)]
    if mapping == "cyclic":
        return [(r % nodes, r // nodes) for r in range(ranks)]
    if mapping == "random":
        slots = [(n, g) for n in range(nodes) for g in range(gpus_per_node)]
        random.Random(seed).shuffle(slots)
        return slots
    raise ValueError(
        f"unknown mapping {mapping!r}: choose from {', '.join(MAPPINGS)}")


def ring_link_counts(slots, rails):
    """Return (cross_node, cross_rail) for a flat ring over these slots."""
    if rails < 1:
        raise ValueError(f"rails must be at least 1, got {rails}")
    ranks = len(slots)
    if ranks <= 1:
        return 0, 0
    cross_node = cross_rail = 0
    for r in range(ranks):
        node_a, local_a = slots[r]
        node_b, local_b = slots[(r + 1) % ranks]
        if node_a != node_b:
            cross_node += 1
            if (local_a % rails) != (local_b % rails):
                cross_rail += 1
    return cross_node, cross_rail


def fabric_load(cross_node, nbytes, ranks):
    """Total cross-node bytes the job pushes into the fabric."""
    if ranks <= 1:
        return 0.0
    if cross_node < 0 or nbytes < 0.0:
        raise ValueError(
            f"link count and message size must be non-negative, got {cross_node} "
            f"links and {nbytes} bytes")
    return cross_node * 2.0 * nbytes * (ranks - 1) / ranks


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

    print("\nTODO 4 answers are in README.md, section 'Drill 4'.")


if __name__ == "__main__":
    main()

# TODO 4 answers (full reasoning in README.md):
# a) The pace is a maximum over links and the fabric load is a sum over them. A flat
#    ring's slowest link was already inter-node under block mapping, so making the rest
#    inter-node too does not slow the chain in a contention-free model. The number for
#    the bug report is the fabric load: 8x the bytes on the contended tier, which
#    becomes congestion, backpressure and variance rather than a clean slowdown.
# b) Cyclic puts consecutive ranks on adjacent NODES but at the same local index
#    (local = r // N changes only once per pass), so its 512 cross-node flows are
#    same-rail one-leaf-hop flows and only G = 8 of them cross a rail. Block crosses a
#    node boundary just 64 times, but each of those joins local G-1 to local 0, so all
#    64 are cross-rail. Random destroys both properties at once: 506 cross-node and 438
#    cross-rail. Block, every time -- 8x less fabric load, and the answer depends on the
#    collective because a HIERARCHICAL algorithm needs intra-node locality to exist at
#    all, which cyclic and random do not provide.
# c) Snake the ring: traverse node k's GPUs 0..G-1, then node k+1's G-1..0, alternating.
#    Every node boundary then joins two ranks with the SAME local index, so the
#    cross-node count stays at N and the cross-rail count falls to zero. Nothing moved
#    between devices; only the order the ring visits them changed.
