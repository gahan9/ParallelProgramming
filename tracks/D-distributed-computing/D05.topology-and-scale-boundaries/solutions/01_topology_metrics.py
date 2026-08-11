"""Solution 1 - diameter and bisection, and the dimension-count trade.

Same harness as ../exercises/01_topology_metrics.py with the four TODOs filled in.
Commentary, and the answers in full, are in README.md next to this file.

Run: python 01_topology_metrics.py
"""

BANNER = ("analytic topology model, not measured: the graph properties are exact, the "
          "link rate is your assumption")

NODES = 4096
LINK_GBS = 50.0
MESSAGE_BYTES = 1073741824.0  # 1 GiB per endpoint, for the all-to-all bound


def integer_root(value, degree):
    """Exact integer `degree`-th root of `value`, or None when there is not one."""
    if value < 1 or degree < 1:
        return None
    estimate = int(round(value ** (1.0 / degree)))
    for candidate in range(max(1, estimate - 2), estimate + 3):
        if candidate**degree == value:
            return candidate
    return None


def torus_metrics(nodes, dims, mesh=False):
    """Return (diameter, bisection_links) for a d-dimensional torus or mesh."""
    if dims < 1:
        raise ValueError(f"dims must be at least 1, got {dims}")
    if nodes < 1:
        raise ValueError(f"nodes must be at least 1, got {nodes}")
    side = integer_root(nodes, dims)
    if side is None:
        raise ValueError(
            f"no integer side length satisfies n**{dims} == {nodes}: nodes {nodes} "
            f"cannot form a {dims}-dimensional cube")
    if mesh:
        return dims * (side - 1), float(side ** (dims - 1))
    return dims * (side // 2), 2.0 * side ** (dims - 1)


def fat_tree_metrics(nodes, radix, levels, oversubscription):
    """Return (link_traversals, bisection_fraction) for a folded Clos."""
    if radix < 2:
        raise ValueError(f"radix must be at least 2, got {radix}")
    if levels < 1:
        raise ValueError(f"levels must be at least 1, got {levels}")
    if oversubscription < 1:
        raise ValueError(
            f"oversubscription must be at least 1 (1 means non-blocking), "
            f"got {oversubscription}")
    capacity = 2 * (radix // 2) ** levels
    if capacity < nodes:
        raise ValueError(
            f"{levels} levels of radix {radix} hold {capacity} endpoints, "
            f"not {nodes}")
    return 2 * levels, 1.0 / oversubscription


def all_to_all_floor_seconds(nodes, nbytes, bisection_links, link_gbs):
    """Section 2.2's lower bound on a uniform all-to-all, in seconds."""
    if bisection_links <= 0.0 or link_gbs <= 0.0:
        raise ValueError(
            f"bisection and link rate must be positive, got {bisection_links} "
            f"link-units at {link_gbs} GB/s")
    return (nodes * nbytes / 2.0) / (bisection_links * link_gbs * 1e9)


def main():
    print(BANNER)
    print(f"N = {NODES} endpoints, B_link = {LINK_GBS:g} GB/s, all-to-all message "
          f"1 GiB per endpoint")
    print(f"full bisection = N/2 = {NODES // 2} link-units\n")

    header = (f"{'arrangement':<24}{'links/node':>11}{'diameter':>10}"
              f"{'bisection':>11}{'of full':>9}{'a2a floor ms':>14}")
    print(header)
    print("-" * len(header))
    for dims in (3, 4, 6):
        result = torus_metrics(NODES, dims)
        side = integer_root(NODES, dims)
        label = f"{dims}D torus {side}^{dims}"
        if result is None:
            print(f"{label:<24}  not implemented (TODO 1)")
            continue
        diameter, bisection = result
        floor = all_to_all_floor_seconds(NODES, MESSAGE_BYTES, bisection, LINK_GBS)
        floor_ms = "n/a" if floor is None else f"{floor * 1e3:.3f}"
        print(f"{label:<24}{2 * dims:>11}{diameter:>10}{bisection:>11g}"
              f"{bisection / (NODES / 2) * 100:>8.1f}%{floor_ms:>14}")
    print("-" * len(header))

    print("\nfat-tree, three levels of radix 64:")
    for oversubscription in (1, 2):
        result = fat_tree_metrics(NODES, 64, 3, oversubscription)
        if result is None:
            print(f"  {oversubscription}:1  not implemented (TODO 2)")
            continue
        traversals, fraction = result
        print(f"  {oversubscription}:1  {traversals} link traversals, "
              f"{fraction * 100:.1f}% of full bisection")

    print("\nsection 3.3 assertions (each must hold, or one of us has a bug):")
    checks = [
        ("3D torus at 4096 is 25% of full bisection", lambda: torus_metrics(4096, 3),
         lambda r: abs(r[1] / 2048 - 0.25) < 1e-12),
        ("6D torus at 4096 is 100% of full bisection", lambda: torus_metrics(4096, 6),
         lambda r: abs(r[1] / 2048 - 1.0) < 1e-12),
        ("3D torus diameter is 24", lambda: torus_metrics(4096, 3),
         lambda r: r[0] == 24),
        ("6D torus diameter is 12", lambda: torus_metrics(4096, 6),
         lambda r: r[0] == 12),
        ("16x16x16 MESH diameter is 45, not 24",
         lambda: torus_metrics(4096, 3, mesh=True), lambda r: r[0] == 45),
        ("2:1 fat-tree keeps 6 traversals and halves bisection",
         lambda: fat_tree_metrics(4096, 64, 3, 2),
         lambda r: r[0] == 6 and abs(r[1] - 0.5) < 1e-12),
    ]
    for label, produce, holds in checks:
        try:
            value = produce()
        except ValueError as exc:
            print(f"  {label:<52} ValueError: {exc}")
            continue
        if value is None:
            print(f"  {label:<52} not implemented")
        else:
            print(f"  {label:<52} {'OK' if holds(value) else 'FAILS'}")

    print("\nrejection checks (each must raise ValueError, not return a number):")
    cases = (
        ("torus_metrics(4000, 3)", lambda: torus_metrics(4000, 3)),
        ("torus_metrics(4096, 0)", lambda: torus_metrics(4096, 0)),
        ("torus_metrics(0, 3)", lambda: torus_metrics(0, 3)),
        ("fat_tree_metrics(4096, 1, 3, 1)", lambda: fat_tree_metrics(4096, 1, 3, 1)),
        ("fat_tree_metrics(4096, 64, 3, 0.5)",
         lambda: fat_tree_metrics(4096, 64, 3, 0.5)),
        ("fat_tree_metrics(4096, 64, 1, 1)", lambda: fat_tree_metrics(4096, 64, 1, 1)),
    )
    for label, produce in cases:
        try:
            outcome = produce()
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        else:
            outcome = "no ValueError raised" if outcome is None else f"returned {outcome}"
        print(f"  {label:<36} -> {outcome}")

    print("\nTODO 4 answers are in README.md, section 'Drill 1'.")


if __name__ == "__main__":
    main()

# TODO 4 answers (full reasoning in README.md):
# a) Twelve links per node instead of six, so twice the per-node port count, twice the
#    cables and twice the routing logic -- and the 3D machine still wins on cost per
#    node and is just as good at halo exchange, which is what its workloads do. Six
#    dimensions buy global operations, and a stencil code does not run any.
# b) 64^3: bisection 2*64^2 = 8192 link-units against a full bisection of 131072, so
#    6.25% -- a quarter of the 4096-node fraction for a 64x larger machine. The
#    all-to-all bound grows as N^(1/d) = N^(1/3): 64^(1/3) = 4x slower.
# c) A multi-pair bandwidth test with enough CONCURRENT cross-leaf senders to exceed
#    the uplink count distinguishes them. Nothing latency-shaped can: identical
#    diameter means identical hop count, so ping, small-message and two-rank tests
#    return identical numbers on both fabrics.
