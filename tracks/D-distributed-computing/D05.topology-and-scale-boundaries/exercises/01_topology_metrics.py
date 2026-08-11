"""Exercise 1 - diameter and bisection, and the dimension-count trade.

Two numbers characterize any fabric: the diameter, which bounds the latency term, and
the bisection, which is a hard ceiling on every all-to-all pattern. This drill makes
you compute both from the graph rather than from a vendor slide, then reproduce the
N = 4096 table in section 3.3 of the module README -- the table that says a 6D torus
reaches full bisection with half the diameter of a 3D one, and twelve links per node.

Your job:
  TODO 1  implement torus_metrics, rejecting inputs the shape has no meaning for
  TODO 2  implement fat_tree_metrics
  TODO 3  implement all_to_all_floor_seconds, the section 2.2 lower bound
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
That gap is the exercise. Worked solution with commentary: ../solutions/README.md

Run: python 01_topology_metrics.py
"""

BANNER = ("analytic topology model, not measured: the graph properties are exact, the "
          "link rate is your assumption")

NODES = 4096
LINK_GBS = 50.0
MESSAGE_BYTES = 1073741824.0  # 1 GiB per endpoint, for the all-to-all bound


def integer_root(value, degree):
    """Exact integer `degree`-th root of `value`, or None when there is not one.

    Provided, because the trap here is not the algebra. `round(4096 ** (1/6))` is
    4.0 only because `round` hides that the float is 3.9999999999999996, and a
    topology tool that silently accepts a side length of 3 has described a machine
    with 729 nodes instead of 4096. Search around the estimate and verify exactly.
    """
    if value < 1 or degree < 1:
        return None
    estimate = int(round(value ** (1.0 / degree)))
    for candidate in range(max(1, estimate - 2), estimate + 3):
        if candidate**degree == value:
            return candidate
    return None


def torus_metrics(nodes, dims, mesh=False):
    """TODO 1: return (diameter, bisection_links) for a d-dimensional torus or mesh.

    With side n = nodes ** (1/dims):
        diameter  = dims * (n // 2)   for a torus, because the wraparound halves the
                                      worst distance
                  = dims * (n - 1)    for a mesh, which has no wraparound
        bisection = 2 * n ** (dims - 1) link-units for a torus: cutting one dimension
                    severs TWO sheets of n**(dims-1) links, one of them the wraparound
                  = n ** (dims - 1)   for a mesh, which severs one sheet

    Raise ValueError with a ONE-LINE message naming the offending value when
    `dims < 1`, or when `nodes` is not a perfect `dims`-th power -- use integer_root
    above and do not round.
    """
    return None


def fat_tree_metrics(nodes, radix, levels, oversubscription):
    """TODO 2: return (link_traversals, bisection_fraction) for a folded Clos.

    A switch with radix k puts k//2 ports down and k//2 up, so L levels reach
    2 * (k//2)**L endpoints and the worst-case path is host-up-...-down-host:

        link_traversals    = 2 * levels          (2*levels - 1 switch traversals)
        bisection          = nodes / (2 * oversubscription)  link-units
        bisection_fraction = bisection / (nodes / 2) = 1 / oversubscription

    Note what is NOT in that first line: `nodes`. The diameter of a fat-tree is flat
    in the endpoint count within a level count, which is the whole reason latency
    tests cannot see oversubscription.

    Raise ValueError with a one-line message when `radix < 2`, when `levels < 1`, when
    `oversubscription < 1` (below 1 is not a ratio this model means), or when
    `levels` levels of that radix cannot hold `nodes` endpoints.
    """
    return None


def all_to_all_floor_seconds(nodes, nbytes, bisection_links, link_gbs):
    """TODO 3: the section 2.2 lower bound on a uniform all-to-all, in seconds.

    Every endpoint spreads `nbytes` over all destinations, half of which are across
    any balanced cut, so nodes * nbytes / 2 bytes must cross it:

        seconds >= (nodes * nbytes / 2) / (bisection_links * link_gbs * 1e9)

    This is a bound, not an estimate: it holds independently of the routing, the
    collective algorithm and the quality of the library. Return a float, and treat
    1 GB as 1e9 bytes as every link vendor does.
    """
    return None


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

    print("\nTODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) The 6D torus reaches full bisection at 4096 nodes with half the")
    print("     diameter of the 3D one. State what it costs, and why anyone still")
    print("     builds the 3D version.")
    print("  b) Take the 3D torus from 4096 to 262144 nodes (64^3). Compute the new")
    print("     bisection fraction, and state the all-to-all penalty as a power of N.")
    print("  c) The 2:1 fat-tree has the same diameter as the 1:1 one. Name the")
    print("     measurement that distinguishes them, and the one that cannot.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
