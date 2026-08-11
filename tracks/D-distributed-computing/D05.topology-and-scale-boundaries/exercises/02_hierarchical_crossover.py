"""Exercise 2 - flat ring against a two-level all-reduce, and where they cross.

The interesting fact about the hierarchical algorithm is that it does NOT move fewer
bytes. Patarasuk and Yuan proved 2m(P-1)/P per rank is a lower bound, and the
two-level version hits it exactly. What it changes is which tier carries the bytes.
This drill makes you compute both costs, prove the byte identity, and then find the
node count where the two-level version starts winning.

Your job:
  TODO 1  implement flat_ring_cost
  TODO 2  implement two_level_cost, with a cross-node term that is ZERO at nodes == 1
  TODO 3  implement crossover_nodes by scanning for a bracket and then bisecting
  TODO 4  read the output and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
That gap is the exercise. Worked solution with commentary: ../solutions/README.md

Run: python 02_hierarchical_crossover.py
"""

BANNER = ("analytic collective model, not measured: exact ring arithmetic on assumed "
          "tier bandwidths")

GPUS_PER_NODE = 8
MESSAGE_BYTES = 1073741824.0        # 1 GiB per rank
INTRA_BW = 100e9                    # bytes/s on the intra-node fast link
INTER_BW = 25e9                     # bytes/s per GPU across the network
ALPHA_INTRA = 1e-6
ALPHA_INTER = 3e-6

# The crossover only exists if the extra phases cost something. With zero overhead the
# two-level algorithm ties at one node and wins everywhere above it, so a crossover
# hunt has nothing to find -- see question (c).
PHASE_OVERHEAD = 20e-3

NODE_LADDER = (1, 2, 8, 64, 128, 512)
GIB = 1024.0**3


def flat_ring_cost(nodes, gpus_per_node, nbytes, intra_bw, inter_bw, alpha_intra,
                   alpha_inter, phase_overhead=0.0):
    """TODO 1: cost of one ring over all P = nodes * gpus_per_node ranks.

    Return a dict with keys total_bytes, slow_bytes, slow_steps, seconds.

        steps        = 2 * (P - 1)
        total_bytes  = 2 * nbytes * (P - 1) / P        per rank
        seconds      = total_bytes / bandwidth + steps * alpha + phase_overhead

    The pacing rule is the point of the drill: a ring is one chain, so it runs at the
    speed of its SLOWEST link. If nodes > 1 at least one link leaves a node, so the
    whole ring runs at inter_bw and ALL of total_bytes is slow-tier. If nodes == 1
    every link is intra-node: use intra_bw, and slow_bytes and slow_steps are zero.

    Return zeros when P <= 1: an all-reduce over one rank is a no-op, and a formula
    that returns a negative time there wins every comparison silently.
    """
    return None


def two_level_cost(nodes, gpus_per_node, nbytes, intra_bw, inter_bw, alpha_intra,
                   alpha_inter, phase_overhead=0.0):
    """TODO 2: reduce-scatter in node, all-reduce per rail across nodes, all-gather.

    Return a dict with the same four keys, so the two are directly comparable.

        intra_steps = 2 * (gpus_per_node - 1)
        intra_bytes = 2 * nbytes * (gpus_per_node - 1) / gpus_per_node
        inter_steps = 2 * (nodes - 1)
        inter_bytes = 2 * (nbytes / gpus_per_node) * (nodes - 1) / nodes

        seconds = intra_bytes/intra_bw + intra_steps*alpha_intra
                + inter_bytes/inter_bw + inter_steps*alpha_inter
                + phases * phase_overhead

    Phase 2 is `gpus_per_node` concurrent node-count all-reduces, one per rail, and
    the rails are disjoint switch trees -- so its cost is that of ONE of them, not the
    sum. At nodes == 1 the cross-node term must be exactly zero, never negative, and
    the algorithm degenerates to the same intra-node ring the flat version runs.

    total_bytes is intra_bytes + inter_bytes, and the identity to check is that it
    equals the flat ring's 2m(P-1)/P exactly.
    """
    return None


def crossover_nodes(gpus_per_node, nbytes, intra_bw, inter_bw, alpha_intra,
                    alpha_inter, phase_overhead=0.0, ceiling=1 << 20):
    """TODO 3: smallest node count at which the two-level cost strictly wins.

    `flat.seconds - two_level.seconds` is increasing in the node count: the flat ring
    adds 2*gpus_per_node slow steps per extra node where the two-level version adds 2,
    and the flat ring's slow-tier byte term is gpus_per_node times larger. So:

      * SCAN the powers of two up to `ceiling` for the first one where two-level wins,
        keeping the previous rung as a lower bound
      * then BISECT between the two, keeping the invariant "low loses, high wins"
      * return None when two-level never wins in range -- the honest answer when the
        tiers are equally fast and the phase overhead is real

    Do not compare the two times for strict inequality without slack: they are EQUAL
    at nodes == 1 when the overhead is zero, and floating point will pick a winner
    from the last bit of a division if you let it.
    """
    return None


def describe(cost, unit):
    if cost is None:
        return "not implemented"
    return (f"{cost['slow_bytes'] / unit:>9.4f}{cost['slow_steps']:>7}"
            f"{cost['seconds'] * 1e3:>10.2f}")


def main():
    print(BANNER)
    print(f"G = {GPUS_PER_NODE}, m = {MESSAGE_BYTES / GIB:g} GiB per rank, intra "
          f"{INTRA_BW / 1e9:g} GB/s, inter {INTER_BW / 1e9:g} GB/s, per-phase overhead "
          f"{PHASE_OVERHEAD * 1e3:g} ms\n")

    header = (f"{'nodes':>6}{'ranks':>7}{'flat GiB':>10}{'steps':>7}{'flat ms':>10}"
              f"{'2L GiB':>10}{'steps':>7}{'2L ms':>10}{'winner':>11}")
    print(header)
    print("-" * len(header))
    for nodes in NODE_LADDER:
        args = (nodes, GPUS_PER_NODE, MESSAGE_BYTES, INTRA_BW, INTER_BW, ALPHA_INTRA,
                ALPHA_INTER, PHASE_OVERHEAD)
        flat = flat_ring_cost(*args)
        two = two_level_cost(*args)
        if flat is None or two is None:
            print(f"{nodes:>6}{nodes * GPUS_PER_NODE:>7}  not implemented (TODO 1, 2)")
            continue
        gap = flat["seconds"] - two["seconds"]
        winner = "tie" if abs(gap) < 1e-15 else ("two-level" if gap > 0 else "flat")
        print(f"{nodes:>6}{nodes * GPUS_PER_NODE:>7}"
              f"{describe(flat, GIB)}{describe(two, GIB)}{winner:>11}")
    print("-" * len(header))

    print("\nthe byte identity (must hold at every node count, exactly):")
    for nodes in NODE_LADDER:
        args = (nodes, GPUS_PER_NODE, MESSAGE_BYTES, INTRA_BW, INTER_BW, ALPHA_INTRA,
                ALPHA_INTER)
        flat = flat_ring_cost(*args)
        two = two_level_cost(*args)
        if flat is None or two is None:
            print(f"  nodes={nodes:<5} not implemented")
            continue
        delta = abs(flat["total_bytes"] - two["total_bytes"])
        verdict = "identical" if delta < 1e-6 else f"DISAGREE by {delta:g} bytes"
        print(f"  nodes={nodes:<5} flat {flat['total_bytes'] / GIB:.6f} GiB, "
              f"two-level {two['total_bytes'] / GIB:.6f} GiB -> {verdict}")

    print("\nthe crossover, and what moves it:")
    scenarios = (
        ("1 GiB, tier ratio 4x", MESSAGE_BYTES, INTER_BW),
        ("4 KiB, tier ratio 4x", 4096.0, INTER_BW),
        ("1 GiB, tier ratio 1x", MESSAGE_BYTES, INTRA_BW),
    )
    for label, nbytes, inter_bw in scenarios:
        crossover = crossover_nodes(GPUS_PER_NODE, nbytes, INTRA_BW, inter_bw,
                                    ALPHA_INTRA, ALPHA_INTER, PHASE_OVERHEAD)
        shown = "not implemented (TODO 3), or two-level never wins" if crossover is None \
            else f"N = {crossover}"
        print(f"  {label:<22} -> {shown}")
    zero_overhead = crossover_nodes(GPUS_PER_NODE, MESSAGE_BYTES, INTRA_BW, INTER_BW,
                                    ALPHA_INTRA, ALPHA_INTER, 0.0)
    print(f"  {'same, no phase overhead':<22} -> "
          f"{'not implemented, or never wins' if zero_overhead is None else f'N = {zero_overhead}'}")

    print("\nrejection checks (each must raise ValueError, not return a number):")
    cases = (
        ("nodes = 0", lambda: flat_ring_cost(0, 8, 1.0, INTRA_BW, INTER_BW, 0.0, 0.0)),
        ("nodes = -4", lambda: two_level_cost(-4, 8, 1.0, INTRA_BW, INTER_BW, 0.0, 0.0)),
        ("gpus_per_node = 0",
         lambda: two_level_cost(4, 0, 1.0, INTRA_BW, INTER_BW, 0.0, 0.0)),
        ("inter_bw = 0", lambda: flat_ring_cost(4, 8, 1.0, INTRA_BW, 0.0, 0.0, 0.0)),
    )
    for label, produce in cases:
        try:
            outcome = produce()
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        except ZeroDivisionError:
            outcome = "ZeroDivisionError -- validate instead of dividing"
        else:
            outcome = "no ValueError raised" if outcome is None else "returned a cost"
        print(f"  {label:<20} -> {outcome}")

    print("\nTODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) Total bytes per rank are identical and Patarasuk and Yuan proved that")
    print("     count is a lower bound. State exactly where the speedup comes from.")
    print("  b) Raise the inter-node bandwidth to the intra-node bandwidth. Say what")
    print("     happens to the advantage and why, in one sentence, from the formulas.")
    print("  c) With zero phase overhead the crossover is pinned at 2 nodes. Explain")
    print("     what that says about the model, and what a real library pays that this")
    print("     model does not charge for.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
