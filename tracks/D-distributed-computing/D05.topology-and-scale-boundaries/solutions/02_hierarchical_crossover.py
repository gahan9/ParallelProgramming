"""Solution 2 - flat ring against a two-level all-reduce, and where they cross.

Same harness as ../exercises/02_hierarchical_crossover.py with the four TODOs filled
in. Commentary, and the answers in full, are in README.md next to this file.

Run: python 02_hierarchical_crossover.py
"""

BANNER = ("analytic collective model, not measured: exact ring arithmetic on assumed "
          "tier bandwidths")

GPUS_PER_NODE = 8
MESSAGE_BYTES = 1073741824.0
INTRA_BW = 100e9
INTER_BW = 25e9
ALPHA_INTRA = 1e-6
ALPHA_INTER = 3e-6
PHASE_OVERHEAD = 20e-3

NODE_LADDER = (1, 2, 8, 64, 128, 512)
GIB = 1024.0**3
TIE_TOLERANCE_S = 1e-15


def validate(nodes, gpus_per_node, nbytes, intra_bw, inter_bw):
    if nodes < 1:
        raise ValueError(f"nodes must be at least 1, got {nodes}")
    if gpus_per_node < 1:
        raise ValueError(f"gpus_per_node must be at least 1, got {gpus_per_node}")
    if nbytes < 0.0:
        raise ValueError(f"bytes must be non-negative, got {nbytes}")
    if intra_bw <= 0.0 or inter_bw <= 0.0:
        raise ValueError(
            f"both tier bandwidths must be positive, got intra {intra_bw} and "
            f"inter {inter_bw}")


def flat_ring_cost(nodes, gpus_per_node, nbytes, intra_bw, inter_bw, alpha_intra,
                   alpha_inter, phase_overhead=0.0):
    """One ring over all P ranks, paced by its slowest link."""
    validate(nodes, gpus_per_node, nbytes, intra_bw, inter_bw)
    ranks = nodes * gpus_per_node
    if ranks <= 1:
        return dict(total_bytes=0.0, slow_bytes=0.0, slow_steps=0, seconds=0.0)
    steps = 2 * (ranks - 1)
    volume = 2.0 * nbytes * (ranks - 1) / ranks
    crosses_nodes = nodes > 1
    bandwidth = inter_bw if crosses_nodes else intra_bw
    alpha = alpha_inter if crosses_nodes else alpha_intra
    return dict(
        total_bytes=volume,
        slow_bytes=volume if crosses_nodes else 0.0,
        slow_steps=steps if crosses_nodes else 0,
        seconds=volume / bandwidth + steps * alpha + phase_overhead,
    )


def two_level_cost(nodes, gpus_per_node, nbytes, intra_bw, inter_bw, alpha_intra,
                   alpha_inter, phase_overhead=0.0):
    """Reduce-scatter in node, one all-reduce per rail across nodes, all-gather."""
    validate(nodes, gpus_per_node, nbytes, intra_bw, inter_bw)
    intra_steps = 2 * (gpus_per_node - 1)
    intra_bytes = 2.0 * nbytes * (gpus_per_node - 1) / gpus_per_node
    if nodes > 1:
        inter_steps = 2 * (nodes - 1)
        inter_bytes = 2.0 * (nbytes / gpus_per_node) * (nodes - 1) / nodes
        phases = 3 if gpus_per_node > 1 else 1
    else:
        inter_steps, inter_bytes = 0, 0.0
        phases = 2 if gpus_per_node > 1 else 0
    return dict(
        total_bytes=intra_bytes + inter_bytes,
        slow_bytes=inter_bytes,
        slow_steps=inter_steps,
        seconds=(intra_bytes / intra_bw + intra_steps * alpha_intra
                 + inter_bytes / inter_bw + inter_steps * alpha_inter
                 + phases * phase_overhead),
    )


def crossover_nodes(gpus_per_node, nbytes, intra_bw, inter_bw, alpha_intra,
                    alpha_inter, phase_overhead=0.0, ceiling=1 << 20):
    """Smallest node count at which the two-level cost strictly wins."""
    def wins(nodes):
        args = (nodes, gpus_per_node, nbytes, intra_bw, inter_bw, alpha_intra,
                alpha_inter, phase_overhead)
        gap = flat_ring_cost(*args)["seconds"] - two_level_cost(*args)["seconds"]
        return gap > TIE_TOLERANCE_S

    low, high, rung = 1, None, 1
    while rung <= ceiling:
        if wins(rung):
            high = rung
            break
        low = rung
        rung *= 2
    if high is None:
        return None
    while high - low > 1:
        mid = (low + high) // 2
        if wins(mid):
            high = mid
        else:
            low = mid
    return high


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

    print("\nTODO 4 answers are in README.md, section 'Drill 2'.")


if __name__ == "__main__":
    main()

# TODO 4 answers (full reasoning in README.md):
# a) From WHICH links carry the bytes. Both algorithms move 2m(P-1)/P per rank -- the
#    identity above holds exactly -- but the flat ring puts all of it on the slow tier
#    because a chain runs at its slowest link, while the two-level version leaves only
#    2m/G there and hands the rest to the intra-node link.
# b) It collapses: at intra_bw == inter_bw the two algorithms move identical bytes over
#    identical-rate links, so only the step count is left (2(N+G-2) against 2(NG-1)),
#    and the whole benefit was the tier bandwidth ratio.
# c) It says the model charges nothing for structure. Three phases with two extra
#    synchronization boundaries cost real launches, barriers and buffer handling, and a
#    real library also pays for uneven node sizes and a second reduction order. With
#    those costs at zero the two-level algorithm cannot lose above one node, so the
#    crossover a library actually tunes is a property of its overheads, not of the ring
#    arithmetic.
