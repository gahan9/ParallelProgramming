# D05 Solutions — Topology & the Scale Boundary

Worked answers for the four drills in [`../exercises/`](../exercises/), with the reasoning
that matters. Read your own attempt first: the code here is short, and the commentary is
the part being assessed.

Every block below is the body a `TODO` asked for, dropped into the starter file unchanged.
The runnable versions sit next to this file — [`01_topology_metrics.py`](01_topology_metrics.py),
[`02_hierarchical_crossover.py`](02_hierarchical_crossover.py),
[`03_oversubscription_check.py`](03_oversubscription_check.py),
[`04_rank_placement_cost.py`](04_rank_placement_cost.py) — and every table here is that
file's own output. Nothing was typed by hand, and nothing was measured on a cluster.

| Solution | Drill | The one thing to take away |
|---|---|---|
| [Drill 1](#drill-1--diameter-and-bisection) | Diameter and bisection | Bisection is a *minimum over cuts*, and dimensions buy it at the price of per-node ports |
| [Drill 2](#drill-2--the-hierarchical-crossover) | Hierarchical crossover | The bytes are identical; only the *tier* changes, so the win is exactly the tier ratio |
| [Drill 3](#drill-3--the-oversubscription-cliff) | Oversubscription | The cliff is at the uplink count and nowhere near where anybody tests |
| [Drill 4](#drill-4--what-a-mapping-costs) | Placement cost | Fabric load moves by `G` while the modelled pace does not move at all |

---

## Drill 1 — diameter and bisection

```python
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
```

```
arrangement              links/node  diameter  bisection  of full  a2a floor ms
-------------------------------------------------------------------------------
3D torus 16^3                     6        24        512    25.0%        85.899
4D torus 8^4                      8        16       1024    50.0%        42.950
6D torus 4^6                     12        12       2048   100.0%        21.475
-------------------------------------------------------------------------------

fat-tree, three levels of radix 64:
  1:1  6 link traversals, 100.0% of full bisection
  2:1  6 link traversals, 50.0% of full bisection
```

**Why `integer_root` instead of `round(nodes ** (1/dims))`?** Because the float root of a
perfect power is not reliably the integer. `4096 ** (1/6)` evaluates to
`3.9999999999999996`, and `4000 ** (1/3)` evaluates to `15.874...`, which rounds to 16 —
so a rounding implementation accepts 4,000 nodes as a `16^3` cube and reports the
bisection of a 4,096-node machine. That is not a floating-point curiosity, it is a wrong
answer about a different computer. Search a small window around the estimate and verify
`candidate ** degree == value` in integers.

**Why is the torus bisection `2 * n^(d-1)` and not `n^(d-1)`?** Because cutting one
dimension of a *torus* severs two sheets of links, not one: the direct links crossing the
cut plane, and the wraparound links that close the ring the other way round. A mesh has no
wraparound and severs one sheet, which is also why its diameter is `d*(n-1)` instead of
`d*(n//2)`. The factor of two is the whole reason a 6D torus reaches full bisection at
4,096 nodes: `2 * 4^5 = 2048 = N/2`.

**Why does `fat_tree_metrics` not take `nodes` into its diameter at all?** Because it does
not appear there. `2 * levels` link traversals is the worst case for *any* pair in the
machine, and adding endpoints under the same level count does not add a hop. Latency and
scale are decoupled in a Clos and coupled in a torus, and that single structural fact is
what sections 2.1 and 3.1 are for.

**The questions.**

- **(a)** The 6D torus costs **twelve links per node instead of six** — twice the per-node
  port count, twice the cables, twice the routing state, on every one of 4,096 nodes. The
  3D machine is still cheaper per node and is *exactly as good* at nearest-neighbour halo
  exchange, which is what its workloads do: a stencil code's communication graph is the
  torus, one hop, no contention, at any size. Dimensions buy **global** operations, and a
  code that runs none is being charged for something it will not use. That trade is the
  argument in [Ajima et al.](../../../../docs/REFERENCES.md#tofu-6d-meshtorus-interconnect-ajima-et-al-2009).
- **(b)** At `64^3 = 262144` nodes the bisection is `2 * 64^2 = 8192` link-units against a
  full bisection of `131072`, so **6.25%** — a quarter of the 4,096-node fraction, for a
  64x larger machine. The all-to-all bound grows as `N^(1/d)`, so `N^(1/3)` at `d = 3`:
  `64^(1/3) = 4x` slower for the same per-node message. This is question 2 of the
  conceptual set with the arithmetic done.
- **(c)** A **multi-pair bandwidth test with more concurrent cross-leaf senders than the
  leaf has uplinks** distinguishes them. Nothing latency-shaped can, and that is not a
  matter of degree: the two fabrics have *identical* diameters, so ping, small-message
  latency and any two-rank test return the same number on both. Drill 3 is the same fact
  from the other side.

---

## Drill 2 — the hierarchical crossover

```python
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
```

```
 nodes  ranks  flat GiB  steps   flat ms    2L GiB  steps     2L ms     winner
------------------------------------------------------------------------------
     1      8   0.0000      0     38.80   0.0000      0     58.80       flat
     2     16   1.8750     30    100.62   0.1250      2     84.18  two-level
     8     64   1.9688    126    104.94   0.2188     14     88.24  two-level
    64    512   1.9961   1022    108.80   0.2461    126     89.75  two-level
   128   1024   1.9980   2046    111.95   0.2480    254     90.22  two-level
   512   4096   1.9995   8190    130.45   0.2495   1022     92.59  two-level
------------------------------------------------------------------------------

the byte identity (must hold at every node count, exactly):
  nodes=128   flat 1.998047 GiB, two-level 1.998047 GiB -> identical

the crossover, and what moves it:
  1 GiB, tier ratio 4x   -> N = 2
  4 KiB, tier ratio 4x   -> N = 953
  1 GiB, tier ratio 1x   -> N = 953
  same, no phase overhead -> N = 2
```

(The table above carries the drill's `PHASE_OVERHEAD = 20 ms`, which is why the flat ring
wins the one-node row. Without it the two rows are an exact tie.)

**Why is the byte identity exact rather than approximate?** Because it is algebra, not
accounting:

```
2m(G-1)/G + 2m(N-1)/(NG) = 2m[ N(G-1) + (N-1) ] / (NG)
                         = 2m(NG - 1)/(NG)
                         = 2m(P-1)/P
```

The module states it as "about `2m`" for both; it is in fact the same number to the last
bit, and a drill that reports a difference has a term wrong. This is what makes the
hierarchical algorithm interesting: [Patarasuk and
Yuan](../../../../docs/REFERENCES.md#bandwidth-optimal-all-reduce-patarasuk-and-yuan-2009)
proved `2m(P-1)/P` is a lower bound, and the two-level algorithm neither beats it nor
loses to it.

**Why does the flat ring use one bandwidth for the whole chain?** Because a ring is a
chain and a chain runs at the pace of its slowest link. Under any mapping with `N > 1`, at
least one of its links leaves a node, so the *entire* ring proceeds at the inter-node
rate — including the `(G-1)/G` of its hops that are between GPUs in the same chassis and
could have run an order of magnitude faster. That waste is not a bug in the ring; it is
the thing hierarchical collectives exist to recover.

**Why must `nodes == 1` be handled explicitly?** Two reasons, and the second is the
interesting one. First, `2*(nodes-1)` is 0 there and `(nodes-1)/nodes` is 0, so a
carelessly written cross-node term can go negative and win every comparison silently.
Second, at one node the two algorithms are *the same algorithm*: the two-level version's
phase 2 is empty and its phases 1 and 3 are exactly the flat intra-node ring. They tie,
and a tool that declares a winner there is reporting the last bit of a division.

**Why is the crossover pinned at `N = 2` with no phase overhead?** Because in this model
the two-level algorithm weakly dominates. It moves identical total bytes, it puts a
`1/G` share of them on the slow tier, and it takes `2(N+G-2)` slow-tier-and-fast-tier
steps against the flat ring's `2(NG-1)`. With `intra_bw >= inter_bw` there is no node
count above one where flat can win, so there is nothing for a crossover to find. Getting
this answer is *correct*; the module's section 3.6 narrative about flat winning "for a
handful of nodes" is only true once the extra phases cost something, which is why the
drill and `tools/collective_hierarchy.py` both carry an explicit
`--phase-overhead-us` term. Give the phases a cost and the crossover appears and moves
with both the message size and the tier ratio, exactly as the module says.

**The questions.**

- **(a)** From **which links carry the bytes**, not how many bytes there are. The identity
  above says the counts are equal, so the lower bound is respected. The flat ring puts all
  `2m(P-1)/P` of them on the slow tier; the two-level version puts `2m/G` there and hands
  `2m(G-1)/G` to the intra-node link. At `G = 8` that is an 8x reduction in slow-tier
  bytes, which is why the modelled times differ by about 3x rather than not at all.
- **(b)** It **collapses**. With `intra_bw == inter_bw` both algorithms move identical
  bytes over identical-rate links, so the byte terms cancel exactly and only the step
  counts are left: `2(N + G - 2)` against `2(NG - 1)`. The remaining advantage is latency,
  worth 1.00x to 1.02x in the shipped scenario. The entire benefit was the tier bandwidth
  ratio, which is a property of the machine and not of the algorithm.
- **(c)** It says the model **charges nothing for structure**. Three phases with two extra
  synchronization boundaries cost real launches, real barriers and real buffer management;
  a real library also pays for a second reduction order (so bit-exactness across
  algorithms is gone) and for uneven node sizes, where phase 2 waits for the largest
  shard. Those costs are what put a library's threshold at a node count above two, and
  they are properties of the implementation, not of the ring arithmetic. The honest
  version of "hierarchical wins above N" is "above N *on this library and this machine*",
  which is why section 3.6 sends you to the
  [MPICH decision-tree paper](../../../../docs/REFERENCES.md#collective-algorithms-in-mpich-thakur-rabenseifner-and-gropp-2005)
  rather than quoting a constant.

---

## Drill 3 — the oversubscription cliff

```python
def achievable_rate(hosts_per_leaf, uplinks, host_rate, uplink_rate,
                    concurrent_senders, local_fraction=0.0):
    """Per-sender off-leaf rate in GB/s with `concurrent_senders` hosts sending."""
    check_rates(host_rate, uplink_rate, local_fraction)
    if uplinks < 1:
        raise ValueError(f"uplinks must be at least 1, got {uplinks}")
    if concurrent_senders < 1:
        raise ValueError(
            f"concurrent senders must be at least 1, got {concurrent_senders}")
    if concurrent_senders > hosts_per_leaf:
        raise ValueError(
            f"{concurrent_senders} concurrent senders on a leaf with "
            f"{hosts_per_leaf} hosts")
    offered = host_rate * (1.0 - local_fraction)
    return min(offered, uplinks * uplink_rate / concurrent_senders)


def saturation_point(uplinks, host_rate, uplink_rate, local_fraction=0.0):
    """Largest sender count the uplink pool still carries at the offered rate."""
    check_rates(host_rate, uplink_rate, local_fraction)
    if uplinks < 1:
        raise ValueError(f"uplinks must be at least 1, got {uplinks}")
    offered = host_rate * (1.0 - local_fraction)
    return int((uplinks * uplink_rate) // offered)


def implied_bisection_fraction(hosts_per_leaf, uplinks, host_rate, uplink_rate):
    """Fraction of full bisection left for cross-leaf traffic: 1/ratio, capped at 1."""
    check_rates(host_rate, uplink_rate, 0.0)
    if hosts_per_leaf < 1 or uplinks < 1:
        raise ValueError(
            f"hosts and uplinks must be at least 1, got {hosts_per_leaf} hosts and "
            f"{uplinks} uplinks")
    ratio = (hosts_per_leaf * host_rate) / (uplinks * uplink_rate)
    return min(1.0, 1.0 / ratio)


def classify(hosts_per_leaf, uplinks, host_rate, uplink_rate, concurrency,
             local_fraction=0.0):
    """Verdict for one traffic pattern on one leaf."""
    check_rates(host_rate, uplink_rate, local_fraction)
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")
    ratio = (hosts_per_leaf * host_rate) / (uplinks * uplink_rate)
    if ratio <= 1.0:
        return "non-blocking"
    if concurrency <= saturation_point(uplinks, host_rate, uplink_rate,
                                       local_fraction):
        return "safe for this pattern"
    return "oversubscribed and exposed"
```

```
 senders k  uplink demand  per-host GB/s  of line rate
------------------------------------------------------
         1             50          50.00        100.0%
         8            400          50.00        100.0%
        16            800          50.00        100.0%
        17            850          47.06         94.1%
        32           1600          25.00         50.0%
------------------------------------------------------

saturation point: 16
implied bisection for cross-leaf traffic: 50.0% of full
```

**Why does `saturation_point` refuse the host count?** Because capacity is a property of
the uplinks and nothing else. `floor(U * R_up / R_host)` is how many hosts the pool can
carry at line rate; adding hosts to the leaf changes how many *could* send, not how many
the uplinks can serve. An implementation whose saturation point moves with the host count
has modelled the sharing rule — fairness — rather than the limit, and it will then report
a cliff that moves when you plug in a server, which is not a thing that happens. Leaving
the parameter out of the signature makes the mistake unwriteable.

**Why is the first *degraded* sender count `saturation + 1` and not `saturation`?** Because
at `k = 16` the demand is `16 * 50 = 800 GB/s` and the pool is exactly `800 GB/s`: fully
saturated and still delivering the offered rate to every sender. The pool is at its limit
and nobody has lost anything yet. The cliff is one integer wide, which is exactly why a
table of powers of two walks straight past it.

**Why three verdicts from `classify` rather than two?** Because "oversubscribed" describes
the switch and not the outcome. A 4:1 leaf carrying a pattern that keeps three quarters of
its bytes local is a *good design* — it saved half the optics and the spine ports and
costs its workload nothing. The same leaf under an all-to-all is a permanent tax. Naming
the middle case is what stops the tool from turning a legitimate engineering decision into
an alarm, and it is also the case a procurement document should be arguing about.

**The questions.**

- **(a)** It licenses exactly one claim: **one host can reach line rate off-leaf when
  nothing else is sending.** It says nothing whatever about the ratio, because `k = 1`
  sits below the saturation point of every leaf ever built — a 1:1 leaf and a 4:1 leaf
  return the identical number. If the report's conclusion was "the fabric is
  non-blocking", the evidence does not reach it.
- **(b)** Raise the **concurrency**, not the message size or the node count: have all 32
  hosts on the leaf send off-leaf simultaneously and compare the per-host rate against
  line rate. A bigger message does not help because the cliff is a *capacity* limit rather
  than a protocol effect — one host sending 1 GiB still only occupies one host-link's
  worth of the pool, and the other fifteen uplinks stay idle no matter how long it sends.
  The OSU multi-pair tests give you the concurrency knob directly.
- **(c)** The radix counts every port as if it were an uplink, so it reports the
  **non-blocking** bisection: 2,048 link-units where the as-built figure is 1,024. The
  error is **high by exactly the oversubscription ratio** — 2x here, 4x on the 8-uplink
  variant — and high is the dangerous direction, because it is the direction in which a
  design gets approved. Section 2.3 flags this as the most common way a bisection number
  in a design document is wrong.

---

## Drill 4 — what a mapping costs

```python
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
```

```
mapping     cross-node  cross-rail   fabric GiB   spine GiB  vs block
---------------------------------------------------------------------
block               64          64       127.75      127.75     1.00x
cyclic             512           8      1022.00       15.97     8.00x
random             506         438      1010.02      874.29     7.91x
---------------------------------------------------------------------
```

**Where the counts come from, by hand.** The ring's links are `(r, (r+1) mod P)` for every
`r`, the wrap included. Under **block**, `node(r) = r // G` changes exactly when
`(r+1) % G == 0`, which happens `N` times — one per node boundary — and each of those
links joins local index `G-1` to local index `0`. Under **cyclic**, `node(r) = r % N`
changes at *every* link, so all `P` are cross-node, while `local(r) = r // N` changes only
when `(r+1) % N == 0`, which is `G` times. Check both on `N = 3, G = 2`, where the ring is
six links long and you can write the whole thing out; a 512-rank table cannot tell you
whether you got the wrap link right.

**Why is cross-rail counted only among cross-node links?** Because two GPUs in the same
chassis talk over the intra-node
[fast link](../../../../docs/GLOSSARY.md#nvlink-xgmi-and-infinity-fabric) and never reach
a leaf switch at all. Asking which rail they are on is asking about a switch neither of
them used. A count that includes intra-node links marks every link of every mapping as
cross-rail and measures nothing.

**Why seed the shuffle?** Because this table is cited by the module's exercise criteria,
and the repository does not publish a number it cannot reproduce. `random.Random(seed)`
also keeps the drill's randomness out of the global generator, so a caller's own
random state is untouched — which matters the moment this function lands in a larger tool.

**The surprise, stated plainly.** Cyclic mapping is 8x worse on fabric load and *better*
on cross-rail — 8 links against block's 64. That inverts the direction section 3.6
originally suggested, and it is worth understanding rather than memorizing: cyclic keeps
consecutive ranks at the same local index, and same-index cross-node traffic is precisely
what a [rail-optimized](../../../../docs/GLOSSARY.md#rail-optimized-network) fabric makes
cheap. Block crosses a node boundary only `N` times but does it between mismatched
indices every time. The mapping that is bad on *both* axes is **random**, which is what a
scheduler hands you when nothing has asked for locality. The module's section 9 drill text
and section 3.6 were corrected against this output.

**The questions.**

- **(a)** The pace is a **maximum** over links and the fabric load is a **sum** over them.
  A flat ring's slowest link was already inter-node under block mapping, so making the
  remaining links inter-node too does not slow the chain in a contention-free
  single-tenant model — hence the identical 88.80 ms. What changed by a factor of `G` is
  the bytes the job pushes onto the contended tier: 127.75 GiB against 1,022 GiB. That is
  the number for the bug report, because congestion, backpressure, interference with other
  tenants and adaptive-routing non-determinism all scale with it, and they show up as a
  worse mean *and* much higher variance rather than as a clean slowdown.
- **(b)** Both facts follow from the index arithmetic above: cyclic changes the node on
  every link and the local index only `G` times, so its cross-node flows are same-rail
  one-leaf-hop flows; block changes the node `N` times and always between local `G-1` and
  local `0`, so all of its cross-node links climb to the spine. Random destroys both
  properties at once — 506 cross-node and 438 cross-rail. I would rather be handed
  **block**: 8x less fabric load is the larger effect, and the answer depends on the
  collective because a *hierarchical* algorithm needs intra-node locality to exist at all.
  Under cyclic or random there is no such thing as "the ranks in this node", so the
  two-level algorithm of drill 2 is not merely less effective, it is strictly worse than
  flat — it has added phases and kept the slow tier.
- **(c)** **Snake the ring.** Traverse node `k`'s GPUs `0..G-1`, then node `k+1`'s
  `G-1..0`, alternating. Every node boundary then joins two ranks with the *same* local
  index, so the cross-node count stays at `N` and the cross-rail count falls to **zero**.
  Not one rank moved between devices; only the order in which the ring visits them
  changed. This is the cheapest topology-aware optimization in the module, and it is the
  reason a collective library wants to know the fabric shape rather than just the rank
  count.

---

## Grading yourself

You have the right answer if you can defend all four of these:

1. **Every rejected input has a one-line message that names the value.** A tool built on
   these functions must be able to print it and exit non-zero. If a user sees a traceback,
   the function was written for its author.
2. **Every edge case is deliberate.** `nodes == 1` in drill 2, `k = saturation` in drill 3,
   the ring's wrap link in drill 4, a node count that is not a perfect power in drill 1:
   state what each returns and why that is the honest answer.
3. **You separated the exact part from the assumed part.** Diameters, bisections and link
   counts are exact graph properties. Every millisecond in these tables is one of those
   multiplied by a bandwidth somebody chose. Saying which is which is most of what makes a
   model usable in a review.
4. **You said what the model does not know.** These four drills know nothing about
   congestion trees, adaptive-routing non-determinism, other tenants, protocol switches,
   or a collective library's real decision tree. Naming one and saying what you would
   measure to bound it is the difference between a model and a claim.

## Three results worth remembering past this module

**1. Oversubscription is invisible below the uplink count, by construction.** Not
"hard to see" — invisible. Diameter, latency and every two-rank number are identical on a
1:1 and a 4:1 leaf. The only test that distinguishes them is a concurrency test, and the
only defence is knowing the as-built uplink ratio.

**2. The hierarchical all-reduce's win is the tier ratio, and nothing else.** Identical
total bytes, `1/G` of them on the slow tier. Set the tiers equal and the advantage
collapses to a step count; break intra-node locality and it inverts. Any collective
optimization that appears to beat a proven lower bound has this shape of explanation.

**3. Placement changes a sum, not a maximum.** That is why an idealized model says it does
not matter and production says it does. Measure bytes on the contended tier, log the
rank-to-host-to-device mapping every run, and treat "we changed nothing and it got slower"
as a placement hypothesis until the log rules it out.

## Verifying the arithmetic

```powershell
# from the module directory
make run-exercises                     # all four harnesses
make run-solutions                     # the same four, filled in
python tools/topology_metrics.py --compare --nodes 4096
python tools/collective_hierarchy.py --bytes 1000000000 --max-nodes 128
python tools/oversubscription.py --hosts-per-leaf 32 --uplinks 16 --host-gbs 50 --uplink-gbs 50
python tools/rank_placement.py --nodes 64 --gpus-per-node 8 --compare-all --seed 7
```

Remember what these are: **analytic models, not measurements**, each saying so in its own
banner. The graph properties are exact; the times are exact formulas multiplied by assumed
link rates. They are right about ratios, orderings and orders of magnitude, and silent
about congestion, other tenants, and everything else that decides whether a real job
finishes.
