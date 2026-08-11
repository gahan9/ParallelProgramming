# D05 Exercises — Topology & the Scale Boundary

Four coding drills, one per number that decides whether a machine's *shape* suits your
traffic: the diameter, the bisection, the oversubscription ratio, and the rank-to-device
mapping. They are the "Coding & Algorithms" drills from
[section 9](../README.md#9-self-assessment--interview-drills) of the module, turned into
starter files with TODOs.

Each file **runs as it ships**. It prints its table with `not implemented` in every cell
until you fill the TODOs in, so you always have a working harness and never a blank page.
Standard library only — no numpy, no torch, nothing to install.

| Drill | File | What it tests |
|---|---|---|
| 1 | [`01_topology_metrics.py`](01_topology_metrics.py) | Torus and fat-tree diameter and bisection, and the section 2.2 all-to-all bound |
| 2 | [`02_hierarchical_crossover.py`](02_hierarchical_crossover.py) | Flat ring against two-level, the byte identity, and the crossover by bisection |
| 3 | [`03_oversubscription_check.py`](03_oversubscription_check.py) | The concurrency cliff, the implied bisection, and a three-way verdict |
| 4 | [`04_rank_placement_cost.py`](04_rank_placement_cost.py) | Cross-node and cross-rail ring links under block, cyclic and seeded-random mappings |

## How to work these

1. **Do the arithmetic by hand first.** Every formula you need is derived in the module
   README: [section 2](../README.md#2-first-principles) for the two characterizing
   numbers and the oversubscription table,
   [section 3](../README.md#3-deep-dive) for the per-topology formulas, the hierarchical
   algorithm and the placement argument. Four nodes and two GPUs on paper will catch
   every off-by-one in drill 4, and a tool cannot.
2. **Then check against the tools.** `tools/topology_metrics.py`,
   `tools/collective_hierarchy.py`, `tools/oversubscription.py` and
   `tools/rank_placement.py` implement the same formulas, so a disagreement means one of
   the two has a bug. Find out which before moving on.
3. **Answer the questions at the bottom of each file.** They are the part a design review
   actually asks. The code gets you the number; the question is what the number licenses
   you to say.

```powershell
# From the module directory (one level up):
make run-exercises                        # runs all four harnesses
python exercises/01_topology_metrics.py
python tools/topology_metrics.py --help
```

## Acceptance criteria

A drill is done when its harness prints these, and you can defend the answers at the
bottom of the file. Every figure below is what the reference solution prints, so a
mismatch is a bug in one of the two.

**Drill 1 — diameter and bisection.**

- The `N = 4096` table reproduces [section 3.3](../README.md#3-deep-dive) exactly:
  3D `16^3` is 6 links per node, diameter **24**, bisection **512** link-units, **25%**
  of full; 4D `8^4` is 8, **16**, **1,024**, **50%**; 6D `4^6` is 12, **12**, **2,048**,
  **100%**. All three agree with
  `python tools/topology_metrics.py --compare --nodes 4096`.
- At 50 GB/s per link and 1 GiB per endpoint the all-to-all floor is **85.899 ms** on the
  3D torus, **42.950 ms** on the 4D and **21.475 ms** on the 6D. The 6D figure equals
  `m / B_link`, which is section 2.2's full-bisection result: independent of `N`.
- The `16x16x16` **mesh** diameter is **45**, not 24. If you get 24, `mesh=True` is not
  reaching the formula.
- Both fat-tree rows report **6 link traversals**. The 1:1 row is 100% of full bisection
  and the 2:1 row is 50%. Identical diameter, half the bisection — that is the entire
  reason oversubscription hides from latency tests.
- All six rejection cases raise `ValueError` with a one-line message that names the
  offending value. `torus_metrics(4000, 3)` must reject rather than round: 4,000 is not a
  cube, and `int(round(4000 ** (1/3)))` is 16, which would silently model a
  4,096-node machine.

**Drill 2 — the hierarchical crossover.**

- Total bytes per rank are **identical**, at every node count, to the last bit — not
  "within tolerance". `2m(G-1)/G + 2m(N-1)/(NG)` collapses algebraically to
  `2m(NG-1)/(NG)`, so a difference of even one byte means a term is wrong.
- At `N = 128`, `G = 8`, `m = 1 GiB`, intra 100 GB/s, inter 25 GB/s: the flat ring puts
  **1.998 GiB** and **2,046 steps** on the slow tier, the two-level version **0.248 GiB**
  and **254 steps**. That is the `G`-fold difference, and it is the column that predicts
  the time.
- With `--bytes 1000000000` (1 GB, decimal, which is what section 3.6's worked table
  uses) `python tools/collective_hierarchy.py --bytes 1000000000 --max-nodes 128`
  reproduces that table: flat **86.06 ms**, two-level **28.20 ms**, a **3.05x**
  difference. The lab command's `--bytes 1073741824` is 1 GiB and therefore 7.4% larger;
  both are correct, and neither is measured.
- The crossover with **zero** phase overhead is `N = 2` for every message size and every
  tier ratio, and `N = 1` is an exact **tie** — at one node the two-level algorithm's
  cross-node phase is empty and it *is* the flat intra-node ring. This is a property of
  the model, not a bug: see question (c).
- With the shipped `PHASE_OVERHEAD = 20 ms` the crossover moves, which is the point of
  the drill: **`N = 2`** at 1 GiB, **`N = 953`** at 4 KiB, and **`N = 953`** at 1 GiB once
  the tier ratio is 1x. Message size and tier ratio both move it; the overhead is what
  gives them something to move.
- `nodes == 1` returns a zero cross-node term, never a negative one, and the two costs are
  reported as a tie rather than both as winners.

**Drill 3 — the oversubscription cliff.**

- The 32-host, 16-uplink, 50 GB/s leaf reproduces
  [section 2.3](../README.md#2-first-principles) exactly: **50.00 GB/s** per host at
  k = 1, 8 and 16; **47.06 GB/s** (94.1%) at k = 17; **25.00 GB/s** (50%) at k = 32.
  50 GB/s is 400 Gb/s, so these are the README's Gb/s figures in the repo's GB/s
  convention.
- `saturation_point` returns **16** — the uplink count — and it has no host-count
  parameter at all. Halving the uplinks gives **8**; doubling the uplink rate gives
  **32**; 75% locality gives **64**. If your saturation point moves when hosts are added,
  you have modelled fairness instead of capacity.
- `implied_bisection_fraction` is **50%** for this leaf. A 2:1 leaf halves the fabric's
  bisection for every byte that crosses a leaf.
- `classify` returns **non-blocking** for a 1:1 leaf, **oversubscribed and exposed** for
  this leaf at 32 concurrent senders, and **safe for this pattern** for the same leaf
  when 75% of the bytes stay local. All three matter: "oversubscribed" describes the
  switch, not the outcome.
- All four rejection cases raise `ValueError`. Zero senders and 33 senders on a 32-host
  leaf are both nonsense, and neither may produce a division by zero or an infinity.

**Drill 4 — what a mapping costs.**

- At 64 nodes x 8 GPUs, 8 rails, seed 7: **block** gives 64 cross-node links and 64
  cross-rail; **cyclic** gives 512 and 8; **random** gives 506 and 438. Fabric load is
  **1.00x / 8.00x / 7.91x** of the block baseline. These match
  `python tools/rank_placement.py --nodes 64 --gpus-per-node 8 --compare-all --seed 7`.
- Cyclic's cross-node count is exactly `nodes * gpus_per_node` — *every* link in the ring
  — and block's is exactly `nodes`, one per node boundary. That factor of `G` is the
  number that belongs in a bug report.
- The cross-rail column is the one that surprises people, and it is worth deriving by
  hand before you trust it. Under cyclic, `local = r // N` changes only once per pass
  around the nodes, so consecutive ranks share a local index and their cross-node flows
  stay **inside one rail**: only `G` of the 512 cross a rail. Under block, every node
  boundary joins local index `G-1` to local index `0`, so all `nodes` of its cross-node
  links **are** cross-rail. Block still wins overall by 8x on fabric load; it is simply
  not free on the spine, and the README's section 9 drill text was corrected to say so.
- The random mapping is **reproducible**: two calls with the same seed give identical
  counts. An unseeded shuffle can illustrate but never support, and this repository does
  not publish a number it cannot reproduce.
- The modelled ring **pace** is the same 88.80 ms for all three mappings. That is not a
  bug in the model, it is the finding: an idealized single-tenant model is the wrong
  instrument for a placement question.

## Shared assumptions unless a drill says otherwise

| Quantity | Value | Why this one |
|---|---|---|
| Diameter | worst-case hops on a shortest path | bounds the `alpha` term of [the alpha-beta model](../../../../docs/GLOSSARY.md#the-alpha-beta-model) |
| Bisection | minimum over balanced cuts, in units of one link rate | [bisection bandwidth](../../../../docs/GLOSSARY.md#bisection-bandwidth); a minimum, never an average |
| Fat-tree | `k//2` down and `k//2` up per switch, `2L` link traversals | folded Clos, after [Leiserson 1985](../../../../docs/REFERENCES.md#fat-trees-leiserson-1985) |
| Torus | `diameter = d*(n//2)`, `bisection = 2*n^(d-1)` | wraparound halves the distance and doubles the cut sheets; the dimension trade is [Ajima et al. 2009](../../../../docs/REFERENCES.md#tofu-6d-meshtorus-interconnect-ajima-et-al-2009) |
| Dragonfly | `a = 2p = 2h` balanced, 3 hops minimal | [Kim et al. 2008](../../../../docs/REFERENCES.md#dragonfly-topology-kim-et-al-2008) |
| Ring all-reduce | `2(P-1)` steps, `2m(P-1)/P` bytes per rank | [ring all-reduce](../../../../docs/GLOSSARY.md#ring-all-reduce); the byte count is a proven lower bound ([Patarasuk and Yuan 2009](../../../../docs/REFERENCES.md#bandwidth-optimal-all-reduce-patarasuk-and-yuan-2009)) |
| Tier bandwidths | intra-node 100 GB/s, inter-node 25 GB/s per GPU | illustrative round numbers, per module section 3.6 |
| Bandwidth unit | GB/s with 1 GB = 1e9 bytes | the convention link vendors quote |
| Rails | GPU `i` of every node on leaf switch `i` | [rail-optimized](../../../../docs/GLOSSARY.md#rail-optimized-network) wiring, module section 3.5 |

Everything these drills compute is an **analytic model, never a measurement**, and every
harness says so in its banner. The graph properties are exact and re-derivable by hand;
every *time* is an exact formula multiplied by an assumed link rate. They are silent about
congestion, adaptive-routing non-determinism, other tenants, and everything else that
decides whether a real job finishes — which is what
[section 5.2](../README.md#5-performance-analysis) gives you the experiments for.

Reference answers with commentary: [`../solutions/`](../solutions/). Read them *after* you
have written yours down — they contain the reasoning you are being assessed on, not just
the code.
