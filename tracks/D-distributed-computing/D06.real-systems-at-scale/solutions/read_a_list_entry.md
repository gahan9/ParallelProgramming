# Drill 2 solution — Read a list entry like an architect

Exercise: [`../exercises/read_a_list_entry.md`](../exercises/read_a_list_entry.md).

**This solution deliberately contains no system name, no rank, no node count and no FLOPS
figure.** Not because they are secret — they are public — but because any of them would be a
snapshot of one list edition, and a curriculum that bakes in an edition-scoped number is wrong
within six months and keeps sounding confident. That is the failure mode section 2.4 of
[the module](../README.md#2-first-principles) commits itself against, and this solution is held
to the same standard as the module text.

So what follows grades **method**. It tells you what a good filled worksheet looks like, which
answers are defensible and which are not, and how to reason from whatever you found. Compare
your worksheet against the reasoning, not against a key.

---

## Part 1 — Provenance: the two cells people leave blank

Everyone fills in the system name. The two that get skipped are the **list edition** and the
**retrieval date**, and they are the two that make the rest of the sheet usable later.

The reason is not bookkeeping. A system's reported configuration changes between editions:
machines are extended, re-tuned, re-submitted with different node counts, and occasionally
submitted at partial scale. Six months from now, in a review, somebody will ask "is that
current?" and the only answers that carry weight are "yes, edition X, retrieved on date Y" or
"no, let me re-check". "I looked it up once" is not one of them.

A good provenance block also has the **site's own documentation URL** filled in, and if you
found the list row easier to read than the site's documentation, that is a signal you have not
found the site's documentation yet. The list row is a submission summary. The site publishes
user guides, node diagrams and architecture pages, and those are where node composition and
the software stack actually live.

---

## Part 2 — What each of the ten cells is really asking

### 1a, accelerator family

Easy to find, and worth less than people assume. What you actually want from this cell is the
memory capacity and bandwidth per accelerator, because those set whether your problem fits and
how fast it can be fed. Two systems with the same accelerator family and different memory
configurations are different machines for a memory-capacity-bound workload.

**"None" is a real and interesting answer.** A CPU-only design trades peak arithmetic per node
for a uniform programming model and, often, better balance — which is the whole argument of
section 3.5 of the module. If you picked such a system, your design-lesson sentence has a much
clearer shape than most.

### 1b, the CPU-to-accelerator link

This is the cell that separates a machine you can program naively from one where you will
spend a quarter on data movement. Three broad answers, in descending order of how much they
let you ignore:

- **Coherent or unified**, where host and device address the same memory without an explicit
  copy. This removes the staging tax rather than optimizing it (module section 3.3).
- **A high-bandwidth proprietary link** that is not coherent. Copies still exist, but they are
  cheap enough to hide behind compute if you overlap them.
- **PCIe.** Copies are explicit, comparatively slow, and will appear in your timeline as gaps.

If the site publishes a node block diagram, read it rather than the prose; the prose usually
says "high-bandwidth interconnect" and the diagram says which links exist and at what width.

### 2a and 2b, interconnect family and topology class

The family is usually published. **The topology class often is not**, and when it is not, that
is a finding rather than a gap — write "not published" and note that you would ask the site,
because it changes what you can predict. See the
[fat-tree, dragonfly and torus](../../../../docs/GLOSSARY.md#fat-tree-dragonfly-and-torus)
entry, and note the three consequences that follow from the answer:

| If the topology is | Then expect | Why it matters to you |
|---|---|---|
| A [fat-tree](../../../../docs/REFERENCES.md#fat-trees-leiserson-1985) | Uniform-ish hops; behaviour depends heavily on the [oversubscription](../../../../docs/GLOSSARY.md#oversubscription) ratio above the leaf | Ask for the ratio. Without it you know almost nothing |
| A [dragonfly](../../../../docs/REFERENCES.md#dragonfly-topology-kim-et-al-2008) variant | Low diameter, strong global bandwidth, and routing that matters | Placement and adaptive routing decide your tail latency |
| A [torus or mesh](../../../../docs/REFERENCES.md#tofu-6d-meshtorus-interconnect-ajima-et-al-2009) | Excellent nearest-neighbour behaviour, distance-dependent global collectives | A halo-exchange code loves it; a global all-to-all does not |

The honest reading of the topology cell is that it tells you **which workloads this machine was
bought for**. That is the sentence Part 3 is asking you to write.

### 3a and 3b, node composition — and the cell everybody skips

**3b is the most valuable cell on the sheet and the least often filled in.** Accelerators per
node divided by network adapters per node is the injection gap: the factor by which a node's
accelerators must share their path off the node. Eight accelerators behind one adapter means
that when all eight need to talk off-node at once, seven of them are waiting, and no amount of
fabric above the node repairs it. It is the term
[D01](../../D01.scaling-limits-and-communication-cost/README.md#3-deep-dive) introduces and the
`8-64 ranks` row of the ladder in section 6.4 is largely about.

If 3b is not published — it frequently is not — write "not published" and write down the
question you would ask: *how many network adapters per node, and is the path from adapter to
accelerator memory zero-copy?* Those are two questions and both are load-bearing. A node with
enough adapters and a staged copy through host memory has the injection bandwidth and cannot
use it.

### 4a and 4b, the Green500 position

Two things to take from this pair, and neither of them is the position.

First, whether the **two orderings disagree** around your system. They frequently do, and each
disagreement has a design explanation: a machine tuned for peak arithmetic per rack sits higher
on one list, and a machine designed around power delivery, cooling and data movement sits
higher on the other. Section 3.7 of the module is exactly this argument — that energy
efficiency is an architectural property and not an operational afterthought.

Second, that **both lists rank by the same benchmark**, one normalized by power. So the
disagreements tell you about power and packaging, and neither list tells you about your
workload unless your workload is dense linear algebra.

### 5a and 5b, HPCG and the ratio

HPCG is a sparse, latency- and bandwidth-sensitive solver, so the HPCG-to-HPL ratio is the
closest single public number to "how much of this machine's peak survives contact with a real
memory access pattern". It is typically a small fraction, and the interesting part is not the
value but the **comparison between architecturally different machines**, since it moves with
memory bandwidth, network latency and balance rather than with peak arithmetic.

If no HPCG result exists for your system, "not published" is the answer and the follow-up is in
Part 4(a).

---

## Part 3 — What a good design lesson sounds like

The test is a single question: **does the sentence name a constraint, or a component?**

| Not a design lesson | Why it fails |
|---|---|
| "Very fast accelerators." | Names a component. Every system on the list has fast accelerators |
| "Ranked in the top tier." | Names a position. Tells you nothing transferable |
| "Well balanced." | True of nothing in particular. Balanced against what? |

| A design lesson | Why it works |
|---|---|
| "Bought global bandwidth over peak arithmetic, because its flagship workload's step is an all-to-all." | Names the constraint and the workload that creates it |
| "Removed the host-to-device copy from the programming model rather than optimizing it, so the memory hierarchy is the design." | Names what was eliminated, not what was added |
| "Chose per-node balance over per-node peak, so a memory-bound solver keeps a larger fraction of peak." | States the trade explicitly, including what was given up |

The pattern is always: *this design gave up X to defeat constraint Y, which its intended
workload creates.* Every row of the module's section 3.8 table has that shape.

**And the "what is missing" sentence carries as much weight as the lesson.** The public record
does not usually contain achieved application efficiency, MTBF or failure rates, job-mix and
utilization data, or the operational cost of the software stack. Naming which of those you
could not find is the difference between a review and a summary — and it is why the calculator
in this module makes you *supply* an MTBF instead of pretending to know one.

---

## Part 4 — The two prose questions

### (a) What the HPL-to-HPCG ratio licenses you to say

**It licenses a bound and a comparison. It does not license a prediction.**

The bound: your sparse, collective-heavy application will not exceed the machine's HPL
efficiency, and will very likely land nearer the HPCG end. HPL is a dense factorization with
very high [arithmetic intensity](../../../../docs/GLOSSARY.md#arithmetic-intensity), regular
access and little global communication — close to a best case for any machine. Use it to
**bound**, never to predict.

The comparison: between two architecturally different machines, the one with the better ratio
is better balanced for memory- and latency-bound work. That is a real and useful signal.

What it does **not** license: any statement about *your* application. HPCG is one access
pattern with one communication structure. If your workload is dominated by an all-to-all, by
irregular gathers, by capacity rather than bandwidth, or by a Python-side critical path, the
ratio is silent about it.

If no HPCG result exists, the honest answer is that you run your own proxy: the smallest kernel
that reproduces your application's memory access pattern and its dominant collective, on an
allocation large enough to cross the node boundary at least twice — one node tells you nothing
about the fabric, and two nodes tell you almost nothing about the topology tier. A few hundred
ranks is where the placement effects the ladder's `64-1k` row describes begin to show up. And
the [Gordon Bell Prize](../../../../docs/REFERENCES.md#acm-gordon-bell-prize-acm) record is the
better public signal for what real applications had to do to run at scale, which is the
argument of module section 2.3.

### (b) When a rank gap predicts nothing, and when it predicts something

**Predicts nothing:** a workload that is sparse, latency-bound, collective-heavy, or limited by
memory capacity rather than arithmetic. A few positions on a list ordered by dense
factorization throughput carries no information about a step whose cost is an all-to-all or a
scatter into irregular memory. Two adjacent systems can differ by more than their rank gap
suggests in either direction, and the sign is not predictable from the ranking.

**Predicts something:** a workload that resembles the benchmark. Dense, large-matrix,
high-intensity arithmetic with little global communication — dense linear algebra, and to a
lesser extent the large GEMMs inside a compute-bound training step. Even then, read it as "in
the same band", not as a ratio, because the submitted run was tuned by people whose job that
was, on the whole machine, with a problem size chosen to flatter it.

The general rule, which is worth carrying out of this drill: **HPL orders machines by the thing
machines are best at.** That is a legitimate thing to measure and a poor thing to plan with.

---

## What this drill trains you to notice

That the easiest thing to find on the page is the least useful thing on it to know, and that
the two cells nobody fills in — adapters per node, and the retrieval date — are the two that
decide whether your notes are worth anything in a month. When you are done, the sheet should
read like a small architecture review: what this machine was built to defeat, what it gave up,
and what you could not learn about it from the outside.
