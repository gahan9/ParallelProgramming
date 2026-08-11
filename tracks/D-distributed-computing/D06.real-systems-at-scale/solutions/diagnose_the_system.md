# Drill 3 solution — Diagnose the system

Exercise: [`../exercises/diagnose_the_system.md`](../exercises/diagnose_the_system.md).

Eight scenarios, eight diagnoses. Read your own answers first: the diagnosis is the easy half
and the **discriminating measurement** is the half that separates an engineer from a confident
guesser, so grade yourself on that column rather than on the constraint.

Nothing here is a measurement from any real system. The scenarios are constructed to be
internally consistent with the cost model in
[D01 section 3](../../D01.scaling-limits-and-communication-cost/README.md#3-deep-dive) and the
ladder in [section 6.4](../README.md#6-challenges-drawbacks--tradeoffs) of the module.

## The answers at a glance

| # | Scale | Binding constraint | Ladder row | Ambiguous with |
|---|---|---|---|---|
| 1 | 8, one node | Intra-node link on the critical path, at a bad compute-to-communication ratio | 1 node | Undivided per-token work (Amdahl) |
| 2 | 64, eight nodes | Injection gap: adapters per node, not the fabric | 8-64 | Leaf oversubscription |
| 3 | 512 | Placement across topology tiers | 64-1k | Contention from neighbours; one sick node |
| 4 | 4k, small messages | Latency term times step count | 1k-10k | Straggler variance |
| 5 | 4k, compute-heavy | Straggler variance | 1k-10k | **One sick node** — a different fix |
| 6 | 16k, all-to-all | Bisection bandwidth and oversubscription | 10k-100k | Congestion tail; the model's own blind spot |
| 7 | 32k, month-long | Goodput: interruptions, detection, lost work | 100k+ | Queue wait and non-failure stalls |
| 8 | 1k, a claim | **Not a systems problem** | none | nothing |

Five of the eight are genuinely ambiguous. If you marked fewer than three you were
pattern-matching on the scale number; the scale narrows the candidates and never picks between
them.

---

## 1. Eight ranks, one node

**Binding constraint.** The `1 node` row: the intra-node accelerator links
([NVLink, xGMI, Infinity Fabric](../../../../docs/GLOSSARY.md#nvlink-xgmi-and-infinity-fabric))
carrying a per-token all-reduce that sits on the critical path. Note the precise statement,
because it decides the fix: the link is not *defective*, the **ratio** is wrong. Tensor
parallelism divides the arithmetic per token by the number of ranks while the all-reduce cost
per token stays roughly flat or grows, so each doubling buys less compute per unit of
communication.

**Discriminating measurement.** Sweep a **zero-byte** all-reduce over 2, 4 and 8 ranks inside
the node, then repeat with the real message size. If the zero-byte time already grows with rank
count, you are paying step count and not bytes. Then the confirming half: hold the rank count
at 8 and **double the work per token** (a larger batch). If throughput scaling improves
markedly, the diagnosis is the compute-to-communication ratio; if it does not, suspect the link
path itself, and check whether the collective is going peer-to-peer or being staged through
host memory.

**Mitigation, from the `1 node` row.** Raise arithmetic intensity per collective — larger
batches, fused layers, fewer and bigger reductions. Keep tensor parallelism at the width where
it still pays and get the rest of the throughput from replicas instead
([B07](../../../B-gpu-ml-performance/B07.distributed-training-and-sharding/README.md) covers
the split). Use coherent or unified memory where the platform offers it so the collective is
not staged through the host.

**Ambiguous with.** Undivided per-token work — sampling, detokenization, a Python-side critical
path — which does not divide at all and produces the same flattening curve. **Test the zero-byte
sweep first**, because it takes minutes and cleanly separates "communication grows" from
"something does not divide". A profile that shows the all-reduce on the critical path is
consistent with both.

---

## 2. Sixty-four ranks, eight nodes

**Binding constraint.** The `8-64` row, and specifically the **injection gap** rather than the
fabric. The clue is the sentence most people read past: aggregate achieved bandwidth rises when
nodes are added and does not move when more accelerators per node are put to work. That is the
signature of a per-node ceiling. Eight accelerators behind one adapter share one path off the
node, so seven of them wait, and the fabric above is not involved.

**Discriminating measurement.** Hold the node count fixed and vary how many accelerators per
node participate in an isolated collective, on a quiet allocation. If aggregate achieved
bytes per second **plateaus at the same value** for 2, 4 and 8 ranks per node, the ceiling is
the adapter. Compare that plateau against the adapter's line rate: landing near it confirms
injection, landing well below it points at the path to accelerator memory instead — a staged
copy through host memory produces the same plateau at a lower number.

**Mitigation, from the `8-64` row.** One adapter per accelerator, or as close as the node design
allows. RDMA directly to and from device memory rather than a staged copy
([D04](../../D04.interconnects-and-rdma/README.md)). Then bucket and overlap the transfers so
what injection you have is busy continuously rather than in bursts.

**Ambiguous with.** Oversubscription at the leaf, which also bends the curve at the node
boundary. It is distinguishable and cheaply: leaf oversubscription gets **worse as you add
nodes**, while an injection ceiling is flat in node count and flat in ranks per node. Run the
per-node test first — it needs two nodes and no scheduler cooperation.

---

## 3. Five hundred and twelve ranks

**Binding constraint.** The `64-1k` row: topology tier and
[oversubscription](../../../../docs/GLOSSARY.md#oversubscription). Two identical runs differing
by 1.35x with nothing rebuilt is the canonical placement symptom, and the operator's aside
about one run landing inside a rack is the whole diagnosis handed over in passing.

**Discriminating measurement.** **Log the rank-to-node mapping on every run** — this should
already be happening, and the fact that it was an aside from the operator means it was not.
Then force the comparison: run the same job twice with placement constrained, once packed
within a switch group and once deliberately spread, and compare. If wall-clock follows the
placement rather than the calendar, placement is confirmed. If both placements are now slow, or
both fast, you have found something time-varying and you are in the alternative below.

**Mitigation, from the `64-1k` row.**
[Topology-aware](../../D05.topology-and-scale-boundaries/README.md#6-challenges-drawbacks--tradeoffs)
rank ordering so that the ranks exchanging the most data are nearest each other; placement
constraints in the job request so the allocation shape is not left to chance;
[rail-optimized](../../../../docs/GLOSSARY.md#rail-optimized-network) fabric if you are still
designing rather than using; and keeping collectives inside a switch group where the algorithm
allows.

**Ambiguous with.** Two others, both consistent with the same 1.35x. **Contention from
neighbouring jobs** on shared links, which varies with what else is running rather than with
your placement. And **one slow node** in the wider allocation, which would show up as a single
outlier rather than as a uniform slowdown. The mapping log discriminates all three at once and
costs nothing, which is why it goes first: it is a read from data you should already have, and
it turns a mystery into a comparison.

---

## 4. Four thousand ranks, small messages

**Binding constraint.** The `1k-10k` row, latency term. A ring all-reduce takes `2(N-1)` steps,
so the latency term is `2(N-1) * alpha` and grows **linearly in rank count** regardless of
message size. Going from 1,024 to 4,096 multiplies it by roughly four while compute divides by
four, so the crossover arrives, and step time goes up. The observation that quoted bandwidth is
nowhere near saturated is the confirmation, not a puzzle: at a few kilobytes there are not
enough bytes for bandwidth to be the issue.

**Discriminating measurement.** The **zero-byte collective swept over rank count**. With no
bytes in flight, everything left is `steps * alpha`. If the zero-byte time grows roughly
linearly with N and is the same order as the step-time growth you are trying to explain, latency
is confirmed and bandwidth is ruled out in the same experiment.

**Mitigation, from the `1k-10k` row.** Reduced-step and hierarchical collectives: for small
messages, an algorithm with `O(log N)` steps beats a ring's `2(N-1)`, which is precisely the
algorithm-selection argument in
[Thakur, Rabenseifner and Gropp](../../../../docs/REFERENCES.md#collective-algorithms-in-mpich-thakur-rabenseifner-and-gropp-2005).
Reduce inside the node first and only then across nodes. Then reduce the *number* of
collectives: fuse several small reductions into one, and consider whether every step needs a
global reduction at all.

**Ambiguous with.** Straggler variance, which also grows with N. The discriminator is in the
numbers already given: compute fell 4x, and the variance term scales with the step it multiplies,
so variance should have got **cheaper** in absolute milliseconds, not more expensive. A term
that grows in absolute terms while the step shrinks is the latency term. The zero-byte sweep
settles it either way.

---

## 5. Four thousand ranks, compute-heavy step

**Binding constraint.** The `1k-10k` row, straggler variance. A barrier pays the **slowest**
rank, so the step time is the expected maximum of 4,096 draws rather than their mean, and a 9%
excess over the slowest single-rank compute is the right order for modest per-rank jitter. The
detail that pins it down is that the excess **grew only a few percent** from 1,024 to 4,096
ranks: the expected maximum grows slowly, sub-linearly, with the number of draws. A term that
quadrupled would have been latency; a term that crept is variance.

**Discriminating measurement.** **Per-rank arrival timestamps at one barrier, plotted as a
histogram rather than reported as a mean.** The shape answers the question that decides the fix:

- A **continuous right tail spread across many ranks**, with the identity of the slowest rank
  changing from step to step, is fleet-wide jitter.
- A **single isolated mode**, reappearing at the same rank every step, is one sick node wearing
  a distribution's clothing.

The confirming test is to drain the suspect node and re-run. If the 9% collapses to something
near the modelled expected-maximum tax, it was one node. If it stays at 9%, it is the fleet.

**Mitigation, from the `1k-10k` row.** Health-gated scheduling so a node that is measurably
slower than the fleet median never enters an allocation. Then the sources of jitter that are
yours to control: clock and core pinning, interrupt and daemon placement, power and thermal
management left in a fixed mode rather than an adaptive one.

**Ambiguous with.** The single sick node, and this is the most important ambiguity in the drill
because **the two have different fixes**. The dashboard's 6%-above-median node is real evidence
and it is not proof: 6% on one node out of 512 cannot by itself produce a 9% step penalty at
4,096 ranks if the rest of the fleet is tight, so the honest reading is that both are present
and you do not yet know the split. Test the histogram first: it is a read of data you can
already collect, it distinguishes both causes in one look, and draining a node costs capacity.

---

## 6. Sixteen thousand ranks, all-to-all

**Binding constraint.** The `10k-100k` row:
[bisection bandwidth](../../../../docs/GLOSSARY.md#bisection-bandwidth), with the
`64-1k` row's oversubscription as the mechanism. An all-to-all moves data across every cut in
the machine by construction, so it is the one pattern that consumes bisection directly. The
symptom that names it is that the job gets **worse when the allocation is spread** — more of
the traffic has to cross tiers that are narrower than the leaf.

**Discriminating measurement.** Run the identical all-to-all at a **fixed rank count**, packed
within one switch group and then spread over several, and compare. If time tracks the number of
group crossings, bisection and oversubscription are confirmed. Second measurement, to size it:
compare the achieved aggregate rate across the cut against the fabric's specified
oversubscription ratio. A 3x gap against a 4:1 fabric is roughly the ratio; a 3x gap against a
non-blocking fabric is a different problem.

**Mitigation, from the `10k-100k` row.** Avoid the global pattern rather than tuning it:
two-phase or hierarchical exchange that keeps most traffic inside a group and sends only a
digest between groups. Place the communicating partners — experts, in a routed model — so that
the heavy exchanges land inside a group. And where the algorithm permits, a
communication-avoiding formulation that trades arithmetic for cut traffic. Buying a wider
fabric is the design-time answer and is not available to you mid-run.

**Ambiguous with.** Two things, and one of them is your own tool. **Congestion tail latency and
incast**: an all-to-all creates many-to-one bursts, and if adaptive routing is not engaging, or
congestion control is mistuned, you get a multiple of the modelled time without any
oversubscription being involved. And the
[alpha-beta model](../../../../docs/GLOSSARY.md#the-alpha-beta-model) **does not model
contention at all**, so "3x what the model predicts" partly measures the model. The
[Slingshot measurement study](../../../../docs/REFERENCES.md#slingshot-interconnect-analysis-de-sensi-et-al-2020)
is the reference for how much congestion control and routing change real fabric behaviour.
Test the packed-versus-spread comparison first: it is a scheduler flag, needs no code change,
and its two possible outcomes point at different halves of the machine.

---

## 7. Thirty-two thousand ranks, a month-long job

**Binding constraint.** The `100k+` row arriving early: **goodput**. Every per-step number is
fine, which is exactly the point — the step time is not the problem and measuring it harder will
not find the problem. Progress is being lost to interruption, detection, restart and re-doing
work since the last checkpoint. The green dashboards are the trap: a cluster busy recomputing
lost work is fully utilized and is not advancing.

**Discriminating measurement.** The **job event log**, and an accounting of the three weeks of
wall-clock into buckets: useful compute, detection time, restart time, work lost since the last
checkpoint, and requeue wait. Then check it against the arithmetic — expected interruptions is
`job_hours * nodes / node_mtbf_hours`, which the calculator prints and which you should be able
to do on a napkin. If the observed count is near the model's, the design was always going to
land here and nobody did the arithmetic. If it is far above, you have a reliability problem on
top of a planning problem.

**Mitigation, from the `100k+` row.** Design for continuous partial failure, which is
[C03](../../../C-ml-system-design/C03.training-systems/README.md)'s subject: asynchronous and
sharded checkpointing so a checkpoint is cheap enough to take often; checkpoint cadence chosen
against the interruption rate rather than against the epoch boundary; fast automated detection
so an interruption costs minutes and not hours; elastic restart so a lost node does not idle the
allocation; and, at the algorithm level, formulations that tolerate losing a rank.

**Ambiguous with.** Non-failure stalls that look identical in wall-clock: queue and requeue
wait, input-pipeline stalls, synchronous evaluation phases, and checkpoint writes that block.
This matters because the fixes differ — a faster checkpoint does nothing for queue wait. The
event log separates them, which is why it is first and why it is the only diagnosis in this
drill that starts by reading rather than by running.

---

## 8. One thousand ranks, a published result

**Binding constraint.** **None. This is not a systems problem**, and if you diagnosed hardware
here, that is the drill working. Two defects, both in the methodology, and either one is enough
to void the number:

1. **The problem size per rank was held constant.** That is weak scaling reported in the
   vocabulary of strong scaling. Under weak scaling, high efficiency is the expected outcome
   rather than an achievement, and the number says nothing about running *your* problem faster
   on more ranks — which is what the capacity request is asking for.
2. **The baseline is the same code with the parallel path compiled out**, which is not the best
   known single-device implementation. It carries the parallel structure's overheads with none
   of its benefit, so it is slow, so every speedup above it is inflated by an unknown factor.
   Unknown is the operative word: you cannot correct for it, only discount it.

The two together are the pair that section 5.1 of the module warns is the most common way a
scaling claim means the opposite of what it appears to. A weak-scaling curve against a weak
baseline is two multiplications in the same favourable direction.

**Discriminating measurement.** Not a hardware measurement at all — a request. Ask for the
**strong-scaling curve at fixed total problem size**, re-baselined against the best known
single-device implementation, with efficiency (not speedup) reported at the largest rank count.
That is three clauses and one question, and it is the question. Until it is answered the
89% figure is uninterpretable rather than wrong.

**Mitigation.** Recompute the efficiency yourself from whatever numbers are published, then
apply the ladder: at 1,024 ranks the plausible constraints are the latency term and variance,
and a claim that reports neither has probably not looked. The capacity request is not
refused — it is **unanswerable as written**, and the useful output of the review is the list of
conditions under which it would be answerable. That distinction is worth practising, because
"no" ends a conversation and "here is what would make this reviewable" gets you a better
proposal in a week.

**Ambiguous with.** Nothing systemic. Note that weak scaling is a perfectly legitimate thing to
measure and report — many real problems grow with the machine. The defect is presenting it
unlabelled, and then compounding it with a baseline chosen for convenience.

---

## The pattern across all eight

Three measurements resolve most of the drill, and they are worth knowing cold:

| Measurement | Isolates | Scenarios |
|---|---|---|
| Zero-byte collective, swept over rank count | The latency term and step count | 1, 4 |
| A collective in isolation on a quiet allocation, against the link rate | The bandwidth term, injection ceiling, bisection | 2, 6 |
| Per-rank arrival timestamps at one barrier, as a histogram | Variance, and whether it is one node or all of them | 3, 5 |

Two more, both reads rather than runs, and both under-used: **the rank-to-node mapping log**
(scenarios 3 and 6) and **the job event log** (scenario 7).

And the rule the eight scenarios exist to install: **the mitigation belongs to the row that
binds, not to the row you know best.** Hierarchical collectives are the right answer in scenario
4 and do nothing in scenario 7. Checkpointing is the right answer in scenario 7 and does nothing
in scenario 6. The rows are not increasing degrees of the same problem; they are different
problems that share one symptom, which is somebody saying "it stopped scaling".
