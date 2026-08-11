# Drill 3 — Diagnose the system

**Clock: 60 minutes.** Reference answers:
[`../solutions/diagnose_the_system.md`](../solutions/diagnose_the_system.md).

Eight scenarios of the form "at scale X, symptom Y". For each one, give three things:

1. **The binding constraint** — named from the ladder in
   [section 6.4](../README.md#6-challenges-drawbacks--tradeoffs) of the module.
2. **The discriminating measurement** — the one observation that would confirm it *and rule
   out the next most likely explanation*. This is the deliverable. A diagnosis you cannot
   falsify is a guess.
3. **The mitigation** — and it must belong to the row you named, not to the row above.

Several scenarios have **more than one consistent explanation**. That is deliberate. Mark
those, say which explanation you would test first, and say why that ordering is cheaper.

Nothing here is a measurement from any real system. The scenarios are constructed to be
internally consistent with the cost model in
[D01 sections 3.1 to 3.5](../../D01.scaling-limits-and-communication-cost/README.md#3-deep-dive)
and the topology ladder in
[D05 section 6.6](../../D05.topology-and-scale-boundaries/README.md#6-challenges-drawbacks--tradeoffs).

## Check your reasoning against the calculator

```bash
python3 ../tools/scale_boundary.py --preset hpc-halo --ranks 65536 --explain
python3 ../tools/scale_boundary.py --preset llm-16k --ranks 16384 --explain
```

`--explain` names the term that dominates each regime and cites the Track D module that owns
it. Use it *after* you have written your answer down. It is an analytic model and it will
happily agree with a wrong diagnosis if you feed it the wrong inputs.

## The three measurements that separate almost everything

Most of the eight resolve with one of these, and knowing them cold is a large part of the
skill:

| Measurement | Isolates | Because |
|---|---|---|
| A **zero-byte** collective, swept over rank count | The latency term | No bytes move, so what is left is `steps * alpha` |
| A collective run **in isolation** on a quiet allocation, compared against the link rate | The bandwidth term | Removes contention with your own compute and with neighbours |
| **Per-rank arrival timestamps** at one barrier, as a histogram rather than a mean | Variance, and whether it is one node or all of them | A mean hides the tail; the barrier only pays the tail |

Two more, less often reached for: **logging the rank-to-node mapping every run** (isolates
placement) and **the job event log** (isolates interruptions from slowness).

---

## The scenarios

### 1. Eight ranks, one node

A tensor-parallel inference service. Going from four accelerators to eight in the same
chassis raises throughput by 1.15x, not 2x. A timeline shows the per-token all-reduce growing
as accelerators are added, and it is on the critical path of every token.

### 2. Sixty-four ranks, eight nodes

Scaling is close to linear up to one node, then bends sharply at the node boundary and
flattens. Each node holds eight accelerators and **one** network adapter. Aggregate achieved
bandwidth rises when nodes are added and does not change when more accelerators per node are
put to work.

### 3. Five hundred and twelve ranks

Two runs of the same code, on the same input, on the same cluster, one hour apart, differ by
1.35x in wall-clock time. Nothing was rebuilt. The operator mentions in passing that the
faster run happened to land inside a single rack.

### 4. Four thousand ranks, small messages

A solver whose per-step collective is a few kilobytes. Going from 1,024 to 4,096 ranks cut
per-rank compute by 4x as expected, and *increased* step time by 1.8x. The fabric's quoted
bandwidth is nowhere near saturated.

### 5. Four thousand ranks, compute-heavy step

Per-rank compute is about 400 ms and the per-step collective is small. Measured step time
is 9% above the slowest single-rank compute time, and that excess grew slowly — a few
percent — when the job moved from 1,024 to 4,096 ranks. A monitoring dashboard shows one
node's mean step time sitting about 6% above the fleet median.

### 6. Sixteen thousand ranks, all-to-all

An expert-routed model whose per-step pattern is an all-to-all. Its measured time is about
3x what the [alpha-beta model](../../../../docs/GLOSSARY.md#the-alpha-beta-model) predicts
from the per-rank link rate, and it gets worse when the scheduler spreads the allocation over
more of the machine rather than packing it.

### 7. Thirty-two thousand ranks, a month-long job

Per-step throughput is within 5% of the target every time anyone measures it. Wall-clock
progress against the plan is 40% short after three weeks. The cluster's utilization
dashboards are green and the job is, most of the time, running.

### 8. One thousand ranks, a published result

A report claims 89% scaling efficiency at 1,024 ranks and asks for four times the capacity on
the strength of it. Reading the methodology: the problem size per rank was held constant
across the sweep, and the single-rank baseline is the same code with the parallel path
compiled out.

---

## Answer template

Copy this eight times and fill it in. Prose, not bullets — a review comment is a sentence
somebody has to act on.

```text
Scenario N
  Binding constraint:        (a row of section 6.4, named)
  Discriminating measurement:(what you would run, and what result would confirm vs refute)
  Mitigation:                (belonging to that row)
  Ambiguous with:            (the other consistent explanation, or "none")
  Test first, because:       (cheapness, or because it changes the next decision)
```

## What "done" looks like

- Eight diagnoses, each naming a row of the ladder rather than a component.
- At least **three** scenarios marked ambiguous with a named alternative. If you found none,
  you are pattern-matching on the scale number instead of reading the symptom.
- Every mitigation belongs to the row you named. The most common error in this drill, and in
  real reviews, is prescribing the previous row's fix — hierarchical collectives for a
  failure-bound job, or checkpointing for an oversubscribed fabric.
- One scenario is not a systems problem at all. If you diagnosed hardware in all eight, read
  [section 5.1](../README.md#5-performance-analysis) again.
