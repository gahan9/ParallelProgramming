# D06 Solutions — Real Systems at Scale

Worked answers for the four drills in [`../exercises/`](../exercises/). Read them **after** you
have written your own down; three of the four are judgement exercises, and the value is in the
gap between your answer and this one, not in this one.

| # | Drill | Solution |
|---|---|---|
| 1 | [`../exercises/scale_boundary.py`](../exercises/scale_boundary.py) | [`scale_boundary.py`](scale_boundary.py) plus the commentary below |
| 2 | [`../exercises/read_a_list_entry.md`](../exercises/read_a_list_entry.md) | [`read_a_list_entry.md`](read_a_list_entry.md) |
| 3 | [`../exercises/diagnose_the_system.md`](../exercises/diagnose_the_system.md) | [`diagnose_the_system.md`](diagnose_the_system.md) |
| 4 | [`../exercises/architecture_review.md`](../exercises/architecture_review.md) | [`architecture_review.md`](architecture_review.md) |

Two of these solutions cannot give you *the* answer, and say so where that is true. Drill 2's
answer depends on which system you picked and which list edition you read, so the solution
grades your **method** and works one item end to end. Drill 4 has several defensible orderings
of its findings; the solution states its own ordering and the reason for it, so you can argue
with the reason rather than with the order.

Everything numeric in this directory is an **analytic model**, never a measurement. Nothing in
this module was run on a machine, and the tool that produced these figures prints that in its
banner every time.

---

## Drill 1 — Place your own workload on the ladder

[`scale_boundary.py`](scale_boundary.py) is the exercise file with all five TODOs filled in,
plus `knee_by_bisection`, which question (b) asks for. Run it:

```bash
python3 scale_boundary.py
python3 ../tools/scale_boundary.py --preset dp-training --ranks 4096
```

The two tables agree row for row, because the constants in the drill *are* the `dp-training`
preset. That agreement is the drill's built-in check: if your version disagrees with either
one, exactly one of the three has a bug, and finding out which is more instructive than the
table.

### The five TODOs, and the design decision inside each

**TODO 1, `step_terms`.** The step is five terms, and the reason to return them as a dict
rather than a total is that the total answers "how slow" while the split answers "why", which
is the only question a mitigation can be chosen from:

```text
serial    = s * C                      does not divide, ever (Amdahl)
compute   = (1 - s) * C / N            the part that does divide
latency   = 2 * (N - 1) * alpha        ring all-reduce, 2(N-1) steps
bandwidth = 2 * m * (N - 1) / N / B    bandwidth-optimal bytes per rank
variance  = (E[max of N] - 1) * (all of the above)
```

The latency and bandwidth shapes are the ring all-reduce from
[D01](../../D01.scaling-limits-and-communication-cost/README.md#3-deep-dive), whose per-rank
byte count is bandwidth-optimal
([Patarasuk and Yuan 2009](../../../../docs/REFERENCES.md#bandwidth-optimal-all-reduce-patarasuk-and-yuan-2009)).
Two consequences people get wrong: the bandwidth term approaches `2m/B` and **stops growing**,
so beyond a few hundred ranks it is a constant tax; and the latency term grows without bound,
so it eventually overtakes any fixed byte count no matter how fast the fabric is.

Variance multiplies the sum rather than adding to it. That is the modelling choice with the
most teaching in it: a barrier pays the *slowest* rank, so jitter inflates whatever the step
already costs instead of contributing its own independent slice.

**TODO 2, `parallel_efficiency`.** `T(1) / (N * T(N))`. Validate before you compute, not
after: `ranks < 1` and a `serial_fraction` outside `[0, 1]` are input errors, and a model that
returns a number for them will be believed. Both raise `ValueError` with a one-line message,
which is also what the module's section 9 drill asks for.

Efficiency is **strictly decreasing** in `N` for a fixed problem, and it is worth seeing why
from the formula rather than from the table: the numerator `T(1)` is constant and the
denominator `N * T(N)` is strictly increasing, because `N * (1-s)C/N` is constant while every
other term is multiplied by a growing `N`. There is no configuration in which buying ranks
raises efficiency on a fixed problem. Weak scaling is a different question and D01 section 2
handles it.

**TODO 3, `binding_constraint`.** Compare the terms and name the biggest, with one wrinkle
that matters more than the comparison: **failure overrides everything**. If the job expects to
be interrupted, its step time is the second problem. That is why the tool reports both a
`step_constraint` and a `constraint`, and prints sentences like "binding constraint goodput:
failures -- the step time is collective bandwidth-bound, which is the second problem and not
the first". A tool that reported only the winner would send you to optimize a collective in a
job that is not going to finish.

The `compute (still scaling)` label exists for the same reason. While overhead is under about
10% of the step, the honest answer is not "compute is the bottleneck" but "nothing is wrong
yet; add ranks".

**TODO 4, `expected_interruptions`.** `job_hours * ceil(ranks / gpus_per_node) / node_mtbf`.
Three properties to check, because each is a different way people misread it:

- It scales **linearly in rank count**. Ranks divided by ranks-per-node is *nodes*, and nodes
  are the failure domain. Eight-fold more accelerators is eight-fold more interruptions.
- It scales **linearly in job length**. A run twice as long expects twice as many.
- It is Poisson with **independent** failures, which is its main lie. Real failures correlate
  through shared power, cooling, racks and switches, so a real fleet sees bursts this model
  cannot produce. The direction of the error is not knowable in advance, which is why the tool
  labels the MTBF an assumption rather than smuggling in a fleet number.

At `dp-training`'s 4,096 ranks the model expects 2.15 interruptions in a one-week job and an
11.6% chance of finishing uninterrupted. That is the whole argument of section 3.9 in one line:
the plan needs a failure budget before it needs a throughput target.

### TODO 5, the questions

**(a) Which row, and which mitigation.** 4,096 ranks sits on the `1k-10k` row by rank count,
but the *binding* constraint at that point is failure, which is the `10k-100k` row's
constraint arriving early because the node count and job length put it there. This is the
lesson that the ladder's rank boundaries are soft: what binds depends on bytes, fabric and job
length, not on the rank number. The mitigation belongs to the row that binds — asynchronous
and sharded checkpointing, fast detection, elastic restart
([C03](../../../C-ml-system-design/C03.training-systems/README.md)) — and hierarchical
collectives, which is the `1k-10k` fix, would do nothing for it.

**(b) The knee by bisection.** Efficiency is monotone decreasing, so the predicate "efficiency
at N is at or above the target" is true for a prefix of the rank counts and false afterwards.
That is exactly the shape bisection needs, and it is why the tool does not sweep: it finds the
boundary in about 40 evaluations instead of N. For `dp-training` the knee is **243 ranks**,
which is 16.9x below the requested 4,096 — the arithmetic version of "half the capacity
delivers most of the value".

The important part is not the algorithm but that it **survives adding terms**. A closed-form
knee exists for a compute-plus-serial model, and the moment you add the latency term it does
not. Bisection keeps working for any monotone efficiency function, including ones with terms
this model does not have yet.

One implementation note: an exact tie at the target can land a few ULPs on the wrong side of a
float comparison, so the predicate compares with a small tolerance. Without it the knee
occasionally moves by one rank between platforms, which looks like a bug in the model and is
not.

**(c) The variance paradox.** The variance *factor* grows with N — the expected maximum of
more draws is further into the tail — and the variance *term* mostly **shrinks**, from 340 ms
at N=2 to 19 ms at N=4,096. Both are true because the factor multiplies a step that is
shrinking faster than the factor grows. Percentage-wise variance is getting worse the whole
way; in milliseconds it is getting cheaper, because there is less step left to inflate.

The trap is concluding that jitter stops mattering at scale. It does not: at the bottom of the
table variance is the *largest* remaining slice of a step that has nowhere else to shrink, and
it is why health-gated scheduling and clock pinning are `1k-10k` mitigations rather than
`8-64` ones. The approximation used here is an extreme-value estimate for the expected maximum
of N draws; it agrees with D01 section 3.4's seeded simulation to within about half a percent,
and it is monotone, which the sanity checks depend on.

**(d) Doubling the bandwidth.** For `dp-training`, doubling `--bandwidth-gbs` from 25 to 50
moves the knee from 243 to 373 ranks — a factor of **1.5, not 2** — and changes the binding
constraint at 4,096 not at all. Halving one term does not halve a sum, and the terms that did
not improve (latency, variance, and the failure rate, which does not depend on the fabric at
all) are what is left. This is recurring lesson 4 arriving as arithmetic: at the top of the
ladder the fabric is not the constraint, and the levers that work are fewer bytes per step, a
different parallelism decomposition, or fewer ranks.

**(e) What the model ignores.** Five things it is silent about, each of which can dominate a
real run: **congestion** and the tail latency it creates under load; the **injection gap**
when a node's accelerators outnumber its network adapters; **topology and oversubscription**,
so a rank-to-node mapping is invisible here and worth 1.35x in
[D05](../../D05.topology-and-scale-boundaries/README.md#6-challenges-drawbacks--tradeoffs);
**algorithm switching** inside a real collective library, which does not use a ring at every
size; and **correlated failure**, discussed above. It also assumes one global collective per
step, which over-states nearest-neighbour codes — the `hpc-halo` preset says so in its own
`--explain` output rather than quietly flattering itself.

Which is the honest summary of the whole tool: it is reliable about **orderings, ratios and
which regime you are in**, and it should never be quoted as a predicted step time.
