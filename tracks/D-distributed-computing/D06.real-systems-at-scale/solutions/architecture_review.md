# Drill 4 solution — Run an architecture review

Exercise: [`../exercises/architecture_review.md`](../exercises/architecture_review.md).

The proposal is fictional and so is every figure in it. Everything computed below is an
**analytic model** from [`../tools/scale_boundary.py`](../tools/scale_boundary.py), not a
measurement, and the memo says so in its own text for the same reason the tool does.

There is more than one defensible ordering of the findings. This solution states its ordering
and the rule behind it — **expected damage to the primary workload's goodput** — so you can
argue with the rule rather than with the order. If your ordering follows a different stated
rule and is internally consistent, it is a good review.

---

## The arithmetic to do before writing

Two computations decide most of the memo, and both come out of the proposal's own numbers.

**1. The failure budget the proposal does not contain.** 4,096 accelerators at 8 per node is
512 nodes. At the stated assumption of a 35,000-hour per-node MTBF, the job-level mean time
between interruptions is `35,000 / 512`, about **68 hours**, against a target run length of six
weeks — 1,008 hours. The model expects roughly **15 interruptions per run** and a **near-zero
chance of finishing one uninterrupted**:

```bash
python3 ../tools/scale_boundary.py --explain \
    --ranks 4096 --model-bytes 1600000000 --compute-ms 90000 --serial-fraction 0.002 \
    --alpha-us 4 --bandwidth-gbs 20 --gpus-per-node 8 \
    --node-mtbf-hours 35000 --job-hours 1008
```

Now put that next to section 6: checkpoints at the end of each epoch, roughly every 14 hours,
written synchronously. An interruption costs, on average, half a checkpoint interval of lost
work — about 7 hours — so 15 interruptions is about **100 hours of recomputation in a 1,008-hour
run, near 10%**, plus detection time and restart time, neither of which the proposal mentions
and both of which are usually the larger half in a system with no automated detection.

**2. Per-accelerator injection, which the proposal states without noticing.** One 400 Gb/s
adapter shared by eight accelerators is **50 Gb/s each, about 6.25 GB/s**. Re-running the model
with the proposal's own implied figure instead of a generous 20 GB/s moves the predicted knee
from **237 ranks to 118** — the design's own numbers halve its own useful scale.

That is the difference between a review that says "consider more NICs" and one that says "your
document contains the number that shows this does not work". Always convert the vendor's
aggregate figure into a per-accelerator figure; it is one division and it is where the injection
gap hides.

Note which inputs were **assumptions rather than the proposal's**: compute per step, serial
fraction, per-rank latency and the node MTBF. The proposal states rank count, node shape, run
length, checkpoint interval, oversubscription ratio and adapter rate; everything else in the
command above is a stated assumption, and a reviewer who does not flag that has made the same
mistake as the author.

---

## The memo

> **To:** Platform Infrastructure
> **Re:** Cluster "Meridian" proposal — review
>
> **What this design is optimized for.** The node is a genuinely good single-node training and
> inference unit. Eight accelerators with a full intra-node mesh and 192 GiB each is the right
> shape for the secondary workload: every one of those jobs fits inside a node, keeps its
> tensor-parallel traffic on the fast intra-node links, and never touches the fabric. The
> two-phase build is sound risk management, the power and cooling reuse is pragmatic, and the
> choice to follow a published reference architecture rather than invent one is the right
> instinct. For 30% of the stated capacity this is a well-judged machine.
>
> **The workload characteristic that makes this the wrong design.** The primary workload's
> communication is an **all-to-all on every layer**, not a reduction. An all-to-all moves data
> across every cut in the machine by construction, so it consumes bisection bandwidth directly
> and cannot be hidden behind compute the way a gradient all-reduce can. This proposal buys a
> 4:1 oversubscribed spine and one network adapter per eight accelerators, which are the two
> cheapest places to save money **if and only if** most traffic stays local. For a routed
> mixture-of-experts model it does not. Combined with a six-week target run length and 512
> failure domains, the design's own numbers predict that the primary workload will spend most
> of its rented capacity idle and will not complete a run without interruption. Both problems
> are fixable at design time and neither is fixable after the purchase order.
>
> Every number below is an analytic model, not a measurement; the tool that produced them says
> so in its banner. They are reliable about ordering and ratios and should not be quoted as
> predicted step times.

### Findings, ordered by expected damage to the primary workload

| # | Finding | Why it matters in production | Fix | Severity |
|---|---|---|---|---|
| 1 | **No failure budget.** 512 nodes, six-week runs, checkpoints only at the ~14 h epoch boundary, written synchronously, and node health checked only at provisioning | The model expects about 15 interruptions per run at a 68 h job MTBI, and near-zero chance of an uninterrupted run. At a 14 h cadence that is roughly 10% of the run lost to recomputation before detection and restart, which are unspecified and are usually the larger half | Add a failure budget to the document: expected interruptions, checkpoint cost, detection and restart time, resulting goodput. Move to asynchronous sharded checkpointing with cadence set against the job MTBI, not the epoch. Add automated detection and elastic restart. Add continuous health gating so a degraded node never enters an allocation | **Blocking** |
| 2 | **Injection gap of 8:1.** One 400 Gb/s adapter per eight accelerators — about 6.25 GB/s each — and a PCIe host link | Every off-node byte from all eight accelerators shares one path, so the fabric above the node is not the limit and cannot be made to be. Using the proposal's own implied per-accelerator rate rather than a generous assumption moves the modelled knee from 237 ranks to 118 | One adapter per accelerator, or at minimum per pair, sized against the actual per-step bytes. Confirm the path from adapter to accelerator memory is zero-copy rather than staged through host memory; with PCIe as the host link, a staged path would waste the adapters you do add | **Blocking** |
| 3 | **4:1 oversubscription under an all-to-all workload.** Section 2 states the traffic pattern; section 4 buys a fabric that assumes locality | An all-to-all consumes bisection by construction, so oversubscription is a direct multiplier on the dominant term of the primary workload's step, every layer, forever. This is the finding that contradicts section 2, and neither section is wrong on its own | Non-blocking, or at worst 2:1, in the tier the primary workload crosses; or a rail-optimized design; or — if the budget is fixed — **buy fewer nodes and a better fabric**, which the modelled knee says would deliver more useful work than the full node count on this fabric | **Blocking** |
| 4 | **The 64-accelerator measurement cannot support the 4,096 extrapolation.** All 8 nodes were under one leaf, and the "91% of traffic stayed within the leaf" figure is a property of a single-leaf allocation, not a prediction about the machine | Efficiency falls monotonically with rank count and the binding constraint *changes identity* along the way, so linear extrapolation of a scaling figure is invalid in principle, not merely optimistic. The one number the test could not produce is the one the design depends on: behaviour across the oversubscribed tier | Re-run the scaling test spanning at least four leaves at the intended oversubscription ratio, report **efficiency** at each rank count rather than speedup, and publish the traffic locality measured on a spread allocation | **High** |
| 5 | **The acceptance test measures a workload the cluster does not run.** A dense linear-algebra benchmark at >70% of peak has very high arithmetic intensity and almost no global communication | The machine can pass acceptance and still be unable to run the primary workload at a usable efficiency. The test exercises exactly the part of the design nobody is worried about | Accept on the real thing: the primary workload's all-to-all pattern at target scale on a spread allocation; a bisection-cut measurement; and an interruption-and-restart drill measuring detection plus restart plus lost work end to end | **High** |
| 6 | **Success criteria are unlabelled and un-derived.** "85% scaling efficiency at full scale" does not say strong or weak, against what baseline; and "aggregate peak throughput" is a multiplication, not a criterion | An unlabelled efficiency target cannot be passed or failed, so it will be interpreted generously at acceptance. Peak is the number this design already leads with and the one that describes it least | State strong or weak, state the baseline, and derive the target from the design's own predicted knee rather than asserting it. Replace the peak criterion with sustained throughput on the primary workload | **High** |
| 7 | **Phase 2 declared identical, so no re-validation.** Two identical phases six months apart | Identical bills of material are not identical hardware six months later: firmware, component revisions and silicon lots move. More importantly, doubling the node count doubles the failure domains and changes which fabric tier the primary workload crosses, so the phase-1 acceptance result does not transfer even if the hardware is bit-identical | Keep an acceptance gate per phase, including a re-run of the scaling and interruption tests at the new scale. This is the cheapest finding on the list to fix and the easiest to lose in a schedule | **Medium** (process) |
| 8 | **Power margin of about 6%.** 2.9 MW estimated against 3.1 MW available, air-cooled, with the estimation method unstated | A margin this thin is consumed by any of: a higher sustained utilization than assumed, warmer inlet air, a denser follow-on accelerator, or an estimate that was made from typical rather than sustained draw. Power is the one constraint that cannot be worked around after the build | State the estimation method and whether it is typical or sustained; measure phase 1's actual draw before committing phase 2; state the headroom explicitly as a number the design owns | **Medium** |

### What the document does not say, which is worth more than what it does

- **No failure budget at all** — finding 1, and the single most valuable absence.
- **No per-step byte count** for the primary workload. The proposal describes the pattern in
  words and never states how many bytes cross the node boundary per layer, which is the number
  findings 2 and 3 both turn on and the one a reviewer cannot supply.
- **No detection or restart time**, so the goodput arithmetic cannot be completed even in
  principle.
- **No statement of what the reference architecture's published results were a result *of*.**
  Copying a design that was validated on a different traffic pattern is the root cause of
  findings 2 and 3, and it is the only finding that explains the others.

---

## How this was graded

| The review does | Because |
|---|---|
| Opens with a specific, true compliment | "Good node for the secondary workload" is verifiable from the document. A review that opens with problems teaches the author to write defensively next time, and the next document will contain less information |
| Names a **workload characteristic**, not a component | "The communication is an all-to-all, not a reduction" survives a change of vendor, generation and rack layout. "Not enough NICs" is a symptom of it and will be argued about on price |
| Orders findings by expected damage, and says so | A findings list without a priority is a to-do list. The stated rule is what makes the order arguable, which is what makes it useful |
| Quantifies with the proposal's own numbers | 6.25 GB/s per accelerator and a 68-hour job MTBI both come out of the document. A finding derived from the author's own figures is very hard to negotiate away |
| Labels its assumptions and its model | Four of the calculator's inputs were assumptions. Not saying so would repeat the author's error in the reviewer's voice, and the memo would deserve the same discount it hands out |
| Includes one process finding | Finding 7 is not technical and will cost real money. Reviews that only look at the architecture miss the half of the risk that lives in the schedule |
| Says one thing the document is silent about | The absent failure budget is worth more than any present number, and only a checklist-driven read finds an absence — you cannot notice a missing section by reading the sections that are there |

**The most common way this drill goes wrong** is a review that finds seven true, small things —
PCIe, air cooling, the thin power margin, the scheduler — and never says the sentence about the
all-to-all. Every one of those findings is correct and their sum does not tell the author to
stop. The purpose of a review is not to demonstrate that you read the document.

**The second most common way** is diagnosing from the scale number. 4,096 accelerators is not
automatically the failure-bound row: it lands there because of the node count, the six-week run
length and the assumed MTBF together, and change any of the three and the binding constraint
moves. Section 6.4 of the module says the rank boundaries are soft; this is what that means in
a review.
