# Drill 4 — Run an architecture review

**Clock: 45 minutes.** Reference answer:
[`../solutions/architecture_review.md`](../solutions/architecture_review.md).

Below is a cluster proposal. It is **fictional** — no organization, vendor, site or figure in
it is real, and it names no specific hardware model — but it is written the way these
documents are actually written, and its mistakes are the ones that are actually made.

Review it against the six-question checklist in
[section 7.1](../README.md#7-real-world-use-cases) of the module, then write the memo you
would really send.

Your deliverable is **two paragraphs plus a table**, in this order:

1. **What the design is optimized for.** State it as a compliment where one is deserved. A
   review that opens with problems trains the author to write defensively.
2. **The one workload characteristic that would make this the wrong design**, said plainly.
   This is the paragraph that saves the money, and it is the one people soften until it
   carries no information. Practise not softening it.
3. A table of findings: what, why it matters in production, the fix, and severity — ordered
   by expected damage. Five to seven rows. A reviewer who reports findings without a
   priority has handed the author a to-do list, not a review.

Run the numbers before you write. The proposal gives you enough to do it:

```bash
python3 ../tools/scale_boundary.py --explain \
    --ranks 4096 \
    --model-bytes 1600000000 \
    --compute-ms 90000 \
    --serial-fraction 0.002 \
    --alpha-us 4 \
    --bandwidth-gbs 20 \
    --gpus-per-node 8 \
    --node-mtbf-hours 35000 \
    --job-hours 1008
```

Those flags are the proposal's own numbers where it states them and a stated assumption where
it does not. Note which is which in your memo; a reviewer who silently invents an input has
made the same mistake as the author.

---
---

# PROPOSAL: Cluster "Meridian" — training and inference capacity

**Author:** Platform Infrastructure. **Status:** For approval. **Reviewers:** you

## 1. Summary

We propose a 512-node cluster, 8 accelerators per node, for a total of 4,096 accelerators.
Aggregate peak throughput is approximately 8.2 exaFLOPS at low precision, which would place
Meridian comfortably in the upper tier of comparable industry deployments. The build follows
the reference architecture of a well-known public training cluster, which we consider
de-risked by its published results.

## 2. Workload

Two workloads share the cluster.

**Primary (70% of capacity).** Pre-training of a mixture-of-experts language model. Expert
routing is the dominant communication pattern: every layer performs an all-to-all to
dispatch tokens to experts and a second to combine the results. Target run length is **six
weeks** of continuous training per model.

**Secondary (30%).** Batch inference and fine-tuning jobs, each fitting inside one node.

## 3. Node design

| Component | Choice | Rationale |
|---|---|---|
| Accelerators | 8 per node, current generation, 192 GiB HBM each | Highest available peak throughput per socket |
| CPU-accelerator link | PCIe | Standard, well-understood, lowest cost |
| Intra-node accelerator fabric | Full mesh between the 8 accelerators | Needed for tensor parallelism |
| Network adapters | **1 per node**, 400 Gb/s | 400 Gb/s exceeds the aggregate demand we measured at small scale |
| Local storage | 8 TB NVMe per node | Scratch and dataset cache |

## 4. Fabric

Two-tier leaf-spine. 32 nodes per leaf, **4:1 oversubscription** from leaf to spine.

We ran a scaling test on 64 accelerators (8 nodes, all under one leaf) and measured 91%
scaling efficiency. Extrapolating linearly, we expect roughly 85-90% efficiency at 4,096
accelerators, which is well within acceptable bounds. The oversubscription saves
approximately 38% of the fabric budget, and since 91% of our traffic at the tested scale
stayed within the leaf, we do not expect the spine to be a bottleneck.

## 5. Power and cooling

Air cooling throughout, reusing the existing facility's hot-aisle containment. The facility
has 3.1 MW available and our estimated draw is 2.9 MW at sustained load.

## 6. Software and operations

Standard scheduler with fair-share queuing. Jobs request whole nodes. Checkpoints are written
by the framework's default synchronous checkpoint at the end of each epoch, roughly every
14 hours, to the shared filesystem.

Node health is checked at provisioning time and after any hardware replacement.

## 7. Cost and phasing

Phase 1 delivers 256 nodes; phase 2 completes the build six months later. Both phases are
identical, so no re-validation is required between them.

## 8. Success criteria

- Aggregate peak throughput as specified in section 1.
- 85% scaling efficiency at full scale on the primary workload.
- Acceptance test: the standard dense linear-algebra benchmark at greater than 70% of peak.

---
---

## Hints, if you want them

Read these only after a first pass.

<details markdown="1">
<summary>Where to look</summary>

- Two sections **contradict each other** when you read them together. Neither is wrong on
  its own.
- One number in section 4 is a correct measurement used to support a claim it cannot
  support. Ask what the measurement was *of*.
- Section 2 states a traffic pattern. Section 4 buys a fabric. Compare them against
  [D05 section 2.2](../../D05.topology-and-scale-boundaries/README.md#2-first-principles) and
  the [bisection bandwidth](../../../../docs/GLOSSARY.md#bisection-bandwidth) entry.
- Section 6 is one sentence long and is the most expensive sentence in the document, given
  section 2's target run length. Do the arithmetic in the command at the top of this file.
- One item is a *process* defect rather than a technical one, and it is in section 7.
- Section 8's acceptance test measures something the primary workload does not do.

</details>

## What "done" looks like

- The compliment is specific and true, not a throat-clear.
- The "wrong design" paragraph names a **workload characteristic**, not a component, and it
  would be understood by someone who joined the company yesterday.
- Every finding has a fix the author could act on this quarter, and the severity ordering is
  defensible out loud.
- At least one finding is about what the proposal does **not** say. The absent failure budget
  is worth more than any of the present numbers.
- You state which of your own inputs were assumptions. If you ran the calculator, say that
  its output is a model rather than a measurement, in the memo, in those words.
