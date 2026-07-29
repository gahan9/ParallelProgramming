# Exercise 2 — Design the quality gate for a quantization rollout

> Reference answer: [`../solutions/02_quality_gate_solution.md`](../solutions/02_quality_gate_solution.md)
> — write yours first.

## The situation

You are shipping **INT8 post-training quantization** of the model behind a **customer-support ticket
router**. The router reads an incoming ticket and assigns it to one of 40 queues; a misroute means
the ticket sits in the wrong queue until a human notices, which is typically hours.

**What quantization buys** (measured on your hardware, not assumed): **1.7× throughput**, cutting the
fleet from 14 GPUs to 9 — about **$9,000/month**.

**What you know about the system**
- Traffic is **multilingual**: 71% English, 12% Spanish, 8% Japanese, 5% German, 4% everything else.
- Ticket length ranges from one line to several pages; the long tail is ~6% of volume.
- Two of the 40 queues are **safety-critical**: `account-compromise` and `billing-dispute`. Together
  they are **0.9%** of volume. A misroute here is a regulatory and trust problem, not an
  inconvenience.
- About 15% of tickets arrive from a mobile app with heavy typos and no punctuation.
- The current model's aggregate routing accuracy is **93.8%**.
- The existing offline eval set is **11,000 tickets sampled uniformly from last year's traffic**.
- Deploys go out weekly. There is a feature-flag system. Nobody has ever rolled back a model.
- The team's proposed evidence is: *"aggregate accuracy on the eval set dropped from 93.8% to 93.6%,
  which is within noise."*

## Your task

**1. Critique the proposed evidence.** The team's one-line justification has at least four distinct
problems. Name them. For each, say what could be hiding behind it.

**2. Design the eval set.** Specify it concretely: size, sampling strategy, how it is versioned, who
can change it, and how you guarantee both variants are scored on the same one. State explicitly what
is wrong with "11,000 tickets sampled uniformly from last year".

**3. Define the slices.** List the slices you would require, and for each say what failure it is
designed to catch. Then handle the hard part: the safety-critical queues are 0.9% of volume, so a
uniform sample of 11,000 contains roughly 100 of them. Say what that does to your ability to detect a
regression there, and what you would do about it.

**4. Set the thresholds.** For the aggregate and for each slice, give a **numeric** pass/fail
threshold and justify it. At least one threshold should not be symmetric with the others — explain
why. State how you would establish that a given delta is inside or outside noise.

**5. Design the rollout.** Shadow, canary, ramp: what percentage, for how long, and what is measured
at each stage. Note which failures each stage can catch that the previous one could not.

**6. Define the rollback trigger.** A specific, automatable condition — a metric, a threshold, and a
time window — plus who owns the decision and how long the procedure takes. Then answer the question
that decides whether any of this is real: **how do you know the rollback works?**

**7. What quantization-specific failure modes would you look for?** Name at least two that a
generic model-change gate would miss, and say which slice would surface each.

**8. State when you would abandon the rollout entirely** rather than iterate on it.

## Constraints on your answer

- Every threshold must be a number with a justification. "Monitor closely" is not a threshold.
- Distinguish what is checked **offline** (before any traffic) from what is checked **online**.
- Your gate must be executable by an on-call engineer at 2 a.m. without judgement calls. If a step
  requires interpretation, it will be interpreted favourably under pressure — design around that.
- One page plus the threshold table.
