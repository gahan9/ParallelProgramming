# Exercise 1 — Design a video content-moderation system

**Clock: 45 minutes.** Set a timer. Reference solution:
[`../solutions/01_design_video_moderation_solution.md`](../solutions/01_design_video_moderation_solution.md).

---

## The prompt

> You work at a video-sharing platform. Users upload **500,000 videos per day**, averaging
> **8 minutes** each. You need a system that detects policy-violating content — graphic
> violence, sexual content, hate speech, and dangerous-acts content — and takes action on it.
>
> The platform employs **200 human reviewers**. Legal requires that certain categories are
> **never removed automatically**; a human must confirm. Regulators in one major market require
> a decision on reported content within **24 hours**.
>
> Design the system end to end.

That is the whole prompt. Everything else you need, you have to establish yourself — which is
the first part of the exercise.

---

## Work the framework

Use the ten steps from [`../README.md` §2](../README.md#2-first-principles) as headings. Time
allocation is a suggestion, not a rule; what matters is that you reach step 10.

| # | Step | Min |
|---|---|---:|
| 1 | Requirements and scope | 5 |
| 2 | Metrics | 4 |
| 3 | Architecture and data flow | 5 |
| 4 | Data, labels, splits | 5 |
| 5 | Model choice and training system | 5 |
| 6 | Serving: throughput budget, hardware sizing | 8 |
| 7 | Evaluation and rollout | 4 |
| 8 | Monitoring, failure modes, degradation | 4 |
| 9 | Cost model and scaling | 3 |
| 10 | Tradeoffs, 10×, omissions | 2 |

---

## What makes this prompt different from the case studies

Three things, and a strong answer engages with all three rather than producing a generic
classification pipeline.

### It is a throughput problem, not a latency problem

Nobody is waiting on a response. The binding constraints are **ingestion throughput**, the
**cost of processing 8 minutes of video 500,000 times a day**, and the **24-hour regulatory
deadline** for reported content. Notice that the deadline applies to a *different queue* than
the bulk upload flow. If you find yourself writing a p99 latency budget for the upload path,
re-read the prompt.

### The human reviewers are part of the system, not a fallback

200 reviewers is a **hard capacity constraint**, and it is the most expensive resource in the
design. Do the arithmetic on what they can actually process, and make the model's job be
*allocating that capacity*, not replacing it. The interesting design question stops being
"what is the model's accuracy" and becomes "what should the model send to a human, and in
what order."

This also means your metric set has to include reviewer-side metrics — queue age, reviewer
throughput, reviewer agreement rate — and your architecture has to include the feedback path
from reviewer decisions back into training data. That feedback path is also a **loop**: the
model chooses what humans see, so human labels are drawn from the model's distribution.

### The cost of an error is asymmetric and unequal across groups

A false negative leaves harmful content up. A false positive removes a creator's legitimate
video, which is a serious harm to that creator and may be a legal problem. **These are not
equally bad, and they are not equally distributed.** Content in a less-represented language,
or referencing a minority cultural context, is systematically more likely to be misclassified
because there is less of it in the training data.

So: fairness here is not a compliance checkbox to mention at the end. It has to appear in your
metrics (step 2), your data strategy (step 4), your evaluation gates (step 7), and your
monitoring (step 8). A design that mentions "we'd check for bias" in step 10 and nowhere else
has not engaged with the prompt.

---

## Questions to answer explicitly

Write an answer to each. If you skip one, you skipped a section.

**Requirements and scope**
1. What are you assuming about the violation base rate? State a number — it drives everything
   downstream, including whether accuracy is even a meaningful metric.
2. Which categories get automated action and which require human confirmation? Why that split?
3. What are the distinct queues, and what is the deadline on each?

**Metrics**
4. What is the business metric, and what offline proxy stands in for it?
5. Precision or recall — which do you optimize, and does the answer differ per category?
6. Which guardrails stop this system from becoming an over-removal machine? Give thresholds.
7. How do you measure fairness across languages, regions, and creator size? Name the metric.

**Architecture and cost**
8. Do the arithmetic: how much video is 500,000 × 8 minutes per day, and what does it cost to
   run a model over all of it? Show the numbers.
9. Do you process every frame? If not, what is your sampling strategy and what does it miss?
10. What is the cascade — which cheap model runs on everything, and what promotes content to
    the expensive model?
11. Do the arithmetic: 200 reviewers, 8-hour shifts — how many items per day can they actually
    review? What fraction of uploads is that?

**Human-in-the-loop**
12. How do you prioritize the review queue? What happens when it is over capacity?
13. How do you handle reviewer disagreement, and what does that tell you about your labels?
14. Reviewer decisions become training labels. What bias does that introduce, and what do you
    do about it?

**Rollout and operations**
15. How do you ship a new model version without a mass false-removal incident?
16. What is the appeals path, and how does it feed back into the system?
17. Name three failure modes and the degradation behaviour for each.

---

## Do the arithmetic

At minimum, produce numbers for these. The reference solution shows one defensible set; yours
will differ, and that is fine as long as the assumptions are labelled.

| Quantity | Notes |
|---|---|
| Hours of video ingested per day | Then per second, so you can reason about steady-state throughput |
| Frames sampled per video | State your sampling rate and defend it |
| GPU-hours per day for the first-pass model | Assume a cost per frame and multiply |
| Human review capacity per day | 200 reviewers → items/day. Compare to upload volume |
| The fraction of uploads you can afford to escalate | This is the number the whole cascade is designed around |
| Cost per 1,000 videos processed | Split into machine cost and human cost, and notice which dominates |

That last row is where the design gets interesting. If you find the human cost dominating by an
order of magnitude, you have discovered the actual objective function.

---

## Common traps

- **Designing a classifier instead of a system.** The model is one box. The queues, the
  escalation policy, the appeals path, and the label pipeline are the design.
- **Ignoring the base rate.** If violations are 0.5% of uploads, a 99% accurate model produces
  far more false positives than true positives. Compute this before quoting any accuracy number.
- **Treating "human review" as an infinite resource.** It is 200 people.
- **Latency-budgeting a throughput system.** Budget the *deadline* on each queue instead.
- **Bolting fairness on at the end.** It has to be in the metrics, the data, and the gates.
- **No rollback story.** A bad moderation model removes thousands of legitimate videos before
  anyone notices. What is your blast-radius limit?
