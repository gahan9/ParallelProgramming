# Exercise 3 — Review a design document

**Clock: 30 minutes.** Reference solution:
[`../solutions/03_review_a_design_solution.md`](../solutions/03_review_a_design_solution.md).

---

## The drill

Below is a design document written by a competent engineer. It is not a strawman: the
architecture is reasonable, the arithmetic that is present is correct, the model choice is
defensible, and most of it would pass a casual review.

It contains **five serious problems**. Each one is the kind that ships, survives review, and
surfaces months later as "the model stopped working and nobody knows why."

Your job:

1. Find all five. They are spread across different sections, and at least one requires
   **cross-referencing two sections** against each other.
2. For each: name it, say what goes wrong in production, and propose a concrete fix.
3. Rank them by expected damage. A reviewer who reports five issues without a priority has
   given the author a to-do list, not a review.

Resist the urge to list everything that is merely *improvable*. There are things in this
document I would do differently that are not defects — a good review distinguishes "this is
wrong" from "I would have chosen otherwise," and says so.

> **Hint on difficulty.** Two of the five are visible within their own section if you are
> reading carefully. Two require you to ask "what happens next month?" rather than "is this
> correct today?". One is only visible if you compare the architecture section against the
> latency section.

---
---

# DESIGN: Proactive Churn Prevention

**Author:** ML Platform team · **Status:** Ready for review · **Reviewers:** you

## 1. Background and goal

StreamCo is a subscription video service with 40 million monthly subscribers on monthly
auto-renew. Voluntary churn currently runs at **3.1% per month**, which is roughly 1.24 million
cancellations. Retention modelling suggests the average subscriber lifetime value is $180, so
each prevented cancellation is worth roughly that much.

Today, retention is reactive: a "we're sorry to see you go" discount is offered on the
cancellation-confirmation page. That is too late — by the time a user reaches that page the
decision is largely made, and the offer accept rate is 6%.

**Goal:** predict which subscribers are at risk of cancelling in the next 30 days, and
intervene *before* they reach the cancellation flow, with a targeted retention offer delivered
in-app and by email.

## 2. Success metrics

| Layer | Metric | Target |
|---|---|---|
| Business | Monthly voluntary churn rate | 3.1% → 2.7% |
| Business | Net revenue retention | +1.5 pts |
| Product | Retention-offer accept rate | > 20% |
| Product | Offers delivered per day | 80,000 – 120,000 |
| Model | AUC on the 30-day churn label | > 0.82 |
| Model | Precision@top-3% of the subscriber base | > 0.35 |
| System | Batch scoring job completes | < 4 hours |
| System | Real-time offer decision p99 | < 200 ms |

We will review these weekly in the retention stand-up. The primary metric for the launch
decision is **monthly voluntary churn rate**, measured over the first full month post-launch.

## 3. Architecture

```
                          ┌──────────────────────┐
   Event streams  ──────▶ │  Feature pipeline    │
   (plays, pauses,        │  (Spark, hourly)     │
    searches, logins,     └──────────┬───────────┘
    billing, support)                │
                                     ▼
                          ┌──────────────────────┐
                          │  Feature warehouse   │
                          │  + online KV store   │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┴────────────────────┐
              ▼                                           ▼
   ┌────────────────────┐                     ┌──────────────────────┐
   │  Nightly batch     │                     │  Real-time offer     │
   │  scoring (all 40M) │                     │  decision service    │
   │  writes risk_score │                     │  (in-app trigger)    │
   └─────────┬──────────┘                     └──────────┬───────────┘
             │                                           │
             ▼                                           ▼
   ┌────────────────────┐                     ┌──────────────────────┐
   │  Campaign service  │                     │  Offer Eligibility   │
   │  (email, next-day) │                     │  Service (external)  │
   └────────────────────┘                     │  + Billing API       │
                                              └──────────────────────┘
```

**Batch path.** Every night at 02:00 we score all 40 M subscribers, write `risk_score` to the
feature store, and hand the top 3% to the campaign service, which sends retention emails the
following morning.

**Real-time path.** When a high-risk user (`risk_score > 0.6`, read from the KV store) opens
the app, the offer decision service decides whether to show an in-app retention offer. Before
showing anything it must call the **Offer Eligibility Service** — a separate team's service
that checks promotional history, regional legal restrictions, and whether the user is already
on a discounted plan — and then the **Billing API** to confirm the account is in good standing
and to reserve the discount code. Only then is the offer rendered.

## 4. Data and features

18 months of historical event data, approximately 4.2 TB.

| Family | Examples |
|---|---|
| Engagement | Sessions in last 7/14/30 days, minutes watched, completion rate, days since last session |
| Content | Genre affinity, whether a followed series ended recently, watchlist additions |
| Account | Tenure, plan tier, price paid, number of profiles, devices |
| Billing | Payment failures, card expiry proximity, prior plan downgrades |
| Support | Tickets opened, sentiment score, unresolved tickets |
| Trend | 7-day-over-28-day deltas on the engagement features |

**Label.** `churned = 1` if the subscriber voluntarily cancelled within 30 days of the
observation date. Involuntary churn (payment failure) is excluded — it is a separate problem
handled by the dunning system.

**Splits.** We take all (subscriber, observation-date) pairs across the 18 months and split
them **80/10/10 at random** into train / validation / test. Random splitting gives us a much
larger and better-balanced training set than a time-based split would, and it avoids the
problem that the most recent months have incomplete 30-day labels. Aggregate features (rolling
engagement averages, lifetime watch minutes) are computed once over the full 18-month window
for efficiency.

**Class balance.** Roughly 3% positive. We downsample negatives 5:1 during training and
recalibrate the output probabilities afterwards with Platt scaling.

## 5. Model

**Gradient-boosted trees** (LightGBM, 800 trees, depth 8) over ~240 engineered features.

We evaluated a deep model with sequence features over the user's play history. It scored 0.006
AUC higher and required a GPU serving fleet. Given that the GBDT trains in 40 minutes on a
single 64-core CPU host and scores 40 M subscribers in under two hours, the extra complexity is
not justified at this stage. We will revisit if the sequence signal proves valuable in offline
experiments.

Retraining is **weekly**, on the trailing 18 months, triggered by an Airflow DAG. The new model
is written to `s3://streamco-models/churn/current/model.txt`, which the scoring job and the
real-time service both read at startup.

## 6. Serving and latency

**Batch scoring.** 40 M subscribers × 240 features. LightGBM inference is ~15 µs per row on one
core; with 64 cores and a 4× overhead factor for feature assembly, that is roughly
`40e6 × 15e-6 × 4 / 64 = 37.5` minutes of pure compute. Feature-warehouse reads dominate; we
budget **2 hours** end to end, comfortably inside the 4-hour window.

**Real-time offer decision.** Budget 200 ms p99:

| Hop | p99 |
|---|---|
| Edge + auth | 15 ms |
| Read `risk_score` and user context from KV store | 20 ms |
| Offer-selection logic (rules over risk band and tenure) | 10 ms |
| Render + return offer payload | 25 ms |
| Reserve for jitter and one retry | 30 ms |
| **Total** | **100 ms** |

Comfortably inside budget with 100 ms of headroom.

## 7. Evaluation and rollout

**Offline.** AUC, precision@k, and calibration on the held-out test split. Launch gate is
AUC > 0.82 and precision@3% > 0.35.

**Online.** We will A/B test at the subscriber level: 50% of subscribers are eligible for
proactive offers (treatment), 50% continue with the existing cancellation-page offer only
(control). Primary metric is monthly voluntary churn rate; we will run for one full month and
read the result.

**Rollout.** New model versions go to a 5% canary for 24 hours. We monitor batch job completion
time, score distribution, and offer volume. If those look healthy we promote to 100% by updating
`s3://streamco-models/churn/current/model.txt`. Both the scoring job and the real-time service
pick up the new file on their next restart.

## 8. Monitoring

Dashboards and alerts:

- Batch job completion time and success/failure
- Score distribution: mean, p50, p95, and the fraction above the 0.6 threshold
- Feature null rates and out-of-range values, per feature
- Offer volume per day, and offer accept rate
- Model AUC recomputed weekly against realised 30-day outcomes
- Real-time service latency p50/p99 and error rate

Alerts page the on-call ML engineer if the batch job fails, if offer volume moves more than 30%
day-over-day, or if the real-time service p99 exceeds 200 ms.

## 9. Cost

| Component | Monthly |
|---|---|
| Feature pipeline (Spark, hourly) | $8,000 |
| Nightly batch scoring | $2,200 |
| Real-time offer service (12 hosts) | $1,900 |
| Weekly retraining | $400 |
| **Total** | **$12,500** |

Against an expected 0.4 pt churn reduction — roughly 160,000 retained subscribers per month at
$180 lifetime value — the system pays for itself many times over.

## 10. Future work

- Sequence model over play history, if offline experiments justify the serving cost
- Uplift modelling instead of pure risk scoring
- Multi-armed bandit over offer types (discount vs free month vs content unlock)

---
---

## Your review

Write it as you would post it on the actual document. For each issue:

- **What** — name the problem precisely.
- **Why it matters** — the concrete production symptom, not a principle.
- **Fix** — specific enough that the author could act on it this sprint.
- **Severity** — and be prepared to defend the ordering.

Then answer the two questions a good reviewer always ends on:

1. **Which single change would you insist on before launch**, if you could only insist on one?
2. **What in this document is fine, and would you say so?** A review that only lists problems
   trains the author to write defensively rather than clearly.
