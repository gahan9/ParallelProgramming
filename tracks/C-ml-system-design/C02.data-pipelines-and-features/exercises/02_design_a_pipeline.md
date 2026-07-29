# Exercise 02 — Design the pipeline

**Goal:** produce a data-pipeline design that a staff engineer would approve: every feature has a
computation regime with a reason, every gate has a failure policy, and the splits match how the model
will be used.

**Time:** 60–90 minutes. **Deliverable:** the six sections below.

---

## The use case

A food-delivery marketplace shows the customer an **estimated delivery time (ETA)** at checkout.

- **Decision:** the ETA displayed at checkout, and the promise window derived from it.
- **Prediction task:** minutes from order placement to delivery at the door, predicted at checkout.
- **Output:** a point estimate plus an upper quantile (the promise is built from the p80, not the mean —
  under-promising costs conversion, over-promising costs trust).
- **Scale:** ~500 orders/second at peak; ~120,000 restaurants; ~200,000 active couriers.
- **Latency:** the ETA call sits in the checkout path with a **120 ms p99 budget end to end**.
- **Label:** actual delivery time, known **~40 minutes** after the order. Fast, clean labels — a
  deliberate contrast with C02's fraud examples.
- **Known problem:** ETAs are systematically optimistic during unexpected local demand spikes, which is
  exactly when accuracy matters most.
- **Feedback loop:** the displayed ETA changes whether the customer orders at all, and long ETAs
  suppress orders from busy restaurants.

## Candidate signals the team has proposed

| Signal | Description |
|---|---|
| `restaurant_median_prep_time_30d` | Median prep time for this restaurant over 30 days |
| `restaurant_median_prep_time_by_hour` | Same, bucketed by hour of week |
| `restaurant_open_orders_now` | Orders currently in this restaurant's kitchen queue |
| `restaurant_prep_time_last_10_orders` | Rolling mean over the restaurant's last 10 completed orders |
| `couriers_available_in_zone` | Couriers idle within the delivery zone right now |
| `zone_orders_last_5_min` | Order volume in the zone over the last 5 minutes |
| `distance_km` | Road distance restaurant → customer |
| `basket_item_count` | Items in this order |
| `basket_prep_complexity` | Weighted score from the item catalogue (pizza vs salad) |
| `is_peak_hour` | Derived from the order timestamp |
| `weather_condition` | Current weather in the zone |
| `customer_building_type` | Apartment / house / office, from address metadata |
| `customer_median_handoff_time` | This customer's historical door-handoff duration |
| `courier_median_speed_30d` | Assigned courier's historical speed (courier assigned *after* checkout) |
| `actual_prep_time` | The restaurant's actual prep time for this order |

Two of those signals are traps. Part 1 is where you catch them.

---

## Part 1 — Feature regime table

For each signal: choose **batch**, **streaming**, **on-demand**, or **reject**, and justify. Freshness
requirement means "how stale can this be before the ETA is wrong?"

| Signal | Regime | Freshness requirement | Why this regime | Cost/complexity note |
|---|---|---|---|---|
| `restaurant_median_prep_time_30d` | | | | |
| `restaurant_median_prep_time_by_hour` | | | | |
| `restaurant_open_orders_now` | | | | |
| `restaurant_prep_time_last_10_orders` | | | | |
| `couriers_available_in_zone` | | | | |
| `zone_orders_last_5_min` | | | | |
| `distance_km` | | | | |
| `basket_item_count` | | | | |
| `basket_prep_complexity` | | | | |
| `is_peak_hour` | | | | |
| `weather_condition` | | | | |
| `customer_building_type` | | | | |
| `customer_median_handoff_time` | | | | |
| `courier_median_speed_30d` | | | | |
| `actual_prep_time` | | | | |

Then answer explicitly:

**1a.** Which two signals must be **rejected**, and why? (One is a leak; the other is subtler.)

```
(your answer)
```

**1b.** Apply the decision rule from README §5 — *upgrade a feature to streaming when the decision
changes within the batch window*. Name the features that pass this test and the features that do not,
and give the counterfactual for one of each ("if this were an hour stale, the ETA would be wrong by
roughly ___ because ___").

```
(your answer)
```

**1c.** The "ETAs are optimistic during demand spikes" problem is a **freshness** problem. Which
specific features address it, and what happens to your cost if you upgrade all of them to streaming?

```
(your answer)
```

---

## Part 2 — Pipeline architecture

Sketch or describe the stages: sources → ingestion → validation → transformation → offline store →
online store → serving → feature log. For each stage state one design decision and its reason.

| Stage | Your design decision | Why |
|---|---|---|
| Sources | | |
| Ingestion | | |
| Validation | | |
| Transformation | | |
| Offline store | | |
| Online store | | |
| Serving retrieval | | |
| Feature logging | | |

Then answer:

**2a.** How do you guarantee the training and serving paths compute `restaurant_median_prep_time_30d`
identically? Give the mechanism, not the intention.

```
(your answer)
```

**2b.** Your online feature retrieval must fit inside a 120 ms p99 end-to-end budget. State your
retrieval budget, how many online features that implies, and one thing you would do if the budget were
exceeded.

```
(your answer)
```

**2c.** Which stages must be idempotent, and what specifically would break if they were not?

```
(your answer)
```

---

## Part 3 — Validation gates

For each check family, state the concrete check, the threshold, and — most importantly — **what happens
when it fires**.

| Check family | Concrete check for this system | Threshold | On failure: block / reject rows / warn? Who is paged? |
|---|---|---|---|
| Schema | | | |
| Volume | | | |
| Per-value rules | | | |
| Distribution (drift) | | | |
| Cross-dataset | | | |
| Freshness / staleness | | | |
| Training-serving skew | | | |

**3a.** Two of these gates protect the *training* path and two protect the *serving* path. Which are
which, and why does that distinction change the failure policy?

```
(your answer)
```

**3b.** `couriers_available_in_zone` goes null for one city because a producer deploys a breaking
change. Walk through what your design does, from the first bad value to the customer-visible outcome.

```
(your answer)
```

---

## Part 4 — Splits and evaluation

**4a.** Choose the split strategy and defend it. Consider: many orders per restaurant, many orders per
customer, strong time-of-day and day-of-week seasonality, and a 40-minute label delay.

```
(your answer)
```

**4b.** Which entity would you group on if you group — restaurant, customer, courier, or zone? Argue for
one and say what you lose.

```
(your answer)
```

**4c.** The label arrives in 40 minutes, so you *could* retrain hourly. Should you? What breaks if you
retrain on a feedback loop where your own ETA influenced whether the order was placed?

```
(your answer)
```

**4d.** Name the offline metric, and name one guardrail (C01's vocabulary) that would catch the model
gaming it.

```
(your answer)
```

---

## Part 5 — Versioning and lineage

State the mechanism for each, in one line:

| Artifact | Your mechanism |
|---|---|
| Dataset version | |
| Feature definition version | |
| Code version | |
| Label definition | |
| Lineage from a deployed model to its inputs | |

Then: a customer disputes a promise made six weeks ago. What must you be able to reconstruct, and does
your design allow it?

```
(your answer)
```

---

## Part 6 — Failure behaviour

The feature store is unavailable. Write the fallback behaviour as a testable requirement, and state
what the customer sees.

```
(your answer)
```

---

## Self-check

- [ ] You rejected exactly two signals and explained both, including the subtle one.
- [ ] **Not everything is streaming**, and you justified each streaming choice against the batch-window
      rule.
- [ ] Every validation gate has a failure *policy*, not just a threshold.
- [ ] You distinguished gates that protect training from gates that protect serving.
- [ ] Your split choice accounts for both time and entity, and you named what you lose.
- [ ] You addressed the feedback loop in Part 4c rather than only the mechanics of retraining.
- [ ] The latency budget produced a number that constrains your *feature set*, not just the model.
- [ ] Failure behaviour is written as something a test could check, and includes what the customer sees.

Reference answer with the reasoning:
[`../solutions/02_design_a_pipeline_solution.md`](../solutions/02_design_a_pipeline_solution.md).
