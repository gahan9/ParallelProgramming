# Solution 02 — Design the pipeline

Reference answer for [`../exercises/02_design_a_pipeline.md`](../exercises/02_design_a_pipeline.md).
Your design will differ; check that you made the moves in [`README.md`](README.md).

---

## Part 1 — Feature regime table

| Signal | Regime | Freshness requirement | Why | Cost note |
|---|---|---|---|---|
| `restaurant_median_prep_time_30d` | **batch** (daily) | 24 h | A 30-day median moves by minutes per day; a day of staleness cannot change the ETA meaningfully | Cheapest possible; one daily job over 120k restaurants |
| `restaurant_median_prep_time_by_hour` | **batch** (daily) | 24 h | Same argument, and hour-of-week buckets need weeks of history to be stable anyway | Same job; more rows, negligible extra cost |
| `restaurant_open_orders_now` | **streaming** | < 30 s | This is the demand-spike signal. A kitchen queue goes from 2 to 15 in minutes and that is worth 15+ minutes of ETA | Stateful stream keyed by restaurant; the main new infrastructure |
| `restaurant_prep_time_last_10_orders` | **streaming** | < 60 s | Captures "this kitchen is running hot today" in a way the 30-day median cannot. Ten events is a short window by design | Same stream, small per-key state |
| `couriers_available_in_zone` | **streaming** | < 30 s | Courier supply is the other half of a spike; a zone can go from 40 idle couriers to 3 in ten minutes | Stateful stream keyed by zone; fewer keys than restaurants |
| `zone_orders_last_5_min` | **streaming** | < 30 s | The window (5 min) is shorter than any affordable batch cadence, so batch cannot express it at all | Same zone stream |
| `distance_km` | **on-demand** | zero | Purely a function of the request (two addresses). Precomputing it would require materializing every address pair | Spends latency, not compute; needs a routing call, so cache aggressively |
| `basket_item_count` | **on-demand** | zero | It is *in* the request | Free |
| `basket_prep_complexity` | **on-demand** (batch catalogue lookup) | catalogue: 24 h | Computed at request time from the basket, using per-item weights that come from a daily batch job over the catalogue | Cheap; one small dictionary lookup |
| `is_peak_hour` | **on-demand** | zero | A function of the request timestamp and the zone's local time | Free. Never precompute a clock |
| `weather_condition` | **batch/streaming hybrid** (poll every 5–10 min per zone) | < 10 min | Weather changes slowly relative to a 40-minute delivery, but "it started raining" matters | Small: a few thousand zones, not per-request. **Never** call the weather API per request |
| `customer_building_type` | **batch** (daily) | 24 h | Address metadata is effectively static | Cheapest |
| `customer_median_handoff_time` | **batch** (daily) | 24 h | A per-customer historical median over many orders; one day of staleness is irrelevant. **Needs a cold-start fallback** to the building-type or zone median for new customers | Cheap; the fallback is the design work |
| `courier_median_speed_30d` | **REJECT** | — | The courier is assigned *after* checkout, so at prediction time there is no courier. This is a time-travel leak | — |
| `actual_prep_time` | **REJECT** | — | It is a component of the label, observed after the fact | — |

### 1a. The two rejections

**`actual_prep_time` is the easy one:** it is part of the thing being predicted, observed after the
prediction moment. Any model given it will report a spectacular offline error and produce nothing
useful.

**`courier_median_speed_30d` is the subtle and more interesting one.** Nothing about the feature is
contaminated — a courier's historical speed is a legitimate, point-in-time-correct, batch-computable
value. The problem is **which courier**. At checkout no courier has been assigned, so the training
join ("attach the courier who actually delivered this order") uses information created *after* the
decision. And it is worse than merely unavailable: courier assignment is not random. Fast couriers get
assigned to urgent orders, so the feature encodes the dispatcher's future decision, and the model
learns a relationship that inverts in production.

The right response is not simply to delete it but to **replace it with the point-in-time-correct
version of the same idea**: `median_speed_of_available_couriers_in_zone`, computed from the couriers
who *were* idle in the zone at checkout. Same information content, legitimately available. Recognizing
that a rejected feature often has a legal twin is the senior move here, and it is the same move as
`region_fraud_rate` → `region_fraud_rate_pit` in exercise 1.

There is also a third signal worth flagging even though it is not a rejection:
`customer_median_handoff_time` is fine but has a **cold-start hole**. A first-time customer has no
history, and first-time customers are exactly the population whose experience matters most for growth.
Ship the fallback with the feature, not after.

### 1b. The batch-window rule

*Upgrade a feature to streaming when the decision changes within the batch window.*

**Passes (must be streaming):** `restaurant_open_orders_now`, `restaurant_prep_time_last_10_orders`,
`couriers_available_in_zone`, `zone_orders_last_5_min`. All four move on a timescale of minutes and all
four move *most* during the demand spikes where the ETA is currently wrong.

**Fails (batch is correct):** both prep-time medians, `customer_building_type`,
`customer_median_handoff_time`, the catalogue weights behind `basket_prep_complexity`.

Counterfactuals:

- **`couriers_available_in_zone`, one hour stale:** during a rain-driven spike, courier supply can fall
  substantially within the hour while the stale feature still reports the pre-spike count. The model
  would use a courier-wait input from before the spike and under-predict the wait by roughly the
  duration of the pickup queue that formed — which is precisely the reported symptom. The error is
  worst exactly when the feature matters most, so the average staleness understates the harm.
- **`restaurant_median_prep_time_30d`, one hour stale:** a 30-day median changes by at most one order's
  worth of influence in an hour, i.e. a rounding error against a 40-minute prediction. Streaming this
  buys nothing and costs a stateful pipeline.

That asymmetry — some features are worthless when stale and others are indifferent — is the whole
argument against "make everything streaming."

### 1c. The demand-spike problem

It is a freshness problem, addressed by the four streaming features above, and the two most important
are `couriers_available_in_zone` and `restaurant_open_orders_now` because a spike is simultaneously a
supply shortage and a kitchen backlog.

What upgrading them costs, honestly: one stateful stream processing job with **two keyed aggregations**
(by restaurant, by zone), which means always-on compute, watermarking and out-of-order handling, state
checkpointing, a replay path for when it falls behind, and a permanent on-call surface. The compute is
not the expensive part; the **operational** cost is. That is also why the answer is four streaming
features rather than fifteen — each one added to the streaming job adds state, latency, and a way for
the job to fall over.

One more thing the freshness fix does *not* solve: if the ETA is optimistic during spikes even with
fresh features, the residual is a **model** problem, and the right lever is the loss function. The
promise is built from the p80, so train with **quantile loss** rather than squared error; under-promising
and over-promising have different costs
([C01 §3](../../C01.problem-framing-and-metrics/README.md#3-deep-dive)), and a symmetric loss cannot
express that.

---

## Part 2 — Pipeline architecture

| Stage | Design decision | Why |
|---|---|---|
| Sources | Order events, courier-location stream, restaurant POS events, catalogue CDC, weather API poller, address metadata | Distinguish streams (order/courier/POS) from slowly-changing dimensions (catalogue, address) — they need different regimes and different validation |
| Ingestion | Append-only raw topics and landing tables, partitioned by **event time**, immutable, schema-on-read | Late courier pings and delayed POS events must land in the partition they happened in, or every as-of query is wrong. Never edit raw |
| Validation | Schema-on-write gate at every curated table; the gate can reject and publish nothing | This is the boundary before anything a model touches. Raw stays permissive; curated is strict |
| Transformation | **One feature definition per feature**, in a shared library/feature-store DSL, materialized to both stores. No feature is implemented twice | The single highest-value decision in the whole design; it removes the dominant failure mode by construction |
| Offline store | Columnar, partitioned by event time, with point-in-time-correct as-of retrieval for training | Training joins must answer "what did we know at checkout?", not "what do we know about this restaurant?" |
| Online store | Key-value, keyed by `restaurant_id`, `zone_id`, `customer_id`; **pre-joined per key** so serving does one lookup per key, not one per feature | Latency: three multi-get lookups is a budget you can hold; 40 round trips is not |
| Serving retrieval | Batch multi-get per key in parallel, hard timeout, explicit per-feature default with a `_was_default` indicator emitted alongside | A default is a *fact about this request* and must be visible to monitoring, never silent |
| Feature logging | Log the exact served feature vector, its defaults mask, model version, feature-definition version, and the ETA shown — for **every** request | Enables training on served features, and makes the skew check a measurement instead of an argument |

### 2a. Guaranteeing identical computation

The mechanism, in order of strength:

1. **One definition, two materializations.** `restaurant_median_prep_time_30d` is declared once (in the
   feature store DSL or a shared library); the offline job and the online materializer both execute
   *that* definition. There is no second implementation to diverge.
2. **Serve from the materialized value, never recompute at request time.** The Go serving path reads a
   number; it does not contain any window logic to get wrong. This eliminates the exact bug in
   exercise 3 (`avg_basket_value_90d` computed over 90 days in one path and 90 *events* in the other).
3. **A continuous skew job.** Sample logged serving vectors, recompute the same features offline as of
   those timestamps, and alert on per-feature disagreement rate. This converts "we believe they agree"
   into a monitored metric.
4. **Train on the logged features.** The strongest guarantee, because then the training distribution is
   the serving distribution by definition — including the defaults.

Item 1 without item 3 is an assumption; item 3 is what makes item 1 verifiable.

### 2b. Latency budget

```
end-to-end p99 budget                          120 ms
  network + serialization + service overhead   ~20 ms
  model inference (GBT or small net)           ~10 ms
  routing/distance call (cached)               ~25 ms
  ---------------------------------------------------
  remaining for feature retrieval              ~65 ms
```

Budget the online store at **p99 ≤ 40 ms**, keeping ~25 ms of headroom for tail interference, and
achieve it by pre-joining features per key so retrieval is **three parallel multi-gets** (restaurant,
zone, customer) rather than one call per feature. Under that structure the feature *count* barely
matters — the number of **keys** does. That is the real constraint on the feature set: adding a fourth
entity type costs a round trip, whereas adding a fifteenth restaurant feature costs bytes.

If the budget is exceeded, in order of preference: (1) drop the lowest-importance key group entirely —
measure the ETA error cost of losing it rather than guessing; (2) move computable features to
on-demand so they need no lookup; (3) cache per-restaurant features in-process with a short TTL, since
a few seconds of extra staleness on a 30-day median is free; (4) serve a degraded ETA from the fallback
path rather than blowing the checkout budget. **Never** let the ETA call block checkout — the fallback
is cheaper than a lost order.

### 2c. Idempotency

Every batch transformation and every backfill must be idempotent; the streaming aggregations must be
replayable to a deterministic state from a checkpoint plus the retained log.

What breaks without it: you cannot **backfill** the 30-day median after fixing a bug (re-running
produces a different table, so you can never reconcile); you cannot **reproduce** the training set that
produced a deployed model, which kills both debugging and audit; and you cannot **recover** from a bad
deploy by replaying, so every incident becomes a manual data-repair job. The usual culprits are
`now()` inside a transformation, append-rather-than-overwrite partition writes, and partitioning by
ingestion time.

---

## Part 3 — Validation gates

| Check family | Concrete check | Threshold | On failure |
|---|---|---|---|
| Schema | Declared columns/types present on every curated table; new undeclared columns reported | Missing/retyped column = error; new column = warning | **Block publication and page the pipeline owner.** Serving keeps the previous day's batch features |
| Volume | Row count per partition vs the trailing band **for the same weekday** | Outside the historical band for that weekday | Block and page. Weekday matters — Saturday is not Tuesday in food delivery |
| Per-value rules | `prep_time > 0` and below a sane ceiling; `distance_km` within the zone's plausible range; `restaurant_id` resolves; no duplicate order ids | Any violation | **Reject the rows**, publish the rest, and alert if the rejection rate exceeds its own threshold. High-volume event data always has some corruption |
| Distribution | PSI per numeric feature vs a trailing reference window; category shares for `weather_condition`, `building_type` | The 0.1/0.25 convention, **tuned to this data's normal weekly movement** | Warn and page during business hours. A real spike *is* a distribution change, so this must not auto-block |
| Cross-dataset | `restaurant_id` and `zone_id` foreign keys resolve; order counts reconcile against the orders service | Any unresolved key above a small tolerance | Block the affected partition |
| Freshness | Per-feature staleness p50/p99 vs the declared requirement in Part 1 | Streaming features: p99 > 60 s. Batch: > 26 h | Page immediately. **Serving marks affected features as defaulted** rather than using a stale value silently |
| Training/serving skew | Sampled logged vectors recomputed offline; per-feature disagreement rate | Any feature above a small disagreement rate | Page, and **block the next model release** until explained |

### 3a. Training-path gates vs serving-path gates

- **Protecting training:** schema, volume, cross-dataset, and the distribution check on the training
  window. A poisoned training table produces a bad model that then runs for weeks, so the cost of a
  bad publish greatly exceeds the cost of a delay. **Policy: block and page.** Yesterday's data is
  better than wrong data.
- **Protecting serving:** freshness and skew. Here blocking is not available — checkout is happening
  right now and there is no "delay" option. **Policy: degrade explicitly and visibly** (mark the
  feature defaulted, fall back, emit the metric) and page.

The distinction changes the policy because **the training path can wait and the serving path cannot.**
Applying a blocking policy to a serving-path gate means taking down checkout; applying a degrade-quietly
policy to a training-path gate means shipping a model trained on garbage. Both are common mistakes and
they are opposites.

### 3b. `couriers_available_in_zone` goes null for one city

1. The producer deploys; courier pings for that city stop arriving or arrive malformed.
2. **Freshness check fires within a minute** — staleness p99 for that zone group crosses 60 s. On-call
   is paged. This is the fast detection, and it is why staleness is monitored per feature rather than
   per job.
3. Simultaneously, the **serving path's timeout** produces a missing value. It substitutes the declared
   default (the zone's trailing typical courier count, *not* zero — zero would say "no couriers", which
   is a confidently wrong statement) and sets `couriers_available_in_zone_was_default = 1`.
4. That indicator is logged and aggregated. A **default-rate** alert for the zone fires, confirming the
   scope is one city rather than global.
5. **Customer impact:** ETAs in that city degrade toward the historical average — they lose their
   spike-awareness but remain sane. Checkout is never blocked. If the default rate stays high, the
   fallback policy from Part 6 widens the promise window for that zone, trading a slightly worse
   headline ETA for not breaking promises.
6. **Training impact:** rows from the affected window are flagged. Because the feature log records the
   defaults mask, the next training run can either exclude them or learn the defaulted regime
   explicitly — a choice you can now make, because the information was recorded rather than silently
   coerced.

Step 3 is the load-bearing one. Defaulting to zero, silently, is the difference between a degraded
service and a mystery incident three weeks later.

---

## Part 4 — Splits and evaluation

### 4a. Split strategy: **temporal, primary; grouped as a secondary evaluation**

Temporal is mandatory: this system predicts the future, there is strong hour-of-day and day-of-week
seasonality, and restaurant behaviour changes with staffing, menu, and season. A random split would let
the model learn from next Friday to predict last Friday and would average away exactly the regime
shifts you need to survive.

Concretely: train on weeks 1–8, validate on week 9, test on weeks 10–11, and hold **whole days** on one
side of each boundary so a day is never split (intra-day correlation is strong). The 40-minute label
delay means the gap between train and test can be trivially small — a nice contrast with the 45-day
horizon in exercise 3, where the gap must be at least one label horizon.

**Also run a grouped-by-restaurant evaluation as a second, reported number**, because new restaurants
join constantly and their ETA quality is a business problem in its own right. Report both: the temporal
number is the launch gate, the grouped number is the cold-start number. When they disagree, believe the
more pessimistic one for the population it describes.

### 4b. Which entity to group on: **restaurant**

Restaurant is the strongest source of both signal and memorization: prep time is dominated by the
kitchen, the per-restaurant features are the ones most likely to be memorized, and new restaurants
onboard continuously — so "how good is the ETA for a restaurant we have never seen?" is a real,
recurring question. Customer is a weaker candidate (customer features contribute much less than
restaurant features), zone is too coarse (a handful of zones means enormous variance), and courier is
irrelevant once `courier_median_speed_30d` is rejected.

**What you lose:** a grouped-by-restaurant split throws away all per-restaurant history for the held-out
restaurants, so it is *pessimistic* for established restaurants — which are the overwhelming majority of
orders. That is why it is a secondary, cold-start-specific number rather than the headline. Reporting it
as the main metric would understate real-world performance just as badly as a random split overstates
it.

### 4c. Should you retrain hourly?

**No, and the reason is the feedback loop, not the cost.** The displayed ETA changes whether the order
is placed, so your training data is a sample *selected by your own predictions*:

- When the model predicts a long ETA, price-sensitive or hungry customers abandon. The orders you
  observe from that restaurant-hour are the ones from customers willing to wait.
- Long ETAs suppress orders to busy restaurants, which reduces their queue, which shortens their
  observed prep times, which makes the model predict shorter ETAs, which brings the orders back. The
  system oscillates, and a tight retrain loop makes the oscillation faster and harder to attribute.
- The abandoned orders have **no label at all** — classic partial feedback. You never learn what would
  have happened.

Practical resolution: **retrain daily, not hourly**, so the loop has a slower time constant than the
phenomena you are modelling; keep a small **randomized holdout** where the displayed ETA is perturbed
slightly (within an honest range) to break the correlation between prediction and selection; and log
abandonment against the displayed ETA so the selection effect is *measured* rather than assumed. If you
genuinely need minute-scale adaptation, put it in the **features** (which is what the streaming
features are for) rather than in the model weights — that adapts fast without closing a training loop
around your own outputs.

### 4d. Metric and guardrail

**Offline target metric: quantile loss at the p80**, because the promise is built from the p80, and
optionally MAE on the point estimate as a diagnostic. Not RMSE: it is minimized by the conditional
mean, and it lets a few catastrophic misses dominate a metric whose business meaning is "did we keep
the promise?"

**The guardrail that catches gaming: promise-breach rate** — the fraction of orders delivered after the
promised window — **reported per zone and per restaurant tier, together with the mean promised ETA.**
The gaming move is obvious once you name it: the model can drive breach rate toward zero by inflating
every ETA. So the pair must be reported together, and mean promised ETA (or its proxy, checkout
conversion) is the counter-metric. This is C01 §3's structure applied directly: one target, a guardrail
that catches the target's most likely abuse, and a counter-metric that moves the wrong way if you cheat.

---

## Part 5 — Versioning and lineage

| Artifact | Mechanism |
|---|---|
| Dataset version | Immutable event-time partitions in a table format with time travel (Delta Lake / Iceberg); training records the snapshot version, not a query |
| Feature definition version | Definitions in version control; a semantic version in feature metadata, and a **new feature name** on any breaking semantic change (never silently redefine `..._30d`) |
| Code version | Git SHA of the training pipeline and the serving service recorded in model metadata |
| Label definition | "Minutes from order placement to door handoff," versioned as an interface document, with the handoff event definition pinned |
| Lineage | A model registry entry linking model artifact → dataset snapshot version → feature definition versions → code SHAs → label definition version. Serving logs the model version and feature-definition version on every request |

**The six-week-old dispute.** You must reconstruct: the exact ETA shown, the promise window derived from
it, the feature vector used (including which values were defaults), the model version, the feature
definition versions, and the actual delivery time. The design supports it **because every request's
feature vector and versions are logged** — this is the same log that enables the skew check and
training-on-served-features, which is why one mechanism pays for itself three times.

What the design would *not* support without the versioned snapshots: recomputing what the features
*should* have been at that moment, in order to distinguish "the model was wrong" from "the features
were wrong." That distinction is the whole reason the dataset version matters.

---

## Part 6 — Failure behaviour

> **Requirement.** If online feature retrieval fails or exceeds its 40 ms p99 budget, the ETA service
> shall respond within the 120 ms end-to-end budget using the fallback estimator — a
> zone-and-hour-of-week median delivery time from the last completed batch, plus a distance term —
> shall widen the promise window by the fallback's historical p80 error margin, shall set
> `eta_source = "fallback"` and `features_defaulted = <mask>` on the response and in the feature log,
> and shall page the on-call owner if the fallback rate for any zone exceeds 1% of requests over five
> minutes. The service shall never block checkout and shall never return no ETA.

**What the customer sees:** an ETA, always — slightly more conservative and less spike-aware than
normal, with a wider promise window. They do not see an error, a spinner, or a checkout they cannot
complete. Internally, every one of those responses is tagged, so the business impact is measurable
after the fact and those rows can be excluded from or explicitly modelled in training.

The design principles behind that requirement, stated plainly: **degrade, never fail**; **be
conservative when uncertain** (a wider promise trades a slightly worse headline number for keeping a
promise); **make the degradation visible in the data**, not just in a log line; and **alert on the
*rate* of degradation**, because a 0.01% fallback rate is normal operations and 5% is an incident.

---

## Common mistakes on this exercise

| Mistake | Why it costs you |
|---|---|
| Everything streaming "for freshness" | Four features need it; eleven do not. The cost is operational and permanent |
| Only rejecting `actual_prep_time` | `courier_median_speed_30d` is the interesting rejection, and it has a legal twin worth naming |
| Deleting the rejected feature without proposing the point-in-time-correct replacement | Same move as exercise 1's `region_fraud_rate_pit`; the information is usually recoverable |
| Gates with thresholds but no failure policy | A threshold with no policy is a dashboard, not a gate |
| The same failure policy for training and serving gates | Training can wait; serving cannot. Opposite policies |
| Defaulting a missing count to 0 silently | Zero is a confident lie. Default to a sane prior and emit an indicator |
| Random split, or a grouped split as the headline | Random overstates; grouped-only understates for the majority population |
| Retraining hourly because labels are fast | The feedback loop, not the label delay, is the binding constraint |
| A latency budget that constrains only the model | It constrains the number of **key groups** you can retrieve, which is a feature-set decision |
| Failure behaviour as "use a cached value" | Cached how stale? Visible how? Alerting at what rate? Write it as a testable requirement |
