# Exercise 03 — Skew postmortem: offline 0.94, online flat

**Goal:** run a structured diagnosis under ambiguity. This is the closest drill in the curriculum to a
real ML on-call incident, and some version of it is asked in most ML system design interviews.

**Time:** 45–60 minutes. **Deliverable:** Part 1's ranked table, Part 3's diagnosis, and Part 4's
incident summary.

**Work Part 1 before reading Part 2.** Part 2 contains the evidence, and the skill being drilled is
generating the hypotheses *before* you have it.

---

## The incident

A payments company deployed a new card-not-present fraud model three weeks ago.

- **Offline:** ROC-AUC **0.94**, PR-AUC 0.41 on a temporal split over three months of warehouse data.
  The previous production model scored 0.89 offline.
- **Online, three weeks in:** caught-fraud value per alert is **statistically indistinguishable** from
  the old model. Analysts say the new alerts "feel more random."
- **No errors.** No alerts firing, no failed jobs, no schema exceptions, latency well within budget.
- **Base rate:** ~0.15% of transactions, unchanged from the training period per the risk team.
- **Label horizon:** a chargeback is confirmed up to **45 days** after the transaction.
- **Deployment:** the new model replaced the old one at 100% traffic on day one. There was no shadow
  period.

What you know about the system:

- Training features are computed by a **Spark SQL** job over the warehouse.
- Serving features are computed by a **Go service** reading a Redis online store.
- The serving path **does** log the exact feature vector it used. Nobody has looked at those logs.
- The offline evaluation used the **temporal split** described above: train on months 1–2, test on
  month 3.
- The model is a gradient-boosted tree on 84 features.

---

## Part 1 — Hypotheses in priority order (do this first)

List **at least seven** candidate causes. Rank by `likelihood × (1 / cost to check)` — the cheap,
likely checks go first — and for each give the **specific** check that would confirm or eliminate it.
"Investigate the data" is not a check; "compare null rates between the logged serving vectors and the
training set, per feature" is.

| Rank | Hypothesis | Specific check | Confirms if... | Eliminates if... |
|---:|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |
| 7 | | | | |

**1a.** Which single check has the highest information value per hour spent, and why?

```
(your answer)
```

**1b.** Two of your hypotheses are "the offline number was never real" and two are "the offline number
was real but production throws it away." Label which are which. Why does that distinction matter for
what you do next?

```
(your answer)
```

**1c.** One hypothesis on your list would mean **nothing is broken at all**. Name it, and say what
evidence would support it.

```
(your answer)
```

---

## Part 2 — The evidence

*Read only after completing Part 1.*

You run your checks. Here is what comes back.

**E1. Score distributions.** Offline scores on the test month: mean 0.021, p99 0.62. Online scores over
the last week: mean 0.008, p99 0.31. The online distribution is shifted materially lower and is
compressed.

**E2. Null rates, top 10 features by importance.** Comparing the training set against the logged
serving vectors:

| Feature | Training null rate | Serving null rate |
|---|---:|---:|
| `amount` | 0.000 | 0.000 |
| `orders_last_30d` | 0.003 | **0.221** |
| `avg_basket_value_90d` | 0.004 | **0.208** |
| `card_country_match` | 0.001 | 0.002 |
| `device_reputation` | 0.011 | 0.019 |
| `txns_last_10_min` | 0.000 | 0.000 |
| `account_age_days` | 0.000 | 0.000 |
| `billing_shipping_distance` | 0.031 | 0.034 |
| `merchant_category_risk` | 0.000 | 0.000 |
| `email_domain_age` | 0.048 | 0.051 |

Both pipelines replace a null with `0` before the model sees it.

**E3. Re-scoring logged vectors.** You take 10,000 logged serving vectors, re-score them with the same
model artifact offline, and compare with the score the serving path returned. **The scores match to
within floating-point tolerance.**

**E4. Feature-value comparison.** You recompute the 84 features offline, from the warehouse, *as of*
each of those 10,000 transaction timestamps, and diff against the logged serving values:

- 81 features agree.
- `avg_basket_value_90d` disagrees on **73%** of rows. Reading the code: Spark SQL computes a mean over
  a **90-day time window**; the Go service computes the mean of the customer's **last 90 order events**,
  which for a high-frequency customer spans two weeks and for a low-frequency customer spans two years.
- `orders_last_30d` disagrees on **22%** of rows — exactly the rows where the serving value was null.
  The Redis key is only populated for customers with activity since the online store was backfilled
  four months ago.
- `merchant_category_risk` disagrees on **4%** of rows: the warehouse has a corrected mapping that the
  serving path's static config file does not.

**E5. Leakage screen.** Timestamp and separation screens over the training table: no feature exceeds
0.99 single-feature AUC; the highest is `txns_last_10_min` at 0.71. No feature timestamp is at or after
the label timestamp.

**E6. Label maturity.** The training set includes transactions through 12 days before the training
cutoff. The label horizon is 45 days.

**E7. Statistical power.** The online comparison has accumulated roughly 40 confirmed-fraud cases in
the treatment arm so far.

---

## Part 3 — Diagnose

**3a.** State the **primary** cause in one sentence, and cite which evidence establishes it.

```
(your answer)
```

**3b.** E3 shows the online and offline scores *match* on the logged vectors. A teammate concludes "so
there is no skew." Explain precisely why that conclusion is wrong, and what E3 actually rules out.

```
(your answer)
```

**3c.** There are **at least three separate defects** in the evidence, of differing severity. Name each,
rank it, and say what it costs.

```
defect 1 (most severe):
defect 2:
defect 3:
```

**3d.** E6 describes a fourth problem that is independent of skew. What is it, which direction does it
bias the model, and how would you fix it?

```
(your answer)
```

**3e.** E7. Can you currently conclude the new model is *not* better? Do the arithmetic and say what
you would tell the product owner about the confidence of the "no improvement" claim.

```
(your answer)
```

**3f.** Why did the **offline temporal split** fail to catch any of this? Be precise — the split was
correctly constructed.

```
(your answer)
```

---

## Part 4 — Incident summary

Write the postmortem body. Keep it under 400 words, blameless, and specific. Cover: impact, root cause,
contributing factors, immediate remediation, and prevention. Prevention items must be **detections and
structural changes**, not "be more careful."

```
## Impact

## Root cause

## Contributing factors

## Immediate remediation

## Prevention
```

---

## Part 5 — The prevention that would have caught it on day one

Rank these five candidate prevention measures by how much of this incident each would have prevented,
and say what each costs to build:

| Measure | Rank | What fraction of this incident it catches | Cost to build |
|---|:---:|---|---|
| Shadow-deploy the new model and compare online score distributions before switching traffic | | | |
| A continuous training/serving skew job: recompute features offline for sampled requests and alert on disagreement rate | | | |
| One shared transformation definition (feature store) for both paths | | | |
| Train on logged serving features instead of warehouse reconstructions | | | |
| A per-feature null-rate comparison between training and serving as a release gate | | | |

Then: **if you could build only one**, which and why?

```
(your answer)
```

---

## Self-check

- [ ] Part 1 has at least seven hypotheses, each with a *specific* check.
- [ ] You separated "the measurement was wrong" from "production throws away a real gain."
- [ ] You included the hypothesis that nothing is broken.
- [ ] You explained why E3 (matching scores) does **not** rule out skew — this is the trap in the
      exercise.
- [ ] You found the `avg_basket_value_90d` window-semantics bug and the `orders_last_30d` coverage bug,
      and ranked them.
- [ ] You noticed the immature-label problem in E6 and stated its direction of bias.
- [ ] You did the power arithmetic in 3e rather than accepting "no improvement" at face value.
- [ ] Your prevention items are detections and structural changes with costs attached.

Reference answer with the reasoning:
[`../solutions/03_skew_postmortem_solution.md`](../solutions/03_skew_postmortem_solution.md).
