# Solution 2 — Rollout plan for `rank-v8`

The point of this drill is a single uncomfortable number, and it arrives in §3. Everything else is
scaffolding around it.

---

## 1. Pre-flight gate — before any traffic

| Check | Automated? | Failure means |
|---|---|---|
| Model artefact loads and produces output on a fixed golden input set | yes | block |
| Feature parity: the three new features compute identically in the training pipeline and the serving path, verified on 10 000 replayed requests | yes | block — this is the [C02](../../C02.data-pipelines-and-features/README.md) skew check and it is the highest-risk item here |
| **The new feature-store table's backfill history** — six weeks of data, but is it *correct* for all six weeks, and is the coverage the same for all tenants/segments? | **manual** | block |
| Null/missing rate for each new feature in serving, per segment, against training | yes | block if any segment exceeds the training-time rate materially |
| Latency benchmark at production prompt-length distribution, p50/p99 | yes | block if p99 > 85 ms |
| Cost per 1 000 requests recomputed at the measured 2.3× | yes | inform, not block |
| Previous version `rank-v7` confirmed loadable within the target rollback time | yes | block |
| Prediction log schema includes `model_version` | yes | block |

**The item that should worry you most is the six-week-old feature pipeline.** Three specific
failure modes, none exotic:

1. **Backfill ≠ live.** The historical values were computed by a backfill job; the live values come
   from a streaming path. These are two different implementations of the same definition, which is
   the textbook source of training/serving skew.
2. **Six weeks is short.** If the feature has any seasonality — weekly is guaranteed here given the
   2.4× weekday/weekend swing, and monthly is likely for a marketplace — the model has never seen a
   full cycle of it.
3. **Coverage skew.** New tenants, new categories or new geographies may have sparse values in the
   backfill and dense values live, or the reverse. Check coverage *per segment*, not globally; a
   global null rate of 2% can hide a 60% null rate in one country.

Verify all three before deploying anything. This is cheap now and very expensive during a canary.

---

## 2. Stage plan

| Stage | Traffic | Duration | Sessions | What it can detect | Exit criteria |
|---|---:|---|---:|---|---|
| **0. Shadow** | 5% mirrored | 48 h | ~105k mirrored | Crashes, latency, feature-null rates, **output divergence** vs v7 | p99 model latency ≤ 85 ms; zero unhandled errors; feature-null rate within 0.5 pp of training; divergence distribution reviewed by a human |
| **1. Canary 1%** | 1% | 24 h | ~21k | System health only — **cannot** detect a quality regression (see §3) | Error rate ≤ baseline × 1.5; p99 page latency ≤ 400 ms; no SRM |
| **2. Canary 5%** | 5% | 72 h (spans a weekend) | ~315k | System health; a *large* (≥ 4% relative) conversion regression | As above, plus conversion CI excludes a −4% relative regression |
| **3. Split 50/50** | 50% | 14 days | ~14.7 M | **The actual experiment.** Detects a 1% relative regression with 80% power | Pre-registered analysis; conversion CI excludes a −1% relative change; guardrails clean |
| **4. Full** | 100% | — | — | — | v7 stays loadable for 30 days |

**Durations, justified:**

- **Shadow 48 h** — long enough to include one full weekday cycle plus the overnight batch window,
  where the feature pipeline behaves differently.
- **Canary 1% at 24 h** — 21k sessions is enough to see a crash, a latency regression, or an SRM. It
  is nowhere near enough to see a conversion change, and the plan says so rather than pretending.
- **Canary 5% at 72 h** — must span a weekend because the weekday/weekend swing is 2.4× and the
  *traffic mix* differs (browsing versus purchasing intent). A Tuesday-to-Thursday canary measures
  Tuesday-to-Thursday users.
- **Stage 3 at 14 days** — two full weekly cycles, which is also roughly what the sample size
  demands, and long enough for novelty effects on a ranking change to begin decaying.

**What shadow mode can and cannot tell you here.** It *can* tell you that v8 runs, how fast, with
what feature-null rates, and how often its ranking differs materially from v7's. It **cannot** tell
you whether the different ranking is *better*, because no user ever sees it — there is no click, no
purchase, no abandonment. Shadow is a safety check. Treating it as evidence of quality is a category
error, and it is a common one.

---

## 3. Sample size — the crux

```bash
python ../tools/ab_calculator.py size --baseline 0.034 --mde-rel 0.01 \
                                      --daily-users 2100000 --traffic-fraction 0.01
```

```
  baseline rate        : 0.0340
  minimum detectable   : 0.0003 absolute (1.0% relative)
  users per arm        : 4,481,521
  users total (2 arms) : 8,963,042
  at 2,100,000 eligible users/day x 1% in the test: 426.8 days
```

**A 1% canary needs 427 days to detect a 1% relative conversion regression.**

That number should reframe the whole plan. Work through the consequences:

| Split | Sessions/day in the test | Days to detect −1% relative |
|---|---:|---:|
| 1% canary | 21 000 | **427** |
| 5% canary | 105 000 | **85** |
| 50/50 | 2 100 000 | **4.3** (but run 14 days for weekly cycles and novelty decay) |
| 5% canary, −2% MDE | 105 000 | **21.4** |

So:

- **Stages 1 and 2 are system-health gates, not quality gates.** Label them that way in the
  document. A plan that lists "monitor conversion rate" as a 1%-canary exit criterion is claiming a
  power it does not have, and it will "pass" a regression by failing to see it.
- **The quality decision is made at stage 3**, at 50/50, over two weeks. That is the only stage with
  the traffic to decide anything at the effect size that matters.
- **The 5% stage can still detect a catastrophe.** At −4% relative it needs ~5 days, which is inside
  its 72-hour window at reduced power and comfortably inside a week. So its exit criterion is
  correctly written as "the confidence interval excludes a −4% relative regression", not "conversion
  is fine".

This is the single most common defect in real rollout plans: percentages chosen for how safe they
*feel*, guarding metrics they cannot possibly move.

---

## 4. Automated rollback triggers

| Trigger | Threshold | Sustain | Action | Owner |
|---|---|---|---|---|
| 5xx rate | canary > 1.5× concurrent baseline | 5 min | **auto-rollback** | serving on-call |
| Model timeout rate | > 0.5% of canary requests | 5 min | **auto-rollback** | serving on-call |
| Page p99 latency | canary > 400 ms absolute (the SLO) | 10 min | **auto-rollback** | serving on-call |
| Model p99 latency | canary > 95 ms | 15 min | page, do not auto-rollback | model owner |
| Conversion rate | canary CI upper bound < baseline − 4% relative | 6 h, ≥ 200k sessions | **auto-rollback** | model owner |
| Search abandonment | canary > baseline + 2 pp | 6 h | page | product |
| Seller-diversity index | canary < baseline × 0.9 | 24 h | annotate + review | product |
| SRM | χ² p < 0.001 on the assignment split | 1 h | **halt** (not rollback — the data is untrustworthy, so stop and investigate) | model owner |
| Cost per 1k requests | > 2.6× baseline | 1 h | annotate | model owner |

**The latency wrinkle.** `rank-v8` is *expected* to be slower: 34 ms → 79 ms is a known, accepted
cost. So a trigger on "latency increased" would fire immediately and correctly-but-uselessly. Two
fixes, both used:

1. Trigger on the **absolute SLO** (page p99 ≤ 400 ms), which is the thing that actually matters to
   users, rather than on a relative change from v7.
2. Trigger on **deviation from the *expected new* baseline** (95 ms, which is the benchmarked 79 ms
   plus headroom), not from the old one. "Slower than v7" is expected; "slower than v8 is supposed
   to be" is a bug.

Encoding the *expected* regression is what separates a real trigger table from a copied one.

**Not everything pages.** Seller diversity is a slow-moving fairness/ecosystem metric where a
24-hour signal deserves a human looking at it, not a 03:00 automatic revert. Auto-rollback is
reserved for triggers that are fast, unambiguous, and where being wrong costs little (a rollback to
a known-good model is cheap).

---

## 5. Rollback mechanics

**Current: 22 minutes. Remaining error budget: 12 minutes. This is the blocking problem and it must
be fixed before stage 1.**

Target: **≤ 90 seconds**, achieved by making rollback a *routing* change rather than a *build*:

1. Both `rank-v7` and `rank-v8` are loaded in the model repository and resident on the replicas
   (26 GB × 2 is affordable here; if it were not, keep v7 on node-local NVMe so the reload is a
   local read).
2. Traffic split is a config value read by the router, changed via API. Rollback = set the v8 weight
   to 0.
3. The prediction cache is keyed by model version, so rollback is effective **immediately** rather
   than after a TTL.
4. Rollback is idempotent and requires no build, no image pull, and no pod restart.

**Verification — the part people skip.** Rehearse it:

- In staging, on every deploy, run an automated "rollback drill" that flips the split and asserts
  that within 90 s, 100% of responses carry the v7 version header. Failing the drill fails the
  deploy.
- Once in production, during stage 1 (1% traffic, low blast radius), **actually trigger a rollback
  on purpose** and time it. If you have never rolled back in production, your rollback time is a
  hypothesis.

---

## 6. The overnight problem

Nobody is on call overnight, and stage 3 runs for 14 days. Three adjustments:

1. **Start every stage transition in the morning**, so the first eight hours at a new exposure are
   observed by humans. Never promote a stage on a Friday afternoon.
2. **Auto-rollback must be genuinely automatic** for the fast triggers (errors, timeouts, absolute
   latency). If it needs a human, it does not work at 03:00, and the overnight window is roughly a
   third of your exposure.
3. **Lower the auto-rollback thresholds overnight**, or freeze promotions overnight. Rolling back
   unnecessarily costs almost nothing (you return to a known-good model); rolling back six hours
   late costs a night of degraded conversion. The asymmetry says be trigger-happy when nobody is
   watching.

A fourth option worth naming: for a two-engineer team, deliberately running stages only during
business hours is defensible. It doubles the calendar duration and removes the overnight risk
entirely. Say which you chose and why.

---

## 7. How this rollout could look successful and still be wrong

1. **Selection bias in the offline evaluation.** NDCG@10 was measured on logged impressions from
   `rank-v7` — a set containing only items v7 chose to show. v8's genuinely better candidates may
   never appear in that set, so the offline number is both biased and not obviously in a known
   direction. (Bottou et al. 2013.)
2. **Interference.** This is a marketplace: rankings allocate scarce buyer attention among sellers.
   If v8 wins by shifting demand *between* listings, the treatment arm's gain is partly the control
   arm's loss, and the effect at 100% will be smaller than the A/B test measured — possibly zero.
   Consider a seller-side or geography-clustered design, or a switchback, for the final read.
3. **Novelty.** A visibly different ranking draws exploratory clicks. A 72-hour canary measures the
   novelty spike; the 14-day stage 3 is partly there to let it decay. Plot the daily treatment effect
   and check that it has flattened before deciding.
4. **The 2.3× cost increase does not appear in any quality metric.** A 1% conversion lift that costs
   2.3× inference may or may not be net-positive. Compute margin per session, not conversion rate
   alone.
5. **The latency regression is measured at the model, not at the page.** 45 ms of extra model latency
   inside a 400 ms budget is fine *in isolation*; on a degraded day when the feature store is slow,
   it may be what pushes the page over the SLO. Check the joint distribution, not the marginal.
6. **The new feature could be leaking.** A feature computed from a table that was backfilled using
   post-hoc information is a classic target leak, and it produces exactly this signature: a large
   offline gain that partly evaporates online. The pre-flight gate is where you catch it.

---

## Summary — the plan in one block

```
  pre-flight : skew check on 3 new features, per-segment coverage, rollback drill
  stage 0    : shadow 5%, 48 h        -> health + divergence only
  stage 1    : canary 1%, 24 h        -> SYSTEM HEALTH GATE (no quality power)
  stage 2    : canary 5%, 72 h        -> health + catastrophic-regression gate (-4%)
  stage 3    : 50/50, 14 days         -> THE EXPERIMENT (-1% detectable, 80% power)
  stage 4    : 100%, v7 loadable 30 d
  rollback   : <= 90 s, routing-only, rehearsed in staging on every deploy
               and once for real during stage 1
```
