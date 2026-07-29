# Solution 02 — Choose metrics for three cost structures

Reference answer for [`../exercises/02_choose_metrics.md`](../exercises/02_choose_metrics.md).

The headline: **all three scenarios are imbalanced, and all three get different metrics.** Imbalance
tells you to stop using accuracy; it does not tell you what to use instead. The **cost asymmetry** and
the **shape of the decision** do that.

---

## Scenario A — Card-not-present fraud

| Field | Answer |
|---|---|
| Prevalence, and what it implies | ~0.15% (stated assumption; confirm with the risk team). At this base rate accuracy is meaningless — "never fraud" scores 99.85% — and ROC-AUC will look flattering while precision is poor |
| Which error costs more, and roughly by what ratio | A miss costs more, but not enormously so — chargeback plus fees, versus a blocked payment plus a support contact and some churn risk. Plausibly in the **5:1 to 20:1** range; elicit it rather than assume it |
| **Target metric** | **Precision@3,000/day** — value caught within the review capacity. Secondary: **recall at the auto-block precision floor** for the automated head |
| Why not accuracy | 99.85% is achievable by doing nothing; it is dominated by the majority class |
| Why not ROC-AUC as the headline | Its axes are within-class rates, so it is nearly invariant to the enormous negative pool. The tool's Lab 3 shows ROC-AUC ≈ 0.92 holding steady while PR-AUC falls to 0.05. Keep ROC-AUC as a *diagnostic* for "did ranking quality change while prevalence drifted?" |
| Threshold basis | **Both, at different tiers.** Auto-block: a precision floor (a wrongly blocked payment is a customer incident, so require very high precision). Review queue: **capacity** — 3,000/day. Auto-allow: everything else. Cost-optimal `p*` sets the *review* boundary only if capacity is not binding, and README §5 shows it usually is |
| Guardrails | p99 scoring latency (inline in the payment path); **false-decline rate**; auto-block precision measured on a sampled audit; cost per prediction; queue age (alerts older than N hours) |
| Counter-metrics | **Customer-initiated appeals/complaints per 1,000 blocks**, and **legitimate-transaction abandonment rate** — both rise if the model games "catch more fraud" by blocking aggressively |
| Required slices | Card type/network, issuing country, transaction amount band, new vs returning customer, merchant category, device/channel. New customers and high-amount bands are the two where a hidden regression hurts most |
| Calibration needed? | **Yes, and it is load-bearing.** The routing decision is expected-loss-based: `p × transaction_amount`. A rank-preserving miscalibration is invisible to AUC and corrupts every routing decision |
| What you would **not** report | Accuracy (meaningless here), F1 (weights the two errors equally, which contradicts the stated asymmetry), and bare PR-AUC without the prevalence baseline beside it |

## Scenario B — Content recommendation

| Field | Answer |
|---|---|
| Classification or ranking? | **Ranking**, and a two-stage one. Consequence: metrics must be position-aware and evaluated at a `k` matching the surface |
| **Target metric**, and at what `k` | **NDCG@20** for the ranked feed (20 slots, graded relevance, multiple acceptable items) |
| Candidate-generation vs ranking metric | Candidate generation: **recall@500** — the only job of the retrieval stage is to not lose the good items before ranking; its precision is irrelevant. Ranking: **NDCG@20** |
| Why not precision/recall on the whole pool | Users see 20 of 50,000 items and the *order* of those 20 determines the experience. A pool-level metric assigns the same value to a relevant item in slot 1 and slot 20; the user does not |
| How the target would be gamed in a sprint | Optimize clicks and the system learns clickbait: sensational thumbnails, misleading titles, cliffhangers, and recommending items the user has already almost finished. CTR rises, quality falls. It will also over-recommend a few extremely popular items, collapsing catalogue diversity — high measured engagement, worse product |
| Guardrails | **Completion / dwell time**, catalogue coverage (share of items ever surfaced), p99 ranking latency, cost per request, **per-slice NDCG** so a language or region is not sacrificed for the aggregate |
| Counter-metrics | **Explicit "hide this" / "not interested" rate**, **next-day and next-week return rate**, unsubscribe/opt-out rate, and a **long-horizon holdout** measuring retention at 90 days. The counter-metric must be one that *worsens* when the target is gamed |
| Bias in implicit labels | **Position bias** dominates: items shown higher get clicked because they were higher, so a metric computed on logged clicks rewards reproducing the current ranker and systematically under-credits anything the current system does not show. Related: **presentation bias** (thumbnail/format effects) and the fact that you have no feedback at all for un-shown items. Mitigations: randomized exploration slots, inverse-propensity weighting, and treating offline NDCG on logged clicks as a *screen*, not a verdict. C02 goes further |
| Required slices | New vs established users (cold start is a different product), language/locale, device class, content category, and creator tier — otherwise the aggregate is dominated by your largest cohort |
| What you would **not** report | **Raw CTR as a headline** (the most gameable metric on the list), any single-number "engagement score" whose weights nobody can defend, and offline NDCG without the position-bias caveat attached |

## Scenario C — Medical triage assistant

| Field | Answer |
|---|---|
| Prevalence, and what it implies | ~4% (stated assumption). Imbalanced enough that accuracy is useless, but not so rare that precision is hopeless |
| Which error costs more, and why the ratio is hard to state | A miss is potentially irreversible; a false positive consumes scarce clinician attention. The ratio is genuinely hard because the costs are not commensurable — you cannot put a defensible dollar value on a missed deterioration, and pretending you can is worse than admitting you cannot. **And the two errors are coupled:** enough false positives cause alert fatigue, which causes misses. So the FP cost is not even constant; it rises with alert volume |
| **Target metric** | **Recall at a fixed alert budget** (equivalently, recall at a floor precision) — e.g. "maximize recall subject to ≤ N alerts per shift." Recall is the thing that matters; the constraint is what keeps it honest |
| Why a precision floor / alert budget belongs *in* the metric | Because unconstrained recall is trivially maximized by flagging everyone, which is exactly the failure mode (alert fatigue) that causes the misses you were trying to prevent. Putting the constraint inside the metric makes the tradeoff impossible to ignore, and it is the honest expression of "a miss is much worse, but not infinitely worse" |
| Calibration: required or optional? | **Required, and the most important of the three scenarios.** A clinician acts on the number: "this patient has a 30% risk in 12 hours" must mean what it says, because the human is combining it with information the model does not have. A miscalibrated score is not merely inaccurate, it actively misinforms an expert who is reasoning correctly from a wrong input |
| Guardrails | Per-slice recall (below), **alerts per shift** (fatigue proxy), **clinician override rate** in both directions, time-to-first-clinician-review for flagged patients, calibration error as a hard release gate |
| Counter-metrics | **Alert dismissal rate** and **dismissal rate over time** — if clinicians increasingly ignore alerts, measured recall is fine and real-world recall is collapsing. Also **time-to-review for *un*flagged patients**: if the model's presence causes non-flagged patients to be deprioritized, the system has introduced a new harm that the recall metric cannot see |
| Required slices, and why longer here | Age band, sex, ethnicity, primary language, insurance/payer status, comorbidity count, arrival mode (ambulance vs walk-in), shift/time-of-day, and site. The list is longer because (a) the harm from a subgroup regression is irreversible, (b) clinical data is known to encode access and documentation disparities, and (c) this is a regulated, audited setting where "the aggregate looked fine" is not a defence |
| Failure behaviour | The model is an **aide**, so the safe fallback is the pre-existing triage protocol. On unavailability: display "risk score unavailable — use standard triage protocol," never a blank or stale score, never a default value that could be read as low risk, and log the outage for the audit trail. The system must degrade to the process that existed before it |
| What you would **not** report | Accuracy, F1 (equal weighting is indefensible when one error is irreversible), and **unconstrained recall** — which is the number a vendor will quote and which is meaningless without the alert volume that produced it |

---

## Cross-scenario answers

### 1. The metric that is a candidate everywhere and a headline nowhere

**F1** (accuracy is a close second, but accuracy is not really a serious candidate in any of the
three).

- **Fraud:** F1 weights a false decline and a missed fraud equally. The business explicitly does not.
- **Recommendation:** F1 is not position-aware at all, so it cannot express the thing that determines
  the user experience.
- **Triage:** equal weighting of a missed deterioration and an extra alert is indefensible, and would
  be indefensible in a review.

F1 remains genuinely useful as a **scalar for model selection** during development — it is a single
number that resists the majority-class shortcut, so it is a reasonable thing to sort a leaderboard by.
That is a different job from being the metric a launch decision hangs on.

### 2. Imbalanced but should not lead with PR-AUC

**Scenario B (recommendation).** Its output is a *ranked list*, so the right family is position-aware
(NDCG@20 for the ranked surface, recall@k for retrieval). PR-AUC summarizes a binary
relevant/not-relevant decision over an entire pool, which throws away both position and graded
relevance — the two things that define the product.

Worth adding: **scenario C should not lead with PR-AUC either**, though for a different reason. PR-AUC
averages over operating points you will never use; triage runs at one operating point, chosen by alert
budget. Report **recall at that point** with its confidence interval. PR-AUC is a fine development
diagnostic in both A and C; only A should carry it near the headline.

### 3. Calibration, ranked most to least important

1. **C — Triage.** A human expert reasons *with* the number; a wrong probability corrupts correct
   clinical reasoning. Also required for any expected-outcome calculation.
2. **A — Fraud.** The routing decision is `p × amount`, so miscalibration directly misprices every
   decision even when the ranking is perfect.
3. **B — Recommendation.** Mostly rank-only, so calibration is cosmetic — *unless* the score feeds an
   auction, a blended objective across content types, or a "we think you'll like this 90%" UI, at
   which point it becomes mandatory. The pivot: **calibration matters exactly when the score is
   multiplied by, compared against, or shown as a number with units.**

### 4. Eliciting the cost ratio without asking for a number

- **Fraud:** *"If we could catch one more fraudulent transaction of average size, how many legitimate
  customers would you accept declining to get it — one? ten? a hundred?"* Any bracket picks a side of
  the curve.
- **Recommendation:** *"If a change raised clicks by 5% but raised 'not interested' taps by 20%,
  would you ship it?"* This elicits the *guardrail*, which is the meaningful quantity here — the raw
  cost ratio is not the operative question for a ranking surface.
- **Triage:** *"How many additional alerts per shift is your team willing to absorb to catch one more
  deteriorating patient — and at what number of alerts would your team start ignoring them?"* The
  second half is the important half: it elicits the alert budget, which is the real constraint, and it
  surfaces the coupling between FP volume and effective recall.

The general technique: when someone cannot state a ratio, offer them **a choice between two concrete
scenarios**. People who cannot produce a number can almost always pick a side, and a few such choices
bracket the ratio well enough to set an operating point.

### 5. Capacity constraints and what the objective becomes

| Scenario | Binding constraint | Objective once it binds |
|---|---|---|
| A — Fraud | 12 analysts × 250 = **3,000 reviews/day** | **Precision@3,000** — maximize value caught inside a fixed review budget. Plus a tiered policy so the head is auto-actioned and the tail auto-allowed |
| B — Recommendation | **20 slots** per feed request (and a p99 latency budget for the ranking stage) | **NDCG@20** — and note that the slot count, not the model, caps the achievable recall |
| C — Triage | **Clinician attention per shift** — a soft limit enforced by alert fatigue rather than a hard queue | **Recall subject to ≤ N alerts/shift.** The subtlety: exceeding the budget does not fail loudly, it silently degrades effective recall as clinicians start dismissing alerts |

The general lesson, and the thing README §5 is built around: **the cost-optimal threshold assumes you
can act on every alert.** When you cannot, the objective changes shape from "minimize expected cost"
to "maximize value within a fixed action budget" — which is a precision@k problem. Presenting a
cost-optimal threshold without checking it against capacity is the most common way a technically
correct answer becomes an operationally useless one.

---

## Common mistakes on this exercise

| Mistake | Why it costs you |
|---|---|
| Same answer for all three ("use PR-AUC, it's imbalanced") | Imbalance rules accuracy *out*; it does not rule anything *in*. The cost structure and output shape decide |
| Listing five target metrics | Listing is not choosing. One target, a few guardrails, everything else diagnostic |
| Guardrails that cannot detect the abuse | "Monitor accuracy" does not catch clickbait. The guardrail must move the opposite way when the target is gamed |
| Treating calibration as universally required or universally optional | It is required precisely when the score is multiplied, compared, or displayed as a number |
| Ignoring the FP→fatigue→FN coupling in triage | The two error types are not independent there; this is the most interesting feature of the scenario |
| Not saying what you would drop | Deliberately excluding a metric, with a reason, is the strongest signal in the whole exercise |
| Forgetting capacity | Every one of the three scenarios has a binding action budget |
