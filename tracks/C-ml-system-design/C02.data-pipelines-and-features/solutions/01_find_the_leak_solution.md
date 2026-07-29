# Solution 01 — Find the leak

Reference answer for [`../exercises/01_find_the_leak.md`](../exercises/01_find_the_leak.md).

Headline result: **6 of 18 features are safe, 9 leak, and 3 cannot be decided without asking someone.**
The three most dangerous ones have correct-looking timestamps, innocent names, and only moderate
separation — so no automated check in the toolkit would flag them.

---

## Part 1 — Classification

| # | Feature | Verdict | Mechanism / question |
|---|---|---|---|
| 1 | `claim_amount` | **SAFE** | Written on the submitted form; exists at `t0` by construction |
| 2 | `policy_age_days` | **SAFE** | A function of two dates both known at `t0` |
| 3 | `prior_claims_closed_24m` | **SAFE** | The dictionary is explicit: *closed before* `t0`. This is what a point-in-time aggregate should look like |
| 4 | `prior_claims_filed_24m` | **LEAK** | "24 months **around** the incident date" is a window straddling `t0`. Claims the claimant files *after* this one are counted, and a burst of later claims is a strong fraud-ring signal that did not exist at decision time |
| 5 | `document_count_at_intake` | **SAFE** | The name pins the observation moment to intake |
| 6 | `document_count_total` | **LEAK** | Grows throughout the investigation. Stamped `t0`, measured whenever the row was last written |
| 7 | `adjuster_hours_logged` | **LEAK** | Timestamp `t0 + 95d`. Also circular: the SIU spends hours *because* the claim is suspicious |
| 8 | `payout_amount` | **LEAK** | Created *by* the outcome. `payout = 0` is close to a synonym for denial, which is close to a synonym for the label. Near-deterministic |
| 9 | `claim_status` | **LEAK** | The nastiest of the obvious ones: the timestamp says `t0` but the *value* is the current status, overwritten as the case progressed. `closed_denied` is effectively the label spelled differently |
| 10 | `siu_referral_flag` | **LEAK** (reconstructable) | Median timestamp `t0 + 12d`: usually a consequence of the investigation, not an input to it. **But** if intake has automatic referral rules, a `siu_auto_referral_at_intake` variant would be legitimately available at `t0`. Reconstruct that; do not use the flag as stored |
| 11 | `region_fraud_rate` | **LEAK** | Computed over the whole table, so it includes this claim's own label *and* claims resolved after `t0`. This is exactly `split_demo`'s `x_user_mean_global`: the contamination is in the feature, and no split can remove it |
| 12 | `region_fraud_rate_pit` | **SAFE** | The correct version of #11 — restricted to claims closed before `t0`. Keep this one and delete #11 |
| 13 | `days_to_first_claimant_contact` | **LEAK** | A future *duration*; unknowable at `t0` by definition. Timestamp `t0 + 6d` confirms it |
| 14 | `claimant_credit_score` | **QUESTION** | The warehouse stores only the latest pull, so the `t0` stamp is **asserted, not measured**. Question: *does the bureau contract provide a historical or as-of pull, and is there an archive of past pulls?* Owner: the vendor-management or data-engineering owner of the bureau integration. If the answer is no, it is unusable as stored |
| 15 | `vehicle_value_estimate` | **QUESTION** | Same class: refreshed monthly and **overwritten in place**, so the historical value is gone. Question: *can we snapshot the valuation table going forward, and does the vendor expose an as-of endpoint for backfill?* Owner: data engineering plus the valuation vendor |
| 16 | `police_report_filed` | **QUESTION** | Reports are usually filed before the claim, but not always. Question: *what is the distribution of `report_filed_at − t0`, and what fraction is positive?* Owner: claims operations plus a one-line SQL query. Cheap to answer, so answer it before deciding |
| 17 | `weather_severity_at_incident` | **SAFE** | The incident precedes `t0` and historical weather for a past time is stable. Worth one confirmation that the provider does not *revise* historical records, but this is low risk |
| 18 | `claimant_has_legal_representation` | **LEAK** | Median timestamp `t0 + 21d`. Legal representation is typically a response to the dispute the label describes |

**Counts: 6 SAFE, 9 LEAK, 3 QUESTION.** Twelve of the eighteen features are unusable as delivered,
which is a normal outcome for an inherited feature table and is why 0.97 was implausible.

---

## Part 2 — How the worst leaks inflate the metric

**Leak 1 — `claim_status` (the most damaging).** The model learns `claim_status = 'closed_denied'
→ fraud` with near-certainty, because denial is largely *how* the label is created. Offline this single
rule reproduces the label, driving AUC toward 1.0 and swamping every genuine signal in the feature
importances. In production every claim is `open` at `t0`, so the feature is constant, the dominant
learned rule contributes nothing, and the model falls back on whatever weak signal it happened to
retain. The tell is a model whose top feature is enormously important offline and has zero variance
online.

**Leak 2 — `payout_amount` (and its cousin `adjuster_hours_logged`).** The model learns
`payout_amount = 0 → fraud`. This is the classic "answer key stapled to the question": an outcome
artifact joined to the decision row. It is the easiest class to catch — the timestamp check flags it,
the separation check flags it, and `payout` reads as an outcome word to any human — and it is the
reason a leakage screen is worth its ten minutes.

**Leak 3 — `region_fraud_rate` (the most instructive).** This one does **not** produce a suspicious
AUC. It quietly adds a few points because each claim's own label is folded into its region's rate, and
in a sparsely populated postcode a single claim can move that rate materially. It also encodes the
*future*: claims resolved after `t0` contribute. The result is a feature that looks like a sensible
prior, tests clean under every split, and reports a metric a few points better than reality — the
partial leak from README §1 that reads as a good week of work rather than a bug. It is also the shape
that `split_demo`'s `x_user_mean_global` row exists to demonstrate: **stable across all three splits
and still leaking.**

### Why 0.97 is itself the signal

Three reasons, and none of them requires knowing the "right" number:

1. **Fraud is adversarial.** Adversaries adapt, so near-perfect ranking is not a stable property of the
   problem. A number that would be remarkable in a benchmark paper appearing in an inherited warehouse
   table is a prior for a bug, not for excellence.
2. **The problem's difficulty has a floor set by information, not by modelling.** At `t0` you have a
   form, a policy record, and some history. Perfect separation would mean fraud is trivially
   identifiable from a submission form, which contradicts the existence of an SIU.
3. **Nobody on the team believes it.** Domain scepticism is evidence. The correct response is not to
   defend the number but to attempt to *break* it, which is what this exercise is.

**What to expect instead:** do not quote an absolute figure — you have no basis for one. State the
comparison instead: the honest number should be benchmarked against **the current production model and
the existing rules baseline on the same honest split**, and against the prevalence-relative PR-AUC
(C01 §3), because at a ~0.15% base rate ROC-AUC will look flattering regardless. "Materially lower, and
we will report it next to the incumbent and the trivial baseline" is the defensible answer.

---

## Part 3 — Which check catches which

| Feature | Timestamp | Separation | Naming | Only human / code review |
|---|:---:|:---:|:---:|:---:|
| `adjuster_hours_logged` | **yes** | likely | no | — |
| `payout_amount` | **yes** | **yes** | **yes** | — |
| `claim_status` | no (stamped `t0`) | **yes** | no | — |
| `siu_referral_flag` | **yes** | likely | no | — |
| `days_to_first_claimant_contact` | **yes** | maybe | no | — |
| `claimant_has_legal_representation` | **yes** | maybe | no | — |
| `document_count_total` | no (stamped `t0`) | maybe | no | mostly |
| `prior_claims_filed_24m` | no (stamped `t0`) | no (moderate) | no | **yes** |
| `region_fraud_rate` | no (stamped `t0`) | no (moderate) | no | **yes** |
| `claimant_credit_score` | no (stamped `t0`) | no | no | **yes** |
| `vehicle_value_estimate` | no (stamped `t0`) | no | no | **yes** |

### The features every automated check would miss

**`prior_claims_filed_24m`, `region_fraud_rate`, `claimant_credit_score`, and
`vehicle_value_estimate`** — and the reason is a single shared property: **their timestamps are asserted
by the pipeline rather than measured by the source.** A column stamped `t0` because that is the row's
key, not because the value was observed at `t0`, defeats the timestamp check completely. Because these
leaks are *partial* (a few points, not perfect separation) they also defeat the separation check, and
their names are unremarkable.

The generalization worth carrying into an interview: **an automated leakage screen catches the
egregious cases and gives you false confidence about the rest.** The screen's real value is triage —
it tells you where to spend the code review, and it takes minutes. It does not replace reading the
transformation SQL and asking the data owner when each column is populated. There is no tool for
"someone overwrites this column in place," which is why §4's answers are structural.

A second generalization worth stating: a *wrong* timestamp is safer than a *stamped* one. A timestamp
that is obviously in the future gets caught; a timestamp that is confidently, precisely wrong does not.

---

## Part 4 — Structural prevention

| Leak class | Example | Pipeline change |
|---|---|---|
| Post-outcome artifact joined to the label row | `payout_amount`, `adjuster_hours_logged` | Physically separate **pre-decision** and **post-decision** tables. The training view may only join the pre-decision table; the post-decision table is not reachable from the feature layer. Structural separation beats vigilance |
| Column back-filled in place | `claim_status`, `document_count_total` | Model these as **event streams**, not current-state columns: append `claim_status_changed` events with their own timestamps and derive `status_as_of(t)` from them. Never expose a mutable current-state column to the feature layer |
| Global aggregate computed before splitting | `region_fraud_rate` | Every aggregate feature must be defined as an **as-of window** with an explicit upper bound (`... WHERE closed_at < :decision_time`). Enforce it: make the aggregate helper *require* an as-of parameter, so an unbounded aggregate does not compile rather than merely being discouraged |
| Slowly-changing dimension with no history | `claimant_credit_score`, `vehicle_value_estimate` | Convert to a **type-2 slowly-changing dimension** — `(entity, valid_from, valid_to, value)` — and snapshot on every refresh. This costs storage and must start *now*; historical values that were overwritten are gone forever, which is why this is the most expensive item on the list |
| Window straddling the decision moment | `prior_claims_filed_24m` | Ban symmetric or "around" windows in feature definitions. Every window is `[t0 − N, t0)` — half-open, ending strictly before the decision. Make this a review checklist item and a lint rule on the feature DSL |
| Feature with no observability timestamp | `claim_id`-style columns, and anything the screen reports as `NOT CHECKED` | Require an observability timestamp per feature value at the schema level. A feature without one is **unauditable, not innocent**, and should fail the gate |

The unifying principle: **make the leak impossible to express, rather than trying to notice it.** A
feature layer that cannot reach post-decision tables and whose aggregate helper demands an as-of
timestamp removes most of this exercise's nine leaks by construction. Every prevention above is a
schema or API change, not a habit — habits do not survive the next contractor.

---

## Part 5 — What you would say on day three

> "I audited the 18 features in `claim_features`. Six are safe as delivered, nine carry information
> that did not exist when the routing decision is made, and three I cannot judge without answers from
> the bureau and valuation vendors. The 0.97 came mostly from `claim_status` and `payout_amount`, which
> are effectively the label written a different way, so it was never a real number rather than a number
> that degraded. I would not quote a replacement figure yet; I will re-measure on the honest feature
> set and report it against the current production model and the rules baseline so we can see whether
> there is a real gain. To proceed I need two things: a decision from data engineering on snapshotting
> the credit-score and valuation tables — starting now, because overwritten history is unrecoverable —
> and thirty minutes with claims operations to confirm when `police_report_filed` and the SIU referral
> flag are actually populated. I'd also like to split the warehouse table into pre-decision and
> post-decision halves so this class of problem cannot recur."

Why it works: it reports without drama, is precise about what was found, refuses to invent a
replacement number, names the two blocking dependencies with owners, flags the one item that is
time-sensitive (overwritten history), and proposes a structural fix rather than a promise to be
careful.

---

## Common mistakes on this exercise

| Mistake | Why it costs you |
|---|---|
| Classifying only by timestamp | Four of the worst leaks have a stamped `t0` and pass that check cleanly |
| Marking everything with a suspicious name as a leak | `region_fraud_rate_pit` is safe and is the *correct* version of a leaky feature. Read the definition, not the name |
| Zero QUESTION verdicts | Three features genuinely cannot be decided from the dictionary. Confidence without information is the failure mode being tested |
| Missing the "stamped, not measured" class | This is the exercise's central lesson and the one no tool can automate |
| Deleting `region_fraud_rate` and its point-in-time twin together | You would throw away a genuinely useful feature; the fix is to keep the as-of version |
| Prevention items that are advice | "Be careful with joins" prevents nothing. Separate the tables so the join cannot be written |
| Quoting a replacement AUC | You have no basis for one. Commit to a comparison against the incumbent and the baseline instead |
