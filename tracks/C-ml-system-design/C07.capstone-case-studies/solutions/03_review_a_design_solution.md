# Solution 3 — Review of the churn-prevention design

Reference answer for [`../exercises/03_review_a_design.md`](../exercises/03_review_a_design.md).

The document contains **five serious defects**. Below they are ranked by expected damage, which
is the ordering a reviewer owes the author — five unranked comments is a to-do list, not a
review.

| Rank | Defect | Section in the doc | Severity |
|---|---|---|---|
| 1 | **The intervention destroys the label** — an unmanaged feedback loop | §1, §4, §7 | 🔴 Blocker |
| 2 | **Random split on temporal data, with leaking aggregate features** | §4 | 🔴 Blocker |
| 3 | **No rollback plan, and no model versioning** | §5, §7 | 🔴 Blocker |
| 4 | **No guardrail metrics** | §2 | 🟠 Major |
| 5 | **Unbudgeted latency path** — two synchronous external calls missing from the budget | §3 vs §6 | 🟠 Major |

---

## 🔴 1. The intervention destroys the label

**What.** The model predicts 30-day churn. Users predicted to churn receive a retention offer.
Some accept and stay. Those users are then labelled `churned = 0` in the next training round.

The model is being trained on outcomes **it caused**. The 18 months of historical data was
generated under a world with no proactive offers; from launch onward, the training data is
generated under a world where the highest-risk users get intervened on. The label
`churned` stops meaning "would have churned" and starts meaning "churned despite our
intervention."

**Why it matters — the concrete trajectory over four months:**

| Month | What happens |
|---|---|
| 1 | Model flags the top 3%. Offers go out. The intervention works; many stay. |
| 2 | Retrained on month 1: those users are labelled *not churned*. Their feature signature — declining engagement, high risk — now carries a **lower** churn weight. |
| 3 | Model ranks them lower. Fewer of them get offers. |
| 4 | They churn. Model relearns they were risky. Offers resume. |

The system oscillates, and every offline metric looks fine throughout — AUC is computed against
the corrupted label and will happily stay above 0.82. §7's weekly "AUC recomputed against
realised outcomes" will not catch it either, because it uses the same corrupted label. **The
monitoring in §8 is structurally incapable of detecting this failure.**

There is a second, deeper problem hiding underneath. Even without the loop, **churn risk is the
wrong quantity**. The document's own §10 names the fix as "future work": what you want is
**uplift** — `P(stay | offer) − P(stay | no offer)` — not `P(churn)`. Many high-risk users are
leaving regardless, and offering them a discount is pure margin loss; some low-risk users are
*persuadable*, and never get an offer. Ranking by risk spends the budget on the wrong people
even when the model is perfectly accurate.

**Fix.**

1. **A permanent randomized control holdout — non-negotiable and it must ship with v1.** 5% of
   high-risk users receive **no offer**. They are the only source of uncounterfactual churn
   labels, and the only unbiased estimate of the model's true performance and the programme's
   true lift. Train on the holdout, or reweight, but never train on intervened users as if they
   were untouched.
2. **Log the treatment.** Every training row must carry `offer_shown`, `offer_type`,
   `offer_accepted`. Without this the historical data cannot be corrected retrospectively, and
   this is the cheapest thing on the list.
3. **Move to uplift modelling.** A two-model or transformed-outcome approach on the
   treatment/control split. It is listed as future work; it is the actual objective.
4. **Change the model metric.** AUC against a treatment-corrupted label is not measuring
   anything. Use **Qini / uplift-at-k** against the randomized holdout.

**Severity: blocker.** It silently degrades over months, the existing metrics cannot see it, and
retrofitting a holdout after launch means throwing away the intervening data.

---

## 🔴 2. Random split on temporal data, with leaking aggregate features

**What.** §4 says: *"split them 80/10/10 at random"* and *"aggregate features … are computed
once over the full 18-month window for efficiency."*

Two leaks, and the second is worse than the first.

**Leak A — random split.** A subscriber contributes many (subscriber, observation-date) rows. A
random split puts January's rows in train and February's in test, so the model learns from a
user's future to predict their past. It also puts near-identical rows for the *same* user on
both sides of the split, so the test set is measuring memorization.

**Leak B — the aggregate features.** "Lifetime watch minutes" and "rolling engagement averages"
computed over the **full 18-month window** means a training row dated month 3 carries a feature
value that summarizes months 1–18. That feature knows whether the user churned in month 4,
because a churned user's lifetime total stops growing. **The label is inside the feature.** This
leak survives even if you fix the split, which is why it is the more dangerous of the two.

The document defends the choice — *"a much larger and better-balanced training set"* and it
*"avoids the problem that the most recent months have incomplete 30-day labels."* That second
concern is legitimate and has a correct answer (embargo the last 30 days), and the document has
solved it in a way that invalidates the model.

**Why it matters.** Offline AUC will be excellent — likely well above the 0.82 gate, which is
how this passes review. Online performance will be materially worse, and the gap will be
attributed to "the offline–online gap" rather than to a bug, because the code is correct and
nothing errors. This is the defect most likely to be shipped and never diagnosed.

**Fix.**

1. **Split by time.** Train on months 1–15, validate on 16, test on 17. Embargo the final 30
   days so labels are complete. Accept the smaller training set; it is the honest one.
2. **Also split by subscriber** for a second evaluation, to measure generalization to users
   never seen in training.
3. **Point-in-time-correct features.** Every aggregate is computed using **only data available
   at the observation date**. This is the expensive fix — it means as-of joins in the feature
   pipeline — and it is not optional.
4. **Prove it.** Before and after the fix, report both AUCs. A large drop is the leak's size,
   and quantifying it is what makes the fix defensible to whoever approved the original numbers.

**Severity: blocker.** The launch gate (AUC > 0.82) is being evaluated against a corrupted
metric, so the gate itself is meaningless.

---

## 🔴 3. No rollback plan, and no model versioning

**What.** §5: the retrained model is written to
`s3://streamco-models/churn/current/model.txt`. §7: promotion to 100% is *"updating"* that same
path, and both consumers *"pick up the new file on their next restart."*

Four distinct problems in two sentences:

1. **The previous version is overwritten.** There is no artefact to roll back *to*.
2. **`current/` is not a version.** You cannot tell which model produced which score, so you
   cannot attribute a regression, reproduce a decision, or answer a subscriber's "why did I get
   this offer."
3. **"On their next restart" is not a deployment mechanism.** Two independent services pick up
   the change at different, unspecified times. During the window, the batch scorer and the
   real-time service are using **different models**, so a user's `risk_score` and the offer
   logic disagree. Nobody can reason about the system's state.
4. **No rollback triggers.** §7 monitors *"batch job completion time, score distribution, and
   offer volume"* — all system health, none of them quality. A model that is fast, produces a
   normal-looking score distribution, and sends the right *number* of offers, but sends them to
   the wrong 3% of subscribers, passes the canary cleanly.

**Why it matters.** A bad model version ships on a Tuesday, sends 100,000 discount offers to the
wrong people, and there is no way to revert and no way to identify who was affected. Recovery
means retraining from scratch, which takes 40 minutes plus a full feature pipeline run, plus the
time to notice — and nobody notices, because the alerts in §8 do not measure targeting quality.

**Fix.**

1. **Immutable versioned artefacts:** `s3://.../churn/2026-07-29T02-00Z-a3f21b/model.txt`, with
   a pointer file naming the active version. Never overwrite.
2. **Explicit, atomic promotion.** Both consumers read the pointer, reload on a signal rather
   than a restart, and log the version they loaded with every score they emit.
3. **Rollback = pointer flip**, target under 5 minutes, and **tested monthly**. An untested
   rollback is not a rollback plan.
4. **Quality-based rollback triggers**, not just health: canary-vs-control offer accept rate;
   population-stability index on the score distribution; overlap between the canary's top 3% and
   the incumbent's top 3% (a low overlap is the clearest possible signal that something changed
   materially); and per-segment offer-volume shifts.
5. **Stamp `model_version` on every score and every offer**, so post-hoc attribution is possible.

**Severity: blocker.** This one is cheap to fix and catastrophic to skip.

---

## 🟠 4. No guardrail metrics

**What.** §2's metric table is complete on the "did it work" axis and empty on the "what did it
cost us" axis. The primary metric is churn rate; the product metric is offer accept rate. There
is nothing that can **fail** a launch that succeeded on those.

**What the design will therefore do, and nobody will notice:**

- **Discount cannibalization.** The largest and most likely harm. Many of the top 3% would have
  stayed anyway; every offer to them is pure margin loss. Churn rate improves, accept rate looks
  great, and **revenue per subscriber falls**. §2 lists "net revenue retention +1.5 pts" as a
  target but no threshold at which the launch is *stopped* — a target is not a guardrail.
- **Offer-seeking behaviour.** Users learn that reducing engagement, or starting a cancellation
  flow, produces a discount. This is a *trained* behaviour with a months-long time constant, and
  it makes the churn signal itself unreliable.
- **Fairness and price discrimination.** The model will find proxies for willingness-to-pay.
  Systematically offering discounts along lines that correlate with a protected characteristic
  is a legal exposure, not just an ethical one.
- **Long-run damage.** Users retained by a discount may churn harder at renewal. A 30-day
  churn metric cannot see a 90-day effect.
- **Notification fatigue.** 80,000–120,000 offers per day, in-app and by email. The document
  budgets the *volume* but never measures the cost of the contact.

**Fix.** Add guardrails with **pre-registered thresholds** and veto power over the primary
metric:

| Guardrail | Threshold |
|---|---|
| Revenue per subscriber (treatment vs control) | No decrease; alert at −0.5% |
| **Incremental margin per offer** (uplift × LTV − discount cost) | Must be positive — this is the real objective function |
| 90-day churn, not just 30-day | No worse than control |
| Offer acceptance and discount rate by demographic proxy | No group > 1.2× the mean |
| Email/push opt-out rate | No increase |
| Repeat-offer rate per subscriber | Hard cap, e.g. 1 per 90 days |

Note that the second row is what §1 should have used as the goal in the first place: the point
is not to reduce churn, it is to make money by reducing churn.

**Severity: major.** Not a blocker only because the randomized holdout from defect 1, once it
exists, gives you the instrument to measure all of these. Without that holdout, none of these
guardrails can be computed — which is why defect 1 outranks it.

---

## 🟠 5. Unbudgeted latency path

**What.** This one requires reading §3 and §6 against each other, which is why it survives
review.

§3 says the real-time offer service, before rendering anything, *"must call the **Offer
Eligibility Service** — a separate team's service … — and then the **Billing API** to confirm
the account is in good standing and to reserve the discount code."*

§6's budget lists edge/auth, a KV read, offer-selection logic, render, and a jitter reserve.
Total 100 ms against a 200 ms budget, described as *"comfortably inside budget with 100 ms of
headroom."*

**Neither external call appears in the budget.** Two synchronous network calls, one to another
team's service and one to a payments API, both on the critical path, both uncosted. There is no
timeout, no retry policy, no circuit breaker, and no fallback.

**Why it matters.** Payment APIs are not fast and their tails are long — 200–500 ms at p99 is
ordinary. Realistically:

```
Stated budget                                    100 ms
+ Offer Eligibility Service p99                   80 ms  (assumption; unmeasured in the doc)
+ Billing API p99, incl. code reservation        250 ms  (assumption; unmeasured in the doc)
                                                 -------
Actual p99                                       430 ms   against a 200 ms budget
```

More than 2× over. And the failure mode is worse than slowness: when the Billing API degrades,
the offer service blocks, its threads exhaust, and — because it sits in the app's render path —
**it takes the app screen with it.** A retention feature causes an outage in the product it was
meant to retain people in. The alert in §8 fires on the offer service's own p99, which tells the
on-call engineer that the offer service is slow, not that Billing is.

**Fix.**

1. **Put every hop in the budget, including other teams' services**, with a stated p99 taken
   from *their* dashboard, not from an assumption. If you cannot get that number, that is the
   first finding.
2. **Hard timeouts on both calls** — e.g. 40 ms and 60 ms — with a circuit breaker.
3. **Fail closed on the offer, open on the app.** On timeout, render the app without the offer.
   A missed offer costs one opportunity; a blocked render costs the session.
4. **Move eligibility off the critical path.** Precompute an `offer_eligible` flag in the same
   nightly batch that computes `risk_score`; the two have the same freshness requirements. This
   removes one call entirely.
5. **Reserve the discount code asynchronously**, after the offer is displayed and before it is
   redeemed. Reserving before display is a synchronous write on a read path.
6. **Alert on dependency latency separately**, so the on-call engineer sees which upstream is
   degrading.

**Severity: major.** It will manifest as a p99 breach on day one — which is at least a *fast*
failure, unlike defects 1 and 2.

---

## Deliberately not flagged

A review that lists everything it would have done differently trains the author to write
defensively. These are fine:

- **LightGBM over the deep model.** 0.006 AUC for a GPU fleet is a bad trade and the document
  says so with a number. Correct call, correctly justified.
- **CPU serving.** Right for a GBDT. No GPU is needed and none is proposed.
- **Nightly batch scoring.** Churn risk moves on a timescale of days; scoring it in real time
  would be engineering for its own sake.
- **Downsampling negatives with Platt recalibration.** Standard, and the recalibration step —
  which people forget — is present.
- **Excluding involuntary churn.** Correct scoping; dunning is a different problem with a
  different intervention.
- **The cost model.** $12,500/month against 160,000 retained subscribers is obviously
  favourable, and the document does not over-claim precision. *(Though once defect 4 is fixed,
  the discount cost — absent from the table — may exceed the infrastructure cost by two orders
  of magnitude. That belongs under defect 4, not here.)*

---

## The two closing questions

**If you could insist on only one change before launch:** the **randomized control holdout**
(defect 1). It is a few days of work, it must exist from day one because it cannot be
constructed retrospectively, and it is the instrument that makes four of the five other fixes
measurable — the true lift, the discount cannibalization, the uplift model, and the honest
model metric all depend on it. Fixing the split (defect 2) is more urgent in the sense that the
model is currently wrong, but a wrong model with a holdout is *detectably* wrong. A right model
without a holdout is unfalsifiable.

**What is good, and worth saying:** the problem framing in §1 is genuinely strong — it names the
current baseline (6% accept rate on the cancellation page), explains *why* it is inadequate
(the decision is already made), and quantifies the opportunity ($180 LTV). The cost model is
honest. The model-choice section shows real judgement in rejecting the deep model on a
cost/benefit basis rather than on preference. The feature taxonomy is thorough and the
trend-delta family shows someone who has thought about what churn actually looks like. This is a
document by someone who knows what they are doing, with five gaps of the kind that are only
obvious once you have been burned by them — which is exactly why review exists.
