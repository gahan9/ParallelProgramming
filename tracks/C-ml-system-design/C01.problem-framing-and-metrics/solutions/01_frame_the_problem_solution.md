# Solution 01 — Frame the problem

Reference answer for [`../exercises/01_frame_the_problem.md`](../exercises/01_frame_the_problem.md).
Your wording will differ; check that you made the *moves* listed in [`README.md`](README.md).

---

## 1. The decision

**Action:** each week, choose one of three treatments for an at-risk account — (a) a Customer Success
outreach touch (call or tailored email), (b) an automated in-product nudge plus a discount offer, or
(c) nothing.

Framing it as a three-way action rather than "predict churn" immediately produces the useful
questions: outreach is capacity-limited (6 people, ~1,000 touches/month), discounts are
budget-limited, and "nothing" is the default that any model must beat. The system's output is
therefore not a score, it is **a weekly ranked worklist for six humans plus a smaller automated
treatment list.**

## 2. Unit and cadence

**One decision per account per week.** Justification is arithmetic, not preference: with ~1,000
touches/month across ~40,000 accounts, you can touch roughly 2.5% of the base per month. A daily
cadence would produce churn every day for a queue that only drains weekly and would let the same
account be touched repeatedly; a monthly cadence loses accounts that decay quickly. Weekly matches
the drain rate of the queue.

Add a **cooldown rule** as part of the decision, not as an afterthought: no account is touched twice
within 30 days. Without it the ranking will keep re-surfacing the same high-risk accounts and starve
everyone else — a real failure mode of ranked worklists.

## 3. The prediction task

> For each **account** that is active as of Monday 00:00 (the "as of" cutoff), predict the probability
> that the account **fully cancels its subscription within the next 60 days**, using only data
> observable at the cutoff.

- **Horizon: 60 days.** Chosen against the *action*, not the data: outreach plus a 6-month discount
  needs weeks to influence a renewal decision, so a 7-day horizon would surface accounts that are
  already gone. 60 days also gives roughly two billing cycles of warning.
- **Output:** a calibrated probability in [0, 1], plus the top contributing factors for the CS rep.
- **Label definition, SQL-ready:** *an account is positive if it has a `subscription_status =
  'cancelled'` event with an effective date in `(cutoff, cutoff + 60 days]`, where the cancellation
  was customer-initiated (excluding involuntary payment-failure churn and excluding accounts migrated
  to a parent contract).*

**The ambiguity you must resolve explicitly.** In seat-based B2B, four different things get called
churn:

| Event | Include as positive? | Reasoning |
|---|---|---|
| Full cancellation | **Yes** | This is the revenue event the VP means |
| Downgrade / seat reduction >50% | **Separate label** | Real revenue loss, different treatment, different predictors — model it separately rather than blurring it |
| Involuntary churn (card failure) | **No** | Predictable from billing data, fixed by dunning emails, not by CS outreach — a different system entirely |
| Silent non-usage (still paying) | **No, but track it** | Not churn yet; it is a *feature* and a leading indicator |

Blurring these into one label is the single most common way a churn model becomes useless: the model
learns to predict the most frequent event (usually involuntary churn), which the CS team cannot act
on. Question to ask, and who owns it: *"which of these four does the retention number in the board
deck actually count?"* — Finance/RevOps, not Customer Success.

## 4. Is ML the right tool?

| Condition | Verdict | Reason |
|---|---|---|
| Complex and changing pattern | **Partly** | Some churn is trivially predictable (usage went to zero, support escalation open, 3 seats removed last month). The residual — accounts that look healthy and leave — is where ML would earn its place, and it is also the hardest part |
| Labels available | **Yes** | Cancellation events are logged and unambiguous once defined. But they arrive with a **60-day lag**, and at ~40,000 accounts an annual churn rate in the low tens of percent gives you thousands of positives — enough, but not lavish |
| Errors tolerable | **Yes** | The worst outcome of a false positive is an unnecessary friendly call and a discount you did not need to give; false negatives are the status quo |
| Scale defeats human judgement | **Weakly** | 40,000 accounts is too many to review individually, but the CS team already has intuitions about their own book of business. This is a *prioritization* problem more than a prediction problem |

**Verdict: ship rules first.** Two of the four conditions are only weakly satisfied, and one (labels)
carries a 60-day lag that will make iteration slow.

**The rules baseline I would actually ship (2 weeks of work):**

```
risk_score = 3 * (weekly_active_users dropped >40% vs 8-week average)
           + 3 * (seats reduced in last 60 days)
           + 2 * (zero logins by the account owner in 21 days)
           + 2 * (open support ticket aged >7 days, or any escalation)
           + 1 * (NPS <= 6 in last 90 days)
           + 1 * (renewal date within 45 days)
```

Rank by `risk_score`, take the top ~250/week, hand it to CS. What this buys you beyond a worklist:
the **feature pipeline**, the **logging of what was surfaced and what was done about it**, the
**metric definitions**, and — crucially — **the treatment/outcome data** that a model will later need
for training. Google's *Rules of ML* Rule #1 exists for exactly this reason.

**What would make me abandon it for a model:** the rules baseline plateaus (CS reports that a
majority of the top-250 list is obvious or already-known), *and* we have ≥2 quarters of logged
treatment/outcome pairs, *and* the residual churn (accounts that leave with a low rules score) is
large enough to be worth chasing. Any one of those alone is insufficient.

## 5. Metric hierarchy

| Layer | Metric | How measured | Cadence |
|---|---|---|---|
| Business | **Net revenue retention over 90 days** for the treated population vs a holdout | Finance/RevOps report against a randomized holdout | Monthly |
| Product / system | **Saves per 100 touches** (accounts that were flagged, touched, and still active at +60 days) and **discount dollars per save** | Product analytics joined to CS activity logs | Weekly |
| Offline model | **Precision@250/week** (the worklist size), with **PR-AUC + prevalence lift** as the secondary and **calibration error** as a gate | Held-out, strictly time-based split | Per model build |

**Assumed links, written to be falsifiable:**

- *We assume that raising precision@250 raises saves-per-100-touches, and we verify it by comparing
  the save rate of the model-ranked list against the rules-ranked list in an A/B test on the CS
  worklist.*
- *We assume that saves-per-100-touches raises 90-day net revenue retention, and we verify it against
  a randomized untreated holdout of ~5% of at-risk accounts, reported monthly by Finance.*
- *If precision@250 improves and saves-per-100-touches does not move within two cycles, we stop
  shipping on offline metrics and investigate the link before doing any more modelling.*

That last sentence is the one most candidates omit, and it is the one that shows you know the proxy
can break.

## 6. Threshold and capacity

**This is a ranking problem, not a threshold problem.** The argument is capacity: CS can execute
~250 touches/week regardless of how many accounts cross any probability threshold. A threshold
produces a list of unpredictable length — 800 one week, 60 the next — which is unusable for staffing.
Ranking produces a fixed-size worklist.

Therefore the metric is **precision@250** (and `saves@250` once you can measure treatment outcomes),
*not* F1, and *not* accuracy. Note this is a real instance of "capacity overrides the cost-optimal
threshold" from README §5.

Two pivots, which are the "it depends on X":

- **If CS doubles to 12 people (~500 touches/week):** the metric becomes precision@500, and because
  precision falls as you go deeper into the ranking, the marginal value of each additional touch
  drops. At some depth the expected save value per touch falls below the cost of the touch — that
  depth, not the headcount, is the right list size. Compute it; don't guess.
- **If outreach becomes automated and effectively unlimited** (an in-product nudge costs ~nothing):
  the constraint moves from *capacity* to *cost and annoyance*. Now it genuinely is a threshold
  problem, `p*` from the cost ratio applies, the model must be **calibrated**, and you need a new
  guardrail — nudge fatigue / opt-out rate — because the failure mode changed from "we ran out of
  people" to "we spammed everyone."

## 7. Testable requirements

| Requirement | Testable form |
|---|---|
| Latency | Batch scoring of all ~40,000 accounts completes within 60 minutes of the Monday 00:00 cutoff; p99 for the ad-hoc single-account lookup in the CS console ≤ 500 ms |
| Throughput | Weekly batch handles 100,000 accounts (2.5× current) without exceeding the 60-minute window |
| Freshness | Scores reflect product events up to the Monday cutoff; a feature pipeline older than 26 hours blocks publication and pages the owner |
| Cost per prediction | ≤ \$0.002 per account-week (≈\$80/week all-in), reviewed if account count doubles |
| **Failure behaviour** | If the model or feature pipeline fails, publish **last week's ranked list re-sorted by the rules baseline**, mark the worklist `degraded: rules-only` in the CS console, and alert the owner. Never publish an empty list and never silently reuse a stale score without labelling it |
| Interpretability | Every listed account shows its top 3 contributing factors in plain language and the date of the most recent triggering event; a CS rep can state why an account is on the list without asking an engineer |

The failure-behaviour row is deliberately specific. "Fall back to rules" is the right instinct;
"...and label the list as degraded so the CS team knows why it looks different" is the part that keeps
trust intact.

## 8. The framing trap

**Risk is not treatment effect.** The obvious framing — "rank by probability of churn, touch the top
250" — spends the entire outreach budget on the accounts *most likely to leave*, which
disproportionately includes accounts that are **already gone**: the decision has been made, the
migration is scheduled, the champion has left. A perfectly accurate risk model would rank those
first, and every touch would be wasted.

What you actually want to rank by is **incremental save probability**:

```
value(account) = P(stays | treated) - P(stays | not treated)   x   account annual value
```

This is uplift/incrementality modelling, and the honest answer for month one is that **you cannot
estimate it yet** — it needs treated/untreated outcome data you do not have.

What I would do in month one, given that constraint:

1. **Ship the rules baseline** and, from day one, **randomize a holdout**: for each account the rules
   surface, withhold treatment for ~10–15% at random. This is the cheapest thing in the entire plan
   and it is what makes every later measurement possible. Skipping it is the mistake that costs a year.
2. **Segment by actionability rather than modelling uplift.** Down-rank accounts showing
   already-decided signals (cancellation form started, data export run, contract end < 14 days with
   no renewal activity) — a crude proxy for "no uplift available," implementable as rules today.
3. **Weight by account value**, so the list is ranked by expected dollars, not probability. A 30%
   risk on a \$120k account outranks an 80% risk on a \$4k one.
4. **Revisit with uplift modelling** once the holdout has accumulated two quarters of data.

Volunteering the randomized holdout in the first meeting is the single highest-leverage thing in this
exercise.

## 9. What you would say in the meeting

> "I think we can help, and I want to make sure we help in a way we can prove. Two things first:
> Finance and I need to agree on exactly what counts as churn — full cancellation is different from a
> downgrade, and involuntary payment churn needs a dunning fix, not a model. Then I'd like to ship a
> simple rule-based at-risk worklist in about two weeks so your team gets value immediately and we
> start logging which outreach actually works. The one thing I need from you is agreement that we
> deliberately *skip* outreach on a small random slice of the at-risk list — without that we will
> never know whether the outreach or the model is doing anything. Once we have a quarter of that data,
> a model becomes a straightforward upgrade rather than a guess."

Why this works: it commits to a two-week deliverable, does not promise a model, names the one hard
ask (the holdout), and reframes the request from "get an AI" to "make retention measurable" without
telling the VP they asked the wrong question.

---

## Common mistakes on this exercise

| Mistake | Why it costs you |
|---|---|
| Starting with "I'd use gradient boosting on the usage features" | Answers the last question first; every reviewer notices |
| Defining churn as "the user churns" | The label ambiguity is the actual work; four different events are in scope |
| Choosing a metric before noticing the 1,000-touch limit | Capacity makes this a ranking problem; a threshold metric is simply wrong here |
| Ranking by risk and stopping | Misses the risk-vs-uplift trap — the highest-risk accounts are often unsaveable |
| No randomized holdout | Nothing downstream is measurable, ever |
| Omitting failure behaviour | The one requirement reviewers reliably ask about |
| Promising a model in the meeting | Sets a date you cannot hit and skips the rules baseline that would have taught you the domain |
