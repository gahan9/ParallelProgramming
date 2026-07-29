# Solution 2 — Diagnosing the loan-approval degradation

**Headline:** covariate shift on `channel` into a region the model does not cover, *plus* concept
drift within that region — and the tell that separates them from everything else is calibration
error moving 3.7× while the prediction distribution barely moves at all.

A single-word answer misses the structure. The degradation is **compositional**: a new
sub-population arrived (covariate shift), and within that sub-population the model's learned
relationship does not hold (concept drift, localized).

---

## 1. Walking the decision tree

**Branch 1 — did anything ship?** No. No model deploy since week 17; no data-platform deploy in
weeks 28–34; model version is `loan-v11` in both windows. System layer is flat: latency 41 → 42 ms,
error rate 0.02% unchanged, fallback rate 0.1% unchanged. **Not a deploy.**

**Branch 2 — did the input distribution move?** Yes, but only in one place. Every numeric feature is
at PSI < 0.06, comfortably "stable" by any convention. `region_code` is at Cramér's V 0.014. But
`channel` is at **Cramér's V 0.211**, and the mechanism is explicit: a new category value,
`partner_mkt`, appears in week 29 and is now 9% of applications. The top-5 features look stable
*in aggregate*, which is exactly what a 9% sub-population does to a global statistic — see §3.

**So: covariate shift is present.** `P(x)` changed by the addition of a new region of input space.

**Branch 3 — did the prediction distribution move?** Barely. Mean 0.163 → 0.169, PSI 0.021, approval
rate 71.4% → 70.9%. The model is producing essentially the same distribution of scores it always
did.

**Branch 4 — the evidence that changes the diagnosis.** ECE went from 0.019 to **0.071**, and
60-day delinquency from 4.1% to **6.8%**. The model's scores are the same *numbers* but they no
longer mean the same *thing*: a 0.15 predicted default probability is now associated with a
materially higher realized default rate. **`P(y | x)` has changed** — for at least part of the
population.

**Diagnosis:** covariate shift (a new 9% sub-population via `partner_mkt`) combined with concept
drift **localized to that sub-population** — the model's input-to-default relationship, learned on
the old channels, does not transfer to marketplace-sourced applicants. Plausible mechanisms: the
partner's applicants are selected differently (people who did not qualify elsewhere), the
partnership's UX changes what applicants self-report, or the partner pre-filters in a way that
breaks the model's implicit assumptions about who applies.

**The quantitative check that confirms it, and the first thing to run:** slice every metric by
`channel`. The prediction is that `partner_mkt` shows a delinquency rate far above 6.8% while the
established channels sit near 4.1%. A global 6.8% at 9% volume implies the new channel is running
somewhere around 30% delinquency:

```
  0.068 = 0.91 × d_old + 0.09 × d_new
  if d_old is still 0.041  ->  d_new ≈ 0.34
```

That single slice, which nobody computed before paging you, is the whole investigation. Run it
first.

---

## 2. Ruling out the alternatives

| Ruled out | The specific number that rules it out |
|---|---|
| **Serving-path incident / deploy** | Model version identical in both windows; no deploys in the window; latency, error rate and fallback rate all flat; **and** every feature's missing rate is unchanged (`employment_months` 1.1% → 1.2% is noise). A broken pipeline shows up as a missing-rate jump or a prediction-distribution jump — the prediction PSI is 0.021. |
| **Pure prior/label shift** | Prior shift means `P(y)` moved with `P(x\|y)` fixed — the same applicants, just more of the risky type. Here the applicant *composition* demonstrably changed (`channel` V = 0.211, a brand-new category at 9% volume), so this is not a pure prior shift. Prior shift would also leave calibration roughly intact after a threshold adjustment; a 3.7× ECE jump is too large for that. |
| **Pure covariate shift** | If the model generalized correctly to the new region, `P(y\|x)` would be unchanged, calibration would hold, and realized delinquency at a given predicted score would match. ECE 0.019 → 0.071 says it does not. Covariate shift is present but is not sufficient to explain the loss. |
| **Pure concept drift across the whole population** | Then `channel` would not be the only feature that moved, and the degradation would not be concentrated. The slice check in §1 distinguishes these definitively: if established channels are still at ~4.1%, the drift is localized, not global. |
| **A label-pipeline bug** | Possible in principle and cheap to check — confirm the 60-day delinquency definition, cohort window and join logic are unchanged. Do this before spending anything, because "the metric broke" is always the cheapest explanation and it is sometimes right. |

---

## 3. How ECE moves 3.7× while prediction PSI stays at 0.021

This is the crux, and it is worth being precise.

**PSI measures the *marginal* distribution of the scores.** ECE measures the *relationship between
scores and outcomes*. They are independent quantities.

Concretely: 91% of applicants are unchanged and produce the same score distribution as before. The
new 9% receive scores drawn from a *similar-looking* range — the marketplace applicants are not
obviously exotic on income, credit score or debt-to-income, which is precisely why the top-5 PSIs
are all under 0.06. So the aggregate score histogram barely moves. PSI is computed on that
histogram, so PSI barely moves.

But within that 9%, a score of 0.15 is followed by default roughly twice as often as it is for
everyone else. Calibration is a *conditional* property, and a 9% sub-population with badly wrong
conditional behaviour damages aggregate ECE substantially while contributing almost nothing to a
marginal-distribution statistic.

**The general lesson, which is the reason this drill exists:** *distributional drift metrics and
calibration metrics answer different questions, and a small mis-served sub-population is visible to
the second and nearly invisible to the first.* If you monitor only PSI, this class of failure is
undetectable. Monitor calibration, and monitor it **per slice**, because even aggregate ECE only
caught this because 9% is a fairly large minority. At 2% it would have hidden there too.

---

## 4. Why nothing paged, and what rule would have caught it

**Was excluding rank-14 features from alerting wrong?** No — as a general policy it is correct.
Alerting on all 48 features at α = 0.05 produces ~2.4 false alarms a day, the dashboard gets muted,
and then nothing is caught (module §5.2). The team made a defensible decision. It was also,
specifically here, insufficient. Both things are true, and pretending otherwise leads to "monitor
everything", which is how you get to zero recall.

**The rule that would have caught it without adding daily noise:**

> **Alert on a *new categorical value* exceeding a volume threshold, on ANY feature, regardless of
> importance rank.**
>
> Trigger: a category value absent from the training data exceeding 2% of requests for 24 hours.

Why this is the right rule:

1. **It is a rare event, not a continuous statistic.** Genuinely new category values appear a
   handful of times a year. This adds roughly zero alerts per month, so it costs nothing from the
   attention budget.
2. **It is unambiguously out-of-distribution.** The model has *never seen* this value. Whatever
   embedding or encoding it receives is arbitrary — for a one-hot encoding it is likely all-zeros or
   an "unknown" bucket that carries training-time semantics that do not apply. There is no
   generalization guarantee whatsoever, at any importance rank.
3. **Importance rank is the wrong gate for this signal.** Rank 14 measures average contribution over
   the *training* distribution. It says nothing about behaviour on a value that was not in the
   training distribution.

**Two supporting rules worth adding at the same time:**

- **Per-slice calibration monitoring** on the largest slices of every categorical feature. This
  would have caught the mechanism directly, though not until labels matured.
- **Cohort-decomposed drift**: compute drift globally *and* on the sub-population present in both
  windows. Here, the same-cohort PSI on everything would be flat, immediately localizing the change
  to the new channel — the same technique that suppresses new-customer false alarms in drill 1, used
  here to localize a real one.

---

## 5. The fix

**The binding constraint is the 60-day label delay.** Whatever you do, you will not know whether it
worked until roughly week 42. That fact should push you toward interventions that are (a) reversible,
(b) verifiable by something other than delinquency, and (c) conservative — because the cost of
over-correcting for two months is far smaller than the cost of under-correcting for two months while
money leaks.

### Next 24 hours — stop the bleeding

1. **Confirm the slice.** Compute delinquency, ECE and approval rate by `channel`. If the diagnosis
   is wrong, everything below is wrong. Thirty minutes of work.
2. **Confirm the label pipeline** is unchanged. Ten minutes, and it is the cheapest possible
   explanation.
3. **Raise the approval threshold for `partner_mkt` applicants only**, to a level whose implied
   risk matches the established channels' realized risk. This is a config change, it is instantly
   reversible, and it does not touch the model. If the estimated `d_new ≈ 0.34` holds, this is a
   large adjustment and it will materially reduce approvals in that channel.
4. **Cap exposure**: reduce the maximum loan amount for the channel while the correction is
   unverified. Reversible, and it bounds the loss during the 60-day blind period.
5. **Do NOT retrain yet.** You have almost no matured labels from the new channel — the loans from
   week 30 mature at week 38. Retraining now trains on four weeks of *immature* outcomes from a
   population you do not understand, and it would bake that misunderstanding in permanently.

Point 5 is the one people get wrong. The reflex is to retrain; the decision tree in module §6 ends
in "do not retrain" on five of seven branches, and this is one of them.

### Next 2 weeks — understand, then correct properly

1. **Talk to the partnership owner.** What does the marketplace's funnel look like? Are applicants
   pre-filtered, and on what? Are they people who were declined elsewhere? Is anything about the
   application form different? This is a fifteen-minute conversation that may explain everything,
   and it is faster than any analysis.
2. **Compare the two populations feature by feature**, within `partner_mkt` versus established
   channels — not just the marginals that PSI already told you were fine, but the *joint* structure.
   A domain classifier (module §3.3) is the efficient way: train a small model to distinguish the
   two channels and read its feature importances.
3. **Decide whether `partner_mkt` needs a separate model or a channel feature.** If the relationship
   genuinely differs, a single model with `channel` as a feature can learn it — *once you have
   labels*. Until then the threshold adjustment stands.
4. **Instrument a small, capped random-approval holdout** in the risky score band for this channel,
   so you observe outcomes in the region you are now declining. Without it, you will never learn
   whether your correction over-shot — this is the §3.5 censoring problem, and it is about to become
   your problem in exactly the region you just changed.
5. **Set an explicit review date at week 40**, when the first `partner_mkt` cohort matures, with a
   named owner. A blind period needs a scheduled end, or it becomes permanent.

### Permanent monitoring changes

| Change | Rationale |
|---|---|
| Alert on new categorical values > 2% for 24 h, any feature | §4 — the rule that would have caught this |
| Per-slice calibration (ECE) on the top categorical slices | catches localized concept drift, which is what actually happened |
| Cohort-decomposed drift | separates compositional change from behavioural change |
| Delinquency by channel, cohort, and score decile on the standing dashboard | the BI team found this before monitoring did, which is the real failure |
| An **early surrogate** — 15-day first-payment-missed rate — validated against the 60-day outcome | shortens detection latency from 60 days to ~15. This is the highest-value item on the list and it is the only one that attacks the floor set by the label delay |

---

## 6. The uncomfortable question

The fix raises the approval threshold for one channel. Before shipping it, check **who those
applicants are**.

**What to look at:**

1. Demographic composition of `partner_mkt` versus established channels, on whatever protected or
   proxy attributes you are permitted to analyse. Marketplace and partnership channels frequently
   over-index on younger, lower-income, thinner-credit-file, or geographically concentrated
   applicants — and those correlate with protected attributes.
2. Whether the threshold change produces a **disparate impact** on any group, measured as the change
   in approval rate by group, not just by channel.
3. Whether the *elevated delinquency itself* is real or an artefact — for example, if the partner's
   applicants systematically receive worse loan terms through some other part of the system, the
   higher default rate is partly *caused* by us and is not a property of the applicants.

**If the answer is uncomfortable — say, `partner_mkt` is 70% one demographic group and your fix cuts
their approval rate by a third — here is what to do:**

- **Ship the exposure cap, hold the threshold change**, and escalate the same day. The loan-amount
  cap reduces risk without changing who gets approved, which buys you time without making a
  consequential distributional decision unilaterally.
- **Escalate to the people who own the decision**: risk, legal, and the product owner for the
  partnership. Bring the numbers: delinquency by channel, the demographic composition, the projected
  approval-rate change by group, and the money at stake per week of delay. Make it easy to decide.
- **Reframe the question correctly.** "Should we approve fewer people from this channel?" is a risk
  question. "Should we approve fewer people from this channel, given that the channel is
  demographically distinct and using it as a decision input may function as a proxy for a protected
  attribute?" is a legal and policy question. **It is not yours to answer alone**, and answering it
  alone at 09:00 on a Tuesday is the mistake — regardless of which way you answer it.
- **Investigate the mechanism, because it changes the ethics.** If marketplace applicants default
  more because they are genuinely higher risk, that is one situation. If they default more because
  the partnership's funnel misleads them about terms, or because they are being routed into worse
  products, then the fix is upstream and declining them is treating a symptom while leaving the
  cause — and the harm — in place.

The general principle: **a channel feature that is highly correlated with a protected attribute is a
proxy, whether or not you intended it as one.** Recognising that before you ship is the job.
