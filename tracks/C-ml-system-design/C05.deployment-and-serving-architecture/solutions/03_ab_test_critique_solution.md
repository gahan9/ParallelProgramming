# Solution 3 — Critique the Smart Reply v2 experiment

The write-up is confident, numerate, and reports p = 1.2 × 10⁻⁶. It is also uninterpretable. That
combination is the lesson: **a tiny p-value on a corrupted design is a very precise measurement of
the wrong thing.**

---

## Part A — The defects

### 1. Randomization unit is the request, not the user — **uninterpretable**

> "Each incoming message was independently assigned to control or treatment with probability 0.5,
> using `random()` at request time."

**Mechanism, three ways at once:**

- **Contamination.** Every active user receives both v1 and v2 suggestions, interleaved. Neither arm
  measures a coherent user experience, and any effect that depends on a user *learning* the feature
  is destroyed.
- **Understated variance.** Suggestion sets from one user are strongly correlated — the same person,
  the same conversations, the same reply habits. Treating 41.2 M suggestion sets as 41.2 M
  independent Bernoulli trials inflates the effective sample size enormously. The standard error is
  therefore far too small, the z-statistic far too large, and **the reported p-value is meaningless
  regardless of whether the effect is real**. The design effect for clustered data of this kind is
  routinely 3–20×; the true standard error may be several times the reported one.
- **No user-level inference is possible.** You cannot ask "did v2 make users happier?" because no
  user was ever in a single arm.

**Fix:** randomize on `hash(user_id + "smart_reply_v2_salt") % 100`. Stable across sessions and
devices, salted so it does not correlate with other running experiments. If a request-level metric
must be analysed under user-level assignment, use a cluster-robust variance estimator or the delta
method — never a naive z-test on request counts.

### 2. Peeking — **optimistic**

> "We watched the dashboard from the first hour. The result crossed p < 0.05 on Tuesday afternoon,
> dipped back on Wednesday morning, and has been comfortably significant since Wednesday evening."

**Mechanism:** each look is its own hypothesis test carrying its own α. Under the null, the
probability that *at least one* of many looks crosses the threshold is far above 5%. The module's
seeded A/A simulation measures 18.1% at ten looks and 24.9% at twenty (`ab_calculator.py peeking`).
Continuous dashboard-watching is effectively unbounded looks. The sentence "dipped back on Wednesday
morning" is itself the tell — that is what a random walk crossing a fixed boundary looks like.

Note the compounding problem: there is **no stated sample size and no stated stopping rule**, so we
cannot even tell whether the decision to stop was data-dependent. Given the narrative, it plainly
was.

**Fix:** fix the sample size in advance from an MDE, or pre-register a group-sequential boundary
(O'Brien–Fleming, Pocock), or use always-valid inference (Johari et al. 2015) if a live dashboard is
genuinely required.

### 3. No guardrail metric — **optimistic, and the most dangerous defect here**

The only metric is suggestion CTR. Smart Reply CTR is an **engagement proxy**, and it is one that a
model can raise while making the product worse in several concrete ways:

- Suggesting replies that are *provocative or emotive* rather than useful — high tap rate, worse
  conversations.
- Suggesting **shorter, lower-information** replies that are easier to tap. CTR up, message quality
  down.
- **Cannibalizing typed replies**: a user who would have written a thoughtful message taps a canned
  one instead. CTR rises and total conversation value falls.
- Safety: a larger model trained on six more months of data can surface inappropriate suggestions in
  sensitive conversations. Nothing in this test would notice.

**Fix — required guardrails:** total messages sent per user, conversation length/continuation rate,
typed-versus-tapped mix, user-level retention, reported/blocked-message rate, an offensive-suggestion
rate from a safety classifier, and the serving cost and latency (see defect 6).

### 4. Four days, Tuesday to Friday — **uninterpretable direction**

**Mechanism:** the test excludes the weekend entirely. Messaging behaviour differs sharply between
weekday and weekend (conversation types, session lengths, who is talking to whom). The measured
effect applies to Tuesday-to-Thursday users, and there is no basis for extrapolating it to the
population. It also fails to cover a full weekly cycle, so any weekly seasonality is aliased into
the treatment effect.

**Fix:** minimum one full week; ideally two, which also lets a novelty effect decay.

### 5. Novelty effect not addressed — **optimistic**

**Mechanism:** v2's suggestions are visibly different. Users try new things. A four-day window
measures the exploration spike, not the steady state. The write-up's own timeline — significant
Tuesday, dipping Wednesday, "comfortably significant" after — is consistent with a novelty curve and
is presented as evidence of stability.

**Fix:** plot the daily treatment effect and require it to be flat before deciding; segment new
versus existing users; run at least two weeks.

### 6. Latency regression dismissed without measuring what matters — **optimistic**

> "p99 suggestion latency went from 41 ms to 88 ms. This is well within our 300 ms budget."

**Mechanism:** the check is against a *component* budget in isolation. Three things are unexamined:
whether the end-to-end p99 (not the component's) still fits; what happens on a degraded day when
other hops are also slow; and whether a suggestion arriving 47 ms later is *shown less often* —
if the UI has a render deadline, slower suggestions may simply not appear, which changes the
denominator of CTR itself. That last one would make the metric incomparable between arms.

Also unmentioned: the larger model costs more to serve, and no cost-per-1k figure appears anywhere.

**Fix:** measure end-to-end p99, measure the suggestion *display* rate per arm (a denominator check),
and report cost per 1 000 suggestion sets.

### 7. No SRM check — **uninterpretable if it fails**

41 204 118 versus 41 198 802 is a difference of 5 316 on ~82.4 M, which is about 0.006% — well
within chance for a fair coin, so **this particular split looks fine**. But the write-up never
checks, and reporting the check is what makes the result trustworthy. On a request-randomized test
the check is also nearly worthless as a bug-detector, because it validates the request split, not
the user split — which is exactly the thing that is broken.

**Fix:** automated SRM check (χ² on the assignment counts) on every experiment, at the randomization
unit, with a p < 0.001 threshold and a hard halt on breach.

### 8. Effect size is presented without its uncertainty, then monetized — **optimistic**

The write-up gives a point estimate (+0.93% relative) and a dollar figure, and no interval. See
Part B: the interval is actually fairly tight here, but the *practice* of converting a point
estimate directly to revenue with no interval, no cost offset, and no stated conversion assumption
is how experiment results become fiction.

### 9. No pre-registered hypothesis or analysis plan — **optimistic**

Nothing states what was decided before launch. With one metric, one comparison and a clean
narrative, this may be fine — or CTR may be the one metric of several that moved. There is no way to
tell from the document, and "no way to tell" is a defect in a document whose purpose is to justify a
decision.

**Fix:** pre-register hypothesis, primary metric, guardrails, MDE, sample size, duration, analysis
plan and stopping rule before launch.

### 10. "Downstream engagement value" is an unexplained conversion — **optimistic**

\$2.1 M/year comes from somewhere. What is the assumed value per incremental tap? Was it validated
against a long-term holdout, or is it a per-click figure someone estimated once? Proxy-to-revenue
conversions are exactly where experiment results acquire authority they have not earned. State the
model and its provenance, or drop the number.

---

## Part B — Is the effect worth shipping?

```bash
python ../tools/ab_calculator.py evaluate --n-control 41204118 --c-control 5767191 \
                                          --n-treatment 41198802 --c-treatment 5819251
```

```
  control   : 41,204,118 users, 5,767,191 conv -> 0.1400
  treatment : 41,198,802 users, 5,819,251 conv -> 0.1412

  absolute lift : +0.00128
  relative lift : +0.92%
  z statistic   : +16.7350
  p-value       : 0.00000
  95% CI (abs)  : [+0.00113, +0.00143]
```

Converting the interval to relative terms against the 13.996% baseline: **[+0.81%, +1.02%]**.

Taken at face value that is a tight, positive interval — the effect is small but precisely
estimated. So why is "worth roughly \$2.1 M/year" still not defensible?

1. **The interval is computed under the independence assumption that defect 1 destroys.** With
   user-level clustering, the true interval is materially wider — plausibly several times wider. The
   sign is probably still positive; the magnitude is not established.
2. **It is an interval on CTR, not on value.** The \$2.1 M conversion is an unexamined multiplication
   (defect 10), and it ignores the 2.3×-style cost increase implied by a larger model (defect 6).
3. **A +0.9% CTR lift is exactly the size of effect that novelty produces** (defect 5). Four days
   cannot distinguish a durable 0.9% from a transient 2% decaying toward 0%.

**What would make it defensible:** re-run user-randomized for two weeks with the guardrails from
defect 3, report the interval with a cluster-robust standard error, subtract the incremental serving
cost, and state the CTR→value model with its provenance. If the lower bound of *that* interval, net
of cost, is still positive, ship it.

---

## Part C — The correct design

```
  hypothesis        : Smart Reply v2 increases the number of messages a user sends per week
                      relative to v1, without reducing conversation quality or safety.
                      (Note the shift: the hypothesis is about the user-level product outcome,
                      not about the tap rate on the widget.)

  primary metric    : messages sent per user per week.
                      Chosen over CTR because CTR can rise while the product gets worse --
                      shorter replies, cannibalized typing, provocative suggestions.
                      CTR becomes a secondary/diagnostic metric.

  guardrail metrics : conversation continuation rate; typed-vs-tapped mix; 7-day user retention;
                      reported/blocked message rate; offensive-suggestion rate (safety classifier);
                      end-to-end p99 suggestion latency; suggestion display rate (denominator
                      integrity check); cost per 1,000 suggestion sets.

  randomization unit: user. hash(user_id + "smart_reply_v2_2026q3") % 100.
                      Stable across sessions and devices; salted per experiment.

  MDE               : +1.0% relative on the primary metric. Set from product value: below 1% the
                      lift does not justify the larger model's serving cost.

  sample size       : computed from the observed baseline and variance of "messages per user per
                      week" -- a count metric, so use its measured mean and variance, not the
                      proportion formula. For the SECONDARY CTR analysis at a 14.0% baseline and
                      1% relative MDE, alpha 0.05, power 0.80:
                        python ../tools/ab_calculator.py size --baseline 0.14 --mde-rel 0.01
                        -> 968,323 users per arm
                      At 25M eligible users/day that is reached in hours, so the binding
                      constraint is DURATION, not sample size. Say so explicitly: this is the
                      comfortable case, and it means there is no excuse for a four-day test.

  duration          : 14 days minimum. Two full weekly cycles (weekday/weekend mix), and long
                      enough for a novelty effect to decay. NOT sample-size-limited -- see above.

  analysis plan     : two-sided test on the primary metric at alpha = 0.05, with a cluster-robust
                      standard error at the user level. Pre-specified segments: new vs existing
                      users, platform, locale, and conversation-volume decile. Daily treatment-
                      effect plot to check for novelty decay. Multiple-comparison correction
                      (Benjamini-Hochberg) across the guardrail set. No other comparisons will be
                      treated as confirmatory.

  stopping rule     : fixed sample size, analysed once at day 14. Interim looks are for GUARDRAILS
                      ONLY and may only stop the test for HARM, never for a win. The asymmetry is
                      deliberate and is stated up front.

  SRM check         : chi-square on the user assignment counts, checked hourly, halt at p < 0.001.
                      Halt means "stop and investigate", not "roll back" -- an SRM says the data
                      is untrustworthy, not that the treatment is bad.
```

---

## Part D — The conversation

> "This is a solid piece of work and the effect direction looks right — thanks for the clear
> write-up, it made the design easy to follow. Before we ship, I want to flag one structural thing
> that I think changes the read: the randomization is per message rather than per user, which means
> the confidence interval is narrower than it should be and we can't attribute a user-level outcome
> to it. Happy to sit down and re-plan it with you — with your traffic, a user-randomized two-week
> version would reach the sample size in hours, so this is mostly a scheduling cost rather than a
> redesign."

Why this works, and why it is graded:

- **It leads with something true and specific**, not a compliment sandwich. The work *is* good.
- **It names one structural issue first**, not ten. The other nine go in the written review; leading
  with a list of ten reads as an attack and gets the whole review dismissed.
- **It gives the fix and its cost in the same breath.** "Mostly a scheduling cost" is the sentence
  that makes the fix acceptable to someone who has already told their manager the number.
- **It offers to do the work together.** The author has to walk back a claim to their manager; the
  reviewer who helps them do that gets better experiments for the next two years. The reviewer who
  is merely right gets ignored.

A technically correct review that lands badly ends with the wrong model shipping anyway. Being right
is necessary and is not sufficient.
