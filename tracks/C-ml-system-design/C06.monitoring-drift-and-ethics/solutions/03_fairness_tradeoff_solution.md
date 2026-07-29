# Solution 3 — The candidate-screening fairness tradeoff

**Headline:** the model violates equal opportunity substantially (TPR 0.681 vs 0.474), the selection
gap is *larger* than the base-rate gap in a way that needs explaining, and — most importantly — every
criterion here is defined relative to a label produced by human interviewers, which makes the label
the weakest link in the entire analysis.

---

## 1. What the numbers say

| Criterion | Definition | Group A | Group B | Difference | Satisfied? |
|---|---|---:|---:|---:|---|
| **Demographic parity** | equal `P(Ŷ=1)` | 0.312 | 0.148 | **0.164** | No — B selected at 47% of A's rate |
| **Equal opportunity** | equal `P(Ŷ=1 \| Y=1)` | 0.681 | 0.474 | **0.207** | No — the largest violation |
| **Equalized odds** | equal TPR **and** FPR | TPR 0.681 / FPR 0.122 | TPR 0.474 / FPR 0.062 | TPR 0.207, FPR 0.060 → **0.207** | No |
| **Predictive parity** | equal precision | 0.742 | 0.673 | **0.069** | Approximately — modest gap |
| **Calibration within groups** | `P(Y=1 \| S=s, A=a)` equal | ECE 0.021 | ECE 0.026 | 0.005 | **Yes** — both well calibrated |

Two observations that shape everything below.

**Observation 1: the model is calibrated and still fails equal opportunity badly.** That is not a
contradiction and it is not a bug — it is the impossibility result (Chouldechova 2017; Kleinberg,
Mullainathan & Raghavan 2016) presenting itself. With base rates of 0.34 and 0.21, calibration and
equalized odds cannot both hold. The model achieved calibration. It therefore could not achieve
equalized odds, and no amount of better modelling changes that.

**Observation 2: B's error profile is asymmetric in a revealing way.** B has *lower* FPR (0.062 vs
0.122) and *much* lower TPR (0.474 vs 0.681). The model is systematically **more conservative** for
group B: it makes fewer wrong advances and far more wrong rejections. In a screening context those
two errors have wildly different costs to the individual, which is the whole argument of §3.

You can reproduce the structure with:

```bash
python ../tools/fairness_metrics.py --base-rate-a 0.34 --base-rate-b 0.21 --n-a 41200 --n-b 9850
```

The absolute numbers differ from the audit table (different data-generating process), but the
diagonal pattern — each intervention zeroing one column and moving the others — reproduces exactly.

---

## 2. Answering the VP's question, completely

The technically accurate partial answer: the groups have different base rates in the training label
(0.34 vs 0.21), so a single threshold applied to a calibrated score necessarily selects a smaller
share of the lower-base-rate group. Nothing about the selection gap on its own proves the model is
doing anything improper.

**But that answer is incomplete, and stopping there would be misleading.** Do the ratios:

```
  selection rate ratio : 0.148 / 0.312 = 0.474
  base rate ratio      : 0.21  / 0.34  = 0.618
```

**The selection gap is substantially larger than the base-rate gap.** If the model were equally good
at identifying qualified candidates in both groups, the selection ratio would track the base-rate
ratio far more closely. It does not, and the TPR row says why: among candidates who *do* go on to
pass the technical interview, the model finds 68% of group A and only 47% of group B.

So the complete answer is: *part* of the selection gap is explained by the base rates, and part is
the model being materially worse at recognising qualified group-B candidates.

**Mechanisms that produce that gap** — worth naming, because they determine the fix:

1. **Representation.** Group B is 19% of applicants; the model has roughly a fifth as much data to
   learn their signal from. Whatever features generalize less well across groups will be fitted
   worse for B.
2. **Measurement bias in the features.** The features may be *less predictive* for group B — e.g.
   a feature derived from prior employer prestige, referral networks, or a keyword vocabulary that
   varies with background. The model is not "biased against B"; the features are less informative
   about B, and the model correctly responds by being less confident, which at a fixed threshold
   means fewer advances.
3. **Measurement bias in the label** — the big one, deferred to §5.
4. **Aggregation bias.** One model for two populations whose feature-outcome relationships differ.

The distinction between (1)/(2) and a genuinely different qualification rate matters enormously,
because a lower TPR at equal calibration is the signature of *noisier features*, not of a fair
model doing its job.

---

## 3. The criterion, and the steelman against it

### Recommendation: **equal opportunity** (equal TPR), monitored alongside calibration.

**Justified from the decision, not from the metrics.** Screening allocates an *opportunity*. Consider
the two error types from the candidate's perspective:

- A **false positive** advances a candidate who then does not pass the technical interview. Cost:
  one interview slot, some of the candidate's time, a rejection later. Recoverable, modest, and
  borne largely by the employer.
- A **false negative** rejects a candidate who *would have* passed. Cost: the job. And it is
  **invisible** — the candidate never learns, the employer never learns, and there is no feedback
  path that would ever correct it. It is also the error type that compounds across a career.

When the two errors are that asymmetric and the costly one falls on the individual, the criterion
that governs is the one on the true-positive rate. That is equal opportunity, and it is precisely
the argument Hardt, Price and Srebro make for it.

**A 0.207 TPR gap means: among group-B candidates who would have passed the technical interview, we
are rejecting about 21 percentage points more of them than we reject of equivalent group-A
candidates.** That is the sentence to put in the memo.

### The strongest counter-argument, stated properly, then rejected

**Steelman for predictive parity / leaving the shared threshold alone:**

> The model is well calibrated in both groups. A score of 0.7 genuinely means a 70% chance of
> passing in either group — the score is *honest*. Any per-group threshold adjustment breaks that
> honesty for the recruiter who reads the score, because "advanced" would then mean different things
> for different candidates. Worse, the recruiter overrides 6% of the time; if they learn that
> advanced group-B candidates are less likely to pass, they will start compensating with their own
> override, which reintroduces the disparity through a channel nobody is measuring. And using group
> membership in the decision rule is legally fraught in employment contexts in several
> jurisdictions. The honest calibrated score is the defensible artefact; equalizing TPR is a
> political adjustment applied on top of a working model.

That is a serious argument and the recruiter-compensation point in particular is a real, documented
failure mode of post-processing interventions.

**Why I still reject it here:**

1. It **treats the label as ground truth**, and §5 argues that it is not. Calibration to a biased
   label is calibration to bias, and its "honesty" is honesty about the wrong quantity.
2. It **accepts a 0.207 TPR gap as a fact about the world** rather than as a symptom of features
   that are less informative for group B. It is not; observation 2 says the model is systematically
   more conservative for B, and that is a property of the model and its features, not of the
   candidates.
3. The recruiter-compensation concern is **an argument for a different implementation**, not for
   inaction. See §4: fixing the features and the training data preserves a single threshold, and it
   is the intervention I actually recommend.

---

## 4. The intervention and its cost

### Preferred: fix it upstream. Fallback: adjust thresholds.

**Tier 1 — upstream, preferred, slow (one to two quarters):**

| Action | Why it is better than post-processing |
|---|---|
| Reweight or oversample group B in training, and evaluate per-group | Attacks representation bias directly; preserves one threshold |
| Audit features for differential informativeness — fit per-group models and compare feature importances | Identifies *which* features are less predictive for B, which is actionable in a way "the model is biased" is not |
| Drop or replace features that are proxies for background rather than capability | Improves the model for everyone, not just the metric |
| Collect better features for the under-served group | The only fix that raises TPR without lowering anything |
| Add a fairness constraint to the training objective | Keeps a single threshold; costs some aggregate accuracy |

This tier is preferred because it makes the model *better*, keeps one threshold, and does not use
group membership at decision time — which dissolves both the recruiter-compensation problem and the
legal exposure.

**Tier 2 — post-processing, fast, use while tier 1 runs:** lower the threshold for group B until
TPR matches (≈0.68 in both).

**The cost, in numbers.** From `fairness_metrics.py` on a comparable synthetic setup, moving from a
shared threshold to equalized TPR:

| Metric | Shared threshold | Equal opportunity | Change |
|---|---:|---:|---|
| Demographic parity difference | 0.209 | 0.071 | improves |
| Equal-opportunity difference | 0.207 | 0.002 | **eliminated (by construction)** |
| FPR difference | 0.110 | 0.005 | improves |
| **Precision (PPV) difference** | 0.047 | **0.148** | **worsens ~3×** |
| Overall accuracy | 0.770 | 0.761 | **−0.9 pp** |

Read the precision row carefully: it is the *quiet casualty*. Under a shared threshold precision is
nearly equal across groups. After equalizing TPR, an advanced group-B candidate is meaningfully less
likely to pass the interview than an advanced group-A candidate. **That is what the recruiter will
notice**, and it is exactly the channel through which post-processing gets undone by human
compensation. If you ship tier 2, you must (a) tell the recruiters what changed and why, and (b)
monitor the override rate by group as a first-class metric.

**Who is actually affected.** Scaled to the audit population, lowering group B's threshold to equalize
TPR moves the group-B selection rate from 0.148 to roughly 0.22:

- **About 700 additional group-B candidates advance per audit period.** Of those, roughly 430 would
  have passed the technical interview and were previously being rejected in error. That is the
  benefit, and it is the number to lead with.
- **Roughly 270 additional group-B candidates advance and do not pass.** They spend time on an
  interview they will not clear; the employer spends interview capacity.
- **Group A is unaffected** under a B-only threshold change. Under a fixed-capacity model — a limited
  number of interview slots — it is *not* unaffected, and roughly 700 marginal group-A candidates
  would be displaced. **Ask whether capacity is fixed before you promise nobody loses**, because if
  it is, this is a reallocation and it must be described as one.

That last point is the difference between a fairness memo that survives contact with the business
and one that does not.

---

## 5. Question the label — the most important section

`Y = "passed the technical interview"` is produced by human interviewers. Every criterion in §1 is
defined relative to `Y`. **If `Y` is biased, satisfying any of them satisfies nothing** — you have
made the model faithfully reproduce a biased process, and you have a metric that says you succeeded.

**What could be wrong with it:**

1. **Interviewer bias.** If technical interviewers systematically under-rate group-B candidates,
   then `Y` under-counts qualified group-B candidates. The observed base-rate gap (0.34 vs 0.21) is
   then partly an artefact of the interview, not a property of the candidates — and the model's
   "correct" calibration is calibration to that artefact.
2. **Selection on the outcome.** `Y` is only observed for candidates who were *advanced*. Everyone
   the model rejected has no `Y` at all. This is the §3.5 censoring problem in full force, and it
   means the base rates themselves are estimated on a model-selected sample. The 0.34 and 0.21 are
   not population base rates; they are base rates *among those the current process advanced*.
3. **The interview may not measure job performance.** Passing a technical interview is a proxy for
   being good at the job. If the interview format itself disadvantages some backgrounds — a
   whiteboard format, an unfamiliar domain framing, a communication-style dependency — then `Y`
   encodes that, and the model learns to predict who does well at *interviews*.
4. **Feedback loop.** The model influences who is interviewed, interview outcomes become labels, and
   the next model learns from them. Module §3.5, applied to people's careers.

**How to check:**

| Check | What it would show |
|---|---|
| **Interviewer-level analysis** — pass rate by (interviewer, group) | If the gap is concentrated in some interviewers, that is interviewer bias, not candidate quality |
| **Blind or structured re-scoring** of a sample of recorded interviews | Whether the gap survives when the assessor cannot see the candidate |
| **Downstream validation**: does passing the interview predict on-the-job performance, and *equally well for both groups?* | The single most informative check. If the interview predicts performance for A and not for B, the label is broken for B |
| **A random-advance holdout**: advance a small random sample the model would have rejected | The only way to observe `Y` in the censored region and get unbiased base rates. Costly, and it is the only clean answer |
| Compare against external base rates for the relevant skill population | Weak evidence, but a large discrepancy is a flag worth chasing |

**If you cannot check it** — which is common, because the downstream performance data often does not
exist or is itself subjective — then say so explicitly and change what you claim. Specifically:

- **Do not claim the model is fair.** Claim that it satisfies criterion X *with respect to the
  interview-outcome label*, and state that the label's validity across groups is unverified. That
  sentence belongs in the model card verbatim.
- **Prefer interventions that are robust to a biased label.** Improving features and representation
  helps regardless. Post-processing to match a possibly-biased label's error rates is a smaller,
  more fragile claim.
- **Push for the random-advance holdout.** It is the only mechanism that produces unbiased base
  rates, and it is worth arguing for even though it costs interview capacity, because without it
  every number in §1 rests on an unverified assumption.
- **Escalate the label question**, because it is a bigger problem than the model. If the interview
  process is biased, the model is a small part of the harm and fixing the model may even reduce the
  pressure to fix the interview.

---

## 6. What is not mine to decide

| Decision | Owner | What I bring them |
|---|---|---|
| Which fairness criterion governs a hiring decision | HR/People leadership with Legal | The tradeoff table, the definition of each criterion in one plain sentence, the cost of each in accuracy and in who is affected, and my recommendation with its reasoning |
| Whether group membership may enter the decision rule | **Legal** | The two implementation options (per-group thresholds vs upstream fixes), what each requires operationally, and an explicit note that I am not qualified to judge the legal exposure |
| Whether the interview process itself is valid | Hiring/interview process owner | The interviewer-level pass-rate analysis, the proposed validation study, and what it would cost |
| Whether to run a random-advance holdout | Hiring leadership | What it buys statistically, how many candidates it involves, and the alternative (permanently unverified base rates) |
| Accepting −0.9 pp accuracy for the intervention | The accountable product/business owner | The before/after table, the affected-candidate counts in both directions, and the capacity question from §4 |
| Whether interview capacity is fixed | Recruiting operations | Why it changes whether this is an addition or a reallocation |

**What is mine:** producing correct numbers, surfacing the tradeoff honestly, flagging the label
problem loudly, recommending a criterion with reasoning, building the per-slice monitoring, and
writing the model card. Making the choice is not mine. Neither is staying quiet about it.

---

## 7. The conversation with the VP

> "Good question, and the answer has two parts. Some of that gap is expected: in our historical
> data, group B candidates pass the technical interview at a lower rate, so a single scoring bar
> will advance fewer of them — that part isn't the model misbehaving. But the gap we're seeing is
> bigger than that explanation covers: among candidates who *do* go on to pass the interview, the
> model finds about 68% of group A and only 47% of group B, so we're missing qualified people in one
> group more than the other. We can close that, and I want to be straight that it's a choice with a
> price — it costs us about a point of overall accuracy and it means some of the extra people we
> advance won't clear the interview. The bigger thing I'd flag is that all of this is measured
> against 'passed our technical interview', and I don't currently have evidence that the interview
> is equally accurate for both groups — if it isn't, we'd be tuning the model to match a biased
> yardstick. I'd like to bring you and Legal a one-page options memo next week with the numbers, and
> separately I'd like to start a small study on the interview itself."

Why this works:

- **It answers the actual question first**, honestly, including the part that exonerates the model.
- **It gives the real finding in one sentence a non-specialist can repeat** — "68% versus 47% of the
  people who would have passed" — which is the sentence that will get repeated in the next meeting
  whether you are there or not.
- **It names the price without hedging.** A VP who later discovers there was a cost you did not
  mention will discount everything else you said.
- **It raises the label problem without making it the headline**, because leading with "our
  interviews may be biased" turns a technical conversation into a defensive one and the point gets
  lost.
- **It ends with two concrete next steps and a date.** The failure mode of fairness conversations is
  that everyone agrees it is important and nothing is scheduled.
