# Solution 03 — Skew postmortem

Reference answer for [`../exercises/03_skew_postmortem.md`](../exercises/03_skew_postmortem.md).

Headline: **three independent feature-skew defects, plus an immature-label problem in training, plus an
online comparison that is far too underpowered to conclude anything.** The trap in the exercise is E3 —
matching online and offline scores feels like it exonerates the pipeline and rules out almost nothing.

---

## Part 1 — Hypotheses in priority order

Ranked by likelihood divided by cost-to-check, so the cheap decisive checks come first.

| Rank | Hypothesis | Specific check | Confirms if | Eliminates if |
|---:|---|---|---|---|
| 1 | **Feature-value skew**: training and serving compute the same feature differently | Recompute all 84 features offline as of the logged request timestamps; diff against the logged serving values, per feature | Any material per-feature disagreement rate | All 84 features agree within tolerance |
| 2 | **Coverage / null-rate divergence**: the online store lacks values the warehouse has | Compare per-feature null (and default) rates between the training set and the logged serving vectors | Any feature's rate differs materially | Rates match per feature |
| 3 | **Score-distribution shift**: online scores do not look like offline scores | Compare mean, p50, p99, and the full histogram of offline test scores vs logged online scores | The distributions differ materially | They overlap | 
| 4 | **Model deployment mismatch**: the served artifact is not the evaluated one, or preprocessing differs | Re-score logged serving vectors offline with the evaluated artifact and diff the scores | Scores differ | Scores match to floating-point tolerance |
| 5 | **Leakage in training**: the offline number was never real | Timestamp and separation screens over the training table; audit any feature above the AUC threshold | A near-perfect or post-label feature exists | No feature is near-perfect and no timestamp is at/after the label |
| 6 | **Immature labels in training**: recent rows labelled "negative so far" | Compare the training cutoff against the label horizon; check the positive rate by row age | Rows exist inside one label horizon of the cutoff | Training excludes the immature window |
| 7 | **Underpowered online comparison — nothing may be broken at all** | Count matured positives per arm; compute the standard error and the minimum detectable effect | The MDE is larger than the effect you expected | Enough matured positives to resolve the expected effect |
| 8 | **Wrong operating point**: the ranking improved but the threshold was carried over from the old model's score scale | Compare alert volume and precision at the deployed threshold against the offline sweep | Alert volume differs materially from the offline prediction | Volumes match |
| 9 | **Prevalence or population shift** between the training window and now | PSI on the input features and the positive rate, training window vs live | Material drift | Stable |

### 1a. Highest information value per hour

**Rank 1 — recompute the features offline as of the logged timestamps and diff.** One query against
data you already have, and it either localizes the problem to specific features or eliminates the
single most likely cause of this exact symptom. Rank 2 (null-rate comparison) is even cheaper and is
really a subset of the same check, so run both together: rank 2 tells you *how many* rows are affected
and rank 1 tells you *which feature and why*.

The reason to prefer these over "look at the model" is that the symptom — good offline, flat online, no
errors — is the canonical signature of skew, and skew is the hypothesis whose check is cheapest.

### 1b. "Never real" vs "thrown away"

- **The offline number was never real:** #5 (leakage), #6 (immature labels), and the split-related
  variants of #6. The fix is to **re-measure**, and then to have a much harder conversation with
  stakeholders about a number that is going to move down.
- **The offline number was real but production throws it away:** #1 (skew), #2 (coverage), #4
  (deployment), #8 (operating point). The fix is **engineering** — the model is fine and the plumbing
  is not.

The distinction matters because it determines your next week *and* your next conversation. If the
number was never real, more engineering effort on serving is wasted and you must reset expectations. If
the gain is real but discarded, there is a recoverable win sitting in the pipeline and you should not
touch the model at all. Diagnosing the wrong one sends the team in exactly the wrong direction, which
is why the cheap discriminating checks come first.

### 1c. The hypothesis where nothing is broken

**#7 — the online comparison is underpowered.** Supporting evidence would be: few matured positive
labels per arm, a wide confidence interval on the difference, a minimum detectable effect larger than
the improvement you expected, and — critically here — **less elapsed time than the label horizon**, so
most outcomes are not yet knowable. "Statistically indistinguishable" is a statement about your
measurement, not about the model, and any postmortem that skips this hypothesis risks "fixing" a
system that was working.

---

## Part 3 — Diagnosis

### 3a. Primary cause

**The training and serving paths compute the same feature definitions differently, so the deployed
model receives inputs it was never fitted to** — established by **E4** (three features disagree when
recomputed from source, one of them on 73% of rows) and corroborated by **E2** (`orders_last_30d` and
`avg_basket_value_90d` null rates roughly 70× higher at serving) and **E1** (online scores shifted
materially lower and compressed, exactly what you expect when two important features are frequently
zeroed).

### 3b. Why E3 does not rule out skew

E3 took the **logged serving feature vectors** and re-scored them offline with the same artifact. The
inputs on both sides of that comparison were *identical* — they were the same vectors. So E3 tests only
the **function**: the model artifact, the feature ordering, the serialization, and the preprocessing
inside the scorer. It confirms all of that is correct, which is genuinely useful and eliminates
hypothesis #4.

What it cannot possibly detect is whether those logged vectors contained the **right values**. Skew is
a property of the *inputs*, and E3 held the inputs constant. The test that addresses it is E4:
recompute the features **from source** as of each request timestamp and diff against what was served.
That is a different comparison with a different second operand, and it is the one that found the bug.

The general principle, worth stating exactly this way in an interview: **re-scoring logged vectors
verifies the model; recomputing features from source verifies the data.** They are different checks and
only the second one tests for skew. Conflating them is how teams spend a week concluding "the
deployment is fine" while the pipeline quietly feeds garbage.

### 3c. The three defects, ranked

**Defect 1 (most severe) — `avg_basket_value_90d`: window semantics differ between paths.**
Spark SQL computes a mean over a **90-day time window**; the Go service computes the mean of the last
**90 order events**. These coincide only for a customer who orders roughly once a day. For a
high-frequency customer the serving window is about two weeks (so it tracks recent behaviour and misses
the seasonal baseline); for a low-frequency customer it spans two years (so it includes purchases from a
different life stage). **Cost:** 73% of rows carry a wrong value on a top-importance feature, and the
error is *systematically correlated with customer frequency* — which is itself correlated with fraud
risk. This is not noise; it is a distortion that varies by segment, so it degrades the model unevenly
and hits some populations much harder than others. Two engineers each wrote a correct implementation of
different specifications, which is why code review did not catch it: neither implementation looks wrong
on its own.

**Defect 2 — `orders_last_30d` (and `avg_basket_value_90d`): online coverage gap, silently coerced to
zero.** The Redis key exists only for customers active since a backfill four months ago, so 22% of
requests have no value. Both paths then replace null with `0`. **This is the silent-coercion pitfall
from README §6**: `orders_last_30d = 0` does not say "unknown", it says "this customer has placed no
orders in 30 days" — which is a confident, false statement that makes an established customer look
brand new. Since account novelty is a fraud signal, the model is being handed an actively misleading
input on more than one request in five. **Cost:** 22% of requests scored on a fabricated customer
profile, and it plausibly explains the downward score shift in E1. The coverage gap is bad; the silent
coercion is what makes it dangerous.

**Defect 3 — `merchant_category_risk`: stale static config at serving.** The warehouse has a corrected
category mapping; the serving path reads a static config file that was never updated. **Cost:** 4% of
rows, lower impact, but it reveals a structural problem — a feature whose definition lives in a config
file that is deployed separately from the model, with no version pinning and no mechanism to detect
divergence. Today it is 4%; after the next mapping change it could be anything.

**Defect 4 (process, contributing) — 100% traffic on day one with no shadow period.** Not a data defect,
but it converted a detectable pre-launch problem into three weeks of degraded production performance.
The score-distribution comparison in E1 is exactly what a shadow deploy produces automatically, before
any customer is affected.

### 3d. E6 — the immature-label problem

The training set includes transactions up to **12 days** before the cutoff, while the label horizon is
**45 days**. So roughly the last 33 days of training data is labelled *"no chargeback yet"* rather than
*"legitimate"*. Any fraud in that window whose chargeback had not yet been filed is labelled negative.

**Direction of bias:** the observed positive rate in the recent window is **too low**, so the model
learns that recent-looking transactions are safer than they are, and — because those rows are the most
recent and therefore the most representative of current fraud patterns — it specifically under-weights
the *newest* fraud signatures. It also depresses the learned prior, which depresses output scores.
Notice that both effects point the same way as E1's downward score shift, so this defect and the skew
defects compound rather than cancel.

**Fix**, in order of preference: **(a)** move the training cutoff back by at least one full label
horizon (train only on rows at least 45 days old), which is simple, correct, and costs you the
freshest six weeks of data; **(b)** if that freshness loss is unacceptable, model label maturation
explicitly — weight recent rows by the empirical fraction of chargebacks that have arrived by that age,
or treat immature rows as censored observations rather than negatives; **(c)** either way, apply the
same rule to the temporal *test* period, and leave a gap of at least one label horizon between train
and test.

### 3e. E7 — can you conclude the new model is not better? **No.**

With ~40 confirmed-fraud cases in the treatment arm, take a rate near `p = 0.5` as the worst case:

```
SE(one arm)         = sqrt(0.5 * 0.5 / 40)      = 0.079    -> +/- 15.5 points at 95%
SE(difference)      = sqrt(2) * 0.079           = 0.112
minimum detectable  = 1.96 * 0.112              ~ 0.219    -> ~22 percentage points

to resolve a 5-point difference near p = 0.5:
n per arm           = 2 * 0.25 * (1.96 / 0.05)^2 ~ 768 confirmed positives per arm
```

You can currently detect only a change of roughly **22 percentage points**. An offline ROC-AUC move
from 0.89 to 0.94 would not plausibly produce anything like that in a caught-value rate, so **the test
is nowhere near able to see the effect it was set up to measure.**

There is a second, worse problem that the arithmetic alone does not show. **Only 21 days have elapsed
against a 45-day label horizon**, so those 40 cases are not a random sample of fraud — they are the
*fastest-confirming* chargebacks, which is a specific and non-representative subset of fraud types. The
comparison is therefore both underpowered *and* biased.

**What to tell the product owner:**

> "We cannot yet say whether the new model is better or worse — the measurement is not capable of
> answering that. We have about 40 confirmed cases, which lets us detect roughly a 22-point swing, and
> the improvement we expected is far smaller than that. Worse, only three weeks have passed against a
> 45-day chargeback horizon, so the cases we do have are the fastest-confirming ones, not a
> representative sample. Separately, we have found real pipeline defects that mean the model is not
> receiving the features it was trained on, and those we should fix regardless of what the A/B test
> eventually says. My recommendation is to fix the feature defects, then run a properly powered
> comparison with a shadow period first — and in the meantime, judge the launch on the leading
> indicators (alert volume, analyst override rate, score distribution) rather than on a chargeback
> metric that cannot have matured yet."

### 3f. Why the temporal split caught none of this

Because **a split decides which rows go where; it says nothing about how features are computed.** Both
the training and test halves were produced by the *same* Spark SQL job over the *same* warehouse, so
both are equally divergent from what the Go service computes. A split can only detect problems whose
mechanism is the partition itself:

- Temporal splits catch learning from the future and non-stationary relationships (`split_demo`'s
  `x_drifting`: 0.5080 random → 0.3310 temporal).
- Grouped splits catch entity memorization (`x_user_mean_trainonly`: 0.6913 → 0.5000).
- **No split catches skew**, because offline evaluation only ever sees the offline pipeline. Skew is a
  difference *between* pipelines, and one of them is not present in the experiment.
- Note that no split catches a leak baked into the feature computation either — `x_user_mean_global`
  stays at 0.77–0.79 under all three.

The corollary is the whole argument for the feature log: **the only offline test that can detect skew is
one whose inputs came from the serving path.** Either recompute-and-diff (E4) or, better, train on the
logged serving features so the two distributions are identical by construction.

---

## Part 4 — Incident summary

> ## Impact
>
> A new card-not-present fraud model ran at 100% of traffic for 21 days while receiving feature values
> that differed materially from its training data. Approximately 22% of requests were scored with a
> fabricated customer-history profile and 73% with an incorrect basket-value average. Fraud
> caught-value per alert was indistinguishable from the previous model; the true impact on fraud losses
> cannot yet be quantified because the 45-day chargeback horizon has not matured. No customer-visible
> errors, no latency regression, no failed jobs.
>
> ## Root cause
>
> Feature definitions were implemented twice — Spark SQL for training, a Go service for serving — with
> no mechanism comparing the two. Three features diverged: `avg_basket_value_90d` used a 90-day time
> window offline and a 90-event window online; `orders_last_30d` was absent from the online store for
> customers inactive since a four-month-old backfill and was silently coerced to `0`;
> `merchant_category_risk` used a serving-side static config that lagged a warehouse mapping
> correction.
>
> ## Contributing factors
>
> 1. Serving feature vectors were logged but never compared against offline recomputation, so the
>    divergence was detectable for the entire period and undetected.
> 2. The model was promoted to 100% of traffic with no shadow period, so the online score-distribution
>    shift (mean 0.021 → 0.008) was not observed before customer impact.
> 3. Null values were coerced to `0` in both paths with no indicator recorded, converting "missing" into
>    a confidently false value and hiding the coverage gap.
> 4. Independently, the training set extended to 12 days before its cutoff against a 45-day label
>    horizon, so roughly 33 days of training rows were labelled "no chargeback yet" — biasing the model
>    toward treating recent transactions as safe.
> 5. The online evaluation was reported as "no difference" while being powered only to detect a ~22
>    point swing, on a label set that could not have matured.
>
> ## Immediate remediation
>
> 1. Reverted to the previous model pending fixes.
> 2. Aligned `avg_basket_value_90d` on the 90-day time-window definition, in one shared definition
>    consumed by both paths.
> 3. Backfilled the online store for all customers and added a coverage metric per feature key.
> 4. Replaced null→0 coercion with an explicit prior plus a `_was_default` indicator, logged per
>    request.
> 5. Pinned `merchant_category_risk`'s mapping to a versioned artifact rather than a deployed config
>    file.
> 6. Moved the training cutoff back one full label horizon and added a horizon-length gap between the
>    temporal train and test periods.
>
> ## Prevention
>
> 1. **Continuous skew detection**: sample logged serving vectors hourly, recompute the same features
>    offline as of those timestamps, alert on per-feature disagreement rate, and **block model releases**
>    while any feature is above threshold.
> 2. **Per-feature null/default-rate comparison** between the training set and live serving traffic, as
>    a hard release gate.
> 3. **Mandatory shadow period** before any traffic shift, gated on agreement between online and offline
>    score distributions.
> 4. **One feature definition, two materializations** — migrate feature computation to a single shared
>    definition so a second implementation cannot exist.
> 5. **Ban silent coercion**: any substituted value must emit an indicator, and default rates are
>    monitored per feature.
> 6. **Label-maturity assertion** in the training pipeline: fail the build if any training row is newer
>    than one label horizon before the cutoff.
> 7. **Power check before any launch decision**: report the minimum detectable effect alongside the
>    online comparison, so "no difference" cannot be stated without it.

---

## Part 5 — Prevention, ranked

| Measure | Rank | What it catches | Cost to build |
|---|:---:|---|---|
| Continuous skew job (recompute sampled requests offline, alert on disagreement) | **1** | All three feature defects, **with localization** to the specific feature and rate. Subsumes the null-rate gate | Moderate: a sampling job, a recompute path, a metric, an alert. Days, not weeks |
| Train on logged serving features rather than warehouse reconstructions | **2** | Removes the *penalty* from all three defects by construction — the model learns the served distribution, including defaults. Does not make the features correct | Moderate–high: needs enough logged history before you can train, and it hides bad features rather than fixing them |
| One shared transformation definition (feature store) | **3** | Prevents defects 1 and 3 structurally and forever, and its materialization layer largely prevents 2 | Highest: a platform migration. Best long-term answer, worst answer to "what do we do this week" |
| Per-feature null-rate comparison as a release gate | **4** | Defect 2 only — but that is 22% of requests, and this is by far the cheapest item on the list | Very low: one query and a threshold. Best value-per-hour of anything here |
| Shadow deploy with score-distribution comparison | **5** | None of the root causes, but catches the **symptom** (E1) before any customer impact, cutting the blast radius from 21 days to zero | Low: mostly a deployment-process change |

### If you could build only one: the continuous skew job

It is the only measure that both **detects** all three defects and **localizes** them to a named feature
with a rate, which is what turns an incident into a ticket. It subsumes the null-rate gate (a
default-rate divergence is a disagreement), it works today without a platform migration, and it
converts "we believe training and serving agree" from an assumption into a monitored metric — which is
the structural change that matters, because assumptions decay and metrics alarm.

Two honest caveats on that choice. If you have **one day**, build the null-rate comparison instead: it
is an hour of work and it would have caught 22% of requests being fabricated. And if you can change
**one process** rather than build anything, mandate the shadow period — it prevents nothing but it
would have reduced this incident from three weeks of degraded production to a pre-launch conversation.
The best answer to "build one thing" and the best answer to "prevent this incident cheaply" are not the
same thing, and saying so is a stronger answer than picking one and defending it absolutely.

---

## Common mistakes on this exercise

| Mistake | Why it costs you |
|---|---|
| Concluding from E3 that there is no skew | The exercise's central trap. Matching scores on identical inputs tests the model, not the data |
| Finding one defect and stopping | There are three feature defects plus an independent label problem plus a process failure |
| Accepting "no improvement" at face value | The comparison can only see a ~22-point swing, on labels that cannot have matured |
| Treating the null→0 coercion as a minor detail | It is what converts a coverage gap into an actively misleading input on 22% of requests |
| Missing the immature-label problem | It is independent of skew and biases the model in the same direction, so it compounds |
| Blaming the temporal split | The split was correctly constructed and is structurally incapable of detecting skew |
| Prevention items that are advice | "Review feature code more carefully" would not have caught two correct implementations of different specifications |
| Not naming who or what would page | A detection with no alerting path is a dashboard nobody reads |
