# Solution 03 — Cost-optimal threshold and the base rate

Reference answer for
[`../exercises/03_threshold_from_costs.md`](../exercises/03_threshold_from_costs.md).

All numbers here are either arithmetic from the stated costs or reproducible output from
[`../tools/metric_explorer.py`](../tools/metric_explorer.py) with `--seed 7`.

---

## Part 1 — The analytic threshold

### 1a. Expected cost of each action

For a single case with calibrated probability `p` of being positive:

```
flag it        : with prob (1 - p) the case is negative -> a false positive
                 expected cost = (1 - p) * C_fp
don't flag it  : with prob p the case is positive       -> a false negative
                 expected cost = p * C_fn
```

True positives and true negatives contribute no *error* cost, which is why they do not appear. If your
cost model gives them non-zero values (a TP has an investigation cost; a TN is free), the derivation
still works — you just carry four terms and the threshold becomes
`p* = (C_fp − C_tn) / (C_fp − C_tn + C_fn − C_tp)`. The two-term version is the common case.

### 1b. Derivation

```
flag when      (1 - p) * C_fp  <  p * C_fn
               C_fp - p*C_fp   <  p * C_fn
               C_fp            <  p * C_fn + p * C_fp
               C_fp            <  p * (C_fp + C_fn)

               p  >  C_fp / (C_fp + C_fn)   =   p*
```

### 1c. The number

```
p* = 8 / (8 + 160) = 8 / 168 = 0.047619  ->  0.0476
```

Flag anything with a predicted fraud probability above ~4.8%.

### 1d. Does `p*` change with the base rate? **No.**

The justification must come from the derivation, and it does: **the prevalence `π` appears nowhere in
it.** The derivation is per-case and conditions on `p`, the probability *for that case*. Once you know
`p`, the base rate has already done its work — it is baked into `p` by the model. So for a calibrated
model the optimal decision boundary in probability space is a function of the cost ratio alone.

This is the correct-but-incomplete answer, and Part 3 is where the "incomplete" gets paid off.

### 1e. `C_fn` rises to \$400

```
p* = 8 / (8 + 400) = 8 / 408 = 0.019608  ->  0.0196
```

The threshold moved **down**, from 0.0476 to 0.0196. That matches intuition: a more expensive miss
means you should be willing to flag on weaker evidence. Note the shape of the dependence — `p*` is
governed by the *ratio* `C_fn / C_fp`, so 8:400 behaves identically to 1:50. Only the ratio matters,
which is why the tool can use `--fp-cost 1 --fn-cost 20` to stand in for \$8 and \$160.

---

## Part 2 — Hand-computed costs at 2% base rate

`cost = FP × 8 + FN × 160`

| Threshold | TP | FP | FN | TN | Precision | Recall | FP cost | FN cost | **Total** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9037 (max F1) | 95 | 117 | 105 | 9683 | 0.4481 | 0.4750 | 936 | 16,800 | **17,736** |
| 0.8223 (min cost) | 145 | 625 | 55 | 9175 | 0.1883 | 0.7250 | 5,000 | 8,800 | **13,800** |
| 0.6420 | 178 | 2764 | 22 | 7036 | 0.0605 | 0.8900 | 22,112 | 3,520 | **25,632** |
| flag nothing | 0 | 0 | 200 | 9800 | — | 0.0000 | 0 | 32,000 | **32,000** |
| flag everything | 200 | 9800 | 0 | 0 | 0.0204 | 1.0000 | 78,400 | 0 | **78,400** |

### 2a. Cheapest

**0.8223**, at \$13,800. The cost curve is U-shaped: too strict and you pay for misses (0.9037), too
loose and you pay for false alarms (0.6420). Both trivial policies are far worse than either.

### 2b. Penalty for optimizing F1 instead of cost

```
17,736 / 13,800 = 1.2852   ->  +28.5% total cost
```

Equivalently, the cost-derived operating point is **22.2% cheaper** (`1 − 13,800/17,736`). Both framings
appear in the README; be careful which denominator you are quoting — this is a real source of confusion
in write-ups.

The interesting part is *why*. F1 does not know your costs; it weights precision and recall equally, so
it lands at a much stricter threshold (precision 0.45 vs 0.19). At a 20:1 cost ratio that strictness is
expensive: moving from 0.9037 to 0.8223 trades **+508 false positives (−\$4,064)** for **−50 false
negatives (+\$8,000)** — a clearly good trade that F1 cannot see. **The model never changed.** Nothing
was retrained. One config value moved.

### 2c. Tool check

```bash
python ../tools/metric_explorer.py confusion --tp 145 --fp 625 --fn 55 --tn 9175 \
    --fp-cost 8 --fn-cost 160
```

```text
  total cost   = 13800   (625 FP x 8 + 55 FN x 160)
  cost / case  =  1.3800
```

### 2d. Predicted relationship to the sweep

**Prediction:** every value in the sweep's `cost` column should be exactly **1/8** of the costs above,
because \$8:\$160 and 1:20 are the same ratio scaled by 8, and total cost is linear in both cost
parameters. The *ranking* of thresholds — and therefore the chosen threshold — must be identical.

**Observed:** 2217 × 8 = 17,736 ✓ · 1725 × 8 = 13,800 ✓ · 3204 × 8 = 25,632 ✓ · 4000 × 8 = 32,000 ✓ ·
9800 × 8 = 78,400 ✓.

This is worth internalizing because of what it implies operationally: **you never need absolute cost
figures to pick a threshold, only the ratio.** That is a far easier thing to extract from a business
owner, and it is the practical reason the elicitation question in exercise 2 asks "how many false
alarms is one miss worth?" rather than "what does a miss cost?"

---

## Part 3 — Base rate drops 10×

### 3a. Why ROC-AUC held and PR-AUC collapsed

- **ROC-AUC (0.9180 → 0.9181):** both of its axes are *within-class* rates — TPR has the positive count
  in its denominator, FPR has the negative count. Adding 90,000 negatives grows FP and the FPR
  denominator together, so the curve is essentially unchanged. It is measuring ranking quality, which
  genuinely did not change.
- **PR-AUC (0.4111 → 0.0969):** precision is `TP / (TP + FP)`, whose denominator **mixes classes**.
  The extra negatives pour into FP with no matching increase in TP, so precision falls at every recall
  level and the area under the curve falls with it.

The one-sentence version for an interview: *ROC-AUC is prevalence-invariant because both its axes are
class-conditional; PR-AUC is not, because precision's denominator is not.*

### 3b. Reconciling "p\* doesn't move" with a score threshold that rose 0.8223 → 0.9240

Both are true, and they are statements about **different spaces**.

Write the decision as a likelihood ratio. Flag `x` when the expected cost of flagging is lower:

```
   P(positive | x) * C_fn  >  P(negative | x) * C_fp

using Bayes with prevalence pi and class-conditional densities f1, f0:

   pi * f1(x) * C_fn  >  (1 - pi) * f0(x) * C_fp

   f1(x) / f0(x)  >  (C_fp / C_fn) * ((1 - pi) / pi)
                     \___________/   \_____________/
                       cost term       prevalence term
```

The **probability-space** threshold `p*` contains no `π` because `P(positive | x)` has already absorbed
it. The **evidence-space** threshold does contain `π`, through `(1 − π)/π`. Dropping prevalence from 2%
to 0.2% multiplies that term by roughly 10:

```
pi = 0.020  ->  (1 - pi)/pi =  49.0
pi = 0.002  ->  (1 - pi)/pi = 499.0     (~10.2x higher evidence bar)
```

So you need materially stronger evidence to justify flagging, and since the raw score is a monotone
function of the evidence, the **score** threshold rises — exactly what the tool reports (0.8223 →
0.9240).

The practical consequence, and this is the sentence that matters in production: **if you deployed a
threshold on a raw score and the base rate then shifted, your threshold is now wrong even though `p*`
did not change.** The two ways out are (a) calibrate the score to a probability and threshold at `p*`,
re-fitting calibration when the prior shifts, or (b) re-run the empirical sweep on fresh data whenever
prevalence moves materially. Option (a) is the one that scales, and it is why README §3 insists
calibration is load-bearing rather than cosmetic.

### 3c. Savings against doing nothing

Working in 1:20 units (multiply by 8 for dollars):

```
scenario A: (4000 - 1725) / 4000 = 2275 / 4000 = 56.9%   ($32,000 -> $13,800, saving $18,200)
scenario B: (4000 - 3170) / 4000 =  830 / 4000 = 20.8%   ($32,000 -> $25,360, saving  $6,640)
```

**What I would tell the product owner:**

> "The model didn't get worse — its ranking quality is identical, ROC-AUC 0.918 in both cases. What
> changed is that fraud got 10× rarer, so we now review roughly ten times as many legitimate
> transactions per fraud we catch. At 2% we removed about 57% of the fraud cost; at 0.2% we remove
> about 21%, and we only catch 36% of the fraud instead of 73%. That 21% is still worth roughly \$6,600
> per 100,000 transactions on the model side — but it is now close enough to 'do nothing' that the
> **operational** cost has to enter the comparison: 590 false alerts per 200 frauds is analyst time we
> are spending. Before we change anything I want two checks: confirm the prevalence drop is real and
> not a labelling or pipeline change, and re-derive the threshold on fresh data. If the drop is real,
> the questions are whether the review capacity is better spent elsewhere and whether a cheaper
> rules-based screen would capture most of the remaining value."

Note the move: the *first* hypothesis for a sudden 10× metric shift is a **data or pipeline change**,
not a real-world change. That instinct is C02's subject.

### 3d. Standard error of the scenario-B recall

```
p = 0.3550, n = 200 positives

SE     = sqrt(0.3550 * 0.6450 / 200) = sqrt(0.00114) = 0.0338
95% CI ~ 0.3550 +/- 1.96 * 0.0338    = 0.3550 +/- 0.066  ->  [0.289, 0.421]
```

**Verdict: no, you cannot distinguish 0.36 from 0.40 on this evaluation set.** A 4-point difference is
well inside a ±6.6-point interval. For two *independent* estimates the standard error of the difference
is about `sqrt(2) × 0.0338 ≈ 0.048`, so you would need a gap of roughly 9–10 points before calling it.

Two refinements worth saying out loud:

- **Paired comparison is much more sensitive.** If you evaluate two models on the *same* 200 positives,
  the estimates are correlated and you should test the disagreements directly (McNemar-style) rather
  than differencing two independent intervals. Always compare models on an identical evaluation set for
  this reason.
- **The binding denominator is the positive count, not the dataset size.** Scenario B has 100,000 rows
  and still only 200 positives, so recall precision is unchanged from scenario A. Getting a tighter
  recall estimate means accumulating more *positives* — a longer evaluation window — not more rows.

---

## Part 4 — The written answer

> Nothing changes in probability space: the cost-optimal threshold is `p* = C_fp/(C_fp + C_fn)`, which
> contains no base-rate term, so if the model is calibrated I keep flagging above 4.8%. What does
> change is everything around it. In evidence space the bar rises: the likelihood-ratio threshold
> scales with `(1 − π)/π`, so a 10× rarer positive needs roughly 10× stronger evidence, and our
> observed score threshold moves from 0.82 to 0.92. Precision falls (0.19 → 0.11) because the negative
> pool grew while true positives did not, recall at the optimum falls from 0.73 to 0.36, and
> alerts-per-catch roughly doubles — so analysts do more work per fraud found. The model's ranking
> quality is unchanged: ROC-AUC stays at 0.918 while PR-AUC drops 4×, which is the signature of a
> prevalence shift rather than a model regression. Is it still worth running? Marginally: we go from
> removing ~57% of fraud cost to ~21%, so the review cost now has to be in the comparison explicitly.
>
> My first check is not statistical, it is operational: confirm the prevalence drop is real and not a
> labelling change, a pipeline gap, or a definition change upstream — a 10× metric move is far more
> often a data bug than a real-world shift. Then re-derive the threshold on fresh post-change data and
> re-fit calibration, because the old calibration was fitted at the old prior.
>
> **It depends on whether the score is calibrated and whether review capacity binds.** If calibrated
> and capacity is free, `p*` needs no change at all. If we are thresholding a raw score, or if the
> analyst queue is the constraint, the operating point must be re-derived — and if capacity binds, the
> objective is not expected cost at all, it is precision at the number of alerts we can actually
> review.

## Part 5 — Capacity check

### 5a. Alerts per day

```
positives/day        = 1,000,000 x 0.02  = 20,000
eval set has 200 positives, so scale its counts by 100x:
  true alerts/day    = 145 x 100         = 14,500
  false alerts/day   = 625 x 100         = 62,500
  total alerts/day                        = 77,000
```

### 5b. Feasible?

**No — 77,000 alerts against 3,000 of capacity, roughly 26× over.** The cost-optimal threshold is
operationally impossible, and any design document that stops at "0.8223" is wrong in a way that will
be discovered by the operations team rather than by engineering.

The four options from README §5, with the arithmetic:

1. **Raise the threshold to fit capacity.** 3,000 alerts/day corresponds to ~30 alerts in the
   200-positive evaluation set, which is a threshold near **0.9566** — precision 0.781, recall
   **0.125**. That is the honest cost of the constraint: fitting 12 analysts drops recall from 0.725 to
   0.125, roughly a **6× loss of catches**, and total cost rises from 1725 to 3507 (still better than
   the 4000 of doing nothing, but only just). Anyone who claims capacity limits are a detail should be
   shown these two numbers.
2. **Tier the queue.** Auto-block the very-high-precision head (precision is 0.78 at 0.9566 and higher
   above it), auto-allow the tail, route only the ambiguous middle to humans. This is the option I
   would pick: it buys most of the recall back without new headcount. The costs are real and must be
   stated — an auto-block error is a customer incident with no human check, so it needs a precision
   floor validated by sampled audit, an appeals path, and a rollback switch.
3. **Raise ranking quality.** The only option that improves both sides of the trade, and the slowest.
   It is a C02/C03 project (better features, better data), not a threshold change.
4. **Add capacity.** Now a clean arithmetic question rather than an argument: at \$160 of loss per
   miss, recovering the ~600 catches/day that capacity is costing us is worth ~\$96,000/day, which
   comfortably funds more analysts *if* the ranking supports that depth at usable precision — and at
   depth 77,000 the precision is 0.19, meaning 4 of every 5 reviews are wasted. That is the number
   that decides it, and it argues for option 2 first.

**Recommendation: tier the queue (2), then revisit capacity (4) with the measured auto-block
precision.** The general principle — and the reason this exercise ends here — is that *the
cost-optimal threshold is the answer to a question about a system with unlimited capacity to act.*
Almost no real system is that system.

---

## Common mistakes on this exercise

| Mistake | Why it costs you |
|---|---|
| "The threshold goes up when the base rate goes down" with no qualification | Half right. `p*` does not move; the *score*/evidence threshold does. Failing to separate the two spaces is the single most common error here |
| "The threshold goes down because positives are rarer, so we should be more aggressive" | Backwards. Rarer positives mean a *weaker* prior, so you need *stronger* evidence |
| Using F1 to pick the operating point | 28.5% more cost for an unchanged model |
| Quoting recall to three decimals on 200 positives | ±6.6 points. State the interval |
| Assuming more rows means a tighter recall estimate | Recall's denominator is the positive count; 100,000 rows with 200 positives is no better than 10,000 with 200 |
| Reporting the cost-optimal threshold without a capacity check | 26× over capacity. Correct arithmetic, unusable answer |
| Treating a 10× prevalence drop as real without checking the pipeline | It is usually a data bug. This is the C02 reflex |
