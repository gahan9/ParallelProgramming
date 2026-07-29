# Exercise 2 — Diagnose the degradation

**Time:** 30–45 minutes · **Deliverable:** ~1 page: the diagnosis with its evidence chain, the
alternatives you ruled out and why, the fix, and its cost.

---

## The page

You are on call. At 08:15 you receive this from the business-intelligence team, not from a monitor.

> **Subject: loan-approval model — approval quality down since roughly week 30**
>
> Our 60-day-lagged repayment cohort analysis finished overnight. For loans approved in weeks 30–33,
> the 60-day delinquency rate is **6.8%**, against a trailing average of **4.1%**. That is a large
> move and it is costing us real money. The model team says nothing has changed. Can someone look?

## The evidence

You pull the monitoring data. Reference window is weeks 20–29; current window is weeks 30–33.

### Layer 1 — system

| Signal | Weeks 20–29 | Weeks 30–33 |
|---|---|---|
| p99 latency | 41 ms | 42 ms |
| Error rate | 0.02% | 0.02% |
| Fallback-tier rate | 0.1% | 0.1% |
| Requests/day | 24 100 | 25 800 |

### Layer 2 — data

| Feature | Importance rank | PSI (ref vs current) | Missing rate ref → cur |
|---|---:|---:|---|
| `applicant_income` | 1 | 0.031 | 0.2% → 0.2% |
| `credit_score` | 2 | 0.024 | 0.0% → 0.0% |
| `debt_to_income` | 3 | 0.048 | 0.4% → 0.4% |
| `employment_months` | 4 | 0.019 | 1.1% → 1.2% |
| `loan_amount` | 5 | 0.052 | 0.0% → 0.0% |
| `region_code` (categorical) | 11 | Cramér's V = 0.014 | 0.0% → 0.0% |
| `channel` (categorical) | 14 | Cramér's V = **0.211** | 0.0% → 0.0% |
| *(41 other features)* | 6–48 | all < 0.06 | unchanged |

### Layer 3 — model

| Signal | Weeks 20–29 | Weeks 30–33 |
|---|---|---|
| Prediction distribution mean | 0.163 | 0.169 |
| Prediction distribution PSI | — | 0.021 |
| Approval rate at the fixed 0.5 threshold | 71.4% | 70.9% |
| Expected Calibration Error (on the 60-day cohort that has matured) | 0.019 | **0.071** |
| Model version | `loan-v11` | `loan-v11` |

### Layer 4 — business

| Signal | Weeks 20–29 | Weeks 30–33 |
|---|---|---|
| 60-day delinquency (approved loans) | 4.1% | **6.8%** |
| Application volume | 24 100/day | 25 800/day |
| Average loan amount | \$14 200 | \$14 400 |

### Context from the room

- The model team is right: no model deploy since week 17.
- The data platform team confirms no schema changes and no pipeline deploys in weeks 28–34.
- Marketing mentions, when asked, that they launched a **partnership with an online marketplace in
  week 29**, which now sends applicants through a new `channel` value.
- The `channel` feature has 6 categories in the reference window and 7 in the current one. The new
  value, `partner_mkt`, accounts for **9% of current applications**.
- Nobody monitored `channel`, because it is ranked 14th by importance and the drift dashboard only
  alerts on the top 10.

---

## Your task

### 1. Name the drift type

Covariate shift, prior/label shift, concept drift, or not-drift-at-all. **Justify it from the
evidence in the tables**, not from the story. Walk the decision tree in §3.3 of the module README
explicitly, saying which branch each piece of evidence sends you down.

Be careful. There is a defensible argument for more than one answer here, and the *best* answer
identifies which parts of the degradation are attributable to which mechanism. A single-word answer
scores poorly even if the word is right.

### 2. Rule out the alternatives

For each type you did **not** choose, state the specific evidence that rules it out. "It's not a
serving bug" is not enough; say which number would have looked different if it were.

### 3. Explain the calibration jump

ECE went from 0.019 to 0.071 while the prediction distribution barely moved (PSI 0.021). Explain how
both can be true at once. This is the crux of the drill.

### 4. Why did nothing page?

The signal that mattered was a categorical feature ranked 14th, excluded from alerting by a
deliberate design choice made to control alert fatigue. Was that choice wrong? Answer properly —
"monitor everything" is not an answer, because §5.2 of the module explains why it produces a muted
dashboard. What specific rule would have caught this *without* adding daily noise?

### 5. The fix

Give the **cheapest intervention that addresses the mechanism you diagnosed**, plus what you would
do if it does not work. Include:

- What you do in the next 24 hours (there is money leaking right now).
- What you do in the next 2 weeks.
- What changes permanently in the monitoring.

Note the constraint that makes this hard: labels take **60 days**. Whatever you deploy, you will not
know whether it worked for two months. How does that change the intervention you choose?

### 6. The uncomfortable question

Your fix probably involves the model treating `partner_mkt` applicants differently. Before you ship
it, what should you check about who those applicants are? Say what you would look at and what you
would do if the answer is uncomfortable.

---

## Self-check before reading the solution

- [ ] Did I walk the decision tree branch by branch, citing numbers?
- [ ] Did I rule out each alternative with a specific number, not a narrative?
- [ ] Can I explain how ECE moves 3.7× while prediction PSI stays at 0.021?
- [ ] Did I propose a monitoring rule that catches this class of event without adding daily noise?
- [ ] Does my fix have a 24-hour action, given money is leaking now?
- [ ] Did I account for the 60-day feedback delay in choosing the intervention?
- [ ] Did I address the fairness question in part 6 rather than skipping it?

Reference answer: [`../solutions/02_diagnose_the_drift_solution.md`](../solutions/02_diagnose_the_drift_solution.md).
