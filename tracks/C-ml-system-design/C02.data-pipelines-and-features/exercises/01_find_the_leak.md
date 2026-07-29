# Exercise 01 — Find the leak

**Goal:** audit a real-looking feature list and separate the features you can train on from the ones
that will make your offline metric a fiction.

**Time:** 45–60 minutes. **Deliverable:** the completed table plus the four written answers.

---

## The system

You are joining a team building a fraud model for an auto-insurance claims department.

- **Decision:** at claim submission (`t0`), route the claim to fast-track auto-approval, standard
  processing, or the Special Investigations Unit (SIU).
- **Label:** `is_fraud`, confirmed by the SIU or by claim denial, typically **90 days** after `t0`.
- **Data:** a warehouse table `claim_features` assembled by a previous contractor. Offline ROC-AUC on
  their temporal split was **0.97**, which nobody on the team believes.

## The feature list

Each row is what the data dictionary says. The "recorded timestamp" column is what the pipeline stamps
on the value.

| # | Feature | What the data dictionary says | Recorded timestamp |
|---|---|---|---|
| 1 | `claim_amount` | Amount requested on the submitted claim form | `t0` |
| 2 | `policy_age_days` | Days between policy inception and claim submission | `t0` |
| 3 | `prior_claims_closed_24m` | Count of the claimant's prior claims **closed before** `t0` | `t0` |
| 4 | `prior_claims_filed_24m` | Count of the claimant's claims filed in the 24 months **around** this claim's incident date | `t0` |
| 5 | `document_count_at_intake` | Documents attached when the claim was submitted | `t0` |
| 6 | `document_count_total` | Total documents ever attached to the claim | `t0` |
| 7 | `adjuster_hours_logged` | Hours the adjuster logged on the case | `t0 + 95d` |
| 8 | `payout_amount` | Amount ultimately paid (0 if denied) | `t0 + 92d` |
| 9 | `claim_status` | Current status: `open`, `closed_paid`, `closed_denied` | `t0` |
| 10 | `siu_referral_flag` | 1 if the claim was referred to the SIU | `t0 + 12d` (median) |
| 11 | `region_fraud_rate` | Fraud rate for the claimant's postcode region, computed over the **whole `claim_features` table** | `t0` |
| 12 | `region_fraud_rate_pit` | Fraud rate for the region, computed from claims **closed before** `t0` | `t0` |
| 13 | `days_to_first_claimant_contact` | Days between submission and the claimant's first inbound contact | `t0 + 6d` (median) |
| 14 | `claimant_credit_score` | Bureau score. The warehouse keeps **only the latest pull**; there is no history table | `t0` (stamped, not measured) |
| 15 | `vehicle_value_estimate` | Third-party valuation, refreshed monthly; **the column is overwritten in place** | `t0` (stamped, not measured) |
| 16 | `police_report_filed` | 1 if a police report exists for the incident | `t0` |
| 17 | `weather_severity_at_incident` | Historical weather severity at the incident location and time | `t0` |
| 18 | `claimant_has_legal_representation` | 1 if a lawyer has contacted the insurer about this claim | `t0 + 21d` (median) |

---

## Part 1 — Classify every feature

Use exactly one of these verdicts per feature:

- **SAFE** — knowable at `t0`; train on it.
- **LEAK** — carries information that did not exist at `t0`; drop or reconstruct it.
- **QUESTION** — cannot be decided from the data dictionary; state the exact question and who answers
  it.

| # | Feature | Verdict | Mechanism / question (one or two sentences) |
|---|---|---|---|
| 1 | `claim_amount` | | |
| 2 | `policy_age_days` | | |
| 3 | `prior_claims_closed_24m` | | |
| 4 | `prior_claims_filed_24m` | | |
| 5 | `document_count_at_intake` | | |
| 6 | `document_count_total` | | |
| 7 | `adjuster_hours_logged` | | |
| 8 | `payout_amount` | | |
| 9 | `claim_status` | | |
| 10 | `siu_referral_flag` | | |
| 11 | `region_fraud_rate` | | |
| 12 | `region_fraud_rate_pit` | | |
| 13 | `days_to_first_claimant_contact` | | |
| 14 | `claimant_credit_score` | | |
| 15 | `vehicle_value_estimate` | | |
| 16 | `police_report_filed` | | |
| 17 | `weather_severity_at_incident` | | |
| 18 | `claimant_has_legal_representation` | | |

---

## Part 2 — How each leak inflates the metric

Pick the **three worst** leaks and, for each, describe the mechanism in terms a reviewer would accept:
what does the model learn, and what happens to that learned rule in production?

```
leak 1:
leak 2:
leak 3:
```

Then answer: **why is a 0.97 AUC itself a signal?** What would you expect an honest fraud model on
this problem to look like, and why does an unusually high number warrant suspicion rather than
celebration?

```
(your answer)
```

---

## Part 3 — Which checks would catch which leak

Run the module's screen for the shape of the output:

```bash
python ../tools/leakage_detector.py --demo
```

Now map this exercise's features onto the three checks. For each of your LEAK verdicts, mark which
check would have caught it:

| Feature | Timestamp check | Separation check | Naming check | Only a human/code review |
|---|:---:|:---:|:---:|:---:|
| | | | | |

Then answer: **which of the 18 features would every automated check miss, and why?** This is the most
important question in the exercise.

```
(your answer)
```

---

## Part 4 — Pipeline changes that prevent the whole class

Do not list "be careful." For each of these leak *classes*, name the concrete pipeline or schema change
that makes the class structurally impossible or immediately visible:

| Leak class | Example from the list above | Pipeline change that prevents it |
|---|---|---|
| Post-outcome artifact joined to the label row | | |
| Column back-filled in place (no history) | | |
| Global aggregate computed before splitting | | |
| Slowly-changing dimension with no history table | | |
| Feature whose window straddles the decision moment | | |
| Feature with no recorded observability timestamp | | |

---

## Part 5 — What you would say on day three

You have found the problem. The contractor is gone, the model was about to ship, and your manager asks
for a status. Write three to five sentences that (a) state what you found without theatrics, (b) give a
realistic expectation for the honest metric, and (c) name what you need to proceed.

```
(your answer)
```

---

## Self-check

- [ ] You used **QUESTION** at least three times — a real audit ends with questions.
- [ ] Every LEAK verdict has a *mechanism*, not just a label.
- [ ] You noticed at least one feature whose recorded timestamp is **stamped rather than measured**,
      and explained why that is worse than an obviously wrong timestamp.
- [ ] You identified the leak that **no automated check** would catch.
- [ ] Part 4 contains schema/pipeline changes, not advice.
- [ ] You gave a realistic expectation for the honest AUC and said why 0.97 was itself the tell.

Reference answer with the reasoning:
[`../solutions/01_find_the_leak_solution.md`](../solutions/01_find_the_leak_solution.md).
