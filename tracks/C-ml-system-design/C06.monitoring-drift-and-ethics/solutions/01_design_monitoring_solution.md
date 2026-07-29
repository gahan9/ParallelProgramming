# Solution 1 — Monitoring plan for TriageBot

The organizing idea: **monitoring is an allocation problem.** You have one on-call engineer covering
a whole platform, so the scarce resource is *attention*, not storage or compute. Every page you add
spends it. The plan below therefore allocates roughly one ML page per week and puts everything else
where it can be read on a Tuesday.

---

## 1. The four layers

Three predictions with three completely different label regimes, so three different designs.

| Prediction | Ground truth | Latency | Consequence for monitoring |
|---|---|---|---|
| Category | agent re-routes | hours | **Near-real-time accuracy is available.** Monitor it directly. This is a luxury; use it. |
| Priority | none | never | No accuracy is computable, ever. Proxies only, and say so out loud. |
| Reply | accept/edit/ignore, then CSAT | instant / 2–5 days at 9% response | Rich immediate proxy, weak and biased delayed truth. |

### Layer 1 — system

| Signal | Notes |
|---|---|
| p99 latency, per model | classifier and generator separately; they fail differently |
| 5xx and timeout rate, per model | |
| Queue depth per replica (generator) | the generator is the one that queues; see C05 §3.2 |
| Fallback-tier rate | when the generator is unavailable, the agent gets no suggestion |
| GPU cost/hour and cost per 1 000 tickets | prompt-length drift shows up here first |

### Layer 2 — data

| Signal | How |
|---|---|
| Missing rate per feature, per day | absolute rate **and** delta vs the training-time rate. This is the null-`account_tier` detector |
| Schema conformance | type, range, and enum membership on every feature-store read |
| Unseen categorical values | any value not present in training, as a share of requests |
| Ticket text length distribution | p50/p95; a jump usually means an integration change (an email client quoting whole threads) |
| PSI on the top-15 features by importance | fixed baseline (training) **and** 30-day rolling, both |
| Cramér's V on **every** categorical feature | cheap; dashboard only. See §4 for why "every" matters |
| Per-organization input volume and mix | the new-customer detector |

### Layer 3 — model

| Signal | Notes |
|---|---|
| **Category re-route rate** | the headline model-quality metric. Available in hours. Overall and per-category |
| Category prediction distribution | 34-way; watch for collapse onto a few classes |
| Category confidence distribution and ECE | miscalibration precedes accuracy loss |
| Priority distribution (P1–P4 mix) | proxy only; see §5 |
| Reply: accept / edit / ignore rates | the immediate proxy for generation quality |
| Reply: edit distance when edited | a rising edit distance means "accepted but rewritten", which the accept rate hides |
| Reply: length, refusal rate, empty-output rate | generator-specific failure modes |
| Per-slice re-route rate | by organization size, product area, language, and org tenure |

### Layer 4 — business

First-contact resolution, median time-to-resolution, CSAT, and re-open rate — all overall and by
organization cohort, all on a dashboard with weekly review.

---

## 2. Alert versus dashboard

**Pages — the whole ML budget, targeting ≈1/week:**

| Signal | Threshold | Sustain | Why it earns a page |
|---|---|---|---|
| 5xx / timeout rate, either model | > 1% | 5 min | the service is broken |
| p99 latency, generator | > 2 000 ms | 10 min | agents are waiting |
| Fallback rate | > 5% | 10 min | agents are getting no suggestions |
| **Missing rate for any top-15 feature** | > training rate + 5 pp | 30 min | **this is the incident-(a) detector** |
| **Category re-route rate** | > 1.4× the 28-day median | 2 h | direct, fast evidence the model got worse |
| Category prediction collapse | > 60% of predictions in one class (training max is 14%) | 30 min | catches a broken classifier immediately |
| Reply empty/refusal rate | > 3% | 15 min | generator failure mode |

That is seven pages' worth of surface. Historical firing rate should be checked after a month and
thresholds loosened if it exceeds one per week.

**Tickets (business hours, no page):** unseen categorical value > 1% of requests; ECE > 1.5×
baseline; cost per 1 000 tickets > 1.3× the 7-day median; any *slice's* re-route rate > 2× its own
28-day median; edit distance up > 25%.

**Dashboard only, weekly review:** all PSI and Cramér's V values; per-organization volume and mix;
priority distribution; CSAT; first-contact resolution; long-tail category performance.

**What I deliberately do NOT alert on, and why:**

1. **PSI on any individual feature.** With ~50 features and daily testing this is a guaranteed
   stream of noise (module §5.2). Drift is a *clue*; the re-route rate is the *symptom*. Alert on
   the symptom.
2. **CSAT.** 9% response rate, 2–5 day delay, and heavily confounded by staffing and by which
   customers respond. It cannot support a threshold that means anything at daily granularity.
3. **Priority-model anything.** No ground truth exists, so any threshold would be arbitrary. It gets
   a dashboard and a monthly review (§5).
4. **Aggregate category accuracy.** Superseded by re-route rate, which is faster and more direct.
5. **Individual organizations' metrics.** 1 200 organizations × any threshold = constant noise. Only
   *cohorts* alert; individual organizations appear on a dashboard, sorted by anomaly.

---

## 3. Runbooks

**`missing_rate_high(feature)`**
1. Compare the missing rate in the feature-store read log against the serving log — if it is missing
   at read, it is the feature store; if it is present at read and null at serving, it is our
   transform.
2. Check the feature store's own pipeline status and last successful run for that table.
3. Check deploys in the last 24 h: ours, the feature store's, and the upstream producer's.
   Escalation: feature-store on-call. Mitigation while diagnosing: if the feature has a safe
   training-time default and the model degrades gracefully without it, keep serving and downgrade to
   a ticket; otherwise route affected tickets to the rules-based fallback router.

**`reroute_rate_high`**
1. Is it concentrated in one category, one organization, or one language? A concentrated spike is an
   input change; a uniform spike is a model or feature change.
2. Any deploy — model, features, or prompt — in the last 24 h? If yes, roll back first and diagnose
   second.
3. Compare today's category prediction distribution against the 28-day band; check whether a new
   organization onboarded (see §4).
   Escalation: model owner. Mitigation: roll back to the previous model version; re-routing is
   annoying but not an outage, so do not panic-mitigate.

**`prediction_collapse`**
1. Check the model version being served against the intended version.
2. Check the missing-rate panel — collapse is usually a features-all-null symptom, so this and the
   missing-rate alert often fire together.
3. Sample ten recent requests and inspect the *post-transform* feature vectors, not the raw
   requests.
   Mitigation: roll back. This one is almost always a deploy.

**`fallback_rate_high`** → check generator replica health, queue depth, and GPU node status; this is
a C05 capacity problem. **`empty_or_refusal_rate_high`** → check for a prompt-template deploy, an
input-length spike (a very long ticket overflowing the context window), and safety-filter changes.

---

## 4. The two known incidents

### Incident (a) — `account_tier` null for 40% of tickets, nine days

- **Fires:** `missing_rate_high(account_tier)` — 0% → 40% is enormously beyond "training rate + 5 pp".
- **Latency:** ~30 minutes (the sustain window). Nine days → thirty minutes.
- **Pages:** yes.
- Secondary confirmation: the category re-route rate would likely also cross its threshold within a
  couple of hours, and prediction-distribution monitoring would show the shift. Three independent
  signals catch it, which is what you want for a failure class that previously took nine days and a
  customer escalation.

### Incident (b) — a large new customer shifts the global category distribution

- **Fires:** nothing pages. PSI on category-related inputs moves and appears on the dashboard.
- **Why it must not page:** it is a legitimate business event. A monitoring system that pages when a
  large customer onboards will be muted before it catches anything real.

**How the design distinguishes "world changed, ignore" from "world changed, broken":**

1. **Alert on the symptom, not the cause.** A new customer changes the *input mix*; it does not make
   the model *wrong* about any individual ticket. The re-route rate — the symptom — stays flat, so
   nothing pages. That single design principle does most of the work here.
2. **Decompose drift by cohort.** Compute PSI both globally and on a fixed cohort of
   organizations that existed in both windows. If global PSI moves and same-cohort PSI does not, the
   change is *compositional* — new customers, not changed behaviour. This is a two-line change to
   the drift job and it converts the most common false alarm in B2B ML into a self-explaining
   dashboard annotation.
3. **Annotate, do not alert.** An onboarding event above a size threshold posts an annotation onto
   the drift dashboard automatically. When a human reviews the weekly PSI panel, the explanation is
   already there.

Point 2 is the answer the drill is looking for.

---

## 5. The priority model, with no ground truth

**What I would monitor:**

| Proxy | What it actually proves | What it does not |
|---|---|---|
| P1–P4 distribution vs the 28-day band | the model has not collapsed or silently re-scaled | nothing about correctness |
| Agent priority-override rate | agents disagree at a rate; a *change* in the rate is informative | agents are not ground truth, and they see the model's prediction — a §3.6 feedback loop |
| SLA breach rate by predicted priority | if P1s breach much more than P2s, the priority ordering carries real signal | confounded by staffing, exactly as the brief says |
| Correlation with post-hoc resolution effort (time, escalations, participants) | a **retrospective** proxy for true urgency | it is a proxy, and effort is partly caused by the priority we assigned |
| **A monthly manually-labelled random sample** (200 tickets, blind to the model's prediction) | genuine accuracy on a small sample, with real confidence intervals | it is 200 tickets; it detects large moves, not small ones |

The last row is the only honest one and it costs about half a day of someone's month. It is worth
it, and the "blind to the model's prediction" qualifier is the load-bearing part — otherwise you are
measuring agreement, not accuracy.

**What I would tell the stakeholder who asks "is the priority model accurate?":**

> "We don't know, and we can't know from production data, because nobody ever records what the
> priority *should* have been. What we can tell you is that its output distribution is stable, that
> agents override it 6% of the time and that rate hasn't changed, and that on a blind sample of 200
> tickets last month it agreed with a human expert 78% of the time — which is our only real accuracy
> number, and it has a margin of about ±6 points. If you need a tighter number we can label a bigger
> sample; here's what that costs."

That answer is better than a confident number, and a stakeholder who has been given a fake number
once will not trust the real ones later.

---

## 6. The reply model

**Immediately available:** accept / edit / ignore rates, edit distance when edited, output length
distribution, refusal and empty-output rates, generation latency, and the rate at which the agent's
final sent message resembles the suggestion at all. These are strong, fast, and free. They are also
proxies: an agent may accept a mediocre suggestion because editing is slow.

**The 9% CSAT response rate is the interesting problem**, and the answer is *non-response bias*, not
sample size. Responders differ systematically from non-responders — typically skewed toward the very
satisfied and the very angry. Consequences and what to do:

1. **Never compare absolute CSAT across slices.** The response rate differs by slice, so a CSAT
   difference between two organizations may be entirely a response-rate difference. Report the
   response rate next to every CSAT number, always.
2. **Track the trend within a slice**, where the response bias is roughly constant, rather than
   levels across slices.
3. **Weight by inverse response propensity** if you can model it from observable ticket features.
   Imperfect, better than nothing, and it makes the assumption explicit.
4. **For any slice where the responses number in the tens, do not report a number.** Report "n too
   small". A CSAT of 3.6 from 14 responses is noise with a decimal point.
5. **Prefer the immediate proxies for anything operational.** Use CSAT to *validate the proxies* on
   a quarterly cadence — that is the job it is actually good for.

---

## 7. Reference windows

| Signal | Reference | Why |
|---|---|---|
| Feature PSI | **both** — fixed (training set) and 30-day rolling | fixed answers "how far from training are we?", rolling answers "what changed this week?". A slow drift is invisible to rolling; an accepted long-term shift makes fixed alert forever |
| Category prediction distribution | 28-day rolling | the class mix genuinely evolves with the product |
| Re-route rate | 28-day rolling median | staffing and product changes shift the level; the recent past is the right comparison |
| Missing rates | **fixed** (training-time rate) | the training-time rate is the contract. A rolling window would silently normalize a nine-day outage into the new baseline — which is precisely how incident (a) survived nine days |
| Per-slice metrics | slice's own 28-day rolling | slices have persistently different levels; comparing a slice to the global average produces permanent false alarms |

The `missing rates → fixed` row is the one that matters. A rolling reference makes a slowly-worsening
problem invisible by construction.

---

## 8. Cost, and the trap

180 000 tickets/day × 3 predictions × ~3 KB (text is bulky) ≈ **1.6 GB/day, ~590 GB/year** at full
fidelity. Not enormous, but the ticket text and the generated reply may carry customer PII, so
retention is a policy question, not a storage question.

**The trap: uniform sampling makes small organizations invisible.** At a 5% sample, an organization
sending 20 tickets/day contributes one row a day, so its per-slice metrics are statistically
meaningless — and small organizations are exactly the ones most likely to be silently mis-served,
because nobody escalates for them.

**Stratified sampling:**

| Stratum | Rate |
|---|---|
| Errors, fallback-served, refusals | 100% |
| Agent-overridden (re-route or priority change) | 100% — rare and maximally informative |
| Organizations under 200 tickets/day | 100% — cheap in absolute volume, and it keeps them measurable |
| Organizations over 200 tickets/day | 5%, with the sampling rate stored per row so metrics can be reweighted |
| New organizations, first 30 days | 100% — the highest-risk period |

Retention: full fidelity 30 days; daily per-slice aggregates 2 years; a 0.5% random archival sample
for 1 year to support long-horizon analyses. Text fields redacted after 30 days unless the
organization's contract says otherwise — and that is a question for legal and the contract owner,
not for the ML team.

---

## What a reviewer should push back on

- The seven page-triggers are an estimate, not a measurement. Run them in shadow for a month, count
  the firings, and loosen whatever exceeds the budget. A plan that has never been calibrated against
  real firing rates is a hypothesis.
- The re-route rate is treated as ground truth for category, but agents re-route for reasons other
  than model error (team capacity, escalation policy). Validate it against a blind labelled sample
  before trusting the 1.4× threshold.
- Nothing here monitors the **feedback loop** on the reply model: agents accept suggestions,
  accepted suggestions become training data, and the model learns to produce what agents accept
  rather than what helps customers. Add catalogue-style diversity monitoring on generated replies,
  and keep a small holdout of tickets where no suggestion is shown — see module §3.5 and Lab 2.
