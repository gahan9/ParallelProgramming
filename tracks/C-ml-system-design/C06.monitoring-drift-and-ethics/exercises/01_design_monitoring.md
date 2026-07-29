# Exercise 1 — Design the monitoring plan

**Time:** 60–90 minutes · **Deliverable:** ~2 pages: a four-layer table, an alert/dashboard split
with numeric thresholds, and a runbook for every alert.

---

## The system

**"TriageBot"** — an ML system inside a B2B customer-support platform. For every incoming support
ticket it predicts:

1. **Category** (one of 34 product areas) — routes the ticket to a team.
2. **Priority** (P1–P4) — sets the SLA clock.
3. **Suggested reply** (generated text) — shown to the agent, who may accept, edit, or ignore it.

### What you know

| Fact | Value |
|---|---|
| Volume | 180 000 tickets/day across 1 200 customer organizations |
| Serving | Two models: a fine-tuned classifier for category+priority (12 ms p99), a 7B generative model for the reply (900 ms p99 TTFT) |
| Inputs | Ticket text, customer metadata, product-usage features from a feature store, and the customer's ticket history |
| Retraining | Monthly, on the last 6 months of tickets |
| Ground truth — category | The agent can re-route a ticket. A re-route is a strong signal the category was wrong. **Available within hours.** |
| Ground truth — priority | Effectively never. Nobody re-labels priority; you only see whether the SLA was met, which is confounded by staffing. |
| Ground truth — reply | Agent accepted / edited / ignored, available immediately. Whether the customer was actually helped: a CSAT survey with a **9% response rate**, arriving 2–5 days later. |
| Business metrics | First-contact resolution rate (currently 61%), median time-to-resolution (4.2 h), CSAT (4.1/5) |
| Team | Four engineers. One is on call for the whole platform, not just ML. |
| Known incidents | (a) Three months ago a feature-store schema change silently set `account_tier` to null for 40% of tickets for nine days; nobody noticed until a customer escalated. (b) Last month a new large customer onboarded, and their ticket volume alone shifted the global category distribution. |

---

## Your task

### 1. The four layers

For each of system, data, model and business, list the specific signals you would collect for
**this** system. Be concrete: not "input drift" but which features, computed how, against which
reference window.

Note the asymmetry the brief hands you: the three predictions have **completely different label
availability** (hours, never, and a 9%-response survey). Your plan must treat them differently.

### 2. Alert versus dashboard

Split every signal into **page**, **ticket**, or **dashboard only**. Give a numeric threshold and a
sustain window for everything that pages.

Budget: **one engineer, on call for the whole platform.** If your plan produces more than about one
ML page per week, it will be muted, and then it produces zero. Say what you chose *not* to alert on
and why — that is the graded part.

### 3. Runbooks

For every signal that pages, write a runbook with the first three things to check, in order. A
runbook that says "investigate the model" is not a runbook.

### 4. Catch the two known incidents

Your plan must detect both past incidents. For each, state:

- Which signal fires.
- **How long** after onset. Be honest — if it is nine hours, say nine hours.
- Whether it pages or merely appears on a dashboard.

Incident (b) is the interesting one: the global category distribution shifted because a legitimate
new customer arrived. That is **not** a problem, and a plan that pages for it is worse than one that
does not. How does your design distinguish "the world changed in a way we should ignore" from "the
world changed in a way that breaks us"?

### 5. The priority model

Priority has effectively no ground truth. Design its monitoring anyway. What proxies would you use,
what would each one *actually* prove, and what would you tell a stakeholder who asks "is the
priority model accurate?"

### 6. The reply model

CSAT has a 9% response rate and a 2–5 day delay. What can you monitor immediately? What is the
response-rate problem going to do to your slice-level analysis, and what would you do about it?

### 7. Reference windows

State, for each drift signal, whether you compare against a fixed baseline, a rolling window, or
both, and why. This is a one-line decision per signal that most plans make by accident.

### 8. Cost

Estimate the prediction-logging volume and say what you sample and what you keep at full fidelity.
There is a trap here involving small customer organizations — find it.

---

## Self-check before reading the solution

- [ ] Does my plan produce fewer than ~1 ML page per week?
- [ ] Did I say explicitly what I chose *not* to alert on?
- [ ] Does every alert have a runbook with three concrete first steps?
- [ ] Does the null-`account_tier` incident fire, and in how long?
- [ ] Does the new-customer incident *not* page, and can I say why?
- [ ] Did I handle three different label-availability regimes differently?
- [ ] Did I choose a reference window per signal, deliberately?
- [ ] Does my sampling keep small organizations statistically visible?

Reference answer: [`../solutions/01_design_monitoring_solution.md`](../solutions/01_design_monitoring_solution.md).
