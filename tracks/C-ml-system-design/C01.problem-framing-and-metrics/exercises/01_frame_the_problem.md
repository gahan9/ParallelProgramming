# Exercise 01 — Frame the problem

**Goal:** convert a vague stakeholder request into a framed ML problem that an engineer could start
building and a reviewer could argue with.

**Time:** 45–60 minutes. **Deliverable:** a one-page design brief, written below.

---

## The request

You are the first ML engineer on a mid-size B2B SaaS product (project-management software, ~40,000
paying company accounts, seat-based monthly subscriptions). In a planning meeting, the VP of Customer
Success says:

> "Churn is killing us. Can we use AI to figure out which customers are going to leave so we can save
> them? I've heard other companies have churn models. Let's get one."

That is everything you have been told. What you *do* know about the environment, from asking around:

- Accounts pay monthly and can cancel at any time; cancellations take effect at the end of the
  billing period.
- The Customer Success team is **6 people**. Each can run roughly **8 meaningful outreach
  touches per day** (a real call or tailored email), so about **1,000 touches per month** total.
- Finance can authorize a **retention discount** of up to 20% for 6 months, but wants the total
  discount budget capped.
- The product logs are decent: logins, feature usage, seats active, support tickets, invoice history,
  NPS survey responses (sparse), and account metadata (industry, size, plan, signup date).
- Nobody currently has a written definition of "churn."

## Your task

Fill in every section below. Where information is missing, **state the question you would ask and who
you would ask it of** — do not invent an answer and do not skip the section.

---

### 1. The decision

What action does this system produce? Be specific about the action, who or what takes it, and the
alternatives to it (including "do nothing").

```
(your answer)
```

### 2. The unit and cadence of decision

One decision per what, evaluated how often? Justify the cadence against the outreach capacity above.

```
(your answer)
```

### 3. The prediction task

Define it precisely enough that two engineers would build the same thing: the exact quantity
predicted, the prediction horizon, the data cutoff ("as of" moment), and the output type
(probability, score, class, ranked list).

Then write the **label definition** as a sentence a data engineer could turn into SQL. Beware: "churn"
in a monthly-subscription B2B product is genuinely ambiguous — full cancellation, downgrade,
seat reduction, and silent non-usage are different events.

```
(your answer)
```

### 4. Is ML the right tool here?

Apply the four conditions from README §2 (complex-and-changing pattern, labels available, errors
tolerable, scale defeats human judgement). Answer each with a yes/no **and a reason**.

Then design the **non-ML alternative you would actually ship first** — a specific rules baseline, not
"some heuristics." State what it would cost to build, what it would catch, and what would make you
abandon it in favour of a model.

```
(your answer)
```

### 5. Metric hierarchy

Fill all three layers and state the **assumed link** between each pair, in the form "we assume X
improves Y, which we will verify by Z."

| Layer | Metric | How measured | Reporting cadence |
|---|---|---|---|
| Business | | | |
| Product / system | | | |
| Offline model | | | |

Assumed links:

```
(your answer)
```

### 6. Threshold and capacity

You have ~1,000 outreach touches per month and roughly 40,000 accounts. Given that:

- Is this a threshold problem or a ranking problem? Defend your answer.
- What metric follows from that choice?
- What happens to your metric choice if Customer Success doubles in size, or if outreach becomes
  automated and effectively unlimited?

```
(your answer)
```

### 7. Testable requirements

Rewrite each of these as a statement a test could check. One line each, with a number.

| Requirement | Your testable form |
|---|---|
| Latency | |
| Throughput | |
| Freshness | |
| Cost per prediction | |
| **Failure behaviour** (model unavailable) | |
| Interpretability / explanation to the CS rep | |

### 8. The framing trap

There is a specific and very common framing error lurking in this request. Name it, explain why the
obvious framing would waste money even with a perfectly accurate model, and say what you would do
about it given that you cannot run a large experiment in month one.

```
(your answer)
```

### 9. What you would say in the meeting

Three to five sentences, in language the VP of Customer Success would accept, that (a) commits to
something concrete, (b) does not promise a model in week one, and (c) names the one thing you need
from them.

```
(your answer)
```

---

## Self-check

- [ ] Section 1 names an **action**, not a score.
- [ ] The label definition is unambiguous enough to write as SQL, and handles downgrade vs cancel.
- [ ] The prediction horizon and the "as of" cutoff are both stated.
- [ ] You designed a real rules baseline you would be willing to ship.
- [ ] All three metric layers are filled *and* the links between them are stated as verifiable
      assumptions.
- [ ] The capacity constraint (1,000 touches) changed your metric choice, not just your commentary.
- [ ] Every requirement in §7 contains a number.
- [ ] Failure behaviour is specified.
- [ ] You named at least one question you cannot answer and who owns the answer.

Reference answer with the reasoning:
[`../solutions/01_frame_the_problem_solution.md`](../solutions/01_frame_the_problem_solution.md).
