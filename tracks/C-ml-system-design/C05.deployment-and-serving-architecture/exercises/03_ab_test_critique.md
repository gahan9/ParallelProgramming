# Exercise 3 — Critique the A/B test

**Time:** 30–45 minutes · **Deliverable:** ~1 page: a numbered list of defects, each with the
*mechanism* by which it corrupts the conclusion, then a corrected test plan.

---

## The document under review

The following arrived in your inbox as a proposal to ship. Read it as a reviewer.

> ### Experiment write-up: Smart Reply v2 — recommend we ship
>
> **Background.** Smart Reply suggests three one-tap replies under each incoming message in our
> messaging product. The v2 model is larger and was trained on six months more data.
>
> **Setup.** We ran the experiment for four days, Tuesday 09:00 through Friday 09:00. Each incoming
> message was independently assigned to control (v1) or treatment (v2) with probability 0.5, using
> `random()` at request time in the suggestion service.
>
> **Metric.** Suggestion click-through rate — the fraction of served suggestion sets where the user
> tapped one of the three replies.
>
> **Results.** Over the four days we served 41.2 M suggestion sets in each arm.
>
> | Arm | Suggestion sets | Clicks | CTR |
> |---|---:|---:|---:|
> | Control (v1) | 41 204 118 | 5 767 191 | 13.996% |
> | Treatment (v2) | 41 198 802 | 5 819 251 | 14.125% |
>
> We ran a two-proportion z-test on the final numbers: **p = 0.0000012**. The lift is **+0.93%
> relative**, which at our volume is worth roughly \$2.1 M/year in downstream engagement value.
>
> **Monitoring.** We watched the dashboard from the first hour. The result crossed p < 0.05 on
> Tuesday afternoon, dipped back on Wednesday morning, and has been comfortably significant since
> Wednesday evening. We are confident the effect is real.
>
> **Serving impact.** v2 is a larger model, so p99 suggestion latency went from 41 ms to 88 ms. This
> is well within our 300 ms budget for the suggestion component, so we did not treat it as a
> blocker.
>
> **Recommendation.** Ship v2 to 100%. The effect is large, highly significant, and stable.

---

## Your task

### Part A — Find the defects

Produce a numbered list. For **each** defect give:

1. What is wrong.
2. **The mechanism** — precisely how it corrupts the conclusion. "Peeking is bad" is not an answer;
   "each interim test carries its own α, so the probability that *some* look crosses the threshold
   under the null is far above 5%" is.
3. Whether it makes the result **optimistic**, **pessimistic**, or **uninterpretable**.
4. How you would fix it.

There are at least three defects the module explicitly names — a randomization-unit error, peeking,
and a missing guardrail metric. There are several more. Aim to find at least six in total.

Prompts, if you get stuck: look hard at *what a "suggestion set" is versus what a user is*; at the
**dates**; at the relationship between the p-value and the **confidence interval**; at what the
latency paragraph does and does not check; at what happens to the *other* things users do when they
tap a suggested reply; and at whether "we did not observe X" is the same as "X did not happen".

### Part B — Is the effect even worth shipping?

Use `../tools/ab_calculator.py evaluate` with the numbers from the table. Report the confidence
interval on the absolute difference, and convert it to a relative range. Then answer: given that
interval, and given that v2 costs more to serve, is "worth roughly \$2.1 M/year" a defensible claim
as written? What would make it defensible?

### Part C — Write the correct design

Produce the experiment plan the team should have written, in the format from §3.6 of the module
README:

```
  hypothesis        :
  primary metric    :
  guardrail metrics :
  randomization unit:
  MDE               :
  sample size       :
  duration          :
  analysis plan     :
  stopping rule     :
  SRM check         :
```

Fill in every line with a specific value and a one-line justification. Use
`../tools/ab_calculator.py size` for the sample size.

### Part D — The conversation

The author is a competent engineer who is genuinely excited and has already told their manager the
number. Write the two or three sentences you would open your review comment with.

This part is graded too. A technically correct review that lands as an attack gets the analysis
ignored, and the wrong model ships anyway.

---

## Self-check before reading the solution

- [ ] Did I find at least six distinct defects?
- [ ] For each, did I state the *mechanism*, not just the name?
- [ ] Did I compute the confidence interval rather than repeating the p-value?
- [ ] Did I notice what the four-day window does and does not cover?
- [ ] Did I identify a guardrail metric that would catch "CTR up, product worse"?
- [ ] Is my corrected plan fully specified — could someone execute it without asking me a question?

Reference answer: [`../solutions/03_ab_test_critique_solution.md`](../solutions/03_ab_test_critique_solution.md).
