# Solution 3 — Build the cost case

Reference answer for [`../exercises/03_build_the_cost_case.md`](../exercises/03_build_the_cost_case.md).

**Headline: approve cache + batching — ~33% saving for 5 engineer-weeks, and both survive the
platform migration. Decline distillation: it breaks even in 6.7 months against a model with about
7 months to live.** Quantization is the interesting borderline case, and the argument that decides it
is lifetime, not break-even.

All figures reproducible with `tools/cost_model.py` (an analytic model; the multipliers here were
measured by the team, which is the only reason they are usable).

---

## 1. Baseline

```
demand        = 18 qps × 900 tokens          = 16,200 tokens/s
usable per GPU= 2,000 × 0.70                 =  1,400 tokens/s
GPUs          = ceil(16,200 / 1,400) = ceil(11.57) = 12        <- rounding matters
monthly       = 12 × $2.80 × 730             = $24,528
requests/month= 18 × 3,600 × 730             = 47.30M
$/1k requests = 24,528 / 47,304 × 1,000      = $0.5185
```

Note the rounding immediately: 11.57 GPUs of demand costs 12 GPUs of money. **0.43 GPUs — $879/month
— is bought and never used**, before anyone optimizes anything. This is why savings arrive in steps.

## 2. Every proposal

| Proposal | GPUs | $/month | Saving/mo | Eng cost | Break-even |
|---|---:|---:|---:|---:|---:|
| Baseline | 12 | 24,528 | — | — | — |
| Exact-match cache (40% hit) | 7 | 14,308 | **10,220** | $18,000 | **1.8 mo** |
| Better batching (×1.25) | 10 | 20,440 | 4,088 | $12,000 | 2.9 mo |
| INT8 quantization (×1.5) | 8 | 16,352 | 8,176 | $36,000 | 4.4 mo |
| Distilled student (×2.5) | 5 | 10,220 | 14,308 | $96,000 | **6.7 mo** |
| Cache + batching + quantization | 4 | 8,176 | 16,352 | $66,000 | 4.0 mo |

Worked example for the cache row, since the pattern is the same for all of them:
`demand × 0.60 = 9,720 → ceil(9,720/1,400) = 7 GPUs → 7 × 2.80 × 730 = $14,308`.
Saving `24,528 − 14,308 = $10,220/month`; break-even `18,000 / 10,220 = 1.8 months`.

## 3. The model-lifetime constraint changes the answer

Break-even alone would rank distillation as merely "slow". Against a **7-month** model life it is
disqualifying, and the reason generalises: an optimization coupled to a specific checkpoint has an
*expiry date*, so its true return is
`saving × (remaining life − delivery time) − engineering cost`, not `saving × 12`.

**Delivery time matters and is usually left out.** Two engineers working in parallel turn engineer-
weeks into roughly half as many calendar weeks, so a 16-engineer-week project is ~2 months before it
saves its first dollar — 2 of the 7 months, gone before the meter starts.

| Proposal | Model-agnostic? | Delivery | Saving months | Net over the remaining 7 months |
|---|---|---|---:|---|
| Cache | **yes** — sits in front of whatever model serves | ~0.4 mo | 6.6 | +$10,220 × 6.6 − $18,000 = **+$49,500**, and it keeps paying afterwards |
| Batching | **yes** — a serving-path property | ~0.3 mo | 6.7 | +$4,088 × 6.7 − $12,000 = **+$15,400**, keeps paying |
| Quantization | **no** — quantized weights are per-checkpoint | ~1.0 mo (incl. the gate) | 6.0 | +$8,176 × 6.0 − $36,000 = **+$13,000**, then pay the cost again |
| Distillation | **no** — the student *is* the checkpoint | ~2.0 mo | 5.0 | +$14,308 × 5.0 − $96,000 = **−$24,500**, a **loss** |

**Distillation loses about $24,500 under the stated timeline.** That is a much stronger statement
than "it takes 6.7 months to break even", and it is the sentence to lead with. It also does not
depend on the quality risk, the eval work, or the re-distillation treadmill — all of which make it
worse.

Quantization survives, but barely, and the shape matters: it returns $13,000 and then the migration
consumes the asset, so you pay the $36,000 again. Against a model with a full life ahead of it the
same project returns roughly $62,000 a year. **Same project, same numbers, and the timing decides
it.**

## 4. The finance ask: 30%

**Yes, and comfortably.**

- **Cache alone: 41.7% reduction** ($24,528 → $14,308) for 3 weeks and $18,000. This clears the ask on
  its own.
- **Cache + batching: 41.7%** — the same 7 GPUs, because after the cache the batching multiplier does
  not remove another whole GPU (`ceil(9,720 / 1,750) = 6`… in fact it does: 6 GPUs, $12,264,
  **50.0%**). Worth 2 more weeks for another ~$2,000/month.

That combination is **5 engineer-weeks ($30,000)**, delivers ~50% against a 30% ask, carries **zero
quality risk** (identical outputs), and survives the platform migration. Delivery: roughly 6 weeks
including the gate-free rollout, so the saving starts in month 2.

**Recompute rather than assuming composition.** The two optimizations do not simply add: caching cuts
demand, and only then does the batching multiplier decide whether another whole GPU comes out. Always
recompute the GPU count for the *combination*.

## 5. Recommendation, including the decline

**Approve:** exact-match cache (3 weeks), then batching (2 weeks). ~50% reduction, $30,000, no
quality risk, both survive the migration.

**Defer:** INT8 quantization. It is a real $8,176/month, but it costs 6 weeks *plus* the quality gate
(see [exercise 2](../exercises/02_quality_gate.md) — that gate is itself substantial work), and it
must be redone after the migration. Revisit **after** the platform migration lands, when the work
attaches to a model with a full life ahead of it.

**Decline: the distilled student.** The number: **$96,000 of engineering, delivered ~2 months in,
against $14,308/month for the ~5 months of model life that remain — a net loss of roughly $24,500**,
before counting the eval work, the maintenance, and the re-distillation the migration forces. It is
not a close call, and it is not a judgement about distillation: the same proposal against a model
with a three-year life would be an easy approval.

## 6. Opportunity cost

Five engineer-weeks is five weeks of roadmap not shipped. Framed for a director who does not care
about GPUs:

> *"Five engineer-weeks buys a permanent $12,000/month reduction in serving cost — about $144,000 a
> year, against a one-off $30,000. It pays back in under three months and keeps paying after the
> platform migration. The question I can't answer alone is whether those same five weeks on the
> roadmap would generate more than $144,000 a year of value. If the feature they displace is worth
> more than that, we should not do this."*

**What I need from product to complete the argument:** the expected annual value of the specific
features displaced, and their timing. Without that, I can only present cost avoidance and let someone
with the revenue picture make the call — and I should say so rather than presenting the cost saving
as though it settles the matter.

Note the asymmetry that makes this an easy approval anyway: the saving is **recurring and certain**
(no quality risk, measured multipliers), while roadmap value is **uncertain**. That does not always
favour cost work, but it does when the payback is under three months.

## 7. Sensitivity

**The single most fragile input is the 40% cache hit rate.** It is the largest saving and the entire
justification for the top recommendation.

- At **40%** (assumed): 7 GPUs, $10,220/month, break-even 1.8 months. Approve.
- At **25%**: demand 12,150 → `ceil(12,150/1,400) = 9` GPUs → $18,396 → saving $6,132/month,
  break-even 2.9 months. **Still approve.**
- At **15%**: demand 13,770 → 10 GPUs → $20,440 → saving $4,088/month, break-even 4.4 months. Now it
  is no better than batching and costs 50% more effort — **the ordering flips**, and batching should
  go first.

**So my answer flips at a hit rate around 15%.** That is a wide margin below the assumed 40%, which
is why I am comfortable — but it is also a one-day measurement (replay a week of the request log
against a simulated cache) and I would run it before committing the three weeks. It would be
embarrassing to spend $18,000 to discover the documents are not re-submitted as often as we thought.

The second-most-fragile input is the **7-month migration date**, which is §8.

## 8. If the migration slips a year

Then distillation's arithmetic changes completely: ~19 months of life, less ~2 months to deliver, so
17 × $14,308 − $96,000 = **+$147,000**, and it becomes the highest-value item on the list. A slip does
not merely soften the decline — it reverses it.

**How to structure the decision so you are not badly wrong in either world:**

1. **Do the model-agnostic work now, unconditionally.** Cache and batching are correct in both
   worlds. They are also the *only* items that require no forecast at all, which is precisely why
   they go first — and it is worth noticing that the ladder's ordering already produced this answer
   without needing the forecast.
2. **Make the model-coupled work a gated decision with a named date.** Write down: *"if the migration
   has not landed or been firmly scheduled by month 4, we reopen quantization; by month 6, we reopen
   distillation."* A decision with a review date is a different artifact from a decision that quietly
   expires.
3. **Ask for the migration's confidence interval, not its date.** "Two quarters" from a platform team
   has a known distribution. If the honest answer is "somewhere between 6 and 18 months", the
   expected-value calculation should use that spread, and I would say so in the memo rather than
   pretending the point estimate is real.
4. **Reduce the coupling.** If distillation is likely eventually, build the distillation *pipeline*
   (traffic logging, teacher scoring, eval harness) now — it is model-agnostic and it makes the
   eventual student a fraction of the 16 weeks. This converts a bet on a date into an investment that
   is correct either way, and it is the most valuable idea in this answer.

**The "it depends on X" pivot:** X is the migration date's *distribution*, not its point estimate.
Firm within 7 months → cache + batching only. Uncertain, plausibly 18 months → add quantization now
and build the distillation pipeline. Point estimates hide exactly the risk that decides this, which
is why the recommendation is written as a schedule of gated decisions rather than a single verdict.

## The email to Finance

> *We can reduce the ML serving line item for this service by about 50% — from $24,500 to roughly
> $12,300 per month — with five engineer-weeks of work (one-off ~$30,000), starting to take effect
> next month. Both changes are caching and request batching, which carry no model-quality risk and
> survive the upcoming platform migration; we are deliberately not pursuing model compression this
> cycle because the current model is due to be replaced before that work could pay for itself.*
