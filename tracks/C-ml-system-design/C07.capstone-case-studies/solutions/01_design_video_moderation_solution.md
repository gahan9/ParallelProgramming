# Solution 1 — Video content-moderation system

Reference design for [`../exercises/01_design_video_moderation.md`](../exercises/01_design_video_moderation.md).

**This is *a* defensible answer, not *the* answer.** Places where a different choice would be
equally right are flagged as **[fork]**. What is not negotiable is that every number traces to
a labelled assumption.

---

## 0. Assumptions register

| # | Assumption | Value | If wrong |
|---|---|---|---|
| A1 | Uploads | 500,000/day, 8 min average (given) | Everything |
| A2 | Reviewers | 200, 8-hour shifts (given) | The central constraint |
| A3 | **Violation base rate** | 0.5% of uploads (2,500/day) | Determines whether any accuracy number is meaningful |
| A4 | Average human review time | 90 s per item — reviewers watch flagged segments, not whole videos | Review capacity |
| A5 | Fully-loaded reviewer cost | $25/hour | The cost model |
| A6 | User reports | 50,000/day | Report-queue triage |
| A7 | Frame sampling | 1 fps, plus scene-change keyframes | Vision cost, and what you miss |
| A8 | GPU | ~300 TFLOP/s achievable on vision inference; ASR at 60× realtime batched | Machine cost |
| A9 | Cost | $3.00/GPU-hour | The cost model |

---

## 1. Requirements and scope

**Functional.** Ingest every upload, detect four violation categories, act automatically where
precision permits, escalate everything else to a prioritized human queue, support user reports
with a 24-hour regulatory deadline, and support appeals.

**Non-functional.**

| Requirement | Target |
|---|---|
| Ingestion throughput | 500,000 videos/day sustained, 2× burst |
| Time to first automated decision | p95 < 10 min from upload (before meaningful distribution) |
| Reported-content decision | **100% within 24 h** — regulatory, not an SLO |
| Appeal decision | p95 < 48 h |
| Auto-removal precision | ≥ 99.5% per category (derived in §2) |
| Availability | 99.9% ingestion; the classifier may degrade, ingestion may not |

**Explicitly: this is a throughput system, not a latency system.** Nobody waits on a response.
The only real-time-ish constraint is getting a decision before a video accumulates views, which
is minutes, not milliseconds. A candidate who writes a p99 latency budget for the upload path
has misread the prompt.

**Three queues, three deadlines.** Proactive (soft, minutes), reported (hard, 24 h), appeals
(soft, 48 h). They compete for the same 200 reviewers, which makes queue allocation a first-class
design decision rather than an operational detail.

**Out of scope:** live-stream moderation (a genuinely different system — the decision must be
made during broadcast), comment moderation, copyright/DMCA, and advertiser suitability.

---

## 2. Metrics — and why the base rate comes first

Do this arithmetic **before** quoting any accuracy number, because it changes what the numbers
mean:

```
Uploads/day                       500,000
Violation base rate (A3)             0.5%  ->   2,500 violations, 497,500 clean

A "99% accurate" model (99% TPR, 99% TNR):
  true positives   0.99 x   2,500  =  2,475
  false positives  0.01 x 497,500  =  4,975
  precision = 2,475 / (2,475 + 4,975) = 33%
```

**Two-thirds of everything that model flags is a legitimate video.** Auto-removing on that
signal takes down 4,975 creators' videos a day. This single calculation reframes the whole
problem: at a 0.5% base rate, "accuracy" is a meaningless metric and precision at the operating
threshold is the only number worth quoting.

To auto-remove at 99.5% precision you need `FP ≤ TP/199`, which at these volumes means fewer
than ~12 false positives a day — achievable only at a very high threshold, and therefore at low
recall. **That is exactly why the human tier exists**, and why the model's real job is triage.

| Layer | Metric |
|---|---|
| Business | Prevalence — violating **views** (not videos) as a share of total views. Views, because one violating video with 10 M views matters more than a thousand with 10 |
| Business | Creator trust: appeal rate, and appeal overturn rate |
| Product | Time-to-action on violating content, weighted by views accrued before action |
| Product | Regulatory compliance rate on the 24-hour queue (must be 100%) |
| Offline | Per-category precision/recall at the operating threshold, and **PR-AUC** — not ROC-AUC, which is misleading at a 0.5% base rate |
| Offline | Reviewer-queue precision: fraction of escalated items that turn out to be violations |

**Guardrails, with thresholds** — the section that makes this a design rather than a classifier:

| Guardrail | Threshold | Prevents |
|---|---|---|
| **Appeal overturn rate**, per category | < 5%, alert at 3% | Silent over-removal. This is the single best proxy for false positives you have, because it is *labelled by humans who disagreed with you* |
| **Per-language and per-region false-positive rate** | No group > 1.5× the global rate | The fairness failure that aggregate metrics hide by construction |
| **Per-creator-size false-positive rate** | Same 1.5× bound | Small creators have no support channel and no leverage; they absorb errors invisibly |
| Auto-removal volume | Alert on > 30% day-over-day change | A model regression that removes at scale before anyone notices |
| Reviewer queue age, p95, per queue | Proactive < 6 h, reported < 12 h | The 24-hour deadline breached in the second half of the month |
| Reviewer agreement rate on double-reviewed items | > 85% | Degrading label quality — if reviewers disagree, your ground truth is noise |

**Fairness is in the metrics, not in a paragraph at the end.** Content in an
under-represented language, or referencing a minority cultural context, is systematically more
likely to be misclassified because it is under-represented in training data. That has to be
measurable (rows 2 and 3), gated at release (§7), and monitored (§8). A design that mentions
bias only in "future work" has not engaged with the prompt.

---

## 3. Architecture

```
UPLOAD
  │
  ├─▶ [Stage 0]  Hash / perceptual-hash match against known-violating corpus
  │              cost: ~0        action: instant block on exact match
  │
  ├─▶ [Stage 1]  Cheap multimodal first pass  (all 500k/day)
  │              1 fps frames + ASR transcript + title/description/metadata
  │              output: per-category risk score
  │              ├─ score > auto_block[c]   ─▶ auto-remove + notify + appeal path
  │              ├─ score < ignore[c]       ─▶ publish
  │              └─ otherwise               ─▶ Stage 2
  │
  ├─▶ [Stage 2]  Expensive specialist models  (~5% of uploads = 25k/day)
  │              temporal video model, per-category specialists, OCR on frames
  │              ├─ high confidence          ─▶ auto action
  │              └─ uncertain                ─▶ Human queue
  │
  └─▶ [Stage 3]  HUMAN REVIEW  (capacity-bound: 64,000 items/day)
                 priority = f(risk, predicted views, category severity, deadline)
                 decision ─▶ action + LABEL ─▶ training set
                                    │
                                    └─▶ feedback loop: humans only ever see
                                        what the model escalated (§6)

REPORTS ──▶ report-validity model ──▶ prioritized into the same human queue
APPEALS ──▶ separate queue, senior reviewers, decisions weighted 2x in training
```

Same funnel logic as the recommender (module README §3.2), for the same reason: spend a lot per
item on very few items. The difference is that **the final stage is people, and its capacity is
fixed**, which inverts the usual optimization. You are not minimizing cost subject to a quality
bar; you are maximizing harm prevented subject to a fixed review budget.

---

## 4. The two arithmetic results that drive the design

### 4.1 Machine cost is small, and ASR dominates it

```
Video ingested       500,000 x 8 min       = 4,000,000 min/day = 66,667 hours/day

VISION  (Stage 1)
  frames at 1 fps    500,000 x 480         = 240 M frames/day  = 2,778 frames/s
  ~5 models/frame at ~17.5 GFLOP each      = 87.5 GFLOP/frame
  daily FLOPs        240e6 x 87.5e9        = 2.1e19 = 21 EFLOP/day
  GPU-hours          2.1e19 / 300e12 / 3600 = 19.4 GPU-hours/day   -> ~1 GPU

AUDIO  (ASR, Stage 1)
  66,667 hours/day at 60x realtime          = 1,111 GPU-hours/day  -> ~46 GPUs

TEXT  (transcript + metadata classification)
  negligible next to the above
```

**ASR is 98% of the machine cost.** That is the non-obvious finding, and it immediately
suggests a lever: do not transcribe everything. Run a cheap audio-event classifier (speech
present? shouting? gunshots?) over all of it, and full ASR only on the ~20% where the audio
signal or the vision score suggests it will matter. That takes ASR from 46 GPUs to ~10.

```
Machine cost   ~50 GPUs x 24 h x $3.00     = $3,600/day  = $108,000/month
```

### 4.2 Human capacity is the constraint, and it costs 11× more

```
Reviewer-hours/day   200 x 8                        = 1,600 h
Items/day            1,600 x 3,600 / 90 s (A4)      = 64,000 items/day
As a share of uploads 64,000 / 500,000              = 12.8%
Reviewer cost        1,600 h x $25 (A5)             = $40,000/day = $1,200,000/month
```

**Human review costs 11× what the machines do**, and it is capacity-capped at 12.8% of uploads.
Every subsequent decision follows from those two facts:

- **The objective function is "maximize harm prevented per reviewer-minute."** Not accuracy, not
  F1. Write that down; it is the single most useful sentence in this design.
- **A high-recall, low-precision first pass is affordable.** Capacity (64,000) is 25× the
  expected violation volume (2,500), so precision as low as ~4% still fills the queue with real
  work. Optimize Stage 1 for *recall*, and let the queue-ranking model handle precision.
- **Review time is a lever equal to model quality.** Cutting A4 from 90 s to 60 s adds 32,000
  items/day of capacity — more than any plausible model improvement delivers. Timestamped
  evidence, pre-clipped segments, and a keyboard-driven UI are ML-system work even though no
  model is involved.

### 4.3 Allocating the 64,000

| Queue | Share | Items/day | Demand | Note |
|---|---:|---:|---|---|
| Reported content | 40% | 25,600 | 50,000 reports (A6) | **Under-capacity by 2×** — needs its own triage model |
| Proactive escalation | 45% | 28,800 | ~25,000 from Stage 2 | Roughly matched |
| Appeals | 10% | 6,400 | Demand-driven | Senior reviewers |
| **Randomized audit sample** | **5%** | **3,200** | — | See §6 — non-negotiable |

The reported queue being under-capacity is a real finding, not a modelling artefact: a
report-validity model must rank the 50,000 reports so the 24-hour deadline is met on the ones
that matter. **[fork]** You could instead argue for auto-closing low-validity reports without
review; that is defensible and legally riskier, and you should say which market's regulator you
checked with.

### 4.4 What 1 fps sampling misses — say this before you are asked

A violation lasting under one second between sampled frames is invisible to Stage 1. Also
missed: content that is only violating in *motion* (a still frame of a stunt looks like a still
frame of a stunt), and slow-building context. Mitigations: scene-change keyframes on top of the
fixed rate; a temporal model in Stage 2 that sees the actual clip; and higher sampling for
uploaders with prior violations. **Residual risk stated explicitly** — it is a sampling design,
and sampling designs miss things. Owning that is better than pretending 1 fps is complete.

---

## 5. Models and training data

**Stage 1** — a shared multimodal encoder (frames + ASR transcript + metadata) with per-category
heads. Multi-task, because the categories share most of their representation and because it
gives you one model to operate instead of four. Calibrated per category, since each has its own
base rate and its own operating threshold.

**Stage 2** — per-category specialists on the ~5% that survives, plus a temporal video model.
Specialists here rather than in Stage 1 because 5% of the volume means you can afford 20× the
compute per item.

**[fork]** A single large multimodal model for both stages is increasingly viable and much
simpler to operate. It costs more per item, so it needs the Stage 1 volume to be cheap enough —
run the numbers for your model before choosing. Simplicity is a real benefit; do not dismiss it.

### Labels, and their three biases

| Source | Volume | Bias |
|---|---|---|
| Reviewer decisions | ~64,000/day | **Only on what the model escalated.** The dominant bias |
| Randomized audit sample | 3,200/day | Unbiased — the only unbiased labels in the system |
| Appeal overturns | Demand-driven | Enriched for false positives — high value per label |
| Known-violating corpus | Fixed | Enriched for previously-caught patterns |

**Split by time, and by uploader.** Policy changes, adversaries adapt, and content trends move,
so a random split leaks the future. Also hold out entire uploaders, because a channel's videos
are highly correlated and a random split will put the same creator's near-duplicate uploads in
both train and test — inflating every metric.

**Adversarial adaptation is the property that makes this different from ordinary
classification.** Bad actors probe the system and route around it. Consequences: the training
distribution shifts *because you deployed*; a held-out test set from last month overstates
today's performance; and retraining cadence is a security parameter, not just a quality one.
Retrain weekly on a rolling window, and keep the randomized audit sample as the only honest
measure of true prevalence.

---

## 6. The feedback loops — there are two, and both matter

**Loop 1: humans only see what the model escalates.** Reviewer labels are conditioned on the
model's decisions, so violations the model never flags never get labelled, and the model never
learns it is missing them. Its measured recall will look excellent and its true recall will
drift downwards invisibly.

**Loop 2: enforcement changes the population.** Removing content and penalizing accounts changes
what gets uploaded next, so the distribution the model was trained on stops existing.

The fix for both is the same instrument: **the 5% randomized audit sample (3,200 items/day) —
uniformly sampled from all uploads, reviewed regardless of model score.** It is the only
unbiased estimate of true prevalence and true recall you will ever have. It costs 5% of the most
expensive resource in the system, and it is the first thing anyone will try to cut in a capacity
crunch. Defend it in advance with the arithmetic:

```
Cost: 3,200 items/day x 90 s / 3,600 x $25 = $2,000/day = $60,000/month
      = 5% of the review budget, 4.6% of total system cost
```

$60k/month to know whether the system works. Without it, every recall number you report is
conditioned on the model's own choices and is therefore unfalsifiable.

**[fork]** Stratified sampling (over-sampling near the decision boundary) gives tighter
estimates of recall for the same reviewer cost, at the price of needing importance weights to
recover prevalence. Reasonable; be explicit about which quantity you are optimizing the estimate
of.

---

## 7. Evaluation and rollout

**Offline gates**, all of which must pass:

1. Per-category precision at the operating threshold ≥ the incumbent.
2. Recall on the randomized audit sample ≥ the incumbent — the only unbiased recall you have.
3. **Per-language, per-region, per-creator-size false-positive rates within 1.5× of global.** A
   model that improves aggregate metrics while regressing one language group does not ship.
4. No regression on a frozen adversarial suite of known evasion techniques.

**Rollout ladder:**

| Stage | What | Gate |
|---|---|---|
| Shadow | Score 100% of traffic, take no action, compare decisions against the incumbent | Disagreement rate and its direction reviewed by policy, not just by ML |
| Canary, escalate-only | New model may **escalate to humans** but not auto-remove | Reviewer-queue precision does not drop |
| Canary, auto-action | 5% of traffic may auto-remove | Appeal overturn rate, per-group FP rates, auto-removal volume |
| Ramp | 25% → 50% → 100%, ≥ 24 h per step | Same four metrics, automatic rollback on any breach |

**The escalate-only canary stage is the important one and it is specific to this domain.** A
moderation model's expensive failure is over-removal, and over-removal is irreversible in the
way that matters — the creator experienced it. Letting the new model prove itself on the
*escalation* decision before it is allowed to *act* costs one extra rollout stage and removes
almost all of the blast radius.

**Rollback:** the previous model stays warm and serving-ready; rollback is a config flag,
< 5 minutes, tested monthly. Plus a **bulk-reinstate tool** that can reverse every auto-removal
made by a given model version in a given window — because the model that ships bad will remove
thousands of videos before the alert fires, and "we rolled back" is not a remedy for the
creators already affected.

**Blast-radius limit:** a hard cap on auto-removals per hour, globally and per category. If the
cap is hit, the system fails **open into the human queue** rather than continuing to remove. One
line of config that turns a catastrophe into a backlog.

---

## 8. Monitoring and degradation

| Signal | Alert | Catches |
|---|---|---|
| Auto-removal volume per category | ±30% day-over-day | Model regression at scale |
| Appeal rate and **overturn rate** | Overturn > 3% | Over-removal — the best FP proxy available |
| Per-group FP rate ratios | Any group > 1.5× global | Fairness regression |
| Queue age p95, per queue | Reported > 12 h | The 24-hour deadline about to be missed |
| Reviewer agreement on double-reviewed items | < 85% | Label quality degrading; policy ambiguity |
| Prevalence from the audit sample | Trend | **True** miss rate — the only unbiased signal here |
| Score distribution per category | Distribution shift | Adversarial adaptation, or an upstream encoder change |
| Stage-2 escalation rate | ±20% | A threshold drift that will silently blow the human budget |

**Degradation ladder**, in order:

1. **Queue backs up** → raise the escalation threshold so only higher-risk items enter. Recall
   falls; the deadline is met.
2. **Reviewers unavailable** (incident, holiday) → prioritize strictly by predicted views ×
   severity, so the reviewer-minutes that exist go where harm is largest.
3. **Stage 2 unavailable** → route its would-be output straight to the human queue; the queue
   backs up and rung 1 handles it.
4. **Stage 1 unavailable** → **publish with a "pending review" flag and reduced distribution.**
   Do *not* block ingestion, and do *not* auto-remove on no signal. Reduced distribution caps
   the harm from a violating video while the pipeline recovers, and this is the rung that most
   designs get backwards.
5. **Everything down** → ingestion continues, all uploads flagged for retro-scan, backlog
   processed on recovery, and the retro-scan is a first-class job that is tested, not a plan.

Note that rung 4 is the design's fail-safe direction: **fail open on ingestion, fail closed on
auto-removal.** Never the reverse.

---

## 9. Cost

| Component | Monthly | Share |
|---|---:|---:|
| **Human review (200 reviewers)** | **$1,200,000** | 91.7% |
| Machine inference (~50 GPUs, ASR-dominated) | $108,000 | 8.3% |
| Storage, pipeline, orchestration | ~$30,000 | (excluded from the split) |
| **Cost per 1,000 videos processed** | **$87** | of which $80 is human |

That table is the design. Three consequences:

1. **A 10% reduction in review time beats a 10% improvement in model AUC**, in dollars, by a
   wide margin. Invest in the reviewer tooling.
2. **The ASR sampling decision (§4.1) is the only machine-side lever worth arguing about** —
   it is 90% of the machine bill.
3. **Any proposal that increases escalation volume must be costed in reviewer-minutes**, not
   in GPU-hours. Reviewer capacity is the currency of this system.

---

## 10. Tradeoffs, 10×, omissions

| Choice | Bought | Paid |
|---|---|---|
| Cheap-first cascade | 12.8% of uploads get human attention affordably | Stage 1 misses set a hard recall ceiling for everything downstream |
| High-recall Stage 1 | Fills the queue with real work | Low precision; the queue-ranking model carries the burden |
| Fail open on ingestion | Uploads never blocked by an ML outage | Violating content is briefly visible, at reduced distribution |
| 5% audit sample | The only unbiased measurement in the system | 5% of the most expensive resource |
| 1 fps sampling | ~46× less vision compute than every frame | Sub-second violations missed; disclosed, not hidden |

**At 10× (5 M uploads/day):**

- Human review does **not** scale to 2,000 reviewers linearly — training, quality consistency,
  and reviewer wellbeing all degrade. The model must absorb the growth, which means auto-action
  precision has to rise, which means Stage 2 gets more expensive per item.
- ASR at 666,000 hours/day makes the "transcribe selectively" decision the single largest cost
  item in the company's ML budget. It gets its own team.
- **Regional models and regional review teams** become necessary — policy differs by
  jurisdiction, and language coverage is the dominant fairness axis.
- Adversarial pressure scales super-linearly with platform size. Expect a dedicated
  counter-abuse function, and expect retraining cadence to become a security SLA.

**Deliberately left out:** live-stream moderation (the decision must be made *during* broadcast,
which is a real-time system with a completely different architecture), the policy-definition
process itself (which is where most disagreements actually originate), reviewer wellbeing
programmes (real, important, and not an ML-systems question), and cross-platform hash sharing.
