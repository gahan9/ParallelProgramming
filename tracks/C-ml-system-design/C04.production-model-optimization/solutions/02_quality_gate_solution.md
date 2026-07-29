# Solution 2 — The quality gate for a quantization rollout

Reference answer for [`../exercises/02_quality_gate.md`](../exercises/02_quality_gate.md).

**Headline: the two safety-critical queues are 0.9% of volume, so a uniform 11,000-ticket eval set
contains about 100 of them — which means it is structurally incapable of detecting the regression you
most care about.** Everything else in this gate is standard practice; that one fact is the design
problem, and an answer that does not confront it has missed the exercise.

---

## 1. What is wrong with "93.8% → 93.6%, within noise"

Four distinct failures, and it is worth separating them because they have different fixes:

1. **It is an aggregate.** A −0.2 point average is arithmetically consistent with `account-compromise`
   accuracy falling from 95% to 60%, because that queue is 0.9% of volume: `0.009 × 35 points ≈ 0.3`
   points of aggregate. **The worst outcome in this system is invisible in the number being quoted.**
2. **"Within noise" is asserted, not measured.** Nobody has established what the noise band *is*. With
   11,000 samples the standard error on a 93.8% proportion is about
   `sqrt(0.938 × 0.062 / 11000) ≈ 0.23` percentage points, so −0.2 is indeed inside one standard
   error — but that also means this eval set **cannot resolve a real 0.2-point regression either**.
   "Within noise" is a statement about the instrument's blindness, not the model's safety.
3. **The threshold was chosen after seeing the result.** If −0.2 had been the pre-declared limit, this
   is a pass. Deciding now, with the number on screen and a $9k/month saving attached, is
   rationalisation. The threshold must be set before the measurement.
4. **Accuracy is not the product metric.** The cost of a misroute varies enormously by queue — hours
   of delay for a routine ticket, a regulatory issue for `account-compromise`. A metric that treats
   all 40 queues as equal does not measure the thing the business cares about.

A fifth, smaller point: no latency or cost measurement is quoted either, so we are not even certain
what we are buying.

## 2. The eval set

| Property | Specification |
|---|---|
| Composition | **Stratified**, not uniform: a traffic-weighted core plus *guaranteed minimum counts* per critical slice (see §3) |
| Size | ~11,000 traffic-weighted core, **plus** an oversampled safety-critical block of ≥1,500 tickets per critical queue |
| Freshness | Sampled from the **last 90 days**, not "last year" — a router's input distribution moves with the product |
| Labels | Human-adjudicated for the safety-critical block; production-resolution labels acceptable for the core |
| Versioning | Content-addressed (hash of the ticket IDs + labels). The eval harness records the hash with every score and **refuses to compare two scores computed under different hashes** |
| Change control | Owned by the team, changed only by pull request; changing the eval set starts a new baseline, it does not silently move the old one |
| Guarantee of comparability | Both variants scored in the same harness run, same hash, same code path — not "we have the baseline from last month" |

**What is wrong with "11,000 sampled uniformly from last year":** it is stale (a year of product
change), it is uniform (so rare-but-critical queues are represented by ~100 examples), and it is
unversioned (so nothing prevents it from drifting between the two scores). All three are fixable, and
the second one is the expensive one.

## 3. Slices, and the 0.9% problem

| Slice | Catches |
|---|---|
| Per-queue (all 40) | queue-specific collapse hidden by the aggregate |
| **`account-compromise`, `billing-dispute` separately** | the regulatory failure |
| Language: en / es / ja / de / other | quantization error concentrating in a script or tokenisation pattern |
| Ticket length: short / medium / long (top 6%) | degradation on long inputs — the classic quantization failure |
| Mobile-app traffic (typos, no punctuation) | degradation on noisy/out-of-distribution text |
| Confidence band: the model's low-confidence decile | where the decision boundary lives, and where small numeric perturbations flip labels |

**The 0.9% problem, stated properly.** ~100 examples of a critical queue gives a standard error of
about `sqrt(0.95 × 0.05 / 100) ≈ 2.2` percentage points. You could not detect a 5-point regression
with any confidence. Three responses, all of which I would do:

1. **Oversample deliberately.** Build a dedicated block of ≥1,500 tickets per critical queue,
   accumulated over a longer window and human-labelled. Report it as its own metric — never folded
   into the aggregate, because folding it back in is what caused the problem.
2. **Report the disagreement rate, not just accuracy.** For any input where fp16 and INT8 disagree,
   you have a direct, high-sensitivity signal that needs no labels at all: run both models over
   100,000 unlabelled recent tickets and count disagreements per queue. This is cheap, it scales, and
   it detects exactly the perturbation quantization causes. **This is the highest-value single test
   in the gate**, and it is the one teams most often miss.
3. **Route-safety asymmetry.** Misrouting *into* a safety-critical queue is a nuisance; misrouting
   *out of* one is the harm. Measure recall on those two queues specifically, not accuracy.

## 4. Thresholds

Pre-declared, numeric, and asymmetric where the harm is asymmetric.

| Metric | Threshold | Justification |
|---|---|---|
| Aggregate accuracy | ≥ −0.3 pp vs baseline | ~1 standard error on the enlarged set; anything larger is a real effect |
| Any non-critical queue | ≥ −2.0 pp | small queues are noisy; −2 pp is detectable and operationally tolerable |
| **`account-compromise` recall** | **≥ −0.5 pp, and no absolute value below 94%** | **asymmetric on purpose**: this is a regulatory/trust failure, so the bar is roughly 4× tighter than other queues and it has an absolute floor as well as a relative one |
| **`billing-dispute` recall** | **≥ −0.5 pp** | same reasoning |
| Any language slice | ≥ −1.0 pp | catches script-specific degradation; Japanese is 8% of volume and tokenises differently |
| Long-ticket slice (top 6%) | ≥ −1.0 pp | the slice most likely to expose activation-range problems |
| fp16↔INT8 disagreement rate | ≤ 2.0% overall, **≤ 0.5% on critical queues** | direct sensitivity to the perturbation; label-free, so it can run on 10× more data |
| Throughput | ≥ 1.6× measured (claim is 1.7×) | if the win is not real, none of the risk is worth taking |
| p99 latency | no regression | quantization should not make latency worse; if it does, something is wrong with the kernel path |

**Establishing the noise band, rather than asserting it:** score the *unchanged* fp16 model twice
under the harness (different shuffle/batching order) and, better, score two same-config training
seeds if available. The observed spread between identical or equivalent configurations **is** the
noise floor. A threshold tighter than that floor will fail randomly; a threshold much looser than it
is not measuring anything.

## 5. The rollout

| Stage | Traffic | Duration | Measured | Catches what the previous stage could not |
|---|---|---|---|---|
| **Offline** | 0% | hours | all of §4 on frozen sets | nothing yet — this is the floor |
| **Disagreement sweep** | 0% (replay) | hours | fp16 vs INT8 on 100k recent unlabelled tickets | rare inputs absent from the eval set |
| **Shadow** | 100% mirrored, no user impact | 3 days | disagreement rate on *live* traffic, latency under real load | today's traffic distribution, real load, real infra interactions |
| **Canary** | 5% | 3 days incl. one weekend | routing accuracy where resolution labels arrive, misroute-reopen rate, latency, cost | the effect on real users and downstream human workflows |
| **Ramp** | 25% → 50% → 100%, ≥24 h at each | ~4 days | same metrics, plus per-queue volume shifts | slow-moving effects and capacity behaviour at full load |

Two details that make this real rather than ceremonial. **Include a weekend in the canary** — support
traffic is different on weekends and it is exactly the kind of distribution shift an offline set
misses. And **watch per-queue volume**, not just accuracy: if `account-compromise` volume drops 30%
overnight, tickets are being routed away from it, and that shows up in volume long before it shows up
in a labelled accuracy metric.

## 6. Rollback trigger, owner, and proof

**Trigger (automatable, no judgement required).** Roll back immediately if **any** of:
- critical-queue disagreement rate > 0.5% over any 1-hour window with ≥200 critical tickets;
- `account-compromise` or `billing-dispute` hourly volume deviates > 25% from the trailing 4-week
  same-hour baseline for 2 consecutive hours;
- aggregate misroute-reopen rate > baseline + 1.0 pp over 6 hours;
- p99 latency > baseline + 20% for 15 minutes;
- any unhandled error-rate increase.

**Owner:** the on-call engineer executes; no approval required to roll back. Approval is required to
roll *forward*. That asymmetry is the point — make the safe action the frictionless one.

**Mechanism:** feature flag, previous fp16 weights kept warm in the serving fleet, target
< 5 minutes.

**How do I know the rollback works?** Because I **rehearsed it**: execute a rollback during the
shadow stage, on purpose, and time it. The problem statement says nobody has ever rolled back a
model, which means the rollback path is untested code that will be run for the first time during an
incident, by a tired person. A rehearsed rollback is the single cheapest risk reduction in this
entire document, and I would not start the canary without one.

## 7. Quantization-specific failure modes a generic gate would miss

1. **Long-input degradation from activation outliers.** Activation ranges grow with sequence length,
   and INT8 scaling that calibrates on typical inputs clips the extremes. Symptom: accuracy fine on
   average, degraded on the top 6% by length. **Caught by:** the long-ticket slice. This is why the
   numerics matter and why [B04](../../B-gpu-ml-performance/B04.precision-and-quantization/) is the
   companion reading.
2. **Calibration-set distribution mismatch.** PTQ calibrates scales on a sample. If that sample is
   71% English, the scales are tuned for English activation ranges and other scripts get worse.
   **Caught by:** the per-language slices — and prevented by calibrating on a traffic-weighted sample
   that includes them.
3. **Decision-boundary flips near the confidence threshold.** Quantization perturbs logits slightly;
   near-tie predictions flip. Aggregate accuracy barely moves because flips go both ways, but the
   *specific* predictions change. **Caught by:** the low-confidence-decile slice and the disagreement
   rate.
4. **Per-tensor vs per-channel scaling regressions** that appear only on particular layers.
   **Caught by:** the disagreement sweep, which is the most sensitive instrument here.

## 8. When I would abandon the rollout

- **Critical-queue recall cannot clear −0.5 pp** after reasonable calibration effort. $9k/month does
  not buy a regulatory incident, and this is a floor, not a negotiation.
- **The measured throughput win is materially below 1.7×** on production shapes. If the win is 1.15×,
  the fleet may not shrink at all after rounding (a real effect — see `cost_model.py`), and we would
  be taking quality risk for nothing.
- **Two iterations of calibration fail to fix a slice regression.** At that point the honest read is
  that this model does not quantize well at INT8; escalate to a less aggressive format
  (weight-only, or a higher-precision format) rather than continuing to tune.
- **The disagreement rate is high but the labelled metrics look fine.** That combination means the
  model's behaviour changed in ways your eval set does not measure — and shipping a change you cannot
  characterise is how you find out about it from a customer.

**The "it depends on X" pivot:** X is *the cost of a misroute in the two critical queues*. If those
queues have a downstream human safety net that catches misroutes within minutes, my thresholds are
too tight and I am blocking $9k/month for little benefit. If they do not, the thresholds are correct
and possibly still too loose. **I would establish that fact before setting a single number** — the
gate's calibration comes from the consequence, not from the statistics.
