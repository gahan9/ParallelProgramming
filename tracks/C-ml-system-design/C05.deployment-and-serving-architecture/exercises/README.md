# C05 Exercises — deployment & serving architecture

Three written design drills. Track C drills produce **documents**, not programs: the artefact a
system-design interview asks for, and the artefact a real design review consumes.

Write your answer first. Reading [`../solutions/`](../solutions/) before attempting the drill turns
a design exercise into a reading exercise, and the two skills are unrelated.

```bash
cd ..
make run-exercises      # lists the drills and verifies each has a reference solution
make run                # run the three analysis tools you will need for the numbers
```

| # | File | Skill it tests | Interview competency |
|---|------|----------------|----------------------|
| 1 | `01_design_the_serving_stack.md` | Choose a serving pattern from requirements; draw the architecture; produce a per-hop latency budget that closes; state the failure behaviour | End-to-end ML system design under a hard latency SLO |
| 2 | `02_rollout_plan.md` | Progressive delivery with numeric, automated rollback triggers and enough traffic per stage for the guardrail to have power | Release engineering and risk management |
| 3 | `03_ab_test_critique.md` | Find the randomization-unit error, the peeking, and the missing guardrail in someone else's test; then state the correct design | Experimental rigour — the skill that separates "we shipped it" from "we know it worked" |

---

## How these are graded

A strong answer is **specific and numeric**. Weak answers are recognizable at a glance: they name
technologies instead of making decisions, they quote an SLO without decomposing it, and they say
"we would monitor for regressions" without naming a metric or a threshold.

| Dimension | Weak | Strong |
|---|---|---|
| Pattern choice | "we'll use a REST API" | "synchronous online, because a human is waiting and the work fits in 250 ms; the explanation text streams separately over SSE" |
| Latency | "we'll target p99 < 250 ms" | a per-hop table summing to 168 ms with 33% headroom, p99 at every hop |
| Scaling | "autoscale on load" | "KEDA on pending-requests-per-replica, target 4, floor 2, warm pool 3 sized from a 40 rps/min ramp and a 180 s cold start" |
| Failure | "add retries" | a four-level fallback chain with stated quality per level, timeouts from downstream p99, full jitter, a retry budget of 10% |
| Rollback | "we can roll back" | "automated on 5xx > 1.5× baseline for 5 min, p99 > 1.2× baseline for 10 min, or guardrail down > 2%; previous version stays resident; rollback is a config change, measured at 90 s in staging" |
| Experiment | "we'll A/B test it" | randomization unit, MDE from business value, computed sample size, duration, guardrails, stopping rule, SRM check |

**Say what you are trading away.** Every one of these drills has a cost dimension, and an answer
that presents only benefits is incomplete. Reviewers notice.

---

## Tools you should use for the numbers

Do not estimate what you can compute:

```bash
python ../tools/ab_calculator.py size --baseline 0.08 --mde-rel 0.03 --daily-users 400000
python ../tools/capacity_autoscale.py --cold-start-s 120 --slo-ms 400
python ../tools/degradation_planner.py --config my_stack.json --shared-fate 0.3
```

An answer with computed numbers and stated assumptions beats an answer with confident adjectives,
every time.

---

## Expected effort

| Drill | Time | Deliverable |
|---|---|---|
| 1 | 60–90 min | ~2 pages: diagram, pattern rationale, latency table, scaling config, fallback chain |
| 2 | 45–60 min | ~1.5 pages: stage table, trigger table, sample-size justification, rollback rehearsal plan |
| 3 | 30–45 min | ~1 page: numbered list of defects with the mechanism for each, then a corrected test plan |
