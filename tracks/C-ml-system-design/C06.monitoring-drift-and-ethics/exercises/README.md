# C06 Exercises — monitoring, drift & ethics

Three written design drills. Track C drills produce **documents**: a monitoring plan, an
investigation write-up, and a fairness decision memo — the three artefacts this job actually
produces.

Write your answer first. Reading [`../solutions/`](../solutions/) before attempting the drill turns
a design exercise into a reading exercise.

```bash
cd ..
make run-exercises      # lists the drills and verifies each has a reference solution
make run                # run the three tools; drills 2 and 3 need their numbers
```

| # | File | Skill it tests | Interview competency |
|---|------|----------------|----------------------|
| 1 | `01_design_monitoring.md` | Four-layer monitoring with alert-versus-dashboard discipline, numeric thresholds and a runbook per alert | Production ML ownership — the difference between "we have dashboards" and "we would find out" |
| 2 | `02_diagnose_the_drift.md` | Reason from evidence to a drift *type*, then to the cheapest fix that addresses it | Debugging under uncertainty with incomplete labels |
| 3 | `03_fairness_tradeoff.md` | Name the violated criterion, quantify the intervention's cost, and justify the choice to a non-technical stakeholder | Technical judgement on a contested question, plus the communication that makes it land |

---

## How these are graded

A strong answer is **specific, numeric, and honest about what it cannot know**. The characteristic
weak answer to all three is confident vagueness: "monitor for drift", "the data changed", "make the
model fairer".

| Dimension | Weak | Strong |
|---|---|---|
| Alerting | "alert on drift" | "page on prediction-distribution mean outside the 30-day ±3σ band for 15 min; dashboard PSI weekly; never page per-feature" |
| Thresholds | "if it degrades significantly" | "PSI > 0.25 on a top-20-importance feature, or fallback rate > 5% for 10 min" |
| Diagnosis | "there's drift" | "concept drift: inputs PSI 0.008, predictions PSI 0.004, accuracy −8 pp; only P(y\|x) can have moved" |
| Fix | "retrain" | "retrain on post-change data with time-decay; reweighting cannot fix a moved target function; expected cost 1 training run + a full canary" |
| Fairness | "we should reduce bias" | "equal opportunity is violated (TPR gap 0.21). Equalizing costs 0.9 pp of accuracy and widens the precision gap to 0.15. Recommend it because a false negative denies an opportunity, and here is what I would tell the VP" |
| Uncertainty | silence | "labels arrive in 34 days, so I cannot confirm concept drift now; here is the proxy I would use and what it would and would not prove" |

**Say what you cannot know.** In this module that is not hedging — the delayed-label problem means
uncertainty is the normal operating condition, and an answer that pretends otherwise is wrong.

---

## Tools you should use for the numbers

```bash
python ../tools/drift_detector.py --config my_windows.json
python ../tools/feedback_loop_sim.py --epsilons 0.0 0.01 0.05
python ../tools/fairness_metrics.py --data my_scores.csv --threshold 0.5
# and, for monitoring power:
python ../../C05.deployment-and-serving-architecture/tools/ab_calculator.py size \
        --baseline 0.12 --mde-rel 0.05
```

---

## Expected effort

| Drill | Time | Deliverable |
|---|---|---|
| 1 | 60–90 min | ~2 pages: four-layer table, alert/dashboard split with thresholds, runbooks |
| 2 | 30–45 min | ~1 page: diagnosis with the evidence chain, ruled-out alternatives, the fix and its cost |
| 3 | 45–60 min | ~1.5 pages: metric analysis, criterion choice with justification, cost statement, and the stakeholder script |
