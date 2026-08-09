# C02 Exercises

Three design drills covering the two failure modes that account for most disappointing ML launches:
**leakage** (your offline number was never real) and **skew** (your offline number was real and the
serving path throws it away).

Do them yourself before reading [`../solutions/`](../solutions/). Write your answers in the exercise
files or copy them somewhere private.

| # | File | Skill it drills |
|---|------|-----------------|
| 1 | [`01_find_the_leak.md`](01_find_the_leak.md) | Audit a real feature list: classify each feature safe / leaking / needs-a-question, and name the inflation mechanism |
| 2 | [`02_design_a_pipeline.md`](02_design_a_pipeline.md) | Choose batch/streaming/on-demand per feature, specify validation gates and failure policy, define splits and versioning |
| 3 | [`03_skew_postmortem.md`](03_skew_postmortem.md) | Structured diagnosis of "offline AUC 0.94, online flat" — causes in priority order, the confirming check for each, and the incident writeup |

Interview mapping: (1) data-correctness reasoning and attention to detail, (2) end-to-end system design
under cost and latency constraints, (3) debugging a production ML system under ambiguity. Exercise 3 is
the closest thing in this curriculum to a real ML on-call incident.

---

## Running the evidence

Exercises 1 and 3 reference tool output. Generate it with:

```bash
cd ..
make run-exercises PYTHON=python      # drop PYTHON= on Linux/macOS if python3 is your default
```

Or run the tools directly:

```bash
python ../tools/leakage_detector.py --demo
python ../tools/split_demo.py --verbose
python ../tools/data_validator.py --demo
```

All three are standard-library Python 3 — no `pip install`. `data_validator.py` and
`leakage_detector.py` **exit 1 when they find violations**: they are pipeline gates, and that is the
intended behaviour, not a failure of the tool.

To experiment on files you can edit:

```bash
cd .. && make demo-files PYTHON=python     # writes _demo/schema.json, _demo/batch.csv, _demo/features.csv
```

---

## Sample tool output you will be reasoning about

All fixtures use fixed seeds, so **these numbers reproduce exactly on any machine.**

### Leakage screen (`leakage_detector.py --demo`)

Three planted leaks, each caught by a **different** check — which is the point of running all three:

| Feature | Timestamp check | Separation check | Naming check |
|---|:---:|:---:|:---:|
| `investigator_notes_len` | **caught** | caught | missed |
| `chargeback_amount` | **caught** | caught | caught |
| `case_status_at_review` | missed | **caught** | missed |

`case_status_at_review` has a *correct* timestamp (stamped at the decision moment) and an innocent
name; only the separation check finds it. Real leakage often arrives with perfect metadata.

### Split comparison (`split_demo.py`)

One dataset, one scoring rule per feature, three splits. Every difference is the split:

```text
  feature                  random   temporal  grouped
  x_signal                 0.7274    0.6976   0.7001
  x_drifting               0.5080    0.3310   0.5207
  x_user_mean_global       0.7879    0.7669   0.7949
  x_user_mean_trainonly    0.6913    0.6589   0.5000
```

> [!NOTE]
> Three separate lessons in that table. `x_user_mean_trainonly` loses **0.19 AUC** going from random to
> grouped — pure entity memorization. `x_drifting` falls to **0.331**, i.e. *anti-predictive*, under a
> temporal split, because its relationship to the label reverses over time. And
> `x_user_mean_global` — a target encoding computed over the whole dataset — stays high under **all
> three** splits, because the leak happened before the split and no partition can undo it. A number
> that is stable across splits is therefore not evidence of validity. Gaps below about 0.04 are
> sampling noise at this test-set size (~1,800 test rows).

### Validation gate (`data_validator.py --demo`)

```text
  [ERROR] drift              amount             PSI 1.7402 exceeds the declared maximum 0.2500; mean 65.5190 -> 225.7662 (+2.38 reference sd)
  [ERROR] drift              country            2 share(s) moved by more than 0.150
            example: IN: 0.063 -> 0.473 (+0.410)
  [ERROR] drift              label              2 share(s) moved by more than 0.150
            example: 1: 0.110 -> 0.293 (+0.183)
  [WARN ] unexpected-column  experiment_arm     present in the data but not declared; a silent upstream schema change
  [INFO ] drift              account_age_days   PSI 0.1190 within tolerance; mean 1391.8567 -> 1490.8167 (+0.12 reference sd)

summary: 12 error(s), 1 warning(s)
GATE: REJECT - do not publish this batch downstream
```

> [!TIP]
> The row that matters most is the **label** drift: the positive rate moved from 11% to 29% with a
> perfectly valid schema. No per-row check would notice, and a retrain on this batch would silently
> shift everything downstream — including the cost-optimal threshold you derived in
> [C01](../../C01.problem-framing-and-metrics/README.md). The
> `account_age_days` INFO row is the control proving the gate is not simply flagging everything.

---

## What "good" looks like

A strong answer to any of these:

1. **Distinguishes "this leaks" from "I need to ask someone."** Most real feature audits end with
   questions, not verdicts, and naming the question and its owner is the senior move.
2. **Names the mechanism, not just the verdict.** *How* would this feature inflate the offline metric?
   Which check catches it? What pipeline change prevents the whole class?
3. **Chooses a regime per feature with a stated reason**, and does not put everything on streaming.
4. **Specifies what happens on validation failure**, per table, including who gets paged.
5. **Ranks hypotheses by likelihood × cost-to-check**, and names the specific check that would
   eliminate each — not "investigate the data."
6. **Separates "the measurement was wrong" from "the system is wrong."** They need completely
   different fixes and completely different conversations with stakeholders.
7. **Gives the "it depends on X" pivot** for every contested choice, with a specific X.
