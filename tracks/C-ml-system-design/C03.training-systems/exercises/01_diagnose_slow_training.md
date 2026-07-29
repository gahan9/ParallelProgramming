# Exercise 1 — Diagnose a slow training run

> Reference answer: [`../solutions/01_diagnose_slow_training_solution.md`](../solutions/01_diagnose_slow_training_solution.md)
> — write yours first.

## The situation

You are on-call. A team pings you: *"Training is slow and we don't know why. We've been tuning the
learning rate for two days."*

Here is everything you are told, verbatim from their message and dashboards:

- The job runs on **128 GPUs**, 16 nodes, synchronous data-parallel.
- It was **2.1× faster last month** on the same code and the same cluster.
- **Mean GPU utilization: 61%.** The utilization chart oscillates — it spikes to ~100%, drops to
  near 0%, and repeats, several times a minute.
- **Mean step time: 1.4 s. p99 step time: 6.8 s.** The p99 got worse when they went from 64 to
  128 GPUs.
- The dataset was **re-exported three weeks ago** "to add a new field". Nobody remembers the format
  changing.
- Host CPU utilization on each node: **~95% across all cores.**
- Host RAM: 60% used. GPU memory: 45% used, flat.
- Loss is decreasing normally. No NaNs. No crashes.
- The job restarts about **twice a day**, each time resuming from a checkpoint, and each restart is
  followed by "a little bump in the loss" that recovers within a few hundred steps.

## Your task

**1. Order the checks.** List the checks you would run, **in order**, and for each one say what
result would confirm or eliminate a hypothesis. The order matters more than the list — justify it.
A good ordering front-loads checks that are cheap, non-invasive, and maximally discriminating.

**2. Name the most likely root cause** and cite the specific evidence above that points to it.
Then name the *second* most likely, because there is more than one problem in this report.

**3. Explain the p99 growth with scale.** Why did the tail get worse going from 64 to 128 GPUs when
the per-rank work did not change? Be precise about the mechanism.

**4. Explain the loss bump at restart.** The team calls it harmless. Is it? What is the most likely
cause, and what would you ask to see to confirm it?

**5. Give the fix list**, ordered by (expected improvement ÷ effort). For each fix, state what it
costs — engineering time, storage, accuracy, or complexity. At least one of your entries should be
something you would explicitly *decline* to do, with the reason.

**6. State the "it depends on X" pivot.** Name one fact you were not told that would materially
change your top recommendation, and say how.

## Constraints on your answer

- No answer may be "add more GPUs". Explain why that is the wrong move here.
- You may use `../tools/throughput_budget.py` to sanity-check the arithmetic. State the numbers you
  fed it and note that its output is a model, not a measurement.
- Keep it to one page. A triage plan nobody reads at 3 a.m. is not a triage plan.
