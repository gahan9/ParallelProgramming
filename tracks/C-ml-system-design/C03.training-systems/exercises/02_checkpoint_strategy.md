# Exercise 2 — Design a checkpoint strategy

> Reference answer: [`../solutions/02_checkpoint_strategy_solution.md`](../solutions/02_checkpoint_strategy_solution.md)
> — write yours first.

## The situation

You own the training infrastructure for a **256-GPU** run that needs **720 GPU-hours per GPU** of
useful compute (roughly 30 days of wall clock if nothing goes wrong).

Measured facts about your environment:

| Fact | Value |
|---|---|
| Mean time between job-killing failures, on-demand cluster | **9 hours** |
| Full checkpoint size | 1.4 TB (weights + Adam optimizer state) |
| Single-writer checkpoint write time | **240 s** |
| Ranks that can write shards in parallel before the filesystem saturates | **16** |
| Restart: detect + requeue + reload | **14 minutes** |
| Step time | 0.9 s |
| On-demand price | $2.40 / GPU-hour |
| Spot price | $0.75 / GPU-hour |
| Mean time between **spot interruptions** | **45 minutes** |
| Spot interruption warning | 120 s |

## Your task

**1. On-demand baseline.** Compute the optimal checkpoint interval `T*` for the on-demand cluster
using `T* = sqrt(2·C·M)`. Show the arithmetic. Express the answer in **both** minutes and steps, and
state the resulting total overhead and goodput.

**2. Make the checkpoint cheaper.** Recompute `T*` and the overhead for:
   - (a) sharded across 16 writers,
   - (b) sharded **and** asynchronous, where the loop blocks only for the device→host snapshot,
     which you measure at 10% of the write time.

   For each, state the *new constraint* the strategy introduces. For (b) in particular: what must be
   true about the interval relative to the background write, and what goes wrong if it is not?

**3. The spot decision.** Redo the analysis at a 45-minute MTBF and $0.75/GPU-hour. Then answer the
actual business question: **is spot cheaper than on-demand for this job?** Show the comparison as
effective dollars per hour of *useful* compute, not per wall hour — that distinction is the whole
exercise.

**4. Use the 120-second warning.** A spot interruption gives you 120 s of notice. Design what the
job does in those 120 seconds. Does it change your answer to (3)? Quantify roughly how much, and
state what could make the graceful-shutdown path fail exactly when you need it.

**5. What must be in the checkpoint.** List every piece of state required for a **bit-exact** resume
of this job. For each, state the failure symptom if it is missing. Then say which ones you would
still include even if you only needed *statistical* resumability, and why.

**6. Retention.** You cannot keep 1.4 TB every 15 minutes. Design a retention policy, and justify it
in terms of what each retained checkpoint is *for*.

**7. The failure mode nobody plans for.** 200 other jobs share your filesystem, and every one of
them uses the same formula. Describe what happens and how you would prevent it.

## Constraints on your answer

- Use `../tools/checkpoint_planner.py` to check your arithmetic, but do the algebra by hand first —
  in an interview you will not have the tool.
- State every assumption you add.
- End with a one-paragraph recommendation you would put at the top of a design doc: a decision, its
  cost, and the condition under which you would revisit it.
