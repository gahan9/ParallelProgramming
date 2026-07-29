# B06 Exercises

Three paper drills. No GPU and no compiler required — but you *do* need to show the arithmetic,
because "we'll batch more" without a number is exactly the answer that fails a serving interview.

Do these yourself before opening [`../solutions/`](../solutions/).

| # | File | Skill it tests |
|---|------|----------------|
| 1 | [`01_capacity_plan.md`](01_capacity_plan.md) | Compute max concurrency from HBM; name the binding resource. |
| 2 | [`02_slo_batching.md`](02_slo_batching.md) | Turn a TTFT/TPOT budget into a batch-size cap and defend it. |
| 3 | [`03_pick_the_optimization.md`](03_pick_the_optimization.md) | Triage five production symptoms; say what you would measure. |

## Checking your work with the models

The three simulators in [`../sim/`](../sim/) will confirm (or embarrass) your arithmetic:

```bash
make -C .. run-exercises        # the exact runs these drills refer to
# Windows: make -C .. run-exercises PYTHON=python
```

Or individually:

```bash
python ../sim/kv_cache_planner.py --preset gqa-70b --gpu mi300x --context 8192 \
    --mean-context 2050 --compute-batch-cap 128
python ../sim/serving_sim.py --policies continuous_chunked --rate 20 \
    --slo-ttft-ms 2000 --slo-tpot-ms 12 --sweep-batch 4,8,16,24,32,48,64,128
python ../sim/speculative_model.py --alpha 0.6 --k 4
```

> [!IMPORTANT]
> Everything the simulators print is a **model output** from an analytic cost model, not a
> measurement on hardware. Use them to check *reasoning* and *ratios*. Never quote their absolute
> milliseconds as if they were benchmark results — that habit is how bad capacity plans get
> approved.

## Rules of engagement

1. **Write the units.** A number without units (`GiB`? `GB`? per GPU or per node?) is not an answer.
2. **State your assumptions explicitly.** Every real capacity plan rests on an assumed sequence-length
   distribution and an assumed arrival rate. If you hide them, nobody can check you.
3. **Name what you would measure.** For every recommendation, give the one metric that would prove
   you right and the one that would prove you wrong.
