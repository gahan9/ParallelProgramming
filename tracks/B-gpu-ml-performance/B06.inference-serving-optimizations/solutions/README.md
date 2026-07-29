# B06 Solutions

Reference answers with the reasoning, not just the results. Read these *after* attempting
[`../exercises/`](../exercises/) — the arithmetic is the point, and reading someone else's arithmetic
does not transfer.

| # | Solution | Headline |
|---|----------|----------|
| 1 | [`01_capacity_plan_solution.md`](01_capacity_plan_solution.md) | 70B fp16 on one MI300X: **67** concurrent sequences paged vs **16** contiguous. The weights eat 73% of HBM, so HBM is the binding resource — and fp8 KV flips it to latency-bound. |
| 2 | [`02_slo_batching_solution.md`](02_slo_batching_solution.md) | The latency curve says cap **81**; simulated goodput peaks at **32**. The gap is the lesson. |
| 3 | [`03_pick_the_optimization_solution.md`](03_pick_the_optimization_solution.md) | Five symptoms → five mechanisms. Only one of them is a speculative-decoding problem. |

## Reproducing every number here

```bash
make -C .. run-exercises        # Windows: append PYTHON=python
```

> [!IMPORTANT]
> Every millisecond, token/s and GiB in these solutions is either (a) exact arithmetic from the
> stated model configuration, or (b) an output of the analytic simulators in [`../sim/`](../sim/).
> **None of it is a hardware measurement.** Where these solutions quote a *published* result — the
> vLLM paper's throughput gain, Orca's, Sarathi-Serve's — it is attributed to that paper and to that
> paper's conditions, and it is not comparable to the simulator's output.

## The three habits these drills are drilling

1. **Name the binding resource before proposing a fix.** Half of all serving work is spent
   optimizing something that was never the limit.
2. **Optimize goodput, not throughput.** Throughput is a number you can always make bigger by
   making users wait. Goodput is the number the business has.
3. **State the falsifier.** "TPOT is high because the batch is too big" becomes engineering only
   when you add "and if I halve the batch and TPOT does not move, I was wrong — it is the ITL tail
   from prefill interference."
