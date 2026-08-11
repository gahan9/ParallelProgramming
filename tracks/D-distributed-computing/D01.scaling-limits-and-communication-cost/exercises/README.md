# D01 Exercises — Scaling Limits & the Cost of Talking

Four coding drills, one per mechanism that bends a scaling curve: the serial fraction,
the latency term, the byte term, and variance. They are the "Coding & Algorithms" drills
from [section 9](../README.md#9-self-assessment--interview-drills) of the module, turned
into starter files with TODOs.

Each file **runs as it ships**. It prints its table with `not implemented` in every cell
until you fill the TODOs in, so you always have a working harness and never a blank page.
Standard library only — no numpy, no torch, nothing to install.

| Drill | File | What it tests |
|---|---|---|
| 1 | [`01_efficiency_bisection.py`](01_efficiency_bisection.py) | Amdahl speedup, efficiency, and the largest rank count that holds a target — by bisection |
| 2 | [`02_crossover_finder.py`](02_crossover_finder.py) | `m* = alpha / beta`, and classifying a message as latency-bound, balanced or bandwidth-bound |
| 3 | [`03_algorithm_selection.py`](03_algorithm_selection.py) | Ring against recursive doubling, the winner, and where the tie sits |
| 4 | [`04_straggler_simulation.py`](04_straggler_simulation.py) | Barrier step time as a maximum, and why jitter is a scaling term |

## How to work these

1. **Do the arithmetic by hand first.** Every formula you need is derived in the module
   README, [section 2](../README.md#2-first-principles) for the two laws and
   [section 3](../README.md#3-deep-dive) for the cost model. Reaching for the tool before
   you have a number practises the wrong skill.
2. **Then check against the tools.** `tools/scaling_model.py`, `tools/message_cost.py` and
   `tools/collective_choice.py` implement the same closed forms, so a disagreement means
   one of the two has a bug. Find out which before moving on.
3. **Answer the questions at the bottom of each file.** They are the part an interview
   actually asks. The code is a means of getting the number; the question is what the
   number licenses you to say.

```powershell
# From the module directory (one level up):
make run-exercises                  # runs all four harnesses
python exercises/01_efficiency_bisection.py
python tools/scaling_model.py --help
```

## Acceptance criteria

A drill is done when its harness prints these, and you can defend the answers at the
bottom of the file. Every figure below is what the reference solution prints, so a
mismatch is a bug in one of the two.

**Drill 1 — efficiency and the wall.**

- The efficiency column agrees with
  `python tools/scaling_model.py --serial-fraction 0.02 --ranks 1024`: 87.7% at 8 ranks,
  44.2% at 64, 4.7% at 1024.
- `largest_ranks_at_efficiency(0.02, 0.5)` returns **51**, and the harness reports that the
  closed form `(1/target - 1 + s)/s` agrees. N=51 is an exact tie — efficiency there is
  precisely 0.5 — so the comparison has to be `>=` and not `>`.
- The tolerance is not decoration. Try `s = 0.001` against a 40% target: the closed form
  says 1501, and a bare `>=` says 1500, because the efficiency at 1501 evaluates to
  0.39999999999999997 in binary floating point. Confirm with
  `python tools/scaling_model.py --serial-fraction 0.001 --ranks 2048 --target-efficiency 0.4`,
  which reports 1501.
- All four rejection cases raise `ValueError` with a one-line message that names the
  offending value. A traceback, a silent `nan`, or a bare `raise ValueError()` all fail.

**Drill 2 — the crossover.**

- `m*` is 300000 bytes on the intra-node link, 62500 on the RDMA fabric, and 1000000 on
  the Ethernet link.
- A 4 KiB message is latency-bound on **all three**. A 64 KiB message is latency-bound on
  the intra-node link, balanced on RDMA, latency-bound on Ethernet. A 1 MiB message is
  bandwidth-bound on both fast links and balanced on Ethernet.
- `crossover_bytes` rejects a non-positive beta, and `classify` rejects a negative size.

**Drill 3 — algorithm selection.**

- Both costs are **exactly** `0.0` at `ranks == 1`, not negative and not one step's worth
  of latency.
- Recursive doubling wins every cell at 1 KiB and 64 KiB. The ring wins at 16 MiB and
  1 GiB for every rank count above 2.
- The tie sits at 550000 bytes for N=8, 1860465 bytes for N=64 and 9039598 bytes for
  N=512, matching the tie printed by
  `python tools/collective_choice.py --ranks 64 --bytes 4096`. A bisection that stops when
  the bracket is under a byte may land one byte low; a bisection that disagrees by more
  than that has an inverted comparison.
- There is no tie at N=1 or N=2, and the function says so rather than returning a number.

**Drill 4 — variance amplification.**

- With the shipped `SEED` and `TRIALS`, the expected slowest rank is 1.030x the mean at
  8 ranks, 1.049x at 64, 1.066x at 1024 and 1.080x at 16384, losing 2.9% to 7.4% of
  throughput across that range. That reproduces the table in
  [section 3.4](../README.md#3-deep-dive) of the module.
- `barrier_step_time` is the maximum, not the mean, and rejects an empty sequence.
- The run is reproducible: two runs of the same file print identical numbers.

## Shared assumptions unless a drill says otherwise

| Quantity | Value | Why this one |
|---|---|---|
| Cost model | `T = alpha + m * beta`, `beta = 1 / bandwidth` | [the alpha-beta model](../../../../docs/GLOSSARY.md#the-alpha-beta-model), from [Hockney 1994](../../../../docs/REFERENCES.md#the-hockney-alpha-beta-model-hockney-1994) |
| Bandwidth unit | GB/s with 1 GB = 1e9 bytes | the convention link vendors quote |
| Reference link | `alpha` 2.5 us, 25 GB/s | inter-node RDMA order of magnitude, per module section 3.1 |
| Ring all-reduce | `2(N-1)` steps, `2m(N-1)/N` bytes per rank | [ring all-reduce](../../../../docs/GLOSSARY.md#ring-all-reduce); the byte count is a proven lower bound ([Patarasuk and Yuan 2009](../../../../docs/REFERENCES.md#bandwidth-optimal-all-reduce-patarasuk-and-yuan-2009)) |
| Recursive doubling | `ceil(log2 N)` steps, `m * ceil(log2 N)` bytes per rank | textbook power-of-two form; real libraries add steps for other rank counts |
| Jitter | per-rank step time Normal, 2% standard deviation | illustrative, and the reason a [straggler](../../../../docs/GLOSSARY.md#straggler) is a scaling term |

Everything these drills compute is an **analytic model, never a measurement**, and every
harness says so in its banner. The models are right about ratios, orderings and orders of
magnitude, and silent about congestion, injection limits, topology and algorithm
switching inside a real collective library.

Reference answers with commentary: [`../solutions/`](../solutions/). Read them *after* you
have written yours down — they contain the reasoning you are being assessed on, not just
the code.
