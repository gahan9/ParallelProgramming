# D03 Exercises - GPU Collectives with NCCL and RCCL

Four drills. Two are Python and run on any laptop; two are HIP/RCCL and need one node
with at least two GPUs. Work them before opening [`../solutions/`](../solutions/): the
reasoning is what transfers, and each solution's header is the part being assessed.

Every starter **compiles and runs as given**, and then **fails its own checks**, because
the parts you are asked to write are marked `TODO` stubs. That gap is deliberate - you
should never be debugging the harness, only your own code. A `FAIL` on the first run
means the setup works.

| # | File | Skill |
|---|------|-------|
| 1 | [`ex1_busbw_convert.py`](ex1_busbw_convert.py) | The traffic factors `f(N)`; algbw versus busbw; refusing bad input without a traceback |
| 2 | [`ex2_allreduce_verify.cpp`](ex2_allreduce_verify.cpp), [`ex2_allreduce_verify.cu`](ex2_allreduce_verify.cu) | A checked, verified collective from scratch; the group calls and the stream synchronization, by ablation |
| 3 | [`ex3_crossover_sweep.cpp`](ex3_crossover_sweep.cpp) | Finding the plateau and the knee on your own hardware, and predicting the knee from alpha-beta |
| 4 | [`ex4_report_auditor.py`](ex4_report_auditor.py) | Auditing a benchmark report you did not produce, including one with an unlabelled bandwidth column |

Each maps to a competency a distributed-training review asks for: bandwidth literacy,
collective correctness under asynchrony, measurement on real hardware, and the ability to
disbelieve a number politely.

## Building and running

The two Python drills self-test. Run them with no arguments and read the failures:

```bash
cd tracks/D-distributed-computing/D03.gpu-collectives-nccl-and-rccl/exercises

python -S -I ex1_busbw_convert.py          # FAIL until you implement the TODOs
python -S -I ex4_report_auditor.py         # FAIL until you implement the TODOs

# then use them like tools, once they pass
python -S -I ex1_busbw_convert.py --collective allreduce --ranks 8 \
       --bytes 1073741824 --time-ms 40
python -S -I ex4_report_auditor.py --input ../synthetic_allreduce.log --ranks 4
```

`-S -I` is not decoration. It disables `site` and isolates the interpreter, so a drill
that quietly grew a dependency on numpy fails here rather than on the reviewer's machine.
These tools must run on a login node with nothing installed.

The two GPU drills are one compiler invocation each. The module `Makefile` one level up
will do it (`make exercises`), or by hand:

```bash
hipcc -O2 -Wall -Wextra ex2_allreduce_verify.cpp -lrccl -o ex2
hipcc -O2 -Wall -Wextra ex3_crossover_sweep.cpp  -lrccl -o ex3
nvcc  -O2 ex2_allreduce_verify.cu -lnccl -o ex2      # NVIDIA instead

./ex2                                                  # needs 2+ GPUs
./ex3 --min-bytes 8 --max-bytes 268435456 --factor 4 --iters 20
```

Warnings count as failures: build with `-Wall -Wextra` and keep it silent. And every
runtime and library call goes through a checking macro - in the drills exactly as in the
labs. That is not style. A collective that fails on one rank and is not checked shows up
as a hang on the other seven, minutes later, with nothing naming the cause.

> [!NOTE]
> There is no committed benchmark log for drill 4, on purpose: a checked-in performance
> file would be a number this repository cannot reproduce. Generate a **synthetic** one
> from the tool's own model
> (`python -S -I ../tools/parse_collective_report.py --emit-example --ranks 4`), or better,
> run lab 2 and audit your own report. Two flags make the fixture exercise the hard paths:
> `--example-error` corrupts one row so the audit has something to catch, and
> `--example-one-column busbw` emits the single unlabelled bandwidth column that the second
> half of drill 4 is about. The synthetic file exists to exercise the parser and nothing
> else: do not quote a number out of it.

## What each drill is really testing

**Drill 1 - the factor, not the formula.** `algbw = m/t` is division. The content is
`f(N)`, and specifically that it is different for every collective and moves with the rank
count, which is why a `busbw` quoted without its `N` cannot be checked by anyone. Two
places where a careless implementation goes wrong. `ranks == 1` makes the all-reduce
factor `2(N-1)/N` equal zero: decide whether that is an error, a factor of 1, or a factor
of 0, and *defend it in a comment* - the drill is the defence, not the branch. And
`allgather` and `reducescatter` share the factor `(N-1)/N` while doing opposite things, so
if your table has them differing you have derived one of them from intuition instead of
from bytes on the wire.

**Drill 2 - the two things you cannot see.** Write lab 1 without looking at it, then take
it apart. Remove `ncclGroupStart`/`ncclGroupEnd` and one thread issuing N collectives
deadlocks, because each call waits for peers that the same thread has not reached yet.
Remove the `hipStreamSynchronize` and the verification reads a buffer the collective has
only been *enqueued* to fill - which frequently passes at small sizes, and that is the
lesson rather than an inconvenience. Record what each omission did **on your machine**,
and notice that neither omission produces an error return: this is exactly the class of
bug that the checking macros cannot catch for you, which is why the ablation is the
exercise.

Verify with a tolerance, not `==`. The library's reduction order depends on rank count,
algorithm and channel count, so an exact comparison is testing something the library never
promised (module README section 3.8).

**Drill 3 - the plateau is not `max(busbw)`.** Taking the maximum of a noisy series
estimates the maximum of the noise; it is biased high by construction, and more so when
the sweep is short or the machine is busy. Use the flat tail, and be willing to report
"this sweep did not plateau, raise `--max-bytes`" instead of a confident number. Then the
prediction: estimate `alpha` from the smallest row, `beta` from the plateau, predict the
90% size, and compare. The measured size usually lands *below* the prediction, because the
model has one `beta` and the library does not - it crosses LL to LL128 to Simple over
exactly that range (README section 3.5), so its effective `beta` improves as the message
grows. Explaining the gap is the deliverable; the number is not.

**Drill 4 - auditing a number you did not produce.** Recompute `busbw` from `size` and
`time` and compare. The subtlety is the tolerance: reports print two decimals, so at small
sizes the rounding quantum is larger than any plausible error and a pure relative
comparison flags every small row. Compare against **both** a relative tolerance and the
column's own quantum. Then the case that motivates the whole tool - a report with one
unlabelled bandwidth column. Test it against both candidates, say which one it looks like,
and say that you are inferring. Note where inference is impossible: at `f(N) ~ 1`, which
is `broadcast`, `reduce`, or all-reduce at two ranks, the two columns are numerically the
same and an honest tool says so rather than guessing.

## Reference

- Parent module: [`../README.md`](../README.md). Section 2.4 (the factor table), 3.5
  (protocols), 3.7 (streams, groups, and the deadlock), 3.8 (non-associativity) and 5.1
  (reading a report) contain everything these drills need.
- Tools to compare against once you are done: [`../tools/busbw.py`](../tools/busbw.py) and
  [`../tools/parse_collective_report.py`](../tools/parse_collective_report.py). Read them
  *after* your version works.
- Terms: [`GLOSSARY.md`](../../../../docs/GLOSSARY.md) -
  [algbw and busbw](../../../../docs/GLOSSARY.md#algbw-and-busbw),
  [collective operation](../../../../docs/GLOSSARY.md#collective-operation),
  [ring all-reduce](../../../../docs/GLOSSARY.md#ring-all-reduce),
  [the alpha-beta model](../../../../docs/GLOSSARY.md#the-alpha-beta-model),
  [stream](../../../../docs/GLOSSARY.md#stream).
- Sources: [`REFERENCES.md`](../../../../docs/REFERENCES.md) -
  [NCCL](../../../../docs/REFERENCES.md#nccl-nvidia) and
  [RCCL](../../../../docs/REFERENCES.md#rccl-amd) for the API and its guarantees,
  [nccl-tests and rccl-tests](../../../../docs/REFERENCES.md#nccl-tests-and-rccl-tests-nvidia-amd)
  for the report format drill 4 parses, and
  [Patarasuk and Yuan](../../../../docs/REFERENCES.md#bandwidth-optimal-all-reduce-patarasuk-and-yuan-2009)
  for why the ring factor is what it is.
- Setup: [`SETUP.md`](../../../../docs/SETUP.md).

Reference answers with full working: [`../solutions/`](../solutions/).
