# D03 Solutions - GPU Collectives with NCCL and RCCL

Worked solutions for the four drills in [`../exercises/`](../exercises/). Read these after
you have written your own: each file carries its argument in a header comment rather than
in the code, and the argument is the part that transfers.

| # | Solution | The one idea to take away |
|---|----------|---------------------------|
| 1 | [`ex1_busbw_convert.py`](ex1_busbw_convert.py) | `algbw` is what your application sees, `busbw` is what the wire sees; only the second one can be compared to a link rate, and only with its rank count attached |
| 2 | [`ex2_allreduce_verify.cpp`](ex2_allreduce_verify.cpp), [`ex2_allreduce_verify.cu`](ex2_allreduce_verify.cu) | The group calls prevent a deadlock and the stream synchronization prevents a wrong answer; neither omission returns an error |
| 3 | [`ex3_crossover_sweep.cpp`](ex3_crossover_sweep.cpp) | The plateau is the flat tail, not the maximum; the 90% size is the bucket size, and it sits well above the textbook crossover |
| 4 | [`ex4_report_auditor.py`](ex4_report_auditor.py) | Compare against the rounding quantum as well as a relative tolerance, and say out loud when you are inferring |

```bash
cd tracks/D-distributed-computing/D03.gpu-collectives-nccl-and-rccl/solutions

python -S -I ex1_busbw_convert.py          # self-test: all cases pass
python -S -I ex4_report_auditor.py         # self-test: all cases pass

hipcc -O2 -Wall -Wextra ex2_allreduce_verify.cpp -lrccl -o ex2
hipcc -O2 -Wall -Wextra ex3_crossover_sweep.cpp  -lrccl -o ex3
nvcc  -O2 ex2_allreduce_verify.cu -lnccl -o ex2      # NVIDIA instead

./ex2                                                  # needs 2+ GPUs, prints PASS
./ex3 --min-bytes 8 --max-bytes 268435456 --factor 4 --iters 20
```

## Drill 1 - algbw, busbw, and the factor

The division is trivial; three decisions are not.

- **The factors are derived from bytes on the wire, not remembered.** The solution stores
  each one as a small callable next to the sentence that justifies it, so
  `allgather = (N-1)/N` reads as "each rank receives every other rank's share exactly
  once" rather than as a magic constant. That is also how you catch the classic error:
  `allgather` and `reducescatter` have the *same* factor while doing opposite things,
  because reduce-scatter is an all-gather run backwards with an add.
- **`ranks == 1` returns a factor of 0, for every collective, and is not an error.** With
  one rank there is no wire, so the bytes on it are zero and `busbw` is genuinely zero -
  including for `broadcast`, where the tempting answer is 1. The alternatives are worse: an
  exception makes a legitimate single-GPU debug run crash a sweep script, and a factor of 1
  reports a bandwidth for a transfer that did not happen. This is a real decision with a
  real cost either way, which is why the drill asks for the defence and not the branch, and
  it is why [`../tools/busbw.py`](../tools/busbw.py) makes the same choice - two tools in
  one module disagreeing about `N = 1` is its own bug.
- **Refusing bad input is part of the deliverable.** One line naming what was wrong, exit
  non-zero, no traceback. A traceback tells the user about *your* code when they wanted to
  know about their input.

The conversion also states which direction it went - measured time in, bandwidths out -
because a tool that consumes measurements must not print the model banner that
[`../tools/multinode_allreduce_model.py`](../tools/multinode_allreduce_model.py) prints.
The distinction between "this is arithmetic about hardware I do not have" and "this is your
measurement, restated" is the whole reason the banner exists.

## Drill 2 - a checked, verified collective, and the two ablations

The code is short. What it has to get right:

- **Every runtime and library call is wrapped**, and the macro prints `file:line`, the
  expression, and the library's own error string. An unchecked failure on one rank becomes
  a hang on the others, minutes later, with nothing naming the cause - the checking is what
  converts a distributed mystery into a local error message.
- **A launch-error check after the kernel launch.** A launch does not return an error
  synchronously; `hipGetLastError` / `cudaGetLastError` immediately after is the only place
  a bad launch configuration is visible before it turns into a wrong result.
- **A bounds guard in the kernel.** The grid is rounded up to whole blocks, so the last
  block always has threads past the end of the buffer.
- **Verification against the closed form with a tolerance.** The expected value is
  `sum(1..N) = N(N+1)/2`, so no host-side reduction is needed and the reference cannot
  itself be wrong. The tolerance is required, not defensive: the library's reduction order
  depends on rank count, algorithm and channel count, so an exact comparison asserts
  something no implementation promises (module README section 3.8).
- **Everything freed, every communicator and stream destroyed**, including on the path
  where verification failed.

Then the two ablations, which are the actual content:

| Removed | What happens | Why the macros cannot catch it |
|---|---|---|
| `ncclGroupStart` / `ncclGroupEnd` | Hangs on most configurations | Each call blocks waiting for peers the same thread has not reached yet. Nothing returned an error; there is nothing to check |
| `hipStreamSynchronize` before the copy | Frequently *passes* at small sizes, fails at large ones | The collective was enqueued, not completed. Both calls succeeded. The bug is in the ordering, which no return code describes |

That table is the reason this drill is a drill. Error checking is necessary and this
module insists on it, but it is not sufficient: the two most common collective bugs are
both invisible to it, and one of them passes your test suite.

## Drill 3 - the plateau, the knee, and why the model misses

- **The plateau is the flat tail, not `max(busbw)`.** A maximum over a noisy series
  estimates the maximum of the noise as much as the level of the signal, so it is biased
  high by construction - worse on a short sweep or a busy machine. The solution takes the
  median of the trailing rows that agree within a few percent, and **refuses to report a
  plateau at all** when the largest sizes are still climbing. "This sweep did not plateau,
  raise `--max-bytes`" is the better answer, because a plateau that is really just the last
  row makes every percentage below it fiction.
- **90% of the plateau, not the textbook crossover.** Where the latency and wire terms are
  equal you are at exactly *half* the plateau - half your time is still latency. Working
  the algebra out gives the 90% point as `(0.9/0.1) * N * alpha * B`, nine times the
  crossover `m* = N*alpha*B` from
  [D01](../../D01.scaling-limits-and-communication-cost/README.md). The practical knee is
  nearly an order of magnitude above the number the model is usually quoted for, and it is
  the knee, not the crossover, that should set your gradient bucket size.
- **The discrepancy is the result.** The model has one `beta`; the library has several. It
  uses LL below a threshold, LL128 above it, and Simple for large messages, and switches
  from tree to ring over roughly the same range (README section 3.5), so its effective
  `beta` *improves* as the message grows and the real curve rises faster than a single-beta
  prediction. The measured 90% size therefore usually lands **below** the prediction. If
  yours lands far above, suspect the other end: the smallest row was not pure latency, so
  `alpha` is overestimated, or something outside the collective - the launch path, another
  tenant - is inside your timing window.

Whatever the sweep says, it is a property of *that* machine at *that* rank count. The
number to carry forward is the method.

## Drill 4 - auditing a report you did not produce

- **Two tolerances, not one.** Reports print bandwidths to two decimals, so a row reading
  `0.00 GB/s` carries a rounding quantum of 0.005 - larger than any error worth flagging.
  Compare against a relative tolerance **and** the column's own quantum, and flag only when
  both are exceeded. Without this the auditor flags every small-message row and gets
  switched off, which is strictly worse than not having it.
- **Recompute from `time`, because `time` is the only measurement.** Everything else in the
  report is derived, so an inconsistency between a reported bandwidth and the raw time is
  either a rounding artefact or a real defect, and the quantum test is what separates them.
- **`#wrong` is a column, not a footnote.** A nonzero entry means the correctness check
  failed, which makes the bandwidth on that row a measurement of broken hardware. It is the
  column people skip, so the tool leads with it.
- **Infer the unlabelled column, and say that you are inferring.** Test the single
  bandwidth column against both candidates and report which it matches. Then the honest
  part: when `f(N) ~ 1` - `broadcast`, `reduce`, or all-reduce at two ranks - the two
  candidates are numerically identical and no inference is possible. The solution detects
  that and says so instead of picking one, because a tool that guesses silently is how a
  wrong `busbw` ends up in someone's design document with a provenance chain that stops at
  your script.

The solution also states what the audit **cannot** tell you: whether `--ranks` matches the
job that actually ran, whether the plateau is good (that needs a link rate, which is a
hardware fact the tool does not have), and whether the smallest row measures the fabric's
latency floor or the launch path's. Naming the limits is what makes the rest credible.
Compare against [`../tools/parse_collective_report.py`](../tools/parse_collective_report.py),
which is the same audit with report-format handling around it.

## Reference

- Parent module: [`../README.md`](../README.md) - sections 2.4, 3.5, 3.7, 3.8 and 5.1.
- Drill statements: [`../exercises/README.md`](../exercises/README.md).
- Terms: [`GLOSSARY.md`](../../../../docs/GLOSSARY.md) -
  [algbw and busbw](../../../../docs/GLOSSARY.md#algbw-and-busbw),
  [all-reduce](../../../../docs/GLOSSARY.md#all-reduce),
  [stream](../../../../docs/GLOSSARY.md#stream).
- Sources: [`REFERENCES.md`](../../../../docs/REFERENCES.md), in particular
  [NCCL](../../../../docs/REFERENCES.md#nccl-nvidia) and
  [RCCL](../../../../docs/REFERENCES.md#rccl-amd) for what the libraries do and do not
  guarantee, and
  [nccl-tests and rccl-tests](../../../../docs/REFERENCES.md#nccl-tests-and-rccl-tests-nvidia-amd)
  for the report format drill 4 audits.
