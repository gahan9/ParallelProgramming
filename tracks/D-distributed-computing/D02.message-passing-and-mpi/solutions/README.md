# D02 Solutions - Message Passing with MPI

Worked solutions for the four drills in [`../exercises/`](../exercises/). Read these after
you have written your own: the reasoning is the part that transfers, and each file carries
its argument in a header comment rather than in the code.

| # | Solution | The one idea to take away |
|---|----------|---------------------------|
| 1 | [`ring_shift.c`](ring_shift.c) | `MPI_Sendrecv` removes the possibility of deadlock, not just the symptom; the ordered fix needs a proof about your topology |
| 2 | [`reduce_scatter_allgather.c`](reduce_scatter_allgather.c) | All-reduce *is* reduce-scatter plus allgather - so if you want the result sharded, stop halfway and pay half the traffic |
| 3 | [`halo_exchange_nonblocking.c`](halo_exchange_nonblocking.c) | Between post and wait the buffer belongs to MPI; violating that is silent below the eager threshold |
| 4 | [`fit_alpha_beta.py`](fit_alpha_beta.py) | Fit the two regimes separately, or you get a confident number that describes neither |

```bash
cd tracks/D-distributed-computing/D02.message-passing-and-mpi/solutions

mpicc -O2 -Wall -Wextra -I../mpi -o ring_shift ring_shift.c
mpicc -O2 -Wall -Wextra -I../mpi -o rsag       reduce_scatter_allgather.c
mpicc -O2 -Wall -Wextra -I../mpi -o halo       halo_exchange_nonblocking.c

mpirun -np 4 ./ring_shift 4194304        # ok
mpirun -np 4 ./rsag 1000003              # ok, both comparisons
mpirun -np 4 ./halo 1048576 262144       # ok
mpirun -np 4 ./halo 1048576 262144 bug   # FAIL - and that is the correct outcome

python -S -I fit_alpha_beta.py --input ../pingpong.csv --classify 1024,65536,16777216
```

> [!IMPORTANT]
> `./halo ... bug` exits non-zero **by design**. It is the only program in this module
> whose failure is the result you want. If it prints `ok`, your message is below the eager
> threshold and the defect is present but invisible - raise the halo size.

## Drill 1 - deadlock-free ring shift

Two programs that look equally reasonable, and only one of which is correct:

```c
MPI_Send(next); MPI_Recv(prev);   /* a cycle of blocking sends: hangs above eager */
MPI_Sendrecv(next, prev);         /* one operation: cannot deadlock at any size */
```

Three points the solution's header develops:

- **The eager threshold is why this is hard.** The first form works at small sizes because
  the implementation buffers the message internally and returns without a matching
  receive. The bug is present the whole time; the schedule hides it.
- **The ordered fix is correct but narrow.** "Even ranks send first" works on a ring with
  an even rank count because such a ring is two-colourable. Run with `-np 3`, or on an
  arbitrary neighbour graph, and you are doing graph colouring inside your application.
- **`N == 1` needs no special case.** Source and destination are both this rank, and a
  self-exchange in a single `MPI_Sendrecv` is legal - the hazard comes from a *blocking
  send that must complete before a receive is posted*, which is precisely what `Sendrecv`
  does not do.

`MPI_Sendrecv` still blocks until the exchange finishes, so it buys correctness and not
overlap. When there is independent work to hide the transfer behind, drill 3's shape is
the one that pays.

## Drill 2 - all-reduce as reduce-scatter plus allgather

The decomposition is two library calls; the work is the counts array.

- **`MPI_Reduce_scatter` and `MPI_Allgatherv`, not the `_block`/non-`v` forms.** Those
  take a single count and assume equal blocks, so they cannot express a count that is not
  divisible by the rank count. Choosing the varying-count forms *is* the drill.
- **Put the remainder on the first blocks, one element each.** Not on the last rank. At
  four ranks the difference is invisible; at 1,024 ranks one systematically larger block
  is a straggler that every other rank waits for at the next collective's implicit
  barrier.
- **A zero-length block is legal and must work.** With `count < size` some ranks receive
  nothing, both collectives accept a count of zero, and no branch is needed.
- **One comparison may use `==`; the other may not.** Small integer-valued floats sum
  exactly regardless of order, so an exact comparison tests the algorithm alone. Mixed
  magnitudes do not: floating-point addition is not associative, your accumulation order
  differs from the library's, and the results *must* differ in the low bits. That is not a
  defect to tolerate - it is why a collective's result is algorithm-dependent, why
  changing rank count can nudge a loss curve, and why bitwise reproducibility across rank
  counts is something you ask a library for rather than assume.

The practical payoff is the factorization itself: if your algorithm wants the reduced
result *sharded* rather than replicated, stop after the reduce-scatter. You have just
halved the traffic for free, which is why sharded optimizers are cheap and is the same
argument [B07](../../../B-gpu-ml-performance/B07.distributed-training-and-sharding/README.md)
makes about ZeRO.

## Drill 3 - nonblocking halo exchange, and the bug it buys you

- **Receives before sends** is about performance, not correctness: nothing blocks before
  the wait, so no ordering can deadlock. A posted receive gives the implementation
  somewhere to put arriving data and can save a copy of the whole halo.
- **One tag per direction.** In this exact code the `(source, tag)` pairs are already
  distinct, so one tag would work - but add `MPI_ANY_SOURCE` or a second exchange in the
  same step and it stops working. Two named constants make the wrong match
  unrepresentable. The structural version of the argument is `MPI_Comm_dup`.
- **`MPI_PROC_NULL` instead of a boundary branch.** An operation with `MPI_PROC_NULL`
  completes immediately, transfers nothing, and leaves the receive buffer untouched, so
  the request array is always four entries and every rank runs identical code. The
  verification asserts the sentinel at the boundaries, which turns that behaviour into
  something tested rather than assumed.
- **One `MPI_Waitall`, not four `MPI_Wait`s.** Four sequential waits impose an order the
  network has no reason to honour. And every request must be completed exactly once:
  abandoning one leaks internal state, and a loop that abandons one per iteration
  exhausts the implementation hours into a run, as an allocation failure with no visible
  cause.

Then the defect. With `bug` on, the program writes to the send buffer between the
`MPI_Isend` and the `MPI_Waitall`, when the buffer belongs to MPI:

| Message size | What the implementation did | What you see |
|---|---|---|
| Below the eager threshold | Already copied your buffer | Correct result. The bug is invisible |
| Above it (rendezvous) | Reads your memory *later*, after your write | Corrupt halo on the neighbour |

Same size-dependence as deadlock, worse symptom: wrong numbers in some fraction of runs
instead of a hang. The three review rules that catch the whole family are in the solution
header - every `MPI_I*` has a visible completion, no buffer is touched between post and
completion, and loops track requests in an array and finish them with one `MPI_Waitall`.

## Drill 4 - fitting alpha and beta

The model `T(m) = alpha + m*beta` is a straight line, so the tempting move is one
least-squares fit over the whole sweep. That is the wrong answer, and why is the drill:

- **The data is not one line.** Over four or five orders of magnitude in `m` there is a
  flat region measuring `alpha`, a linear region measuring `beta`, and usually a step
  between them where the implementation switches from eager to rendezvous - which changes
  the effective `beta`. One fit through all of it returns a slope that is a weighted
  average of two regimes and an intercept that is not `alpha`: a plausible-looking number
  with no referent.
- **Median, not mean, for the latency floor.** Every failure mode of a latency
  measurement makes it slower and none make it faster, so outliers are one-sided and the
  mean chases them.
- **Relative residuals, not absolute.** The largest message is thousands of times slower
  than the smallest, so an absolute residual is dominated by the tail and will happily
  accept a window that has swallowed the flat region or a protocol step.
- **The longest acceptable window, not the best-fitting one.** A two-point window fits
  perfectly and says nothing, so minimising the residual selects the least informative
  answer available. Report how many rows the window covers so a reader can judge it.
- **Refusing bad input is part of the deliverable.** One line, naming the file and the
  line number, exit non-zero. A traceback names your code instead of the user's data.

Compare your version against [`../tools/fit_alpha_beta.py`](../tools/fit_alpha_beta.py),
which is the same fit with a protocol-step detector added. Note what neither version does:
claim that the step it found is "the" eager threshold. That number is a tunable, it differs
between builds and transports, and quoting it as a constant is how it ends up in someone
else's design document.

## Reference

- Parent module: [`../README.md`](../README.md) - sections 2.4, 2.5, 3.2, 3.3 and 6.2.
- Drill statements: [`../exercises/README.md`](../exercises/README.md).
- Terms: [`GLOSSARY.md`](../../../../docs/GLOSSARY.md).
- Sources: [`REFERENCES.md`](../../../../docs/REFERENCES.md), in particular the
  [MPI standard](../../../../docs/REFERENCES.md#mpi-standard-mpi-forum) for completion
  semantics and
  [Thakur, Rabenseifner and Gropp](../../../../docs/REFERENCES.md#collective-algorithms-in-mpich-thakur-rabenseifner-and-gropp-2005)
  for the collective decompositions drill 2 rebuilds by hand.
