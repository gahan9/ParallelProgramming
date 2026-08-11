# D02 Exercises - Message Passing with MPI

Four drills. Three are C programs you complete; the fourth is a Python fit. Do them
before opening [`../solutions/`](../solutions/), and write real, compiling,
edge-case-correct code - the edge cases *are* the exercises here, because every one of
them corresponds to a bug that has shipped.

Every starter **compiles and runs as given**. Each one then **fails its own
verification**, because the part you are asked to write is a marked `TODO` stub. That gap
is deliberate: you should never be debugging the harness, only your own code. A `FAIL`
line on the first run means the setup works.

| # | File | Skill |
|---|------|-------|
| 1 | [`ring_shift.c`](ring_shift.c) | Deadlock-free symmetric exchange with `MPI_Sendrecv`; why the ordered fix does not generalise |
| 2 | [`reduce_scatter_allgather.c`](reduce_scatter_allgather.c) | The all-reduce factorization; uneven and empty blocks; when an exact float comparison is legitimate |
| 3 | [`halo_exchange_nonblocking.c`](halo_exchange_nonblocking.c) | Nonblocking exchange, `MPI_PROC_NULL`, and reproducing the buffer-reuse race on purpose |
| 4 | [`fit_alpha_beta.py`](fit_alpha_beta.py) | Turning a measured sweep into `alpha` and `beta`; refusing bad input without a traceback |

Each maps to an interview competency: deadlock reasoning, collective decomposition,
nonblocking ownership rules, and measurement literacy. The solutions include the
reasoning, not just the answer.

## Building and running

There is no separate `Makefile` here. All three C drills are one `mpicc` line, and the
module `Makefile` one level up builds the labs, not these:

```bash
cd tracks/D-distributed-computing/D02.message-passing-and-mpi/exercises

mpicc -O2 -Wall -Wextra -I../mpi -o ring_shift ring_shift.c
mpicc -O2 -Wall -Wextra -I../mpi -o rsag       reduce_scatter_allgather.c
mpicc -O2 -Wall -Wextra -I../mpi -o halo       halo_exchange_nonblocking.c
```

`-I../mpi` is what supplies [`MPI_CHECK`](../mpi/mpi_check.h). Use it on **every** MPI
call, in the exercises as in the labs: MPI's default error handler aborts the job without
telling you where, and `MPI_CHECK` plus `MPI_ERRORS_RETURN` turns that into a file and a
line. Warnings count as failures - build with `-Wall -Wextra` and keep it silent.

```bash
# Drill 1: the size sweep is the point. The naive version passes the first and hangs on
# the second; yours must pass both.
mpirun -np 4 ./ring_shift 1024
mpirun -np 4 ./ring_shift 4194304
mpirun -np 1 ./ring_shift 1024          # N == 1: source and destination are self
mpirun -np 3 ./ring_shift 1024          # odd N: no even/odd colouring exists

# Drill 2: three counts, and the last two are where hand-written versions break.
mpirun -np 4 ./rsag 1048576             # divisible
mpirun -np 4 ./rsag 1000003             # prime: not divisible by 4
mpirun -np 4 ./rsag 3                   # fewer elements than ranks: empty blocks

# Drill 3: correct first, then the bug, at two sizes.
mpirun -np 4 ./halo 1048576 262144
mpirun -np 4 ./halo 1024    1      bug
mpirun -np 4 ./halo 1048576 262144 bug

# Drill 4: needs a real measurement first, from lab 2 one level up.
cd .. && make all && mpirun -np 2 ./ping_pong | tee pingpong.csv && cd exercises
python fit_alpha_beta.py --input ../pingpong.csv --classify 1024,65536,16777216
```

> [!NOTE]
> Drill 4 has no committed sample CSV, and that is on purpose: a checked-in performance
> file would be a number this repository cannot reproduce. Run lab 2 on your machine and
> fit your own data. If you have no MPI at all, the drill is still worth doing against a
> file you write by hand - but label it as synthetic, and do not quote anything from it.

## What each drill is really testing

**Drill 1 - the bug that only exists above a threshold.** The naive `MPI_Send` then
`MPI_Recv` ring is a circular wait, but below the implementation's eager threshold the
send is satisfied from an internal buffer and returns anyway, so the program works at
every size you are likely to test. Find the boundary by bisecting the count. The
takeaway is not "use `MPI_Sendrecv`" but that **the same source is correct below the
threshold and broken above it**, so a test suite exercising only small messages provides
no evidence at all about this class. Keep an `MPI_Ssend` debug build: a synchronous send
never completes without a matching receive, which converts a size-dependent production
hang into a deterministic test failure.

**Drill 2 - the counts array is the difficulty.** `MPI_Reduce_scatter_block` and
`MPI_Allgather` take one count and assume equal blocks, so neither can express 1,000,003
elements over 4 ranks. The varying-count forms can. While you are there, notice where you
put the remainder: giving the last rank all of it is a load-imbalance bug that is
invisible at 4 ranks and a straggler at 1,024. Then run both comparisons and be able to
say why one is allowed to use `==` and the other is not.

**Drill 3 - ownership, not syntax.** Between `MPI_Isend` and its `MPI_Wait` the send
buffer belongs to MPI. The `bug` flag violates that and the violation is *silent* below
the eager threshold, which makes it strictly worse than deadlock: wrong numbers in some
fraction of runs rather than a hang. Record which size exposes it, then state the review
rule that would have caught it without running anything.

**Drill 4 - a fit is a claim about data.** Resist fitting one line through the whole
sweep: the flat region measures `alpha`, the linear region measures `beta`, and averaging
them produces a confident number describing nothing. And treat the error handling as part
of the deliverable, not as polish - this tool's job is to tell a colleague what is wrong
with their file, and a traceback names your code instead of their data.

## Reference

- Parent module: [`../README.md`](../README.md). Sections 2.4 (deadlock and its three
  fixes), 2.5 (what completion guarantees), 3.2-3.3 (collective costs) and 6.2 (the
  nonblocking buffer-reuse bug) contain everything these drills need.
- Terms: [`GLOSSARY.md`](../../../../docs/GLOSSARY.md) -
  [rank and communicator](../../../../docs/GLOSSARY.md#rank-and-communicator),
  [collective operation](../../../../docs/GLOSSARY.md#collective-operation),
  [ring all-reduce](../../../../docs/GLOSSARY.md#ring-all-reduce),
  [the alpha-beta model](../../../../docs/GLOSSARY.md#the-alpha-beta-model).
- Sources: [`REFERENCES.md`](../../../../docs/REFERENCES.md) - the
  [MPI standard](../../../../docs/REFERENCES.md#mpi-standard-mpi-forum) is normative for
  every completion-semantics question these drills raise.
- Setup: [`SETUP.md`](../../../../docs/SETUP.md).

Reference answers with full working: [`../solutions/`](../solutions/). Read them *after*
you have written yours - the commentary is the part being assessed, and it includes a few
honest notes on where each approach stops being the right one.
