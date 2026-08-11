/* solutions/ring_shift.c - Drill 1 worked solution: deadlock-free periodic ring shift.
 *
 * Build and run (from this directory):
 *   mpicc -O2 -Wall -Wextra -I../mpi -o ring_shift ring_shift.c
 *   mpirun -np 4 ./ring_shift 1024
 *   mpirun -np 4 ./ring_shift 4194304     # 16 MiB per block, above the eager threshold
 *   mpirun -np 1 ./ring_shift 1024        # N == 1: source and destination are self
 *   mpirun -np 3 ./ring_shift 1024        # odd rank count: no even/odd colouring exists
 *
 * WHY THIS SHAPE
 *
 * The whole drill is the difference between two programs that look equally reasonable:
 *
 *     MPI_Send(next); MPI_Recv(prev);          <- deadlocks above the eager threshold
 *     MPI_Sendrecv(next, prev);                <- cannot deadlock, at any size
 *
 * The first is a cycle of blocking sends. Every rank is inside MPI_Send waiting for its
 * right neighbour to post a receive, and no rank can post one until its own send
 * returns. Below the implementation's eager threshold the send is satisfied from an
 * internal buffer and returns anyway, so the cycle never forms and the program works -
 * on your laptop, at your test size, every time. Above it the implementation switches
 * to a rendezvous protocol and the same source hangs. The bug was always there; the
 * schedule was hiding it.
 *
 * MPI_Sendrecv removes the possibility rather than the symptom. It is a single
 * operation, so the implementation is free to order or overlap its two halves however
 * it likes, and it guarantees the pairwise exchange completes. That matters more than
 * it first appears: the ordered fix - "even ranks send first, odd ranks receive first" -
 * is also correct here, but it only exists because a ring with an even rank count
 * happens to be two-colourable. Run with `-np 3` and there is no such colouring; run on
 * an arbitrary neighbour graph and finding one is a graph-colouring problem you do not
 * want inside your application. Sendrecv is correct without a proof about the topology.
 *
 * What it costs: MPI_Sendrecv still blocks until the exchange is done, so it buys
 * correctness and not overlap. When there is independent local work to hide the
 * transfer behind, MPI_Irecv/MPI_Isend with one MPI_Waitall is the version that pays -
 * that is drill 3.
 */

#include "mpi_check.h"
#include <string.h>

/* One step of the ring shift. `send` and `recv` must not overlap: MPI forbids aliasing
 * the two buffers of a Sendrecv, and the caller below double-buffers for that reason. */
static void ring_shift_once(const int *send, int *recv, int count, MPI_Comm comm) {
  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  MPI_CHECK(MPI_Comm_size(comm, &size));

  const int dest = (rank + 1) % size;
  /* Not (rank - 1) % size: in C the result of % takes the sign of the left operand, so
     rank 0 would compute -1 and MPI would reject the rank. Adding `size` first is the
     idiom, and it is worth writing consistently even where it looks redundant. */
  const int source = (rank - 1 + size) % size;
  const int tag = 0;

  /* At size == 1 this is a send to self matched by a receive from self, in one call.
     That is legal and needs no special case: the deadlock hazard of self-messaging
     comes from a blocking send that must complete BEFORE a receive is posted, and
     Sendrecv posts both together. MPI_STATUS_IGNORE because the source, tag and count
     are all already known - there is nothing to learn from the status and filling one
     in is not free. */
  MPI_CHECK(MPI_Sendrecv(send, count, MPI_INT, dest, tag,
                         recv, count, MPI_INT, source, tag,
                         comm, MPI_STATUS_IGNORE));
}

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed\n");
    return EXIT_FAILURE;
  }
  /* Without this the default handler (MPI_ERRORS_ARE_FATAL) aborts the job before
     MPI_CHECK can report a file and a line. */
  MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  const long requested = (argc > 1) ? atol(argv[1]) : 1024;
  if (requested <= 0 || requested > (1L << 26)) {
    if (rank == 0) fprintf(stderr, "usage: ring_shift [count in 1..2^26]\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  const int count = (int)requested;

  int *cur = (int *)malloc((size_t)count * sizeof(int));
  int *next = (int *)malloc((size_t)count * sizeof(int));
  if (cur == NULL || next == NULL) {
    fprintf(stderr, "rank %d: allocation of 2 x %d ints failed\n", rank, count);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  for (int i = 0; i < count; ++i) cur[i] = rank;
  /* A sentinel no rank can stamp, so a receive buffer that was never written fails
     loudly instead of accidentally holding a plausible value. */
  for (int i = 0; i < count; ++i) next[i] = -1;

  int failures = 0;
  for (int step = 1; step <= size; ++step) {
    ring_shift_once(cur, next, count, MPI_COMM_WORLD);

    const int expect = (rank - (step % size) + size) % size;
    for (int i = 0; i < count; ++i) {
      if (next[i] != expect) { ++failures; break; }
    }

    /* Double-buffer rather than shifting in place: the data received this step is what
       gets sent next step, and pointing both arguments of the next Sendrecv at one
       buffer would alias them. */
    int *swap = cur; cur = next; next = swap;
    for (int i = 0; i < count; ++i) next[i] = -1;
  }

  /* size shifts around a ring of size ranks is the identity. Checking the round trip as
     well as each intermediate step is what makes this a test rather than a demo: an
     implementation that shifted the wrong direction would still return home. */
  int home = 0;
  for (int i = 0; i < count; ++i) {
    if (cur[i] != rank) { home = 1; break; }
  }

  int local_bad = (failures != 0 || home != 0) ? 1 : 0;
  int any_bad = 0;
  /* Verify GLOBALLY. A per-rank pass message is worth little in SPMD, where rank 0
     printing "ok" while rank 3 holds garbage is exactly the failure you are testing
     for. One MPI_MAX over the local verdicts costs one collective and removes that. */
  MPI_CHECK(MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

  if (rank == 0) {
    printf("ring_shift: ranks=%d count=%d (%.1f KiB per block) steps=%d  %s\n",
           size, count, (double)count * sizeof(int) / 1024.0, size,
           any_bad ? "FAIL" : "ok");
  }

  free(next);
  free(cur);
  MPI_CHECK(MPI_Finalize());
  return any_bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
