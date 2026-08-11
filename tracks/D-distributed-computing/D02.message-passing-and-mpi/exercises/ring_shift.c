/* exercises/ring_shift.c - Drill 1: a deadlock-free periodic ring shift.
 *
 * Every rank sends its block to rank+1 (mod N) and receives from rank-1 (mod N),
 * for N steps, so a rank-stamped buffer must come back to where it started.
 *
 * This file COMPILES AND RUNS as given. It FAILS its own verification, because the
 * shift does nothing yet - that gap is the exercise. Fill in the TODO.
 *
 * Build and run (from this directory):
 *   mpicc -O2 -Wall -Wextra -I../mpi -o ring_shift ring_shift.c
 *   mpirun -np 4 ./ring_shift 1024        # 4 KiB blocks: below the eager threshold
 *   mpirun -np 4 ./ring_shift 4194304     # 16 MiB blocks: above it, on most builds
 *   mpirun -np 1 ./ring_shift 1024        # N == 1: source and destination are self
 *
 * The size sweep is the point. A naive "send to next, then receive from previous"
 * implementation passes the small case and hangs on the large one, because below the
 * eager threshold the implementation buffers the message and the blocking send
 * returns with no matching receive posted. Above it, every rank waits on its right
 * neighbour and the cycle closes. MPI_Sendrecv cannot deadlock against itself, which
 * is why the module recommends it for symmetric neighbour exchange (README 2.4).
 */

#include "mpi_check.h"
#include <string.h>

/* One step of the ring shift: send `count` ints from `send` to rank+1, receive
 * `count` ints from rank-1 into `recv`. `send` and `recv` must not overlap.
 *
 * TODO 1: implement this with a single MPI_Sendrecv, wrapped in MPI_CHECK.
 *
 *   - The destination is (rank + 1) % size and the source is (rank - 1 + size) % size.
 *     Write the source that way, not as (rank - 1) % size: in C the % operator keeps
 *     the sign of the left operand, so rank 0 would ask for source -1.
 *   - Use the same tag for both halves; one exchange needs only one tag.
 *   - N == 1 needs no special case. Destination and source are both this rank, and
 *     MPI_Sendrecv with self is legal precisely because it is one call rather than a
 *     send that must complete before a receive is posted.
 *   - Pass MPI_STATUS_IGNORE: you already know the source, the tag and the count, so
 *     there is nothing in the status worth the cost of filling it in.
 */
static void ring_shift_once(const int *send, int *recv, int count, MPI_Comm comm) {
  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  MPI_CHECK(MPI_Comm_size(comm, &size));

  /* Delete these four casts once the TODO is implemented; they exist only so the
     unfinished starter compiles warning-clean under -Wall -Wextra. */
  (void)send;
  (void)recv;
  (void)count;
  (void)rank;
  (void)size;
}

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed\n");
    return EXIT_FAILURE;
  }
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

  /* Stamp every element with this rank, so a block always names its origin. */
  for (int i = 0; i < count; ++i) cur[i] = rank;
  /* A sentinel no rank could ever stamp, so an unwritten recv buffer is detectable
     rather than accidentally plausible. */
  for (int i = 0; i < count; ++i) next[i] = -1;

  int failures = 0;
  for (int step = 1; step <= size; ++step) {
    ring_shift_once(cur, next, count, MPI_COMM_WORLD);

    /* After `step` shifts, this rank must hold the block that started on the rank
       `step` places to its left, counted around the ring. */
    const int expect = (rank - (step % size) + size) % size;
    for (int i = 0; i < count; ++i) {
      if (next[i] != expect) { ++failures; break; }
    }

    int *swap = cur; cur = next; next = swap;
    for (int i = 0; i < count; ++i) next[i] = -1;
  }

  /* After exactly `size` shifts the buffer has come all the way round. */
  int home = 0;
  for (int i = 0; i < count; ++i) {
    if (cur[i] != rank) { home = 1; break; }
  }

  int local_bad = (failures != 0 || home != 0) ? 1 : 0;
  int any_bad = 0;
  MPI_CHECK(MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

  if (rank == 0) {
    printf("ring_shift: ranks=%d count=%d (%.1f KiB per block) steps=%d  %s\n",
           size, count, (double)count * sizeof(int) / 1024.0, size,
           any_bad ? "FAIL (expected until TODO 1 is done)" : "ok");
  }

  free(next);
  free(cur);
  MPI_CHECK(MPI_Finalize());
  return any_bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
