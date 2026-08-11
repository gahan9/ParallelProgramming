/* exercises/halo_exchange_nonblocking.c - Drill 3: nonblocking halo exchange, and the
 * buffer-reuse bug that nonblocking buys you (README 6.2).
 *
 * A 1D domain is split across ranks. Each rank owns `interior` cells and keeps a halo
 * of `halo` cells on each side, holding a copy of its neighbour's edge. The exchange is
 * NOT periodic: rank 0 has no left neighbour and rank N-1 has no right neighbour, both
 * expressed as MPI_PROC_NULL rather than as an if-statement.
 *
 * This file COMPILES AND RUNS as given. It FAILS verification, because the exchange is
 * not written yet. Fill in TODO 1.
 *
 * Build and run (from this directory):
 *   mpicc -O2 -Wall -Wextra -I../mpi -o halo halo_exchange_nonblocking.c
 *   mpirun -np 4 ./halo 1024 1              # tiny halo
 *   mpirun -np 4 ./halo 1048576 262144      # 1 MiB halo: above the eager threshold
 *
 * Then the second half of the drill. Add the word `bug` and the program writes to the
 * send buffer after posting MPI_Isend and before MPI_Waitall - the defect from README
 * 6.2, already wired up for you:
 *   mpirun -np 4 ./halo 1024 1        bug   # small: usually still CORRECT
 *   mpirun -np 4 ./halo 1048576 262144 bug  # large: corrupt, on most builds
 *
 * Record which size exposes it. The bug is the same in both runs; only the protocol
 * changed. Below the eager threshold the implementation has already copied your buffer
 * into its own by the time you scribble on it, so the corruption is invisible. Above
 * it, the send is a rendezvous that reads YOUR memory later, after your write. Same
 * size-dependence as deadlock, except the symptom is wrong numbers rather than a hang,
 * in some fraction of runs. A test suite that only ever sends small messages provides
 * no evidence at all about this bug class.
 */

#include "mpi_check.h"
#include <string.h>

/* Exchange halos with the left and right neighbours.
 *
 * Layout of `cells`, length halo + interior + halo:
 *   [ left halo | interior ......................... | right halo ]
 *     0..halo-1   halo..halo+interior-1                halo+interior..
 *
 * You send the EDGES of your interior and receive into the HALOS:
 *   - to the left neighbour:  your first `halo` interior cells, at &cells[halo]
 *   - to the right neighbour: your last  `halo` interior cells, at
 *                             &cells[halo + interior - halo]
 *   - from the left neighbour:  into &cells[0]
 *   - from the right neighbour: into &cells[halo + interior]
 *
 * TODO 1: post four nonblocking operations and complete them with ONE
 *   MPI_Waitall(4, ...). Wrap every call in MPI_CHECK.
 *
 *   - Post both MPI_Irecv calls BEFORE both MPI_Isend calls. Not for correctness -
 *     nothing blocks until the wait, so no ordering can deadlock - but because a
 *     posted receive gives the implementation somewhere to put arriving data and can
 *     let it skip an internal copy.
 *   - Use two different tags, one per direction. With one tag, this rank's message to
 *     its left neighbour and its right neighbour's message to it are indistinguishable
 *     by (source, tag) in the middle of the chain, and a wrong match is possible.
 *   - Do not special-case the ends. `left` and `right` are already MPI_PROC_NULL where
 *     there is no neighbour, and an operation with MPI_PROC_NULL completes immediately,
 *     sends nothing, and leaves the receive buffer untouched - which is exactly the
 *     behaviour the verification below expects at the boundaries.
 *   - Collect all four requests in the array and wait on it once. Waiting on them one
 *     at a time serialises what you just paid to overlap; dropping one leaks internal
 *     state until the implementation runs out.
 */
static void halo_exchange(double *cells, int interior, int halo, MPI_Comm comm,
                          int inject_bug) {
  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  MPI_CHECK(MPI_Comm_size(comm, &size));

  const int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
  const int tag_to_left = 11, tag_to_right = 22;

  MPI_Request req[4];
  for (int i = 0; i < 4; ++i) req[i] = MPI_REQUEST_NULL;

  /* Delete these casts once TODO 1 is implemented. */
  (void)cells;
  (void)interior;
  (void)halo;
  (void)left;
  (void)right;
  (void)tag_to_left;
  (void)tag_to_right;
  (void)req;

  /* ------------------------------------------------------------------------
   * The deliberate defect from README 6.2, enabled with the `bug` argument.
   * This write lands AFTER the MPI_Isend above has been posted and BEFORE the
   * MPI_Waitall that completes it, so between those two points the buffer belongs
   * to MPI and not to you. Writing it is a data race, and its outcome depends on
   * whether the implementation has already copied the data - which depends on the
   * message size and therefore on the protocol.
   *
   * Once TODO 1 is written, move this block so it sits between your MPI_Isend calls
   * and your MPI_Waitall. Then run it at both sizes.
   * ------------------------------------------------------------------------ */
  if (inject_bug && interior > 0) {
    cells[halo] = -999.0;                          /* first interior cell, in flight */
    cells[halo + interior - 1] = -999.0;           /* last interior cell, in flight */
  }
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

  const long want_interior = (argc > 1) ? atol(argv[1]) : 1024;
  const long want_halo = (argc > 2) ? atol(argv[2]) : 1;
  const int inject_bug = (argc > 3 && strcmp(argv[3], "bug") == 0) ? 1 : 0;
  if (want_interior <= 0 || want_halo <= 0 || want_halo > want_interior ||
      want_interior > (1L << 26)) {
    if (rank == 0) {
      fprintf(stderr, "usage: halo <interior in 1..2^26> <halo in 1..interior> [bug]\n");
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  const int interior = (int)want_interior;
  const int halo = (int)want_halo;

  const size_t total = (size_t)halo + (size_t)interior + (size_t)halo;
  double *cells = (double *)malloc(total * sizeof(double));
  if (cells == NULL) {
    fprintf(stderr, "rank %d: allocation of %zu doubles failed\n", rank, total);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  /* Stamp the interior with this rank, and the halos with a sentinel so an unwritten
     halo is detectable rather than accidentally plausible. */
  for (size_t i = 0; i < total; ++i) cells[i] = -1.0;
  for (int i = 0; i < interior; ++i) cells[halo + i] = (double)rank;

  halo_exchange(cells, interior, halo, MPI_COMM_WORLD, inject_bug);

  /* A boundary halo must be untouched (MPI_PROC_NULL wrote nothing), so the sentinel
     is the correct expected value there. An interior halo must hold the neighbour. */
  const double expect_left = (rank == 0) ? -1.0 : (double)(rank - 1);
  const double expect_right = (rank == size - 1) ? -1.0 : (double)(rank + 1);
  int bad_left = 0, bad_right = 0;
  for (int i = 0; i < halo; ++i) {
    if (cells[i] != expect_left) { bad_left = 1; break; }
  }
  for (int i = 0; i < halo; ++i) {
    if (cells[halo + interior + i] != expect_right) { bad_right = 1; break; }
  }

  int local = (bad_left || bad_right) ? 1 : 0;
  int any_bad = 0, bad_ranks = 0;
  MPI_CHECK(MPI_Allreduce(&local, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
  MPI_CHECK(MPI_Allreduce(&local, &bad_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

  if (rank == 0) {
    printf("halo: ranks=%d interior=%d halo=%d (%.1f KiB per message) bug=%s\n",
           size, interior, halo, (double)halo * sizeof(double) / 1024.0,
           inject_bug ? "ON" : "off");
    printf("  halos correct on %d of %d ranks : %s\n", size - bad_ranks, size,
           any_bad ? "FAIL" : "ok");
    if (any_bad && !inject_bug) {
      printf("  FAIL is expected until TODO 1 is implemented.\n");
    }
    if (any_bad && inject_bug) {
      printf("  With bug=ON, FAIL is the lesson: the send buffer was modified in flight.\n");
    }
    if (!any_bad && inject_bug) {
      printf("  bug=ON but the result is correct: this message size is below the eager\n");
      printf("  threshold, so MPI had already copied the buffer. Raise the halo size.\n");
    }
  }

  free(cells);
  MPI_CHECK(MPI_Finalize());
  return any_bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
