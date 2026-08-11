/* solutions/halo_exchange_nonblocking.c - Drill 3 worked solution.
 *
 * 1D nonblocking halo exchange, plus the README 6.2 buffer-reuse defect on a flag so you
 * can watch it appear and disappear with message size.
 *
 * Build and run (from this directory):
 *   mpicc -O2 -Wall -Wextra -I../mpi -o halo halo_exchange_nonblocking.c
 *   mpirun -np 4 ./halo 1024 1                 # correct
 *   mpirun -np 4 ./halo 1048576 262144         # correct
 *   mpirun -np 4 ./halo 1024 1         bug     # small: usually still prints ok
 *   mpirun -np 4 ./halo 1048576 262144 bug     # large: corrupt on most builds
 *
 * WHY POST THE RECEIVES FIRST
 *
 * Not for correctness. Nothing blocks until MPI_Waitall, so no ordering of the four posts
 * can deadlock - that is the whole reason nonblocking is the fix that generalises. The
 * reason is performance: a receive that is already posted when the data arrives gives the
 * implementation somewhere to put it, so it can deliver straight into your buffer instead
 * of into an internal unexpected-message buffer and copying afterwards. Posting receives
 * first is a habit that costs nothing and occasionally saves a copy of the whole halo.
 *
 * WHY TWO TAGS
 *
 * Consider rank 5 in a chain, with one tag for everything. It posts a receive from rank 4
 * and a receive from rank 6, and both neighbours send. Now the (source, tag) pairs are
 * distinct, so this case is in fact safe - but change the code to use MPI_ANY_SOURCE, or
 * add a second exchange in the same step, and the messages become indistinguishable. One
 * tag per direction makes the intent explicit and the wrong match unrepresentable, which
 * is worth two named constants. The structural version of the same argument is section
 * 3.1's: give a subsystem its own communicator with MPI_Comm_dup and its tags become
 * private by construction.
 *
 * WHY MPI_PROC_NULL INSTEAD OF AN IF
 *
 * MPI_PROC_NULL is a rank you may send to and receive from: the operation completes
 * immediately, transfers nothing, and leaves the receive buffer untouched. So the ends of
 * the chain need no branch, the request array is always exactly four entries, and the one
 * MPI_Waitall covers every rank identically. Special-casing the boundaries instead means
 * two code paths, of which the rarely-taken one is where the bugs live.
 *
 * WHY ONE MPI_Waitall RATHER THAN FOUR MPI_Waits
 *
 * MPI_Waitall lets the implementation complete the four operations in whatever order they
 * actually finish. Four sequential MPI_Wait calls impose an order the network has no
 * reason to honour, so the first wait can sit idle while the third operation's data is
 * already there. And every request MUST be completed by exactly one Wait or Test:
 * abandoning one leaks internal state, and a loop that abandons one per iteration
 * eventually exhausts the implementation - usually as a confusing allocation failure
 * hours into a run, with no hint of the cause.
 *
 * THE DEFECT, AND WHY IT IS WORSE THAN DEADLOCK
 *
 * Between MPI_Isend and its MPI_Waitall the send buffer belongs to MPI, not to you. The
 * `bug` path writes to it in that window. Whether that corrupts anything depends on
 * whether the implementation has already copied the data, which depends on the message
 * size and therefore the protocol:
 *
 *   below the eager threshold : already copied, so the write is invisible - prints ok
 *   above it (rendezvous)     : MPI reads YOUR memory later, after the write - corrupt
 *
 * Same size-dependence as deadlock, but the symptom is wrong numbers in some fraction of
 * runs rather than a hang, which makes it strictly harder to find. Note what this does to
 * testing: a suite that only sends small messages provides NO evidence about this bug
 * class, and a green CI run is not evidence of absence.
 *
 * The three review rules that catch the whole family:
 *   1. Every MPI_I* call has a visible matching completion in the same function, or a
 *      documented owner that completes it.
 *   2. No buffer passed to an MPI_I* call is read or written between post and completion.
 *      Send buffers included - the send side is the half people forget.
 *   3. Loops that post requests collect them in an array and complete them with one
 *      MPI_Waitall, so a dropped request is visible as an off-by-one rather than as a
 *      resource leak.
 */

#include "mpi_check.h"
#include <string.h>

static void halo_exchange(double *cells, int interior, int halo, MPI_Comm comm,
                          int inject_bug) {
  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  MPI_CHECK(MPI_Comm_size(comm, &size));

  const int left = (rank == 0) ? MPI_PROC_NULL : rank - 1;
  const int right = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
  const int tag_to_left = 11, tag_to_right = 22;

  /* Layout: [ left halo | interior | right halo ]
     interior_last is written as (halo + interior) - halo, i.e. `halo` cells back from the
     end of the interior, rather than collapsed to `interior`. The two are the same
     address; the uncollapsed form is the one that stays correct if the halo width on the
     two sides is ever allowed to differ. */
  double *const left_halo = cells;
  double *const interior_first = cells + halo;
  double *const interior_last = cells + (halo + interior) - halo;
  double *const right_halo = cells + halo + interior;

  MPI_Request req[4];

  /* Receives first: see the header. A message from the left neighbour was tagged
     `to_right` by the sender, and vice versa - the tag names the DIRECTION of travel,
     so it must be read from the receiver's point of view too. */
  MPI_CHECK(MPI_Irecv(left_halo, halo, MPI_DOUBLE, left, tag_to_right, comm, &req[0]));
  MPI_CHECK(MPI_Irecv(right_halo, halo, MPI_DOUBLE, right, tag_to_left, comm, &req[1]));
  MPI_CHECK(MPI_Isend(interior_first, halo, MPI_DOUBLE, left, tag_to_left, comm, &req[2]));
  MPI_CHECK(MPI_Isend(interior_last, halo, MPI_DOUBLE, right, tag_to_right, comm, &req[3]));

  /* ---- This window is where overlap lives ----
   * Everything from here to the MPI_Waitall runs while the transfers are in flight, so
   * it is where you put the interior computation that does not depend on the halos. That
   * is the entire performance argument for nonblocking: it is not faster on the wire, it
   * just stops you waiting on it. The rule for what may go here is narrow and absolute:
   * anything, EXCEPT touching the four buffers above.
   */

  if (inject_bug && interior > 0) {
    /* WRONG ON PURPOSE. These two cells are inside buffers currently owned by MPI. */
    cells[halo] = -999.0;
    cells[halo + interior - 1] = -999.0;
  }

  MPI_CHECK(MPI_Waitall(4, req, MPI_STATUSES_IGNORE));
  /* After the Waitall, and only after it, the buffers are yours again. */
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

  for (size_t i = 0; i < total; ++i) cells[i] = -1.0;
  for (int i = 0; i < interior; ++i) cells[halo + i] = (double)rank;

  halo_exchange(cells, interior, halo, MPI_COMM_WORLD, inject_bug);

  /* At the ends, MPI_PROC_NULL wrote nothing, so the sentinel IS the correct value.
     Asserting that rather than skipping the check is what makes MPI_PROC_NULL's
     behaviour tested instead of assumed. */
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
    if (any_bad && inject_bug) {
      printf("  Expected with bug=ON at this size: the send buffer was modified while\n");
      printf("  MPI still owned it, and this message is large enough to be rendezvous.\n");
    }
    if (!any_bad && inject_bug) {
      printf("  bug=ON yet correct: this message is below the eager threshold, so MPI had\n");
      printf("  already copied the buffer. The defect is present and invisible. Raise the\n");
      printf("  halo size until it appears, and note that number is not portable.\n");
    }
    if (any_bad && !inject_bug) {
      printf("  Unexpected: the correct version failed. Check tags and buffer offsets.\n");
    }
  }

  free(cells);
  MPI_CHECK(MPI_Finalize());
  /* With the bug deliberately on, a nonzero exit is the CORRECT outcome of the drill, so
     the caller should not read this as the solution being broken. */
  return any_bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
