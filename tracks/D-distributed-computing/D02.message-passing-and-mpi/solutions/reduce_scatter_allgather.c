/* solutions/reduce_scatter_allgather.c - Drill 2 worked solution.
 *
 * Allreduce(MPI_SUM, float) = Reduce_scatter + Allgatherv, correct for any count.
 *
 * Build and run (from this directory):
 *   mpicc -O2 -Wall -Wextra -I../mpi -o rsag reduce_scatter_allgather.c
 *   mpirun -np 4 ./rsag 1048576      # divisible
 *   mpirun -np 4 ./rsag 1000003      # prime: not divisible by 4
 *   mpirun -np 4 ./rsag 3            # fewer elements than ranks: two blocks are EMPTY
 *   mpirun -np 4 ./rsag 0            # degenerate: no elements at all
 *
 * WHY THIS FACTORIZATION
 *
 * MPI_Allreduce IS MPI_Reduce_scatter followed by MPI_Allgather. That is not a curiosity;
 * it is how the bandwidth-optimal all-reduce algorithms are constructed, and it explains
 * two things at once.
 *
 * First, the byte count. Reduce-scatter moves m(N-1)/N per rank and allgather moves the
 * same, giving the 2m(N-1)/N that Patarasuk and Yuan proved is a lower bound. Doing the
 * two halves explicitly gets you the same traffic as a good library all-reduce, which is
 * why the ring in lab 3 is competitive at large messages despite being twenty lines.
 *
 * Second, and more usefully: if your algorithm needs the reduced result SHARDED rather
 * than replicated, stop after the first half. You have just halved the traffic for free.
 * That single observation is why sharded optimizers (ZeRO-style) are cheap rather than
 * expensive - they were going to reduce-scatter anyway, so keeping the scattered result
 * instead of gathering it costs nothing and saves m(N-1)/N per rank per step.
 *
 * WHY THE VARYING-COUNT FORMS
 *
 * MPI_Reduce_scatter_block and MPI_Allgather take ONE count and assume every rank's block
 * is the same size, so they cannot express count=1000003 over 4 ranks. The `v` and
 * non-block forms take an array of counts, which is the only way to describe an uneven
 * split. Getting that array right is the actual content of this drill; the collectives
 * themselves are one line each. Two things fall out of doing it properly:
 *
 *   * Spread the remainder over the FIRST (count % size) blocks, one element each, rather
 *     than giving the last rank all size-1 extra elements. At this scale the difference is
 *     invisible; at 1,024 ranks a single rank with a systematically larger block is a
 *     straggler, and every other rank waits for it at the next collective's implicit
 *     barrier. Load balance is a property you design in, not one you notice later.
 *   * A zero-length block is legal and must work. When count < size some ranks receive
 *     nothing, and both collectives handle a count of 0 correctly - so the code needs no
 *     branch for it, provided the pointers you pass are valid.
 *
 * WHY ONE COMPARISON IS EXACT AND THE OTHER NEEDS A TOLERANCE
 *
 * Case A sums small integer-valued floats. Every partial sum is exactly representable in
 * binary32, so the result cannot depend on the order of accumulation and `==` is the right
 * comparison: it tests the algorithm and nothing else. If a test like this fails, the
 * indices are wrong, full stop.
 *
 * Case B sums values of mixed magnitude, where floating-point addition is NOT associative:
 * (a+b)+c and a+(b+c) genuinely differ. Your decomposition and the library's accumulate in
 * different orders, so they MUST differ in the low bits, and demanding equality would be
 * demanding the wrong thing. This is not a wart. It is why a collective's result is
 * algorithm-dependent, why changing the rank count can shift a training loss curve
 * slightly, and why bitwise reproducibility across rank counts is a feature you have to
 * ask a library for rather than something you get by default.
 */

#include "mpi_check.h"
#include <math.h>
#include <string.h>

/* Split `count` into `size` contiguous, near-equal blocks. */
static void block_layout(int count, int size, int *counts, int *displs) {
  const int base = count / size;
  const int rem = count % size;
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + ((i < rem) ? 1 : 0);
    displs[i] = offset;
    offset += counts[i];
  }
  /* offset == count now, by construction, for every count including 0 and count < size. */
}

static void allreduce_via_rs_ag(const float *sendbuf, float *recvbuf, int count,
                                MPI_Comm comm) {
  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  MPI_CHECK(MPI_Comm_size(comm, &size));

  int *counts = (int *)malloc((size_t)size * sizeof(int));
  int *displs = (int *)malloc((size_t)size * sizeof(int));
  if (counts == NULL || displs == NULL) {
    fprintf(stderr, "rank %d: layout allocation failed\n", rank);
    MPI_Abort(comm, EXIT_FAILURE);
  }
  block_layout(count, size, counts, displs);

  /* +1 so a zero-length block still has a valid, unique, non-NULL address. MPI accepts
     NULL with a count of 0, but a real pointer means no branch is needed below and no
     reader has to check whether this implementation is one of the strict ones. */
  float *myblock = (float *)malloc(((size_t)count + 1) * sizeof(float));
  if (myblock == NULL) {
    fprintf(stderr, "rank %d: block allocation failed\n", rank);
    MPI_Abort(comm, EXIT_FAILURE);
  }

  /* Phase 1: every rank contributes the whole buffer; rank i keeps only block i,
     already summed across all ranks. Traffic per rank: m(N-1)/N. */
  MPI_CHECK(MPI_Reduce_scatter(sendbuf, myblock, counts, MPI_FLOAT, MPI_SUM, comm));

  /* Phase 2: concatenate the N reduced blocks onto every rank. Allgatherv rather than
     Allgather because the blocks differ in length; passing `displs` is what places each
     rank's block at the right offset instead of assuming i*count/size. Same traffic
     again: m(N-1)/N. */
  MPI_CHECK(MPI_Allgatherv(myblock, counts[rank], MPI_FLOAT,
                           recvbuf, counts, displs, MPI_FLOAT, comm));

  free(myblock);
  free(displs);
  free(counts);
}

static int check_exact(int rank, int count, MPI_Comm comm) {
  float *in = (float *)malloc(((size_t)count + 1) * sizeof(float));
  float *mine = (float *)malloc(((size_t)count + 1) * sizeof(float));
  float *lib = (float *)malloc(((size_t)count + 1) * sizeof(float));
  if (in == NULL || mine == NULL || lib == NULL) {
    fprintf(stderr, "rank %d: verification allocation failed\n", rank);
    MPI_Abort(comm, EXIT_FAILURE);
  }
  for (int i = 0; i < count; ++i) in[i] = (float)(rank + 1);
  for (int i = 0; i < count; ++i) mine[i] = -1.0f;

  allreduce_via_rs_ag(in, mine, count, comm);
  MPI_CHECK(MPI_Allreduce(in, lib, count, MPI_FLOAT, MPI_SUM, comm));

  int bad = 0;
  for (int i = 0; i < count; ++i) {
    if (mine[i] != lib[i]) { bad = 1; break; }
  }
  free(lib);
  free(mine);
  free(in);
  return bad;
}

static int check_tolerance(int rank, int count, MPI_Comm comm) {
  float *in = (float *)malloc(((size_t)count + 1) * sizeof(float));
  float *mine = (float *)malloc(((size_t)count + 1) * sizeof(float));
  float *lib = (float *)malloc(((size_t)count + 1) * sizeof(float));
  if (in == NULL || mine == NULL || lib == NULL) {
    fprintf(stderr, "rank %d: verification allocation failed\n", rank);
    MPI_Abort(comm, EXIT_FAILURE);
  }
  for (int i = 0; i < count; ++i) {
    const unsigned mixed = ((unsigned)i * 2654435761u + (unsigned)rank * 40503u) % 100000u;
    in[i] = (float)((double)mixed / 97.0) * ((i % 3 == 0) ? 1.0e-4f : 1.0e3f);
  }
  for (int i = 0; i < count; ++i) mine[i] = -1.0f;

  allreduce_via_rs_ag(in, mine, count, comm);
  MPI_CHECK(MPI_Allreduce(in, lib, count, MPI_FLOAT, MPI_SUM, comm));

  int bad = 0;
  const double rel_tol = 1e-5;
  /* Relative to max(|b|, 1) rather than to |b| alone: a pure relative error explodes
     when the true sum happens to land near zero through cancellation, which would make
     the test fail on arithmetic that is in fact fine. */
  for (int i = 0; i < count; ++i) {
    const double a = (double)mine[i], b = (double)lib[i];
    const double scale = (fabs(b) > 1.0) ? fabs(b) : 1.0;
    if (fabs(a - b) / scale > rel_tol) { bad = 1; break; }
  }
  free(lib);
  free(mine);
  free(in);
  return bad;
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

  const long requested = (argc > 1) ? atol(argv[1]) : (1L << 20);
  if (requested < 0 || requested > (1L << 26)) {
    if (rank == 0) fprintf(stderr, "usage: rsag [count in 0..2^26]\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  const int count = (int)requested;

  const int bad_exact = check_exact(rank, count, MPI_COMM_WORLD);
  const int bad_tol = check_tolerance(rank, count, MPI_COMM_WORLD);

  int local[2], global[2];
  local[0] = bad_exact;
  local[1] = bad_tol;
  MPI_CHECK(MPI_Allreduce(local, global, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

  if (rank == 0) {
    printf("rsag: ranks=%d count=%d  (count %% ranks = %d, %s)\n",
           size, count, (size > 0) ? count % size : 0,
           (count < size) ? "fewer elements than ranks" : "at least one per rank");
    printf("  exact  (integer-valued floats) : %s\n", global[0] ? "FAIL" : "ok");
    printf("  tolerance (mixed magnitudes)   : %s\n", global[1] ? "FAIL" : "ok");
  }

  MPI_CHECK(MPI_Finalize());
  return (global[0] || global[1]) ? EXIT_FAILURE : EXIT_SUCCESS;
}
