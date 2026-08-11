/* exercises/reduce_scatter_allgather.c - Drill 2: Allreduce = Reduce_scatter + Allgather.
 *
 * Build MPI_Allreduce (MPI_SUM over floats) out of its two halves, and make it correct
 * for the two counts that break most hand-written versions: a count that is NOT
 * divisible by the rank count, and a count SMALLER than the rank count.
 *
 * This file COMPILES AND RUNS as given. It FAILS both verifications, because the
 * decomposition is not written yet. Fill in the three TODOs.
 *
 * Build and run (from this directory):
 *   mpicc -O2 -Wall -Wextra -I../mpi -o rsag reduce_scatter_allgather.c
 *   mpirun -np 4 ./rsag 1048576      # divisible: the easy case
 *   mpirun -np 4 ./rsag 1000003      # prime count: not divisible by 4
 *   mpirun -np 4 ./rsag 3            # fewer elements than ranks: some blocks are EMPTY
 *
 * Why this factorization is worth knowing: it is how the bandwidth-optimal all-reduce
 * algorithms are built (README 3.2), and it is why a sharded optimizer is cheap - if
 * you were going to reduce-scatter anyway, keeping the scattered result instead of
 * gathering it is exactly half the traffic, for free.
 */

#include "mpi_check.h"
#include <math.h>
#include <string.h>

/* Split `count` elements into `size` contiguous blocks and fill in the per-rank block
 * sizes and their starting offsets.
 *
 * TODO 1: fill counts[] and displs[] so that:
 *   - sum(counts) == count exactly, for ANY count and any size, including count == 0
 *     and count < size (in which case some counts[] entries are 0 - that is legal and
 *     both collectives below handle it);
 *   - displs[i] is the offset of block i, i.e. the running sum of the earlier counts;
 *   - the blocks are as even as possible. Give the first (count % size) blocks one
 *     extra element rather than dumping the remainder on the last rank: a single rank
 *     holding size-1 extra elements is a load-imbalance bug that shows up as one slow
 *     rank at scale, and every other rank waits for it at the next collective.
 */
static void block_layout(int count, int size, int *counts, int *displs) {
  (void)count;                 /* delete these casts when you implement TODO 1 */
  (void)size;
  (void)counts;
  (void)displs;
}

/* Allreduce(MPI_SUM) built from reduce-scatter then allgather.
 *
 * TODO 2: reduce-scatter. Reduce `sendbuf` elementwise across all ranks and leave rank
 *   i holding only its own block, in `myblock`. Use MPI_Reduce_scatter, which takes the
 *   array of per-rank receive counts you built in TODO 1.
 *
 *   Note MPI_Reduce_scatter_block also exists and is simpler - one count for everyone -
 *   but it requires count to be divisible by size, so it cannot pass the 1000003 case.
 *   Choosing the varying-count form is the whole point of the drill.
 *
 * TODO 3: allgather. Concatenate every rank's reduced block into `recvbuf` on every
 *   rank. Because the blocks differ in length you need MPI_Allgatherv, which takes both
 *   the per-rank counts and the per-rank displacements. MPI_Allgather (no v) assumes
 *   equal blocks and would silently misplace data.
 *
 * Wrap every call in MPI_CHECK.
 */
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

  /* One extra element so a zero-length block still gets a valid, non-NULL pointer.
     Passing NULL for a count of 0 is legal in MPI, but keeping the buffer real means
     the code below needs no special case. */
  float *myblock = (float *)malloc(((size_t)count + 1) * sizeof(float));
  if (myblock == NULL) {
    fprintf(stderr, "rank %d: block allocation failed\n", rank);
    MPI_Abort(comm, EXIT_FAILURE);
  }

  (void)sendbuf;               /* delete these casts when you implement TODO 2 and 3 */
  (void)recvbuf;

  free(myblock);
  free(displs);
  free(counts);
}

/* Case A: small exact integers. Every partial sum is representable in float, so the
 * result is independent of the order of accumulation and an EXACT comparison is
 * legitimate. That tests the algorithm and nothing else. */
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

/* Case B: values that do not sum exactly. Now your result and the library's differ in
 * the low bits, because floating-point addition is NOT associative and the two
 * algorithms accumulate in different orders. The comparison needs a RELATIVE
 * tolerance, and that is not a weakness of either implementation: it is why a
 * collective's result is algorithm-dependent, why changing rank count can nudge a
 * training loss curve, and why bitwise reproducibility across rank counts is a feature
 * you must ask for rather than a default. */
static int check_tolerance(int rank, int count, MPI_Comm comm) {
  float *in = (float *)malloc(((size_t)count + 1) * sizeof(float));
  float *mine = (float *)malloc(((size_t)count + 1) * sizeof(float));
  float *lib = (float *)malloc(((size_t)count + 1) * sizeof(float));
  if (in == NULL || mine == NULL || lib == NULL) {
    fprintf(stderr, "rank %d: verification allocation failed\n", rank);
    MPI_Abort(comm, EXIT_FAILURE);
  }
  /* A deterministic, rank-dependent, badly-conditioned spread: mixing magnitudes is
     what makes the summation order visible. No RNG, so runs are reproducible. */
  for (int i = 0; i < count; ++i) {
    const unsigned mixed = ((unsigned)i * 2654435761u + (unsigned)rank * 40503u) % 100000u;
    in[i] = (float)((double)mixed / 97.0) * ((i % 3 == 0) ? 1.0e-4f : 1.0e3f);
  }
  for (int i = 0; i < count; ++i) mine[i] = -1.0f;

  allreduce_via_rs_ag(in, mine, count, comm);
  MPI_CHECK(MPI_Allreduce(in, lib, count, MPI_FLOAT, MPI_SUM, comm));

  int bad = 0;
  const double rel_tol = 1e-5;                /* generous for fp32 over `size` terms */
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
    if (global[0] || global[1]) {
      printf("  FAIL is expected until the three TODOs are implemented.\n");
    }
  }

  MPI_CHECK(MPI_Finalize());
  return (global[0] || global[1]) ? EXIT_FAILURE : EXIT_SUCCESS;
}
