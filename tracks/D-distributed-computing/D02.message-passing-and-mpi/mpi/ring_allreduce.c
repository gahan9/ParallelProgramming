/* mpi/ring_allreduce.c - hand-written ring all-reduce vs MPI_Allreduce. */
#include "mpi_check.h"
#include <string.h>

/* Ring all-reduce (MPI_SUM over floats): reduce-scatter then allgather.
   Handles any count and any rank count, including count < size. */
static void ring_allreduce_float(const float *sendbuf, float *recvbuf, int count,
                                 MPI_Comm comm) {
  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(comm, &rank));
  MPI_CHECK(MPI_Comm_size(comm, &size));

  memcpy(recvbuf, sendbuf, (size_t)count * sizeof(float));
  if (size == 1) return;

  /* Segment boundaries. The first (count % size) segments carry one extra
     element, so the segments cover count exactly for any count. */
  const int base = count / size, rem = count % size;
  int *starts = (int *)malloc((size_t)size * sizeof(int));
  int *counts = (int *)malloc((size_t)size * sizeof(int));
  float *inbox = (float *)malloc((size_t)(base + 1) * sizeof(float));
  if (starts == NULL || counts == NULL || inbox == NULL) {
    fprintf(stderr, "rank %d: ring_allreduce allocation failed\n", rank);
    MPI_Abort(comm, EXIT_FAILURE);
  }
  int offset = 0;
  for (int i = 0; i < size; ++i) {
    counts[i] = base + ((i < rem) ? 1 : 0);
    starts[i] = offset;
    offset += counts[i];
  }

  const int next = (rank + 1) % size;
  const int prev = (rank - 1 + size) % size;

  /* Phase 1: reduce-scatter, size-1 steps, m/N bytes per step.
     After the loop, rank r holds the fully reduced segment (r + 1) % size. */
  for (int step = 0; step < size - 1; ++step) {
    const int send_seg = (rank - step + size) % size;
    const int recv_seg = (rank - step - 1 + size) % size;
    MPI_CHECK(MPI_Sendrecv(recvbuf + starts[send_seg], counts[send_seg], MPI_FLOAT, next, 0,
                           inbox, counts[recv_seg], MPI_FLOAT, prev, 0,
                           comm, MPI_STATUS_IGNORE));
    for (int i = 0; i < counts[recv_seg]; ++i) {
      recvbuf[starts[recv_seg] + i] += inbox[i];
    }
  }

  /* Phase 2: allgather, size-1 steps, m/N bytes per step. */
  for (int step = 0; step < size - 1; ++step) {
    const int send_seg = (rank - step + 1 + size) % size;
    const int recv_seg = (rank - step + size) % size;
    MPI_CHECK(MPI_Sendrecv(recvbuf + starts[send_seg], counts[send_seg], MPI_FLOAT, next, 1,
                           recvbuf + starts[recv_seg], counts[recv_seg], MPI_FLOAT, prev, 1,
                           comm, MPI_STATUS_IGNORE));
  }

  free(inbox);
  free(counts);
  free(starts);
}

static double time_reps(void (*fn)(const float *, float *, int, MPI_Comm),
                        const float *in, float *out, int count, int reps) {
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  const double t0 = MPI_Wtime();
  for (int r = 0; r < reps; ++r) fn(in, out, count, MPI_COMM_WORLD);
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  return (MPI_Wtime() - t0) / (double)reps;
}

static void library_allreduce_float(const float *sendbuf, float *recvbuf, int count,
                                    MPI_Comm comm) {
  MPI_CHECK(MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, comm));
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

  const int count = (argc > 1) ? atoi(argv[1]) : (1 << 20);
  const int reps = (argc > 2) ? atoi(argv[2]) : 20;
  float *in = (count > 0) ? (float *)malloc((size_t)count * sizeof(float)) : NULL;
  float *mine = (count > 0) ? (float *)malloc((size_t)count * sizeof(float)) : NULL;
  float *theirs = (count > 0) ? (float *)malloc((size_t)count * sizeof(float)) : NULL;
  if (count <= 0 || reps <= 0 || in == NULL || mine == NULL || theirs == NULL) {
    if (rank == 0) fprintf(stderr, "usage: ring_allreduce [count>0] [reps>0]\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  /* Small exact integers: the sum is representable in float regardless of the
     order of accumulation, so an exact comparison tests the algorithm and not
     the floating-point model. */
  for (int i = 0; i < count; ++i) in[i] = (float)(rank + 1);
  const float expected = (float)(size * (size + 1) / 2);

  const double t_mine = time_reps(ring_allreduce_float, in, mine, count, reps);
  const double t_lib = time_reps(library_allreduce_float, in, theirs, count, reps);

  int bad = 0, any_bad = 0;
  for (int i = 0; i < count; ++i) {
    if (mine[i] != expected || theirs[i] != expected) { bad = 1; break; }
  }
  MPI_CHECK(MPI_Allreduce(&bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));

  if (rank == 0) {
    const double mib = (double)count * sizeof(float) / (1024.0 * 1024.0);
    printf("ranks=%d count=%d (%.1f MiB)  verify=%s\n", size, count, mib,
           any_bad ? "FAIL" : "ok");
    printf("  hand-written ring : %8.3f ms   modelled bytes/rank = %.1f MiB\n",
           t_mine * 1e3, 2.0 * mib * (size - 1) / size);
    printf("  MPI_Allreduce     : %8.3f ms\n", t_lib * 1e3);
  }

  free(theirs);
  free(mine);
  free(in);
  MPI_CHECK(MPI_Finalize());
  return any_bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
