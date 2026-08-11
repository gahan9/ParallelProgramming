/* mpi/ping_pong.c - half round-trip latency across a size sweep. */
#include "mpi_check.h"
#include <string.h>

#define MAX_BYTES (1 << 22)   /* 4 MiB */

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed\n");
    return EXIT_FAILURE;
  }
  MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  int rank = 0, size = 0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  if (size != 2) {
    if (rank == 0) fprintf(stderr, "ping_pong requires exactly 2 ranks, got %d\n", size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  char *buf = (char *)malloc(MAX_BYTES);
  if (buf == NULL) {
    fprintf(stderr, "rank %d: allocation of %d bytes failed\n", rank, MAX_BYTES);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  memset(buf, 0, MAX_BYTES);

  const int peer = 1 - rank;
  const int tag = 0;
  if (rank == 0) printf("bytes,iters,half_roundtrip_us\n");

  for (int bytes = 0; bytes <= MAX_BYTES; bytes = (bytes == 0) ? 1 : bytes * 2) {
    const int iters = (bytes < (1 << 16)) ? 1000 : 100;
    const int warmup = iters / 10 + 1;
    double t0 = 0.0;

    for (int i = 0; i < warmup + iters; ++i) {
      if (i == warmup) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        t0 = MPI_Wtime();
      }
      /* Ordered exchange (fix 1): safe at every size, including above the
         eager threshold, which is exactly why lab 4 is a separate program. */
      if (rank == 0) {
        MPI_CHECK(MPI_Send(buf, bytes, MPI_BYTE, peer, tag, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Recv(buf, bytes, MPI_BYTE, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      } else {
        MPI_CHECK(MPI_Recv(buf, bytes, MPI_BYTE, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Send(buf, bytes, MPI_BYTE, peer, tag, MPI_COMM_WORLD));
      }
    }

    const double elapsed = MPI_Wtime() - t0;
    if (rank == 0) {
      printf("%d,%d,%.3f\n", bytes, iters, 0.5 * elapsed / (double)iters * 1e6);
      fflush(stdout);
    }
  }

  free(buf);
  MPI_CHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}
