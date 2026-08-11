/* mpi/deadlock.c - one program, five exchange patterns, two ranks. */
#include "mpi_check.h"
#include <string.h>

enum pattern { PAT_DEADLOCK, PAT_SSEND, PAT_ORDERED, PAT_SENDRECV, PAT_NONBLOCKING, PAT_INVALID };

static const char *const kPatternNames[] = {
    "deadlock", "ssend", "ordered", "sendrecv", "nonblocking", NULL};

static enum pattern parse_pattern(const char *name) {
  for (int i = 0; kPatternNames[i] != NULL; ++i) {
    if (strcmp(name, kPatternNames[i]) == 0) return (enum pattern)i;
  }
  return PAT_INVALID;
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
  if (size != 2) {
    if (rank == 0) fprintf(stderr, "deadlock requires exactly 2 ranks, got %d\n", size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  const enum pattern pat = (argc < 3) ? PAT_INVALID : parse_pattern(argv[1]);
  const long bytes = (argc < 3) ? -1 : atol(argv[2]);
  if (pat == PAT_INVALID || bytes < 0 || bytes > (1L << 30)) {
    if (rank == 0) {
      fprintf(stderr, "usage: deadlock "
                      "<deadlock|ssend|ordered|sendrecv|nonblocking> <bytes in [0, 2^30]>\n");
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  char *out = (char *)malloc((size_t)bytes + 1);
  char *in = (char *)malloc((size_t)bytes + 1);
  if (out == NULL || in == NULL) {
    fprintf(stderr, "rank %d: allocation of %ld bytes failed\n", rank, bytes);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  memset(out, rank + 1, (size_t)bytes + 1);
  memset(in, 0, (size_t)bytes + 1);

  const int peer = 1 - rank;
  const int tag = 0;
  const int n = (int)bytes;

  printf("rank %d: entering pattern %s with %ld bytes\n", rank, argv[1], bytes);
  fflush(stdout);

  switch (pat) {
    case PAT_DEADLOCK:
      /* Both ranks send first. Works below the eager threshold, hangs above it. */
      MPI_CHECK(MPI_Send(out, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD));
      MPI_CHECK(MPI_Recv(in, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      break;

    case PAT_SSEND:
      /* Synchronous send: no eager buffer can hide the bug. Hangs at any size. */
      MPI_CHECK(MPI_Ssend(out, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD));
      MPI_CHECK(MPI_Recv(in, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      break;

    case PAT_ORDERED:                              /* Fix 1 */
      if (rank == 0) {
        MPI_CHECK(MPI_Send(out, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Recv(in, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      } else {
        MPI_CHECK(MPI_Recv(in, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Send(out, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD));
      }
      break;

    case PAT_SENDRECV:                             /* Fix 2 */
      MPI_CHECK(MPI_Sendrecv(out, n, MPI_BYTE, peer, tag,
                             in, n, MPI_BYTE, peer, tag,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      break;

    case PAT_NONBLOCKING: {                        /* Fix 3 */
      MPI_Request req[2];
      MPI_CHECK(MPI_Irecv(in, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD, &req[0]));
      MPI_CHECK(MPI_Isend(out, n, MPI_BYTE, peer, tag, MPI_COMM_WORLD, &req[1]));
      MPI_CHECK(MPI_Waitall(2, req, MPI_STATUSES_IGNORE));
      break;
    }

    case PAT_INVALID:                              /* rejected above; keeps -Wswitch quiet */
      break;
  }

  const int ok = (n == 0) || (in[0] == (char)(peer + 1));
  printf("rank %d: completed, payload %s\n", rank, ok ? "correct" : "WRONG");
  fflush(stdout);

  free(in);
  free(out);
  MPI_CHECK(MPI_Finalize());
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
