/* mpi/mpi_check.h - included by every sample in this module. */
#ifndef MPI_CHECK_H
#define MPI_CHECK_H

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Use after MPI_Init has succeeded. Requires MPI_ERRORS_RETURN to be set on the
   communicator, otherwise the default handler aborts before we can report. */
#define MPI_CHECK(call)                                                       \
  do {                                                                        \
    const int err_ = (call);                                                   \
    if (err_ != MPI_SUCCESS) {                                                \
      char msg_[MPI_MAX_ERROR_STRING];                                        \
      int len_ = 0;                                                            \
      MPI_Error_string(err_, msg_, &len_);                                     \
      fprintf(stderr, "MPI error at %s:%d: %s\n", __FILE__, __LINE__, msg_);   \
      MPI_Abort(MPI_COMM_WORLD, err_);                                         \
    }                                                                          \
  } while (0)

#endif /* MPI_CHECK_H */
