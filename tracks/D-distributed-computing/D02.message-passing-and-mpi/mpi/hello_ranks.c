/* mpi/hello_ranks.c */
#include "mpi_check.h"

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init failed\n");
    return EXIT_FAILURE;
  }
  MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));

  int world_rank = 0, world_size = 0;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  char node_name[MPI_MAX_PROCESSOR_NAME];
  int name_len = 0;
  MPI_CHECK(MPI_Get_processor_name(node_name, &name_len));

  /* Section 3.1: split into one communicator per shared-memory domain. */
  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                                MPI_INFO_NULL, &node_comm));
  int node_rank = 0, node_size = 0;
  MPI_CHECK(MPI_Comm_rank(node_comm, &node_rank));
  MPI_CHECK(MPI_Comm_size(node_comm, &node_size));

  /* A dup'd communicator: same ranks, different context (section 3.1). */
  MPI_Comm private_comm = MPI_COMM_NULL;
  MPI_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &private_comm));
  int dup_size = 0;
  MPI_CHECK(MPI_Comm_size(private_comm, &dup_size));

  printf("world %d/%d | node-local %d/%d | dup size %d | node %.*s\n",
         world_rank, world_size, node_rank, node_size, dup_size, name_len, node_name);
  fflush(stdout);

  MPI_CHECK(MPI_Comm_free(&private_comm));
  MPI_CHECK(MPI_Comm_free(&node_comm));
  MPI_CHECK(MPI_Finalize());
  return EXIT_SUCCESS;
}
