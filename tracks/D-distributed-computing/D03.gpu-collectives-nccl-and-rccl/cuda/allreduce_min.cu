// Single-process, multi-GPU all-reduce with NCCL, verified against a CPU reference.
// Build: nvcc -O2 cuda/allreduce_min.cu -lnccl -o allreduce_min_cuda
//
// This is hip/allreduce_min.cpp with the mechanical substitution from README
// section 4: the runtime wrapper changes, every nccl* call is byte-identical.
// If you are diffing the two files and see a difference in the collective code
// itself, one of them has drifted and this one is the copy.
#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(cmd)                                                       \
  do {                                                                        \
    cudaError_t e_ = (cmd);                                                   \
    if (e_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd, \
                   cudaGetErrorString(e_));                                   \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

#define NCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    ncclResult_t r_ = (cmd);                                                  \
    if (r_ != ncclSuccess) {                                                  \
      std::fprintf(stderr, "NCCL %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd, \
                   ncclGetErrorString(r_));                                   \
      std::exit(EXIT_FAILURE);                                                \
    }                                                                         \
  } while (0)

// Each device fills its buffer with a rank-dependent value, so the expected
// all-reduce result is exactly sum_{r=0}^{N-1} (r + 1) = N(N+1)/2.
__global__ void fill_with_rank(float* buf, int n, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[idx] = value;
  }
}

int main(int argc, char** argv) {
  const int count = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
  if (count <= 0) {
    std::fprintf(stderr, "element count must be positive\n");
    return EXIT_FAILURE;
  }
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return EXIT_FAILURE;
  }
  const size_t bytes = (size_t)count * sizeof(float);
  std::printf("all-reduce over %d devices, %d floats (%.1f MiB)\n", ndev, count,
              (double)bytes / (1024.0 * 1024.0));

  std::vector<ncclComm_t> comms(ndev);
  std::vector<cudaStream_t> streams(ndev);
  std::vector<float*> sendbuf(ndev, nullptr);
  std::vector<float*> recvbuf(ndev, nullptr);

  // This shortcut exists only for the single-process case; multi-process uses
  // ncclCommInitRank with a broadcast unique ID (section 3.2).
  NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, /*devlist=*/nullptr));

  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaStreamCreate(&streams[dev]));
    CUDA_CHECK(cudaMalloc(&sendbuf[dev], bytes));
    CUDA_CHECK(cudaMalloc(&recvbuf[dev], bytes));
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(
        sendbuf[dev], count, (float)(dev + 1));
    CUDA_CHECK(cudaGetLastError());  // launch errors surface here, not later
  }

  // Without the group, device 0's collective cannot complete before device 1's
  // call is enqueued, and one thread issuing them serially deadlocks (3.7).
  NCCL_CHECK(ncclGroupStart());
  for (int dev = 0; dev < ndev; ++dev) {
    NCCL_CHECK(ncclAllReduce(sendbuf[dev], recvbuf[dev], (size_t)count,
                             ncclFloat, ncclSum, comms[dev], streams[dev]));
  }
  NCCL_CHECK(ncclGroupEnd());

  // Nothing above has necessarily executed yet. This is where it completes.
  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaStreamSynchronize(streams[dev]));
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  int failures = 0;
  std::vector<float> host((size_t)count);
  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMemcpy(host.data(), recvbuf[dev], bytes,
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < count; ++i) {
      // Tolerance, not equality: the reduction order is not guaranteed and
      // floating-point addition is not associative (section 3.8).
      if (std::fabs((double)host[i] - expected) > 1e-5 * expected &&
          ++failures <= 5) {
        std::fprintf(stderr, "dev %d elem %d: got %.9g want %.9g\n", dev, i,
                     (double)host[i], expected);
      }
    }
  }

  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(sendbuf[dev]));
    CUDA_CHECK(cudaFree(recvbuf[dev]));
    CUDA_CHECK(cudaStreamDestroy(streams[dev]));
    NCCL_CHECK(ncclCommDestroy(comms[dev]));
  }

  std::printf("%s (expected %.1f on every element of every device)\n",
              failures == 0 ? "PASS" : "FAIL", expected);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
