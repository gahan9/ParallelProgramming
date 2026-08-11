// Single-process, multi-GPU all-reduce with RCCL, verified against a CPU reference.
// Build: hipcc -O2 hip/allreduce_min.cpp -lrccl -o allreduce_min
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(cmd)                                                       \
  do {                                                                       \
    hipError_t e_ = (cmd);                                                   \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd, \
                   hipGetErrorString(e_));                                   \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define RCCL_CHECK(cmd)                                                       \
  do {                                                                        \
    ncclResult_t r_ = (cmd);                                                  \
    if (r_ != ncclSuccess) {                                                  \
      std::fprintf(stderr, "RCCL %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd, \
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
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return EXIT_FAILURE;
  }
  const size_t bytes = (size_t)count * sizeof(float);
  std::printf("all-reduce over %d devices, %d floats (%.1f MiB)\n", ndev, count,
              (double)bytes / (1024.0 * 1024.0));

  std::vector<ncclComm_t> comms(ndev);
  std::vector<hipStream_t> streams(ndev);
  std::vector<float*> sendbuf(ndev, nullptr);
  std::vector<float*> recvbuf(ndev, nullptr);

  // This shortcut exists only for the single-process case; multi-process uses
  // ncclCommInitRank with a broadcast unique ID (section 3.2).
  RCCL_CHECK(ncclCommInitAll(comms.data(), ndev, /*devlist=*/nullptr));

  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&streams[dev]));
    HIP_CHECK(hipMalloc(&sendbuf[dev], bytes));
    HIP_CHECK(hipMalloc(&recvbuf[dev], bytes));
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(
        sendbuf[dev], count, (float)(dev + 1));
    HIP_CHECK(hipGetLastError());  // launch errors surface here, not later
  }

  // Without the group, device 0's collective cannot complete before device 1's
  // call is enqueued, and one thread issuing them serially deadlocks (3.7).
  RCCL_CHECK(ncclGroupStart());
  for (int dev = 0; dev < ndev; ++dev) {
    RCCL_CHECK(ncclAllReduce(sendbuf[dev], recvbuf[dev], (size_t)count,
                             ncclFloat, ncclSum, comms[dev], streams[dev]));
  }
  RCCL_CHECK(ncclGroupEnd());

  // Nothing above has necessarily executed yet. This is where it completes.
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  int failures = 0;
  std::vector<float> host((size_t)count);
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMemcpy(host.data(), recvbuf[dev], bytes,
                        hipMemcpyDeviceToHost));
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
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipFree(sendbuf[dev]));
    HIP_CHECK(hipFree(recvbuf[dev]));
    HIP_CHECK(hipStreamDestroy(streams[dev]));
    RCCL_CHECK(ncclCommDestroy(comms[dev]));
  }

  std::printf("%s (expected %.1f on every element of every device)\n",
              failures == 0 ? "PASS" : "FAIL", expected);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
