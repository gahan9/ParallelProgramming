// Drill 2, NVIDIA form - a verified multi-GPU all-reduce from scratch (D03 lab 1).
//
// Build and run (from this directory):
//   nvcc -O2 ex2_allreduce_verify.cu -lnccl -o ex2_cuda
//   ./ex2_cuda 1048576
//
// This is ex2_allreduce_verify.cpp with the mechanical substitution from README
// section 4: cuda* for hip*, nccl.h for rccl/rccl.h, -lnccl for -lrccl. Every
// nccl* call is identical, which is the portability claim made concrete.
//
// It COMPILES AND RUNS as given and FAILS its own verification: four pieces are
// marked TODO. Every runtime and library call is already wrapped in a checking
// macro; TODO 1 is to make those macros actually check. A hollow checking macro
// is worse than none, because it looks like diligence in review.
//
//   TODO 1  make CUDA_CHECK and NCCL_CHECK report file, line, expression and the
//           library's error string, then exit non-zero.
//   TODO 2  write the fill kernel, including the bounds guard.
//   TODO 3  issue the all-reduce for every device inside one group.
//   TODO 4  verify against the closed-form CPU reference, with a tolerance.
#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TODO 1: these compile and check nothing. Replace both bodies.
#define CUDA_CHECK(cmd)  do { (void)(cmd); } while (0)
#define NCCL_CHECK(cmd)  do { (void)(cmd); } while (0)

// TODO 2: fill buf[0 .. n) with `value`. The launch rounds the grid up to a whole
// block, so the last block has threads with idx >= n and needs a guard.
__global__ void fill_with_rank(float* buf, int n, float value) {
  (void)buf;
  (void)n;
  (void)value;
}

// TODO 4: return the number of mismatching elements on this device. `expected` is
// ndev(ndev+1)/2. Compare with a relative tolerance, not with ==.
static int verify_device(const float* host, int count, double expected, int dev) {
  (void)host;
  (void)count;
  (void)expected;
  (void)dev;
  return -1;  // -1 means "not implemented"
}

int main(int argc, char** argv) {
  const int count = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
  if (count <= 0) {
    std::fprintf(stderr, "usage: ex2_cuda [element count > 0]\n");
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

  NCCL_CHECK(ncclCommInitAll(comms.data(), ndev, /*devlist=*/nullptr));

  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaStreamCreate(&streams[dev]));
    CUDA_CHECK(cudaMalloc(&sendbuf[dev], bytes));
    CUDA_CHECK(cudaMalloc(&recvbuf[dev], bytes));
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(sendbuf[dev], count,
                                                         (float)(dev + 1));
    CUDA_CHECK(cudaGetLastError());  // launch errors surface here, not later
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(recvbuf[dev], count,
                                                         -1.0f);
    CUDA_CHECK(cudaGetLastError());
  }

  // TODO 3: one ncclGroupStart, one ncclAllReduce per device on that device's
  // stream, one ncclGroupEnd.
  NCCL_CHECK(ncclGroupStart());
  NCCL_CHECK(ncclGroupEnd());

  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaStreamSynchronize(streams[dev]));
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  int failures = 0;
  bool unimplemented = false;
  std::vector<float> host((size_t)count);
  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMemcpy(host.data(), recvbuf[dev], bytes,
                          cudaMemcpyDeviceToHost));
    const int bad = verify_device(host.data(), count, expected, dev);
    if (bad < 0) {
      unimplemented = true;
    } else {
      failures += bad;
    }
  }

  for (int dev = 0; dev < ndev; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(sendbuf[dev]));
    CUDA_CHECK(cudaFree(recvbuf[dev]));
    CUDA_CHECK(cudaStreamDestroy(streams[dev]));
    NCCL_CHECK(ncclCommDestroy(comms[dev]));
  }

  if (unimplemented) {
    std::printf("FAIL: verify_device() is still the stub (TODO 4). Expected "
                "until the TODOs are done.\n");
    return EXIT_FAILURE;
  }
  std::printf("%s (expected %.1f on every element of every device)\n",
              failures == 0 ? "PASS" : "FAIL", expected);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
