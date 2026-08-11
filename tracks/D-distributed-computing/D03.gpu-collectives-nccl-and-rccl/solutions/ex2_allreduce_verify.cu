// Drill 2 solution, NVIDIA form - a verified multi-GPU all-reduce with NCCL.
//
// Build and run:
//   nvcc -O2 ex2_allreduce_verify.cu -lnccl -o ex2_cuda
//   ./ex2_cuda 1048576
//
// This is ex2_allreduce_verify.cpp with the mechanical substitution from README
// section 4: cuda* for hip*, nccl.h for rccl/rccl.h, -lnccl for -lrccl. Every
// nccl* call in the body is byte-identical, which is what the compatibility claim
// in section 3.1 means in practice. The reasoning behind each of the four
// requirements - check every call, check after the launch, verify with a
// tolerance, free on every path - is in the .cpp header and is not repeated here.
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

__global__ void fill_with_rank(float* buf, int n, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {  // the grid is rounded up to a whole block; this is not optional
    buf[idx] = value;
  }
}

static int verify_device(const float* host, int count, double expected, int dev) {
  const int kMaxReports = 5;
  int bad = 0;
  for (int i = 0; i < count; ++i) {
    if (std::fabs((double)host[i] - expected) > 1e-5 * expected) {
      if (bad < kMaxReports) {
        std::fprintf(stderr, "dev %d elem %d: got %.9g want %.9g\n", dev, i,
                     (double)host[i], expected);
      }
      ++bad;
    }
  }
  if (bad > kMaxReports) {
    std::fprintf(stderr, "dev %d: %d further mismatches not shown\n", dev,
                 bad - kMaxReports);
  }
  return bad;
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
    CUDA_CHECK(cudaGetLastError());
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(recvbuf[dev], count,
                                                         -1.0f);
    CUDA_CHECK(cudaGetLastError());
  }

  NCCL_CHECK(ncclGroupStart());
  for (int dev = 0; dev < ndev; ++dev) {
    NCCL_CHECK(ncclAllReduce(sendbuf[dev], recvbuf[dev], (size_t)count,
                             ncclFloat, ncclSum, comms[dev], streams[dev]));
  }
  NCCL_CHECK(ncclGroupEnd());

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
    failures += verify_device(host.data(), count, expected, dev);
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
