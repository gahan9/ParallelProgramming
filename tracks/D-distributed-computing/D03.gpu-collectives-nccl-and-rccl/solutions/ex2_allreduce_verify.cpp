// Drill 2 solution - a verified multi-GPU all-reduce with RCCL (D03 lab 1).
//
// Build and run:
//   hipcc -O2 -Wall -Wextra ex2_allreduce_verify.cpp -lrccl -o ex2
//   ./ex2 1048576
//
// Four things this program does that a first attempt usually does not, and each
// is the reason a reviewer asks for it:
//
// 1. EVERY runtime and library call is checked. Not the interesting ones: all of
//    them. The calls that fail in the field are the boring ones - hipSetDevice on
//    a device that has fallen off the bus, hipMalloc under memory pressure - and
//    an unchecked failure there produces plausible garbage a long way from the
//    cause. The macro prints file, line, the stringised expression and the
//    library's own error string, which is the difference between a bug report and
//    a bisect.
// 2. hipGetLastError() AFTER the launch. A kernel launch is asynchronous and
//    returns void, so a bad launch configuration is only visible here. Checking
//    it costs nothing and is the single most commonly skipped check in HIP code.
// 3. The result is verified against a CPU reference with a TOLERANCE. The
//    reference is a closed form - every device contributes (dev+1) to every
//    element, so a correct sum is ndev(ndev+1)/2 - which means the check needs no
//    second reduction of its own to be wrong in the same way. The tolerance is
//    not laziness: the library picks its reduction order from the discovered
//    topology, floating-point addition is not associative, and an exact
//    comparison would be asserting a guarantee neither library makes (README 3.8).
// 4. Everything is released on every path, including the failing one.
//
// The two ablations the drill asks for, and what each does:
//
//   Remove ncclGroupStart/ncclGroupEnd: on most configurations this HANGS. The
//   collective for device 0 cannot complete until every rank has arrived, and a
//   single thread issuing them serially never reaches device 1's call. The group
//   defers the launch until all members are known (README 3.7).
//
//   Remove the post-collective hipStreamSynchronize loop: at a small count it
//   usually still PASSES, because the collective finishes before the host reads
//   the buffer. At a large count it FAILS. That is the worst failure signature
//   there is - correct code and incorrect code are indistinguishable at test
//   sizes - and it is why hip/stream_trap.cpp exists as its own lab.
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

__global__ void fill_with_rank(float* buf, int n, float value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {  // the grid is rounded up to a whole block; this is not optional
    buf[idx] = value;
  }
}

// Returns the number of mismatching elements. Prints at most `kMaxReports` of
// them: a wrong buffer is wrong everywhere, and a million lines of it buries the
// one line that says which device.
static int verify_device(const float* host, int count, double expected, int dev) {
  const int kMaxReports = 5;
  int bad = 0;
  for (int i = 0; i < count; ++i) {
    // Relative tolerance. `expected` is a small integer here so the sum is in
    // fact exact, but writing the check this way is what generalises to real
    // data, where the reduction order is the library's business and not yours.
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
    std::fprintf(stderr, "usage: ex2 [element count > 0]\n");
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

  RCCL_CHECK(ncclCommInitAll(comms.data(), ndev, /*devlist=*/nullptr));

  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&streams[dev]));
    HIP_CHECK(hipMalloc(&sendbuf[dev], bytes));
    HIP_CHECK(hipMalloc(&recvbuf[dev], bytes));
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(sendbuf[dev], count,
                                                         (float)(dev + 1));
    HIP_CHECK(hipGetLastError());
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(recvbuf[dev], count,
                                                         -1.0f);
    HIP_CHECK(hipGetLastError());
  }

  RCCL_CHECK(ncclGroupStart());
  for (int dev = 0; dev < ndev; ++dev) {
    RCCL_CHECK(ncclAllReduce(sendbuf[dev], recvbuf[dev], (size_t)count,
                             ncclFloat, ncclSum, comms[dev], streams[dev]));
  }
  RCCL_CHECK(ncclGroupEnd());

  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  int failures = 0;
  std::vector<float> host((size_t)count);
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMemcpy(host.data(), recvbuf[dev], bytes, hipMemcpyDeviceToHost));
    failures += verify_device(host.data(), count, expected, dev);
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
