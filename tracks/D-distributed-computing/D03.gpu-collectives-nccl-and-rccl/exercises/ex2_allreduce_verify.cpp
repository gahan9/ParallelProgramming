// Drill 2 - a verified multi-GPU all-reduce, written from scratch (D03 lab 1).
//
// Build and run (from this directory):
//   hipcc -O2 -Wall -Wextra ex2_allreduce_verify.cpp -lrccl -o ex2
//   ./ex2 1048576
//
// This file COMPILES AND RUNS as given and FAILS its own verification, because
// four pieces are marked TODO. A FAIL on the first run means the setup works.
//
// Note what is already here and what is not. Every runtime and library call is
// ALREADY wrapped in HIP_CHECK or RCCL_CHECK - that part is not optional and not
// a stylistic preference. What is missing is the checking INSIDE those macros:
// TODO 1 is to make them do their job. A hollow checking macro is worse than no
// macro, because it looks like diligence in review. Shipping the stubs below is
// the anti-pattern this drill inoculates you against.
//
//   TODO 1  make HIP_CHECK and RCCL_CHECK report file, line, expression and the
//           library's own error string, then exit non-zero.
//   TODO 2  write the fill kernel, including the bounds guard.
//   TODO 3  issue the all-reduce for every device inside one group.
//   TODO 4  verify against the closed-form CPU reference, with a tolerance.
//
// Then, once it passes, run the two ablations and write down what each does on
// YOUR machine:
//   a) delete ncclGroupStart/ncclGroupEnd and re-run;
//   b) delete the post-collective hipStreamSynchronize loop and re-run, at a
//      small count and then at a large one.
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// TODO 1: these compile and check nothing. Replace both bodies.
//
// A finished HIP_CHECK evaluates its argument exactly once, compares against
// hipSuccess, and on failure prints __FILE__, __LINE__, the stringised
// expression and hipGetErrorString() before exiting non-zero. The do/while(0)
// wrapper is what lets the macro sit inside an unbraced if without changing
// which statement the else binds to. RCCL_CHECK is the same shape against
// ncclSuccess and ncclGetErrorString().
#define HIP_CHECK(cmd)   do { (void)(cmd); } while (0)
#define RCCL_CHECK(cmd)  do { (void)(cmd); } while (0)

// TODO 2: fill buf[0 .. n) with `value`.
//
// The launch below rounds the grid up to a whole block, so the last block has
// threads with idx >= n. Without a guard those threads write past the end of the
// allocation, which is a silent heap corruption on a good day.
__global__ void fill_with_rank(float* buf, int n, float value) {
  (void)buf;
  (void)n;
  (void)value;
}

// TODO 4: return the number of mismatching elements on this device.
//
// `expected` is the closed form: every device contributed (dev + 1) to every
// element, so a correct sum is 1 + 2 + ... + ndev = ndev(ndev+1)/2 everywhere.
// Compare with a RELATIVE TOLERANCE, not with ==: the library chooses the
// reduction order from the topology and floating-point addition is not
// associative (README 3.8). Report at most a few mismatches, then stop printing
// but keep counting - a wrong buffer produces millions of lines otherwise.
static int verify_device(const float* host, int count, double expected, int dev) {
  (void)host;
  (void)count;
  (void)expected;
  (void)dev;
  return -1;  // -1 means "not implemented"; delete this once TODO 4 is written
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

  // ncclCommInitAll is the single-process shortcut. Multi-process code cannot use
  // it: it needs ncclGetUniqueId, an out-of-band broadcast of that id, and
  // ncclCommInitRank on every rank (README 3.2).
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
    HIP_CHECK(hipGetLastError());  // launch errors surface here, not later
    // A sentinel no correct result can hold, so an untouched recvbuf is
    // detectable rather than accidentally plausible.
    fill_with_rank<<<blocks, threads, 0, streams[dev]>>>(recvbuf[dev], count,
                                                         -1.0f);
    HIP_CHECK(hipGetLastError());
  }

  // TODO 3: one ncclGroupStart, one ncclAllReduce per device on that device's
  // stream, one ncclGroupEnd. Without the group, the collective for device 0
  // cannot complete until device 1's call is enqueued, and a single thread
  // issuing them serially deadlocks (README 3.7).
  RCCL_CHECK(ncclGroupStart());
  RCCL_CHECK(ncclGroupEnd());

  // Nothing above has necessarily executed yet. This is where it completes, and
  // this loop is what makes the read below defined.
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  int failures = 0;
  bool unimplemented = false;
  std::vector<float> host((size_t)count);
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipMemcpy(host.data(), recvbuf[dev], bytes, hipMemcpyDeviceToHost));
    const int bad = verify_device(host.data(), count, expected, dev);
    if (bad < 0) {
      unimplemented = true;
    } else {
      failures += bad;
    }
  }

  // Free everything on every path. A leak on the failing path is still a leak,
  // and this program is expected to fail until the TODOs are done.
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipFree(sendbuf[dev]));
    HIP_CHECK(hipFree(recvbuf[dev]));
    HIP_CHECK(hipStreamDestroy(streams[dev]));
    RCCL_CHECK(ncclCommDestroy(comms[dev]));
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
