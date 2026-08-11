// Lab 4: the asynchrony trap. ncclAllReduce ENQUEUES and returns; when it
// returns the result is not in recvbuf, sendbuf is not safe to reuse or free,
// and no rank has synchronized with any other (README section 3.7).
//
// Build: hipcc -O2 hip/stream_trap.cpp -lrccl -o stream_trap
// Run:
//   ./stream_trap --no-sync --count 268435456   # large buffer: expect FAIL
//   ./stream_trap --no-sync --count 1024        # small buffer: often PASSES
//   ./stream_trap --sync    --count 268435456   # correct: PASS
//   ./stream_trap --no-sync --free-early --count 268435456   # worse than wrong
//
// The middle invocation is the lesson: incorrect code that passes is how this
// bug reaches production. A FAIL from --no-sync is the program working.
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(cmd)                                                       \
  do {                                                                       \
    hipError_t e_ = (cmd);                                                   \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd, \
                   hipGetErrorString(e_));                                   \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                       \
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

__global__ void fill(float* buf, size_t n, float value) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {  // bounds guard: the grid is rounded up to a whole block
    buf[idx] = value;
  }
}

static void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--sync | --no-sync] [--count N] [--free-early]\n"
               "  --sync        synchronize the streams before reading (correct)\n"
               "  --no-sync     read the buffer straight after enqueue (the bug)\n"
               "  --free-early  free sendbuf before synchronizing (worse: the\n"
               "                library may still be reading it)\n"
               "  --count N     elements per device, 1 .. 2^30\n",
               argv0);
}

int main(int argc, char** argv) {
  bool synchronize = true;
  bool free_early = false;
  long long requested = 1 << 20;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--sync") == 0) {
      synchronize = true;
    } else if (std::strcmp(argv[i], "--no-sync") == 0) {
      synchronize = false;
    } else if (std::strcmp(argv[i], "--free-early") == 0) {
      free_early = true;
    } else if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
      char* end = nullptr;
      requested = std::strtoll(argv[++i], &end, 10);
      if (end == argv[i] || *end != '\0') {
        std::fprintf(stderr, "--count needs an integer, got '%s'\n", argv[i]);
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (requested < 1 || requested > (1LL << 30)) {
    std::fprintf(stderr, "--count must be in 1 .. 2^30, got %lld\n", requested);
    return EXIT_FAILURE;
  }
  const size_t count = (size_t)requested;
  const size_t bytes = count * sizeof(float);

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return EXIT_FAILURE;
  }

  std::vector<ncclComm_t> comms(ndev);
  std::vector<hipStream_t> streams(ndev);
  std::vector<float*> sendbuf(ndev, nullptr);
  std::vector<float*> recvbuf(ndev, nullptr);

  RCCL_CHECK(ncclCommInitAll(comms.data(), ndev, /*devlist=*/nullptr));

  const int threads = 256;
  const size_t blocks = (count + threads - 1) / threads;
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&streams[dev]));
    HIP_CHECK(hipMalloc(&sendbuf[dev], bytes));
    HIP_CHECK(hipMalloc(&recvbuf[dev], bytes));
    fill<<<(unsigned int)blocks, threads, 0, streams[dev]>>>(
        sendbuf[dev], count, (float)(dev + 1));
    HIP_CHECK(hipGetLastError());
    // A sentinel no correct result can hold, so an unwritten recvbuf is
    // detectable rather than accidentally plausible.
    fill<<<(unsigned int)blocks, threads, 0, streams[dev]>>>(recvbuf[dev], count,
                                                            -1.0f);
    HIP_CHECK(hipGetLastError());
  }
  // The setup must be complete before the collective, always. This lab is about
  // the synchronization AFTER the collective, not about skipping this one.
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
  }

  std::vector<float> host(count);
  std::printf("stream_trap: %d devices, %zu elements (%.1f MiB), %s%s\n", ndev,
              count, (double)bytes / (1024.0 * 1024.0),
              synchronize ? "synchronizing (correct)" : "NOT synchronizing (the bug)",
              !free_early ? ""
              : synchronize ? ", freeing sendbuf after the sync (defined, so no trap)"
                            : ", freeing sendbuf early (undefined behaviour)");

  // ---- begin the block quoted in README section 4, lab 4 ----
  RCCL_CHECK(ncclGroupStart());
  for (int dev = 0; dev < ndev; ++dev) {
    RCCL_CHECK(ncclAllReduce(sendbuf[dev], recvbuf[dev], (size_t)count,
                             ncclFloat, ncclSum, comms[dev], streams[dev]));
  }
  RCCL_CHECK(ncclGroupEnd());

  if (synchronize) {
    for (int dev = 0; dev < ndev; ++dev) {
      HIP_CHECK(hipSetDevice(dev));
      HIP_CHECK(hipStreamSynchronize(streams[dev]));
    }
  }

  // With `synchronize == false` this reads a buffer whose contents are
  // undefined: the collective was enqueued, not completed. The copy is ordered
  // on a different stream, which is why the bug is intermittent rather than
  // deterministic -- the worst possible failure signature.
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMemcpy(host.data(), recvbuf[0], (size_t)count * sizeof(float),
                      hipMemcpyDeviceToHost));
  // ---- end of the quoted block ----

  if (free_early) {
    // Deliberately wrong WHEN --no-sync: the library may still be reading
    // sendbuf. Expect a fault, a hang, or silent corruption -- which of the
    // three you get is a property of timing, not of the code. With --sync the
    // streams have already drained, so this same free is perfectly legal; the
    // flag is only a trap in combination, which is the point worth taking away.
    std::printf(synchronize
                    ? "freeing sendbuf after synchronizing: legal, nothing to see\n"
                    : "freeing sendbuf with the collective still possibly "
                      "reading it\n");
    for (int dev = 0; dev < ndev; ++dev) {
      HIP_CHECK(hipSetDevice(dev));
      HIP_CHECK(hipFree(sendbuf[dev]));
      sendbuf[dev] = nullptr;
    }
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  size_t mismatches = 0;
  size_t first_bad = 0;
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs((double)host[i] - expected) > 1e-5 * expected) {
      if (mismatches == 0) {
        first_bad = i;
      }
      ++mismatches;
    }
  }

  // Everything must be released even on the failing path: this program is
  // expected to fail, and a leak on the failure path is still a leak.
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
    if (sendbuf[dev] != nullptr) {
      HIP_CHECK(hipFree(sendbuf[dev]));
    }
    HIP_CHECK(hipFree(recvbuf[dev]));
    HIP_CHECK(hipStreamDestroy(streams[dev]));
    RCCL_CHECK(ncclCommDestroy(comms[dev]));
  }

  if (mismatches == 0) {
    std::printf("PASS: every element is %.1f\n", expected);
    if (!synchronize) {
      std::printf("  ...and that is the trap. The collective happened to finish\n"
                  "  before the host read the buffer. The code is still wrong;\n"
                  "  raise --count until it fails, then note how large that had\n"
                  "  to be. In production the buffer grows for you.\n");
    }
  } else {
    std::printf("FAIL: %zu of %zu elements wrong, first at index %zu "
                "(got %.9g, want %.1f)\n",
                mismatches, count, first_bad, (double)host[first_bad], expected);
    if (!synchronize) {
      std::printf("  This FAIL is the correct outcome for --no-sync: the read\n"
                  "  raced the collective. Re-run with --sync and it passes.\n");
    }
  }
  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
