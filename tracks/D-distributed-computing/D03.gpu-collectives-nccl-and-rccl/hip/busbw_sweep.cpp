// Lab 3: an all-reduce bandwidth sweep across message sizes, on RCCL.
//
// Build: hipcc -O2 hip/busbw_sweep.cpp -lrccl -o busbw_sweep
// Run:   ./busbw_sweep --min-bytes 8 --max-bytes 1073741824 --factor 2 --iters 20
//
// Every number this program prints is a MEASUREMENT of the machine it ran on.
// Nothing here is modelled, and nothing here is portable between machines: the
// plateau height is your fabric's effective all-reduce bandwidth and the knee is
// your minimum useful message size (README section 5.2).
//
// The timed region is the block quoted in README section 4, lab 3, verbatim. If
// you change one, change the other.
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

// Rank-dependent fill, so the expected sum is the closed form N(N+1)/2 and the
// CPU reference needs no communication of its own.
__global__ void fill_with_rank(float* buf, size_t n, float value) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {  // bounds guard: the grid is rounded up to a whole block
    buf[idx] = value;
  }
}

struct Options {
  size_t min_bytes = 8;
  size_t max_bytes = (size_t)1 << 26;  // 64 MiB: finishes quickly, shows the knee
  int factor = 4;
  int iters = 20;
  int warmup = 5;
};

static void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s [--min-bytes N] [--max-bytes N] [--factor F]\n"
               "          [--iters N] [--warmup N]\n"
               "  sizes are in bytes and are rounded down to a multiple of 4;\n"
               "  --factor must be >= 2, --iters >= 1, --warmup >= 0\n",
               argv0);
}

// Returns 0 on success. Refuses bad input with one line rather than guessing.
static int parse_args(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const char* flag = argv[i];
    const bool needs_value = (std::strcmp(flag, "--min-bytes") == 0 ||
                              std::strcmp(flag, "--max-bytes") == 0 ||
                              std::strcmp(flag, "--factor") == 0 ||
                              std::strcmp(flag, "--iters") == 0 ||
                              std::strcmp(flag, "--warmup") == 0);
    if (std::strcmp(flag, "--help") == 0 || std::strcmp(flag, "-h") == 0) {
      usage(argv[0]);
      return 1;
    }
    if (!needs_value) {
      std::fprintf(stderr, "unknown argument '%s'\n", flag);
      return 1;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "'%s' needs a value\n", flag);
      return 1;
    }
    const char* text = argv[++i];
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= 0.0) ||
        value > 1.0e18) {
      std::fprintf(stderr, "'%s' is not a usable value for %s\n", text, flag);
      return 1;
    }
    if (std::strcmp(flag, "--min-bytes") == 0) {
      opt->min_bytes = (size_t)value;
    } else if (std::strcmp(flag, "--max-bytes") == 0) {
      opt->max_bytes = (size_t)value;
    } else if (std::strcmp(flag, "--factor") == 0) {
      opt->factor = (int)value;
    } else if (std::strcmp(flag, "--iters") == 0) {
      opt->iters = (int)value;
    } else {
      opt->warmup = (int)value;
    }
  }
  if (opt->factor < 2) {
    std::fprintf(stderr, "--factor must be at least 2\n");
    return 1;
  }
  if (opt->iters < 1) {
    std::fprintf(stderr, "--iters must be at least 1\n");
    return 1;
  }
  if (opt->warmup < 0) {
    std::fprintf(stderr, "--warmup cannot be negative\n");
    return 1;
  }
  opt->min_bytes -= opt->min_bytes % sizeof(float);
  opt->max_bytes -= opt->max_bytes % sizeof(float);
  if (opt->min_bytes < sizeof(float)) {
    opt->min_bytes = sizeof(float);
  }
  if (opt->max_bytes < opt->min_bytes) {
    std::fprintf(stderr, "--max-bytes (%zu) is below --min-bytes (%zu)\n",
                 opt->max_bytes, opt->min_bytes);
    return 1;
  }
  return 0;
}

// Geometric progression that always finishes ON --max-bytes rather than the last
// power below it. The largest row is the one a plateau estimate leans on, so
// stopping an octave short would quietly move the answer the sweep exists to give.
static size_t next_size(size_t bytes, size_t max_bytes, int factor) {
  if (bytes >= max_bytes) {
    return max_bytes + 1;  // out of range: ends the loop
  }
  if (bytes > max_bytes / (size_t)factor) {
    return max_bytes;
  }
  return bytes * (size_t)factor;
}

int main(int argc, char** argv) {
  Options opt;
  if (parse_args(argc, argv, &opt) != 0) {
    return EXIT_FAILURE;
  }

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least 2 GPUs, found %d\n", ndev);
    return EXIT_FAILURE;
  }

  std::vector<ncclComm_t> comms(ndev);
  std::vector<hipStream_t> streams(ndev);
  std::vector<float*> send(ndev, nullptr);
  std::vector<float*> recv(ndev, nullptr);

  RCCL_CHECK(ncclCommInitAll(comms.data(), ndev, /*devlist=*/nullptr));

  // One allocation per device at the largest size, reused for every row. Per-size
  // allocation would put buffer registration inside the sweep, which is exactly
  // the cost the warmup loop exists to keep out of the timings (README 6.2).
  const size_t max_count = opt.max_bytes / sizeof(float);
  const int threads = 256;
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&streams[dev]));
    HIP_CHECK(hipMalloc(&send[dev], opt.max_bytes));
    HIP_CHECK(hipMalloc(&recv[dev], opt.max_bytes));
    HIP_CHECK(hipMemsetAsync(recv[dev], 0, opt.max_bytes, streams[dev]));
    const size_t blocks64 = (max_count + threads - 1) / threads;
    fill_with_rank<<<(unsigned int)blocks64, threads, 0, streams[dev]>>>(
        send[dev], max_count, (float)(dev + 1));
    HIP_CHECK(hipGetLastError());  // launch errors surface here, not later
  }
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  std::printf("MEASURED on this machine: all-reduce over %d devices, "
              "%d iters per size (%d warmup)\n",
              ndev, opt.iters, opt.warmup);
  std::printf("busbw = algbw * 2(N-1)/N = algbw * %.4f  (README 2.4)\n",
              2.0 * (ndev - 1) / ndev);
  std::printf("%14s %14s %12s %12s %12s  %s\n", "bytes", "elements",
              "time us", "algbw GB/s", "busbw GB/s", "verify");
  std::printf("--------------------------------------------------"
              "--------------------------------\n");

  // Verification samples rather than reading back every element of every size:
  // a full device-to-host copy of a 1 GiB buffer per device per row would cost
  // more than the collective being measured and would change what is timed.
  // Every element of a correct result is identical here, so a strided sample
  // that always includes the first and last element detects any corruption of a
  // chunk. The count checked is printed so the claim stays honest.
  const size_t max_samples = 1u << 12;
  std::vector<float> host(max_samples + 1);

  int failures = 0;
  double best_busbw = 0.0;
  size_t best_bytes = 0;

  for (size_t bytes = opt.min_bytes; bytes <= opt.max_bytes;
       bytes = next_size(bytes, opt.max_bytes, opt.factor)) {
    const size_t count = bytes / sizeof(float);
    if (count == 0) {
      continue;
    }

    for (int it = 0; it < opt.warmup; ++it) {
      RCCL_CHECK(ncclGroupStart());
      for (int dev = 0; dev < ndev; ++dev) {
        RCCL_CHECK(ncclAllReduce(send[dev], recv[dev], count, ncclFloat,
                                 ncclSum, comms[dev], streams[dev]));
      }
      RCCL_CHECK(ncclGroupEnd());
    }
    for (int dev = 0; dev < ndev; ++dev) {
      HIP_CHECK(hipSetDevice(dev));
      HIP_CHECK(hipStreamSynchronize(streams[dev]));
    }

    const int iters = opt.iters;
    // ---- begin the block quoted in README section 4, lab 3 ----
    hipEvent_t t0, t1;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0, streams[0]));
    for (int it = 0; it < iters; ++it) {
      RCCL_CHECK(ncclGroupStart());
      for (int dev = 0; dev < ndev; ++dev) {
        RCCL_CHECK(ncclAllReduce(send[dev], recv[dev], count, ncclFloat, ncclSum,
                                 comms[dev], streams[dev]));
      }
      RCCL_CHECK(ncclGroupEnd());
    }
    HIP_CHECK(hipEventRecord(t1, streams[0]));
    for (int dev = 0; dev < ndev; ++dev) {
      HIP_CHECK(hipSetDevice(dev));
      HIP_CHECK(hipStreamSynchronize(streams[dev]));
    }
    float ms = 0.0f;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    HIP_CHECK(hipEventDestroy(t0));
    HIP_CHECK(hipEventDestroy(t1));

    const double secs = (double)ms / 1000.0 / (double)iters;
    const double algbw = (double)bytes / secs;              // section 2.4
    const double busbw = algbw * 2.0 * (ndev - 1) / ndev;   // f(N) for all-reduce
    // ---- end of the quoted block ----

    const size_t samples = (count < max_samples) ? count : max_samples;
    const size_t stride = count / samples;  // >= 1 by construction
    int row_failures = 0;
    for (int dev = 0; dev < ndev && row_failures == 0; ++dev) {
      HIP_CHECK(hipSetDevice(dev));
      for (size_t s = 0; s < samples; ++s) {
        const size_t idx = s * stride;
        HIP_CHECK(hipMemcpy(&host[s], recv[dev] + idx, sizeof(float),
                            hipMemcpyDeviceToHost));
        if (std::fabs((double)host[s] - expected) > 1e-5 * expected) {
          ++row_failures;
          std::fprintf(stderr, "dev %d elem %zu: got %.9g want %.9g\n", dev, idx,
                       (double)host[s], expected);
          break;
        }
      }
      // The last element is the one a truncated final chunk would corrupt, so
      // check it explicitly rather than hoping the stride landed on it.
      float tail = 0.0f;
      HIP_CHECK(hipMemcpy(&tail, recv[dev] + (count - 1), sizeof(float),
                          hipMemcpyDeviceToHost));
      if (std::fabs((double)tail - expected) > 1e-5 * expected) {
        ++row_failures;
        std::fprintf(stderr, "dev %d last elem %zu: got %.9g want %.9g\n", dev,
                     count - 1, (double)tail, expected);
      }
    }
    failures += row_failures;

    if (busbw > best_busbw) {
      best_busbw = busbw;
      best_bytes = bytes;
    }
    std::printf("%14zu %14zu %12.2f %12.2f %12.2f  %s\n", bytes, count,
                secs * 1e6, algbw / 1e9, busbw / 1e9,
                row_failures == 0 ? "ok" : "WRONG");
  }

  std::printf("--------------------------------------------------"
              "--------------------------------\n");
  std::printf("peak busbw %.2f GB/s at %zu bytes; %zu elements sampled per "
              "device per row\n",
              best_busbw / 1e9, best_bytes,
              (size_t)((max_samples < max_count) ? max_samples : max_count));

  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipFree(send[dev]));
    HIP_CHECK(hipFree(recv[dev]));
    HIP_CHECK(hipStreamDestroy(streams[dev]));
    RCCL_CHECK(ncclCommDestroy(comms[dev]));
  }

  std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
  std::printf("Read the busbw column: flat and low, then a steep rise, then a "
              "plateau. The\nplateau is this fabric's effective all-reduce "
              "bandwidth; the knee is the smallest\nmessage worth sending. "
              "Neither number transfers to another machine.\n");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
