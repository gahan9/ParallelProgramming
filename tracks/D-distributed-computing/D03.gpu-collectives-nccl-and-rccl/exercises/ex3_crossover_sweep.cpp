// Drill 3 - find the crossover on your own hardware (D03 lab 3, README 5.2).
//
// Build and run (from this directory):
//   hipcc -O2 -Wall -Wextra ex3_crossover_sweep.cpp -lrccl -o ex3
//   ./ex3 --min-bytes 8 --max-bytes 268435456 --factor 4 --iters 20
//
// The measurement half is written for you, verification included, because timing
// a collective correctly is not the drill. The ANALYSIS is the drill: four marked
// TODOs that turn a table of times into the two numbers that decide your gradient
// bucket size.
//
//   TODO 1  derive algbw and busbw for one row (README 2.4).
//   TODO 2  estimate the plateau from the measured rows. Think about what you are
//           claiming: the largest single row is one sample of a noisy quantity.
//   TODO 3  report, per row, the percentage of that plateau reached, and print the
//           SMALLEST size reaching 90% of it. That size is the practical crossover.
//   TODO 4  predict it from the alpha-beta model instead of measuring it: take
//           alpha from the smallest row and beta from the plateau, then compare.
//           Explain any discrepancy using the protocol transitions in README 3.5.
//
// This file COMPILES AND RUNS as given and reports the analysis as unfinished. The
// measured columns will be real from the first run.
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

struct Row {
  size_t bytes = 0;
  double seconds = 0.0;   // measured, per iteration
  double algbw = 0.0;     // TODO 1
  double busbw = 0.0;     // TODO 1
  bool verified = false;  // measured
};

// TODO 1: fill in row.algbw and row.busbw from row.bytes, row.seconds and ndev.
// algbw is m/t. busbw is algbw * f(N), and f(N) for all-reduce is 2(N-1)/N.
static void derive_bandwidths(Row* row, int ndev) {
  (void)row;
  (void)ndev;
}

// TODO 2: return the plateau bus bandwidth in bytes per second, or 0.0 if the
// sweep never plateaus.
//
// The lazy answer is max(busbw). Consider instead what "plateau" means: the
// region where busbw has stopped growing with size. A defensible estimate uses
// more than one row - for instance the median or mean of the largest few rows,
// or the largest rows whose busbw is within a few percent of each other - and it
// should be able to say "this sweep did not reach a plateau" rather than
// confidently returning the last row.
static double estimate_plateau(const std::vector<Row>& rows) {
  (void)rows;
  return 0.0;
}

// TODO 3: return the smallest byte size whose busbw reaches `fraction` of the
// plateau, or 0 if no row does.
static size_t smallest_reaching(const std::vector<Row>& rows, double plateau,
                                double fraction) {
  (void)rows;
  (void)plateau;
  (void)fraction;
  return 0;
}

// TODO 4: predict the same size from the alpha-beta model.
//
// The ring costs T = 2(N-1)*alpha + 2m(N-1)/N * beta. The smallest row is almost
// pure latency, so it estimates 2(N-1)*alpha. The plateau is pure bandwidth, so
// it estimates 1/beta on the wire. busbw reaches `fraction` of the plateau when
// the wire term is fraction/(1-fraction) times the latency term - work that out
// on paper before you code it, and write the algebra in a comment. Then compare
// with TODO 3's measured answer: the model assumes ONE beta, and README 3.5 says
// the library changes protocol as the message grows, so the two will not agree.
// The size of the disagreement is the interesting result.
static size_t predict_reaching(const std::vector<Row>& rows, double plateau,
                               double fraction, int ndev) {
  (void)rows;
  (void)plateau;
  (void)fraction;
  (void)ndev;
  return 0;
}

// ---------------------------------------------------------------------------
// Measurement harness below. Real, verified, and not part of the drill.
// ---------------------------------------------------------------------------

__global__ void fill_with_rank(float* buf, size_t n, float value) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {  // bounds guard: the grid is rounded up to a whole block
    buf[idx] = value;
  }
}

struct Options {
  size_t min_bytes = 8;
  size_t max_bytes = (size_t)1 << 26;
  int factor = 4;
  int iters = 20;
  int warmup = 5;
  double target = 0.90;
};

static int parse_args(int argc, char** argv, Options* opt) {
  for (int i = 1; i < argc; ++i) {
    const char* flag = argv[i];
    if (std::strcmp(flag, "--help") == 0 || std::strcmp(flag, "-h") == 0) {
      std::fprintf(stderr,
                   "usage: ex3 [--min-bytes N] [--max-bytes N] [--factor F]\n"
                   "           [--iters N] [--warmup N] [--target 0.90]\n");
      return 1;
    }
    const bool known = (std::strcmp(flag, "--min-bytes") == 0 ||
                        std::strcmp(flag, "--max-bytes") == 0 ||
                        std::strcmp(flag, "--factor") == 0 ||
                        std::strcmp(flag, "--iters") == 0 ||
                        std::strcmp(flag, "--warmup") == 0 ||
                        std::strcmp(flag, "--target") == 0);
    if (!known) {
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
    if (end == text || *end != '\0' || !(value >= 0.0) || value > 1.0e18) {
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
    } else if (std::strcmp(flag, "--warmup") == 0) {
      opt->warmup = (int)value;
    } else {
      opt->target = value;
    }
  }
  if (opt->factor < 2 || opt->iters < 1 || opt->warmup < 0) {
    std::fprintf(stderr, "--factor >= 2, --iters >= 1, --warmup >= 0\n");
    return 1;
  }
  if (!(opt->target > 0.0 && opt->target < 1.0)) {
    std::fprintf(stderr, "--target must be strictly between 0 and 1\n");
    return 1;
  }
  opt->min_bytes -= opt->min_bytes % sizeof(float);
  opt->max_bytes -= opt->max_bytes % sizeof(float);
  if (opt->min_bytes < sizeof(float)) {
    opt->min_bytes = sizeof(float);
  }
  if (opt->max_bytes < opt->min_bytes) {
    std::fprintf(stderr, "--max-bytes is below --min-bytes\n");
    return 1;
  }
  return 0;
}

// Always finish ON --max-bytes rather than the last power below it: the largest row
// is the one your plateau estimate will lean on.
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

  const size_t max_count = opt.max_bytes / sizeof(float);
  const int threads = 256;
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamCreate(&streams[dev]));
    HIP_CHECK(hipMalloc(&send[dev], opt.max_bytes));
    HIP_CHECK(hipMalloc(&recv[dev], opt.max_bytes));
    const size_t blocks = (max_count + threads - 1) / threads;
    fill_with_rank<<<(unsigned int)blocks, threads, 0, streams[dev]>>>(
        send[dev], max_count, (float)(dev + 1));
    HIP_CHECK(hipGetLastError());
    fill_with_rank<<<(unsigned int)blocks, threads, 0, streams[dev]>>>(
        recv[dev], max_count, -1.0f);
    HIP_CHECK(hipGetLastError());
  }
  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipStreamSynchronize(streams[dev]));
  }

  const double expected = (double)ndev * (ndev + 1) / 2.0;
  std::vector<Row> rows;
  int failures = 0;

  for (size_t bytes = opt.min_bytes; bytes <= opt.max_bytes;
       bytes = next_size(bytes, opt.max_bytes, opt.factor)) {
    const size_t count = bytes / sizeof(float);
    if (count == 0) {
      continue;
    }
    for (int it = 0; it < opt.warmup; ++it) {
      RCCL_CHECK(ncclGroupStart());
      for (int dev = 0; dev < ndev; ++dev) {
        RCCL_CHECK(ncclAllReduce(send[dev], recv[dev], count, ncclFloat, ncclSum,
                                 comms[dev], streams[dev]));
      }
      RCCL_CHECK(ncclGroupEnd());
    }
    for (int dev = 0; dev < ndev; ++dev) {
      HIP_CHECK(hipSetDevice(dev));
      HIP_CHECK(hipStreamSynchronize(streams[dev]));
    }

    hipEvent_t t0, t1;
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0, streams[0]));
    for (int it = 0; it < opt.iters; ++it) {
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

    Row row;
    row.bytes = bytes;
    row.seconds = (double)ms / 1000.0 / (double)opt.iters;

    // Verify the first, middle and last element on every device. Every element of
    // a correct result is identical, so this catches a corrupted chunk without a
    // full readback dominating the measurement it is checking.
    row.verified = true;
    for (int dev = 0; dev < ndev; ++dev) {
      HIP_CHECK(hipSetDevice(dev));
      const size_t probes[3] = {0, count / 2, count - 1};
      for (int p = 0; p < 3; ++p) {
        float value = 0.0f;
        HIP_CHECK(hipMemcpy(&value, recv[dev] + probes[p], sizeof(float),
                            hipMemcpyDeviceToHost));
        if (std::fabs((double)value - expected) > 1e-5 * expected) {
          std::fprintf(stderr, "dev %d elem %zu: got %.9g want %.9g\n", dev,
                       probes[p], (double)value, expected);
          row.verified = false;
          ++failures;
        }
      }
    }

    derive_bandwidths(&row, ndev);
    rows.push_back(row);
  }

  for (int dev = 0; dev < ndev; ++dev) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipFree(send[dev]));
    HIP_CHECK(hipFree(recv[dev]));
    HIP_CHECK(hipStreamDestroy(streams[dev]));
    RCCL_CHECK(ncclCommDestroy(comms[dev]));
  }

  const double plateau = estimate_plateau(rows);
  std::printf("MEASURED on this machine: %d devices, %d iters per size\n", ndev,
              opt.iters);
  std::printf("%14s %12s %12s %12s %10s  %s\n", "bytes", "time us", "algbw GB/s",
              "busbw GB/s", "% plateau", "verify");
  for (const Row& row : rows) {
    const double pct = (plateau > 0.0) ? 100.0 * row.busbw / plateau : 0.0;
    std::printf("%14zu %12.2f %12.2f %12.2f %10.1f  %s\n", row.bytes,
                row.seconds * 1e6, row.algbw / 1e9, row.busbw / 1e9, pct,
                row.verified ? "ok" : "WRONG");
  }

  if (plateau <= 0.0) {
    std::printf("\nestimate_plateau() returned 0 (TODO 2 not implemented, or the "
                "sweep never plateaus)\n");
  } else {
    const size_t measured = smallest_reaching(rows, plateau, opt.target);
    const size_t predicted = predict_reaching(rows, plateau, opt.target, ndev);
    std::printf("\nplateau %.2f GB/s busbw\n", plateau / 1e9);
    std::printf("smallest size at %.0f%% of plateau: measured %zu, "
                "alpha-beta prediction %zu\n",
                opt.target * 100.0, measured, predicted);
    if (measured == 0 || predicted == 0) {
      std::printf("(a 0 above means that TODO is still a stub)\n");
    }
  }
  std::printf("%s\n", failures == 0 ? "measurement verified"
                                    : "MEASUREMENT NOT VERIFIED");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
