// Drill 3 solution - the crossover sweep and its alpha-beta prediction.
//
// Build and run:
//   hipcc -O2 -Wall -Wextra ex3_crossover_sweep.cpp -lrccl -o ex3
//   ./ex3 --min-bytes 8 --max-bytes 268435456 --factor 4 --iters 20
//
// The measurement is the same shape as hip/busbw_sweep.cpp. What this file adds is
// the analysis, and the analysis is where the judgement lives.
//
// THE PLATEAU IS NOT max(busbw). Taking the maximum of a noisy series estimates
// the maximum of the NOISE as much as the level of the signal: it is biased high by
// construction, and it is biased high by more when the sweep is short or the
// machine is busy. This solution takes the median of the trailing rows that are
// within a few percent of each other, and refuses to report a plateau at all when
// the last two rows are still climbing. "This sweep did not plateau; raise
// --max-bytes" is a better answer than a confident number, because a plateau
// estimate that is really the last row makes every percentage below it meaningless.
//
// THE 90% SIZE IS THE ANSWER YOU ACTUALLY WANT. The crossover m* where the latency
// and wire terms are equal (D01, and tools/busbw.py's factor arithmetic) is only
// 50% of the plateau: at m* half your time is still latency. The size that reaches
// 90% of the plateau is the smallest message worth sending, and therefore the right
// gradient bucket size. The algebra, worked here so the code can be one line:
//
//   T(m)     = L + W(m),  L = 2(N-1)*alpha,  W(m) = 2m(N-1)/N * beta
//   busbw(m) = m*f(N)/T(m), and as m grows this tends to f(N)/(2(N-1)/N * beta)
//              = 1/beta, the plateau. So busbw(m)/plateau = W(m)/(L + W(m)).
//   Setting that to `fraction` gives W = L * fraction/(1-fraction), hence
//              m = (fraction/(1-fraction)) * N * alpha / beta
//                = (fraction/(1-fraction)) * N * alpha * plateau_bandwidth
//   At fraction = 0.5 this is m* = N*alpha*B, matching D01. At 0.9 it is 9x that,
//   which is the practical point: the knee is nearly an order of magnitude above
//   the textbook crossover.
//
// WHY THE PREDICTION MISSES, AND WHY THAT IS THE RESULT. The model has one beta.
// The library does not: it uses LL below a threshold (about half the wire spent on
// inline flags), LL128 above it (about 94% payload), and Simple for large messages
// (README 3.5), and it switches algorithm from tree to ring over roughly the same
// range. So the effective beta IMPROVES as the message grows, which makes the real
// curve rise faster than a single-beta model predicts, and the measured 90% size
// usually lands BELOW the prediction. If instead your measured size is far ABOVE
// the prediction, suspect the other direction: the small-size rows were not pure
// latency (so alpha is overestimated) or something outside the collective - the
// launch path, a busy machine - is in the timing.
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <algorithm>
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
  double seconds = 0.0;
  double algbw = 0.0;
  double busbw = 0.0;
  bool verified = false;
};

static void derive_bandwidths(Row* row, int ndev) {
  if (row->seconds <= 0.0 || ndev < 2) {
    row->algbw = 0.0;
    row->busbw = 0.0;
    return;
  }
  row->algbw = (double)row->bytes / row->seconds;
  row->busbw = row->algbw * 2.0 * (ndev - 1) / ndev;  // f(N) for all-reduce
}

// Median of the trailing rows that agree within `flatness`. Returns 0.0 when the
// sweep is still climbing at its largest size, which is a real answer.
static double estimate_plateau(const std::vector<Row>& rows,
                               double flatness = 0.05) {
  if (rows.size() < 3) {
    return 0.0;
  }
  const Row& last = rows.back();
  const Row& second_last = rows[rows.size() - 2];
  if (second_last.busbw <= 0.0) {
    return 0.0;
  }
  // Still climbing steeply at the top of the sweep: there is no plateau in this
  // data, and inventing one would make every percentage below it fiction.
  if (last.busbw > second_last.busbw * (1.0 + 2.0 * flatness)) {
    return 0.0;
  }
  std::vector<double> tail;
  for (size_t i = rows.size(); i-- > 0;) {
    if (rows[i].busbw <= 0.0) {
      break;
    }
    if (rows[i].busbw < last.busbw * (1.0 - flatness)) {
      break;
    }
    tail.push_back(rows[i].busbw);
  }
  if (tail.size() < 2) {
    return last.busbw;  // one flat row only: report it, and the caller sees why
  }
  std::sort(tail.begin(), tail.end());
  const size_t mid = tail.size() / 2;
  return (tail.size() % 2 == 1) ? tail[mid]
                                : 0.5 * (tail[mid - 1] + tail[mid]);
}

static size_t smallest_reaching(const std::vector<Row>& rows, double plateau,
                                double fraction) {
  if (plateau <= 0.0) {
    return 0;
  }
  for (const Row& row : rows) {
    if (row.busbw >= fraction * plateau) {
      return row.bytes;
    }
  }
  return 0;
}

// m = (fraction/(1-fraction)) * N * alpha * plateau, derived in the header.
static size_t predict_reaching(const std::vector<Row>& rows, double plateau,
                               double fraction, int ndev) {
  if (rows.empty() || plateau <= 0.0 || ndev < 2 || fraction <= 0.0 ||
      fraction >= 1.0) {
    return 0;
  }
  // The smallest row is assumed to be pure latency: T = 2(N-1)*alpha. That
  // assumption is exactly what the comparison below tests.
  const double t_small = rows.front().seconds;
  if (t_small <= 0.0) {
    return 0;
  }
  const double alpha = t_small / (2.0 * (ndev - 1));
  const double ratio = fraction / (1.0 - fraction);
  const double bytes = ratio * (double)ndev * alpha * plateau;
  if (!(bytes > 0.0) || bytes > 1.0e18) {
    return 0;
  }
  return (size_t)bytes;
}

// ---------------------------------------------------------------------------
// Measurement harness, identical to ../exercises/ex3_crossover_sweep.cpp.
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

// Always finish ON --max-bytes rather than the last power below it: estimate_plateau
// reads the trailing rows, so stopping an octave short changes the answer.
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
    std::printf("\nNo plateau in this sweep: busbw is still climbing at the "
                "largest size.\nRaise --max-bytes and re-run. Reporting a "
                "plateau here would be reporting\nthe last row and calling it "
                "a limit.\n");
  } else {
    const size_t measured = smallest_reaching(rows, plateau, opt.target);
    const size_t predicted = predict_reaching(rows, plateau, opt.target, ndev);
    const double alpha = rows.front().seconds / (2.0 * (ndev - 1));
    std::printf("\nplateau            %.2f GB/s busbw (median of the flat tail)\n",
                plateau / 1e9);
    std::printf("alpha from the %zu-byte row: %.2f us per ring step\n",
                rows.front().bytes, alpha * 1e6);
    std::printf("smallest size at %.0f%% of plateau: measured %zu bytes, "
                "alpha-beta prediction %zu bytes\n",
                opt.target * 100.0, measured, predicted);
    if (measured > 0 && predicted > 0) {
      std::printf("  ratio measured/predicted %.2f. Below 1 means the library's "
                  "effective beta\n  improved with size - the LL to LL128 to "
                  "Simple transitions of README 3.5.\n  Above 1 means the "
                  "smallest row was not pure latency, or something outside\n"
                  "  the collective is in the timing.\n",
                  (double)measured / (double)predicted);
    }
    std::printf("Use the measured number as your minimum useful message size, "
                "and therefore\nas your gradient bucket size. It is a property "
                "of THIS machine.\n");
  }
  std::printf("%s\n", failures == 0 ? "measurement verified"
                                    : "MEASUREMENT NOT VERIFIED");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
