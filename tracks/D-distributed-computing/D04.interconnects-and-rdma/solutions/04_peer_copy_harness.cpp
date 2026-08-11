// Solution 4 - peer-copy bandwidth sweep, cross-checked against the topology classes.
//
// Extends hip/peer_copy_bw.cpp from one ordered pair at one size to every ordered pair
// at three sizes, then flags pairs whose measured bandwidth disagrees with the class
// exercise 1's classifier gave them. That flag is the automated form of the silent
// fallback in README section 3.5.
//
// Build: hipcc -O2 solutions/04_peer_copy_harness.cpp -o peer_copy_harness
// Run:   ./peer_copy_harness [classes.txt]
//
// The optional classes file is one ordered pair per line, "src dst class", with '#'
// comments and blank lines ignored:
//
//     # produced from tools/topology_report.py --json
//     0 1 direct-link
//     0 2 cross-socket
//
// Without it the harness falls back to the two classes it can see for itself --
// peer-capable and not -- and says so, because a check that quietly invents its
// reference is worse than one that admits it has none.
//
// NOT COMPILED, NOT RUN. There is no GPU toolchain on the machine this was authored on.
// It is hand-reviewed only; the first build is part of the exercise.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
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

namespace {

// 1 MiB separates latency from bandwidth badly, 256 MiB reaches the asymptote, and the
// middle size is there so a curve with a knee in it cannot masquerade as a flat line.
const size_t kSizes[] = {1u << 20, 16u << 20, 256u << 20};
const size_t kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);
const size_t kBudgetBytes = 256u << 20;  // per size, so small sizes get more iterations
const int kMinIters = 5;
const int kWarmupIters = 3;
const unsigned char kPattern = 0xAB;

struct Measurement {
  int src;
  int dst;
  size_t bytes;
  double gbs;
  int peer;
  std::string klass;
};

typedef std::map<std::pair<int, int>, std::string> ClassMap;

// Returns false only if the file was named and could not be opened or parsed. An absent
// argument is not an error; an unreadable file is.
bool LoadClasses(const char* path, ClassMap* out) {
  std::FILE* file = std::fopen(path, "r");
  if (file == NULL) {
    std::fprintf(stderr, "cannot open classes file '%s'\n", path);
    return false;
  }
  char line[512];
  int lineno = 0;
  while (std::fgets(line, sizeof(line), file) != NULL) {
    ++lineno;
    char* hash = std::strchr(line, '#');
    if (hash != NULL) {
      *hash = '\0';
    }
    int src = -1;
    int dst = -1;
    char klass[128];
    const int fields = std::sscanf(line, "%d %d %127s", &src, &dst, klass);
    if (fields == 0 || fields == EOF) {
      continue;  // blank or comment-only
    }
    if (fields != 3 || src < 0 || dst < 0) {
      std::fprintf(stderr, "%s:%d: expected 'src dst class'\n", path, lineno);
      std::fclose(file);
      return false;
    }
    (*out)[std::make_pair(src, dst)] = std::string(klass);
  }
  std::fclose(file);
  if (out->empty()) {
    std::fprintf(stderr, "%s: no pairs in file\n", path);
    return false;
  }
  return true;
}

int IterationsFor(size_t bytes) {
  const size_t wanted = kBudgetBytes / bytes;
  return wanted < (size_t)kMinIters ? kMinIters : (int)wanted;
}

// One ordered pair at one size. Peer access is enabled here and disabled before
// returning: leaving every pair enabled for the whole run would let a later measurement
// inherit a mapping it did not ask for, and would make the run's result depend on the
// order the pairs were visited. The cost is the enable/disable pair per measurement,
// which is why the warm-up is inside this function rather than once at the top.
void MeasurePair(int src_dev, int dst_dev, size_t bytes,
                 std::vector<unsigned char>* host, double* gbs_out, size_t* wrong_out) {
  int can_access = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dst_dev, src_dev));

  HIP_CHECK(hipSetDevice(dst_dev));
  if (can_access) {
    hipError_t e = hipDeviceEnablePeerAccess(src_dev, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr, "hipDeviceEnablePeerAccess(%d->%d): %s\n", dst_dev, src_dev,
                   hipGetErrorString(e));
      std::exit(EXIT_FAILURE);
    }
    (void)hipGetLastError();  // clear the recorded already-enabled error
  }

  unsigned char* src = NULL;
  unsigned char* dst = NULL;
  HIP_CHECK(hipSetDevice(src_dev));
  HIP_CHECK(hipMalloc(&src, bytes));
  HIP_CHECK(hipMemset(src, kPattern, bytes));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipSetDevice(dst_dev));
  HIP_CHECK(hipMalloc(&dst, bytes));
  HIP_CHECK(hipMemset(dst, 0x00, bytes));

  hipStream_t stream;
  hipEvent_t t0, t1;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));

  for (int i = 0; i < kWarmupIters; ++i) {
    HIP_CHECK(hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  const int iters = IterationsFor(bytes);
  HIP_CHECK(hipEventRecord(t0, stream));
  for (int i = 0; i < iters; ++i) {
    HIP_CHECK(hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  HIP_CHECK(hipEventRecord(t1, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  const double secs = (double)ms / 1000.0;
  *gbs_out = secs > 0.0 ? (double)bytes * (double)iters / secs / 1e9 : 0.0;

  // Every pair, every size. A peer copy that transferred nothing is very fast.
  HIP_CHECK(hipMemcpy(&(*host)[0], dst, bytes, hipMemcpyDeviceToHost));
  size_t wrong = 0;
  for (size_t i = 0; i < bytes; ++i) {
    if ((*host)[i] != kPattern) {
      ++wrong;
    }
  }
  *wrong_out = wrong;

  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dst));
  if (can_access) {
    HIP_CHECK(hipDeviceDisablePeerAccess(src_dev));
  }
  HIP_CHECK(hipSetDevice(src_dev));
  HIP_CHECK(hipFree(src));
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return -1.0;
  }
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  return values.size() % 2 == 1 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

// THE CONSISTENCY RULE, in two sentences, all of it relative to this run:
//   1. A pair is flagged if it reaches less than kSlowFraction of the median of the
//      OTHER pairs sharing its class, which catches one bad link among siblings.
//   2. A direct-link pair is flagged if it fails to reach kTierFactor times the median
//      of the fastest non-direct class, which catches a whole tier being lost.
// Both need a population, so both stay silent on a class with fewer than three members;
// the blind spot that matters more is a UNIFORMLY slow machine, where every pair agrees
// with every other and this check reports nothing at all. Only an absolute expectation
// would catch that, and this program does not have one -- that is what lab 3 and the
// vendor's own bandwidth test are for.
const double kSlowFraction = 0.6;
const double kTierFactor = 1.5;

void ReportInconsistencies(const std::vector<Measurement>& all, size_t largest) {
  std::vector<Measurement> rows;
  for (size_t i = 0; i < all.size(); ++i) {
    if (all[i].bytes == largest) {
      rows.push_back(all[i]);
    }
  }

  std::map<std::string, std::vector<double> > by_class;
  for (size_t i = 0; i < rows.size(); ++i) {
    by_class[rows[i].klass].push_back(rows[i].gbs);
  }

  double best_non_direct = -1.0;
  for (std::map<std::string, std::vector<double> >::const_iterator it = by_class.begin();
       it != by_class.end(); ++it) {
    if (it->first == "direct-link" || it->first == "self" || it->first == "unknown") {
      continue;
    }
    const double median = Median(it->second);
    if (median > best_non_direct) {
      best_non_direct = median;
    }
  }

  std::printf("\nconsistency check (relative, within this run only)\n");
  int alerts = 0;
  for (size_t i = 0; i < rows.size(); ++i) {
    const Measurement& row = rows[i];
    const std::vector<double>& siblings = by_class[row.klass];

    if (siblings.size() >= 3) {
      std::vector<double> others;
      bool dropped = false;
      for (size_t j = 0; j < siblings.size(); ++j) {
        if (!dropped && siblings[j] == row.gbs) {
          dropped = true;  // leave this pair out of its own reference
          continue;
        }
        others.push_back(siblings[j]);
      }
      const double median = Median(others);
      if (median > 0.0 && row.gbs < kSlowFraction * median) {
        std::printf("  ALERT (%d,%d) %s: %.1f GB/s against a class median of %.1f\n",
                    row.src, row.dst, row.klass.c_str(), row.gbs, median);
        ++alerts;
      }
    }

    if (row.klass == "direct-link" && best_non_direct > 0.0 &&
        row.gbs < kTierFactor * best_non_direct) {
      std::printf("  ALERT (%d,%d) reported direct-link but measured %.1f GB/s, near the "
                  "%.1f GB/s of the slower classes -- suspect a silent fallback\n",
                  row.src, row.dst, row.gbs, best_non_direct);
      ++alerts;
    }

    if (row.klass == "unknown") {
      std::printf("  NOTE  (%d,%d) has class 'unknown', so it is measured but not "
                  "checked\n", row.src, row.dst);
    }
  }

  for (std::map<std::string, std::vector<double> >::const_iterator it = by_class.begin();
       it != by_class.end(); ++it) {
    if (it->second.size() < 3) {
      std::printf("  quiet: class '%s' has %d pair(s), too few to compare against\n",
                  it->first.c_str(), (int)it->second.size());
    }
  }
  if (alerts == 0) {
    std::printf("  no pair contradicts its class. This says nothing about whether the "
                "machine as a whole is slow.\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  ClassMap classes;
  bool have_classes = false;
  if (argc > 2) {
    std::fprintf(stderr, "usage: %s [classes.txt]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    if (!LoadClasses(argv[1], &classes)) {
      return EXIT_FAILURE;
    }
    have_classes = true;
  }

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) {
    std::fprintf(stderr, "need at least two GPUs for a peer sweep; found %d\n", ndev);
    return EXIT_FAILURE;
  }

  std::printf("MEASURED on this machine, %d devices, %d ordered pairs x %d sizes\n", ndev,
              ndev * (ndev - 1), (int)kNumSizes);
  if (have_classes) {
    std::printf("classes read from %s\n", argv[1]);
  } else {
    std::printf("no classes file given: falling back to peer-capable / not, which cannot "
                "tell a switch-local PCIe path from a cross-socket one\n");
  }

  size_t largest = 0;
  for (size_t i = 0; i < kNumSizes; ++i) {
    if (kSizes[i] > largest) {
      largest = kSizes[i];
    }
  }
  std::vector<unsigned char> host(largest);

  std::printf("\n%4s %4s %11s %9s %6s  %s\n", "src", "dst", "bytes", "GB/s", "peer",
              "class");
  std::vector<Measurement> all;
  size_t total_wrong = 0;
  for (int src_dev = 0; src_dev < ndev; ++src_dev) {
    for (int dst_dev = 0; dst_dev < ndev; ++dst_dev) {
      if (src_dev == dst_dev) {
        continue;
      }
      int can_access = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dst_dev, src_dev));

      std::string klass;
      if (have_classes) {
        ClassMap::const_iterator it = classes.find(std::make_pair(src_dev, dst_dev));
        klass = it == classes.end() ? std::string("unknown") : it->second;
      } else {
        klass = can_access ? "peer-capable" : "no-peer";
      }

      for (size_t s = 0; s < kNumSizes; ++s) {
        double gbs = 0.0;
        size_t wrong = 0;
        MeasurePair(src_dev, dst_dev, kSizes[s], &host, &gbs, &wrong);
        total_wrong += wrong;
        if (wrong != 0) {
          std::fprintf(stderr, "VERIFY FAILED (%d,%d) at %zu bytes: %zu wrong\n", src_dev,
                       dst_dev, kSizes[s], wrong);
        }
        Measurement row;
        row.src = src_dev;
        row.dst = dst_dev;
        row.bytes = kSizes[s];
        row.gbs = gbs;
        row.peer = can_access;
        row.klass = klass;
        all.push_back(row);
        std::printf("%4d %4d %11zu %9.2f %6s  %s\n", src_dev, dst_dev, kSizes[s], gbs,
                    can_access ? "yes" : "no", klass.c_str());
      }
    }
  }

  ReportInconsistencies(all, largest);

  if (total_wrong != 0) {
    std::fprintf(stderr, "\n%zu wrong bytes in total: the numbers above are void\n",
                 total_wrong);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

// Answers to the questions in ../exercises/04_peer_copy_harness.md:
//
// a) 8% apart is not a difference until you know the spread. Two numbers have no spread,
//    so the honest answer is that the harness cannot tell. What I would add: repeat each
//    measurement at least five times, keep the individual timings rather than one
//    average, and report the median with the min and max -- or the median with the
//    interquartile range if the run is long enough to afford more repeats. What I would
//    report instead of a single number is that triple, because a reader can see from it
//    whether an 8% gap sits inside the noise. If the spread is wide, the interesting
//    output is not the bandwidth at all but WHY it is wide: another tenant, a clock or
//    power cap, or a warm-up that was too short.
//
// b) Reasons not to leave every pair enabled: each enabled direction consumes mapping
//    resources on the device and there is a limit, so a large node can fail an enable
//    late in the sweep and make the result depend on the order pairs were visited; and a
//    measurement of pair (0,1) taken while (0,2) and (0,3) are also mapped is not the
//    same experiment as one taken alone, which quietly destroys comparability between
//    the first row and the last. Enabling and disabling around each measurement costs a
//    pair of runtime calls plus the path setup, which is exactly what the warm-up
//    iterations inside MeasurePair are there to absorb -- it costs run time, not
//    accuracy, and it buys an experiment whose rows are independent of each other.
//
// c) The stream should belong to the device that will do the copy, and for
//    hipMemcpyPeerAsync as written that is the destination device, whose context the
//    copy engine is being asked for; the call is legal with the source device's stream
//    too, and what changes is which device's engine and which stream's ordering the copy
//    joins -- so it can serialise against unrelated work on the wrong device, and on
//    some paths a different engine ends up doing the transfer. How I would test rather
//    than argue: run the sweep twice, once with the stream created on each device, and
//    compare the matrices; then run it again with a long unrelated kernel occupying the
//    source device's default stream, which will show up as a slowdown in exactly one of
//    the two variants. That experiment answers it for the machine in front of me, which
//    is the only place the answer is worth anything.
//
// d) Three candidates from README section 3.7: the link trained to a lower generation
//    than both ends support; fewer lanes than expected, from bifurcation or an
//    electrically narrower slot; and a shared switch uplink that this pair crosses and
//    its siblings do not. The command I would run first is `lspci -vv` on the two
//    devices and read LnkCap against LnkSta -- one output separates the first two
//    causes immediately, because a trained-down link and a narrow link each show as a
//    LnkSta that does not match LnkCap, in the speed field and the width field
//    respectively. If both match, the cause is not the endpoint's link and I would go
//    to `lspci -tv` for the bridge tree and look for the shared uplink.
