// Unidirectional peer-to-peer copy bandwidth between two GPUs, verified.
// Build: hipcc -O2 hip/peer_copy_bw.cpp -o peer_copy_bw
// Run:   ./peer_copy_bw [src_dev] [dst_dev] [bytes]
#include <hip/hip_runtime.h>

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

int main(int argc, char** argv) {
  const int src_dev = (argc > 1) ? std::atoi(argv[1]) : 1;
  const int dst_dev = (argc > 2) ? std::atoi(argv[2]) : 0;
  const size_t bytes =
      (argc > 3) ? (size_t)std::strtoull(argv[3], nullptr, 10)
                 : ((size_t)256 << 20);
  const int iters = 20;

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2 || src_dev < 0 || dst_dev < 0 || src_dev >= ndev ||
      dst_dev >= ndev || src_dev == dst_dev) {
    std::fprintf(stderr, "need two distinct valid device ids; found %d\n", ndev);
    return EXIT_FAILURE;
  }
  if (bytes == 0) {
    std::fprintf(stderr, "byte count must be positive\n");
    return EXIT_FAILURE;
  }

  // The teaching point of this lab: when this is 0, the copy below still
  // succeeds -- it silently stages through host memory (section 3.5).
  int can_access = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access, dst_dev, src_dev));
  std::printf("device %d reading device %d: %s\n", dst_dev, src_dev,
              can_access ? "direct peer path" : "NO direct path, will stage");

  HIP_CHECK(hipSetDevice(dst_dev));
  if (can_access) {
    hipError_t e = hipDeviceEnablePeerAccess(src_dev, 0);
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr, "hipDeviceEnablePeerAccess: %s\n",
                   hipGetErrorString(e));
      return EXIT_FAILURE;
    }
    (void)hipGetLastError();  // clear the recorded already-enabled error
  }

  unsigned char* src = nullptr;
  unsigned char* dst = nullptr;
  HIP_CHECK(hipSetDevice(src_dev));
  HIP_CHECK(hipMalloc(&src, bytes));
  HIP_CHECK(hipMemset(src, 0xAB, bytes));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipSetDevice(dst_dev));
  HIP_CHECK(hipMalloc(&dst, bytes));
  HIP_CHECK(hipMemset(dst, 0x00, bytes));

  hipStream_t stream;
  hipEvent_t t0, t1;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));

  for (int i = 0; i < 3; ++i) {  // warm-up: first copy pays path setup
    HIP_CHECK(hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipEventRecord(t0, stream));
  for (int i = 0; i < iters; ++i) {
    HIP_CHECK(hipMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  HIP_CHECK(hipEventRecord(t1, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  const double secs = (double)ms / 1000.0;
  const double gbs = (double)bytes * (double)iters / secs / 1e9;

  // A fast wrong answer is the worst outcome, so check every byte.
  std::vector<unsigned char> host(bytes);
  HIP_CHECK(hipMemcpy(host.data(), dst, bytes, hipMemcpyDeviceToHost));
  size_t wrong = 0;
  for (size_t i = 0; i < bytes; ++i) {
    if (host[i] != 0xAB) {
      ++wrong;
    }
  }

  std::printf("%zu MiB x %d iters: %.1f GB/s unidirectional, %zu wrong bytes\n",
              (size_t)(bytes >> 20), iters, gbs, wrong);

  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dst));
  if (can_access) {
    HIP_CHECK(hipDeviceDisablePeerAccess(src_dev));
  }
  HIP_CHECK(hipSetDevice(src_dev));
  HIP_CHECK(hipFree(src));

  return wrong == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
