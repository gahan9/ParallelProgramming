// Unidirectional peer-to-peer copy bandwidth between two GPUs, verified.
//
// This is hip/peer_copy_bw.cpp under the mechanical substitution D03 section 4
// tabulates: hip -> cuda throughout, hipErrorPeerAccessAlreadyEnabled ->
// cudaErrorPeerAccessAlreadyEnabled, hipcc -> nvcc. Nothing about the logic
// changes, and it is kept line-for-line parallel on purpose so a diff between
// the two files shows only the substitution.
//
// Build: nvcc -O2 cuda/peer_copy_bw.cu -o peer_copy_bw_cuda
// Run:   ./peer_copy_bw_cuda [src_dev] [dst_dev] [bytes]
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e_ = (cmd);                                                    \
    if (e_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA %s:%d '%s': %s\n", __FILE__, __LINE__, #cmd,  \
                   cudaGetErrorString(e_));                                    \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char** argv) {
  const int src_dev = (argc > 1) ? std::atoi(argv[1]) : 1;
  const int dst_dev = (argc > 2) ? std::atoi(argv[2]) : 0;
  const size_t bytes =
      (argc > 3) ? (size_t)std::strtoull(argv[3], nullptr, 10)
                 : ((size_t)256 << 20);
  const int iters = 20;

  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
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
  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, dst_dev, src_dev));
  std::printf("device %d reading device %d: %s\n", dst_dev, src_dev,
              can_access ? "direct peer path" : "NO direct path, will stage");

  CUDA_CHECK(cudaSetDevice(dst_dev));
  if (can_access) {
    cudaError_t e = cudaDeviceEnablePeerAccess(src_dev, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr, "cudaDeviceEnablePeerAccess: %s\n",
                   cudaGetErrorString(e));
      return EXIT_FAILURE;
    }
    (void)cudaGetLastError();  // clear the recorded already-enabled error
  }

  unsigned char* src = nullptr;
  unsigned char* dst = nullptr;
  CUDA_CHECK(cudaSetDevice(src_dev));
  CUDA_CHECK(cudaMalloc(&src, bytes));
  CUDA_CHECK(cudaMemset(src, 0xAB, bytes));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaSetDevice(dst_dev));
  CUDA_CHECK(cudaMalloc(&dst, bytes));
  CUDA_CHECK(cudaMemset(dst, 0x00, bytes));

  cudaStream_t stream;
  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));

  for (int i = 0; i < 3; ++i) {  // warm-up: first copy pays path setup
    CUDA_CHECK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaEventRecord(t0, stream));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream));
  }
  CUDA_CHECK(cudaEventRecord(t1, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  const double secs = (double)ms / 1000.0;
  const double gbs = (double)bytes * (double)iters / secs / 1e9;

  // A fast wrong answer is the worst outcome, so check every byte.
  std::vector<unsigned char> host(bytes);
  CUDA_CHECK(cudaMemcpy(host.data(), dst, bytes, cudaMemcpyDeviceToHost));
  size_t wrong = 0;
  for (size_t i = 0; i < bytes; ++i) {
    if (host[i] != 0xAB) {
      ++wrong;
    }
  }

  std::printf("%zu MiB x %d iters: %.1f GB/s unidirectional, %zu wrong bytes\n",
              (size_t)(bytes >> 20), iters, gbs, wrong);

  CUDA_CHECK(cudaEventDestroy(t0));
  CUDA_CHECK(cudaEventDestroy(t1));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dst));
  if (can_access) {
    CUDA_CHECK(cudaDeviceDisablePeerAccess(src_dev));
  }
  CUDA_CHECK(cudaSetDevice(src_dev));
  CUDA_CHECK(cudaFree(src));

  return wrong == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
