// Solution 2 - Segmented scan.
//
// The whole fix is two lines, and both are load-bearing:
//
//   if (have && sf[tid] == 0u) {   // still connected to my neighbour?
//       sv[tid] += add_v;         // then accumulate
//       sf[tid] |= add_f;         // and inherit "a head lives in my window"
//   }
//
// Why the flag has to travel with the value: at step `s`, lane `i` is combining with the
// partial result of the window [i-2s+1 .. i-s]. If a segment head sits anywhere in THAT
// window, lane i must stop accumulating from now on - but lane i cannot see that head
// directly, because it only ever reads its neighbour's aggregate. Propagating the flag with
// `|=` is how the head's existence reaches lane i in log2(n) steps instead of never.
//
// Read the pair (value, flag) as the element of a monoid and the combine
//   (v1,f1) (+) (v2,f2) = (f2 ? v2 : v1+v2, f1|f2)
// which is associative - and associativity is the entire licence to run this as a tree
// (README S2). This is Blelloch's segmented scan, and it is why "scan" in the literature is
// stated over an arbitrary associative operator, not over addition.
//
// Correctness of the flag update: only a lane that accumulates updates its flag, and only
// lane `tid` ever writes sv[tid]/sf[tid], so reading its own sf[tid] after the barrier is
// safe. The operands come from registers captured BEFORE the barrier, so the step still
// sees pre-step values only.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_segmented_scan_solution.cpp -o 02_segmented_scan_solution.exe
// Run:   ./02_segmented_scan_solution.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256
#define SEGMENT_STRIDE 37

__global__ void segmented_scan_inclusive(const float* in, const unsigned int* flags,
                                         float* out, int n) {
    __shared__ float        sv[BLOCK];
    __shared__ unsigned int sf[BLOCK];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * BLOCK + tid;

    sv[tid] = (idx < n) ? in[idx] : 0.0f;
    sf[tid] = (idx < n) ? flags[idx] : 1u;          // past-the-end lanes start their own segment
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset *= 2) {
        const bool         have  = (tid >= offset);
        const float        add_v = have ? sv[tid - offset] : 0.0f;
        const unsigned int add_f = have ? sf[tid - offset] : 0u;
        __syncthreads();

        if (have && sf[tid] == 0u) {
            sv[tid] += add_v;
            sf[tid] |= add_f;
        }
        __syncthreads();
    }

    if (idx < n) out[idx] = sv[tid];
}

int main() {
    const int n = BLOCK;

    std::vector<float>        h_in(static_cast<size_t>(n));
    std::vector<unsigned int> h_flags(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        h_in[static_cast<size_t>(i)]    = static_cast<float>(i % 5 + 1);
        h_flags[static_cast<size_t>(i)] = (i == 0 || i % SEGMENT_STRIDE == 0) ? 1u : 0u;
    }

    std::vector<float> h_expect(static_cast<size_t>(n));
    float running = 0.0f;
    int   segments = 0;
    for (int i = 0; i < n; ++i) {
        if (h_flags[static_cast<size_t>(i)] != 0u) {
            running = 0.0f;
            ++segments;
        }
        running += h_in[static_cast<size_t>(i)];
        h_expect[static_cast<size_t>(i)] = running;
    }

    float*        d_in    = nullptr;
    float*        d_out   = nullptr;
    unsigned int* d_flags = nullptr;
    HIP_CHECK(hipMalloc(&d_in, sizeof(float) * static_cast<size_t>(n)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(float) * static_cast<size_t>(n)));
    HIP_CHECK(hipMalloc(&d_flags, sizeof(unsigned int) * static_cast<size_t>(n)));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), sizeof(float) * static_cast<size_t>(n),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_flags, h_flags.data(), sizeof(unsigned int) * static_cast<size_t>(n),
                        hipMemcpyHostToDevice));

    segmented_scan_inclusive<<<1, BLOCK>>>(d_in, d_flags, d_out, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> h_got(static_cast<size_t>(n));
    HIP_CHECK(hipMemcpy(h_got.data(), d_out, sizeof(float) * static_cast<size_t>(n),
                        hipMemcpyDeviceToHost));

    int first_bad = -1, bad = 0;
    for (int i = 0; i < n; ++i) {
        if (h_got[static_cast<size_t>(i)] != h_expect[static_cast<size_t>(i)]) {
            if (first_bad < 0) first_bad = i;
            ++bad;
        }
    }

    printf("segmented inclusive scan: n = %d, %d segments (head every %d elements)\n",
           n, segments, SEGMENT_STRIDE);
    printf("result : %s (%d mismatches)\n", bad == 0 ? "PASSED" : "FAILED", bad);
    if (first_bad >= 0)
        printf("first mismatch at i = %d: got %.1f, expected %.1f\n", first_bad,
               h_got[static_cast<size_t>(first_bad)], h_expect[static_cast<size_t>(first_bad)]);

    // Show the reset explicitly: the element at each head must equal its own value.
    printf("segment heads (i: value -> scan): ");
    for (int i = 0; i < n; i += SEGMENT_STRIDE)
        printf("%d: %.0f -> %.0f  ", i, h_in[static_cast<size_t>(i)],
               h_got[static_cast<size_t>(i)]);
    printf("\n");

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_flags));
    return bad == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
