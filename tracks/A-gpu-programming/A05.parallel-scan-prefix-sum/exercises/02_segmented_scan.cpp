// Exercise 2 - Segmented scan: one launch, many independent scans.
//
// Real batches are ragged. A batch of 7 sequences of different lengths is stored as one
// flat array plus segment metadata, and you want a prefix sum that RESTARTS at every
// sequence boundary (per-sequence token offsets, per-row KV-cache positions, per-document
// cumulative counts). Launching one scan per segment wastes the GPU on short segments;
// a SEGMENTED scan does all of them in one pass.
//
// The trick is that the segmented combine is still associative, so the same Hillis-Steele
// tree works if you carry the flag along with the value:
//
//   value : accumulate the neighbour's partial sum ONLY IF this lane has not yet been
//           cut off from it by a segment head
//   flag  : once a segment head appears anywhere in a lane's window, that lane must stop
//           accumulating - so the head marker propagates up the tree with the value
//
// YOUR TASK (TODO): make the combine segment-aware. The starter runs a plain unsegmented
// scan, so it is correct inside the first segment and wrong in every later one - the
// mismatch report below shows you exactly where it starts drifting.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 02_segmented_scan.cpp -o 02_segmented_scan.exe
// Run:   ./02_segmented_scan.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define BLOCK 256
#define SEGMENT_STRIDE 37                           // a head every 37 elements: uneven tail

// Inclusive segmented scan of one tile. flags[i] == 1 marks the FIRST element of a segment.
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
        // Stage the operands in registers before anyone writes: this is the same
        // read-write hazard the lab's in-place Hillis-Steele kernel closes.
        const bool         have  = (tid >= offset);
        const float        add_v = have ? sv[tid - offset] : 0.0f;
        const unsigned int add_f = have ? sf[tid - offset] : 0u;
        __syncthreads();

        // TODO: accumulate only while this lane is still connected to its neighbour, and
        //       propagate add_f so a head found deeper in the tree keeps cutting later steps.
        if (have) {
            sv[tid] += add_v;                       // starter: unsegmented -> wrong at boundaries
        }
        (void)add_f;                                // unused until the TODO is done
        __syncthreads();
    }

    if (idx < n) out[idx] = sv[tid];
}

int main() {
    const int n = BLOCK;                            // one tile keeps the focus on the combine

    std::vector<float>        h_in(static_cast<size_t>(n));
    std::vector<unsigned int> h_flags(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        h_in[static_cast<size_t>(i)]    = static_cast<float>(i % 5 + 1);
        h_flags[static_cast<size_t>(i)] = (i == 0 || i % SEGMENT_STRIDE == 0) ? 1u : 0u;
    }

    // CPU reference: a running sum that resets at every head. Segment sums stay small,
    // so every value is exact in float and any mismatch is a real bug.
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
    if (first_bad >= 0) {
        printf("first mismatch at i = %d (segment head at %d): got %.1f, expected %.1f\n",
               first_bad, (first_bad / SEGMENT_STRIDE) * SEGMENT_STRIDE,
               h_got[static_cast<size_t>(first_bad)], h_expect[static_cast<size_t>(first_bad)]);
        printf("An unsegmented scan is correct until the first head, then carries the\n"
               "previous segment's total forward. That is the bug the flags must stop.\n");
    }

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_flags));
    return bad == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
