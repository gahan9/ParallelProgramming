// Solution 1 - Convert between exclusive and inclusive scan.
//
// Both conversions are elementwise; the content is entirely in the boundary.
//
//   exclusive -> inclusive : incl[i] = excl[i] + in[i]
//       No boundary case at all: excl[0] is the identity, so incl[0] == in[0] falls out.
//       This is the direction to prefer - it needs no shift and no extra buffer, and it is
//       why libraries hand you an exclusive scan and let you add the input back if you
//       wanted inclusive.
//
//   inclusive -> exclusive : excl[i] = (i == 0) ? identity : incl[i - 1]
//       Two traps. (1) excl[0] is the IDENTITY (0 for +), not incl[0]. (2) The shift must
//       write to a DIFFERENT buffer: in place, thread i-1 races thread i for the same slot
//       and you get a scan corrupted in a scheduling-dependent way. If you must do it in
//       place, stage incl[i-1] in a register and put a barrier between the read and the
//       write - the same fix the lab's in-place Hillis-Steele kernel uses.
//
//   grand total = excl[n-1] + in[n-1]
//       An exclusive scan does not contain the total anywhere; the last combine "falls off"
//       the end. Kernels that need it (stream compaction needs exactly this to size its
//       output) must either add the tail back, as here, or have the scan write the total
//       to a separate slot - which is why blelloch_scan_tile takes a `tile_sums` pointer.
//
// Build: hipcc -O3 --offload-arch=gfx942 -I../hip 01_exclusive_to_inclusive_solution.cpp -o 01_exclusive_to_inclusive_solution.exe
// Run:   ./01_exclusive_to_inclusive_solution.exe
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <hip/hip_runtime.h>
#include "check.h"

#define TILE_ELEMS 256
#define BL_THREADS (TILE_ELEMS / 2)
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(i) ((i) >> LOG_NUM_BANKS)

// --- GIVEN (from the lab): Blelloch exclusive scan of one tile. ---
__global__ void blelloch_scan_tile(const float* in, float* out, int n) {
    extern __shared__ float temp[];
    const int tid   = threadIdx.x;
    const int elems = 2 * blockDim.x;

    const int ai = tid;
    const int bi = tid + blockDim.x;
    const int pad_a = CONFLICT_FREE_OFFSET(ai);
    const int pad_b = CONFLICT_FREE_OFFSET(bi);
    temp[ai + pad_a] = (ai < n) ? in[ai] : 0.0f;
    temp[bi + pad_b] = (bi < n) ? in[bi] : 0.0f;

    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int li = offset * (2 * tid + 1) - 1;
            int ri = offset * (2 * tid + 2) - 1;
            li += CONFLICT_FREE_OFFSET(li);
            ri += CONFLICT_FREE_OFFSET(ri);
            temp[ri] += temp[li];
        }
        offset *= 2;
    }

    if (tid == 0) {
        const int last = (elems - 1) + CONFLICT_FREE_OFFSET(elems - 1);
        temp[last] = 0.0f;
    }

    for (int d = 1; d < elems; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int li = offset * (2 * tid + 1) - 1;
            int ri = offset * (2 * tid + 2) - 1;
            li += CONFLICT_FREE_OFFSET(li);
            ri += CONFLICT_FREE_OFFSET(ri);
            const float left = temp[li];
            temp[li] = temp[ri];
            temp[ri] += left;
        }
    }
    __syncthreads();

    if (ai < n) out[ai] = temp[ai + pad_a];
    if (bi < n) out[bi] = temp[bi + pad_b];
}

// SOLVED TODO 1.
__global__ void exclusive_to_inclusive(const float* excl, const float* in, float* incl, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        incl[idx] = excl[idx] + in[idx];
    }
}

// SOLVED TODO 2. Separate output buffer: the shift is a race if done in place.
__global__ void inclusive_to_exclusive(const float* incl, float* excl, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        excl[idx] = (idx == 0) ? 0.0f : incl[idx - 1];
    }
}

static int mismatches(const float* d_out, const std::vector<float>& expect, int count) {
    std::vector<float> got(static_cast<size_t>(count));
    HIP_CHECK(hipMemcpy(got.data(), d_out, sizeof(float) * count, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < count; ++i)
        if (got[static_cast<size_t>(i)] != expect[static_cast<size_t>(i)]) ++bad;
    return bad;
}

int main() {
    const int    n     = TILE_ELEMS;
    const size_t bytes = sizeof(float) * static_cast<size_t>(n);

    std::vector<float> h_in(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) h_in[static_cast<size_t>(i)] = static_cast<float>(i % 5 + 1);

    std::vector<float> h_excl(static_cast<size_t>(n)), h_incl(static_cast<size_t>(n));
    float running = 0.0f;
    for (int i = 0; i < n; ++i) {
        h_excl[static_cast<size_t>(i)] = running;
        running += h_in[static_cast<size_t>(i)];
        h_incl[static_cast<size_t>(i)] = running;
    }
    const float h_total = running;

    float *d_in = nullptr, *d_excl = nullptr, *d_excl2 = nullptr;
    float *d_incl = nullptr, *d_incl_ref = nullptr;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_excl, bytes));
    HIP_CHECK(hipMalloc(&d_excl2, bytes));
    HIP_CHECK(hipMalloc(&d_incl, bytes));
    HIP_CHECK(hipMalloc(&d_incl_ref, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_incl_ref, h_incl.data(), bytes, hipMemcpyHostToDevice));

    const size_t shmem = sizeof(float) * (TILE_ELEMS + CONFLICT_FREE_OFFSET(TILE_ELEMS - 1) + 1);
    blelloch_scan_tile<<<1, BL_THREADS, shmem>>>(d_in, d_excl, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const int bad_scan = mismatches(d_excl, h_excl, n);

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    exclusive_to_inclusive<<<blocks, threads>>>(d_excl, d_in, d_incl, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const int bad_incl = mismatches(d_incl, h_incl, n);

    inclusive_to_exclusive<<<blocks, threads>>>(d_incl_ref, d_excl2, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    const int bad_excl = mismatches(d_excl2, h_excl, n);

    // SOLVED TODO 3: the total the exclusive scan dropped off its end.
    float last_excl = 0.0f;
    HIP_CHECK(hipMemcpy(&last_excl, d_excl + (n - 1), sizeof(float), hipMemcpyDeviceToHost));
    const float total_from_excl = last_excl + h_in[static_cast<size_t>(n - 1)];

    printf("n = %d\n", n);
    printf("given  blelloch exclusive scan : %s (%d mismatches)\n",
           bad_scan == 0 ? "PASSED" : "FAILED", bad_scan);
    printf("TODO 1 exclusive -> inclusive  : %s (%d mismatches)\n",
           bad_incl == 0 ? "PASSED" : "FAILED", bad_incl);
    printf("TODO 2 inclusive -> exclusive  : %s (%d mismatches)\n",
           bad_excl == 0 ? "PASSED" : "FAILED", bad_excl);
    printf("TODO 3 grand total             : %s (got %.1f, expected %.1f)\n",
           total_from_excl == h_total ? "PASSED" : "FAILED",
           total_from_excl, h_total);

    const bool ok = (bad_scan == 0) && (bad_incl == 0) && (bad_excl == 0) &&
                    (total_from_excl == h_total);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_excl));
    HIP_CHECK(hipFree(d_excl2));
    HIP_CHECK(hipFree(d_incl));
    HIP_CHECK(hipFree(d_incl_ref));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
