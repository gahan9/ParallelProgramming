// Teaching core — A01 bounds guard (HIP twin of CUDA lab).
// Full files:
//   tracks/A-gpu-programming/A01.foundations-and-programming-model/hip/vector_add.cpp
//   tracks/A-gpu-programming/A01.foundations-and-programming-model/exercises/01_saxpy.cpp
// Drill (A01 §9): N=1000, block=256 → 4 blocks, 1024 threads, 24 with idx >= n.

__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {                 // skip the partial last block — not optional
        C[idx] = A[idx] + B[idx];
    }
}

// Launch uses ceil division so every element is covered:
const int threads = 256;
const int blocks  = (n + threads - 1) / threads;   // ceil(n / 256)
vector_add<<<blocks, threads>>>(d_A, d_B, d_C, n);

// Pair: ceil without the guard → extra threads write past the allocation.
//       floor (n / 256)        → last elements never computed (FAILED, visible).
// You need both.

// Triton spelling of the same guard:
//   mask = offsets < n
//   tl.load(..., mask=mask); tl.store(..., mask=mask)

// Stretch: comment out `if (idx < n)`, launch N=1000 / block=256, and run under
// NVIDIA compute-sanitizer (or a device debugger on AMD). The CPU check of
// C[0..n) can still match. The illegal accesses are at idx 1000..1023.
