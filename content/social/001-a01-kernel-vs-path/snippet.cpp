// Pursue: tracks/A-gpu-programming/A01.foundations-and-programming-model/hip/vector_add.cpp
// Build:  cd .../A01.foundations-and-programming-model && make hip && make run-hip
// Profile: make profile-hip
// Stretch: keep A,B,C resident; time kernel-only vs a second full H2D+kernel+D2H — compare.

__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host teaching core (error checks omitted — see full lab file):
hipMalloc(&d_A, bytes);
hipMalloc(&d_B, bytes);
hipMalloc(&d_C, bytes);
hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice);
hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice);
vector_add<<<blocks, 256>>>(d_A, d_B, d_C, n);
hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost);
// Then: verify vs CPU reference; time with hipEvent — not a CPU clock across async launch.
