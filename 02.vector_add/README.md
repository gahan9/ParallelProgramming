# Vector Addition - GPU Profiling Guide

This directory contains two versions of a vector addition program:
- **`vector_add.cpp`**: Baseline version with fixed array size (N=1024)
- **`vector_add_dynamic.cpp`**: Enhanced version that dynamically queries GPU resources and optimizes array size

## Building

```bash
make all              # Build both executables
make                  # Same as 'make all'
make test             # Build and run baseline version
make test_dynamic     # Build and run dynamic version
```

## Profiling

### Quick Start

```bash
# Profile baseline version
make profile

# Profile dynamic version
make profile_dynamic

# Profile both versions for comparison
make profile_both

# Profile with detailed metrics
make profile_detailed

# Profile kernel execution time
make profile_kernel
```

### Manual Profiling

You can also run profiling manually:

```bash
# Basic profiling
rocprofv3 --summary --output-format csv ./vector_add_dynamic.exe

# Detailed profiling with system and HSA traces
rocprofv3 --summary --sys-trace --hsa-trace --output-format csv -d profiling_output ./vector_add_dynamic.exe

# Kernel-specific profiling
rocprofv3 --summary --kernel-trace --output-format csv -d profiling_output ./vector_add_dynamic.exe
```

### Profiling Output

Profiling results are stored in `profiling_output/` directory:
- **`baseline/`**: Results from baseline version
- **`dynamic/`**: Results from dynamic version
- **`detailed/`**: Detailed performance counter results
- **`kernel/`**: Kernel execution timing results

Key files to examine:
- `results.csv`: Summary statistics
- `results.json`: Detailed trace data (if JSON format used)
- `kernel_trace.csv`: Kernel execution timeline

## What to Watch Out For

### 1. **Memory Bandwidth Utilization**

**What to check:**
- Look for `MemoryBandwidth` or `MemoryThroughput` metrics in profiling output
- Compare achieved bandwidth vs. theoretical peak bandwidth

**Red flags:**
- Memory bandwidth utilization < 50% suggests memory-bound operations
- Large gaps between memory transfers indicate poor overlap

**How to interpret:**
```
Achieved Bandwidth / Peak Bandwidth = Efficiency
- > 80%: Excellent
- 50-80%: Good
- < 50%: Needs optimization
```

### 2. **Compute Utilization**

**What to check:**
- `GPUUtilization` or `ComputeUnitUtilization` metrics
- Kernel occupancy (threads per compute unit)

**Red flags:**
- GPU utilization < 60% indicates underutilization
- Low occupancy suggests thread block size issues

**How to interpret:**
- High utilization with low performance = compute-bound
- Low utilization = likely memory-bound or synchronization issues

### 3. **Kernel Execution Time**

**What to check:**
- `KernelDuration` or `KernelTime` in profiling output
- Compare kernel time vs. total execution time

**Red flags:**
- Kernel time << total time = overhead from memory transfers
- Very short kernel times (< 1μs) = may not be representative

**Optimization targets:**
- Kernel should dominate execution time for compute-bound workloads
- For memory-bound: focus on reducing transfer overhead

### 4. **Memory Transfer Overhead**

**What to check:**
- `hipMemcpy` timing vs. kernel execution time
- Host-to-device and device-to-host transfer times

**Red flags:**
- Transfer time > kernel time = memory transfer bottleneck
- Multiple small transfers instead of batched transfers

**Best practices:**
- Overlap transfers with computation when possible
- Use pinned memory for faster transfers
- Batch small transfers into larger ones

### 5. **Work Distribution**

**What to check:**
- Number of threads vs. array size
- Threads per block configuration

**Red flags:**
- Threads per block not a multiple of warp/wavefront size (32/64)
- Too many or too few blocks per grid

**Optimal configuration:**
- Threads per block: 256 or 512 (multiple of 64 for AMD)
- Enough blocks to fill all compute units

### 6. **Error Checking**

**Always verify:**
- No HIP errors in profiling output
- Correctness of results (program should verify this)
- Memory allocation success

## Baseline Analysis

### Baseline Version (`vector_add.cpp`)

**Characteristics:**
- Fixed array size: N = 1024 elements
- Total memory: 3 × 1024 × 4 bytes = 12 KB
- Threads per block: 256
- Blocks per grid: (1024 + 256 - 1) / 256 = 4 blocks

**Expected Performance:**
- **Compute operations**: 1024 additions (1 FLOP per element)
- **Memory operations**: 
  - Read: 2 × 1024 × 4 bytes = 8 KB (A and B)
  - Write: 1 × 1024 × 4 bytes = 4 KB (C)
  - Total: 12 KB transferred
- **Arithmetic Intensity**: 1 FLOP / 12 bytes = 0.083 FLOP/byte (very low)

**Limitations:**
- Too small to effectively utilize GPU
- Memory transfer overhead dominates
- Underutilizes compute units
- Not representative of real-world workloads

### Dynamic Version (`vector_add_dynamic.cpp`)

**Characteristics:**
- Array size determined by GPU capabilities
- Typically uses 80% of available free memory
- Optimized for both memory and compute utilization
- Scales with GPU resources

**Advantages:**
- Better GPU utilization
- More representative of real workloads
- Adapts to different GPU configurations
- Better for performance analysis

## Roofline Model Analysis

### Understanding the Roofline Model

The Roofline model is a performance analysis tool that relates:
- **Arithmetic Intensity** (AI): FLOPs per byte transferred
- **Performance**: Achieved GFLOP/s
- **Peak Performance**: Limited by compute or memory bandwidth

### Vector Addition Roofline Analysis

#### Arithmetic Intensity Calculation

For vector addition: C[i] = A[i] + B[i]

**Operations per element:**
- 1 addition (1 FLOP)
- 2 reads (A[i], B[i])
- 1 write (C[i])

**Memory transferred per element:**
- 3 × sizeof(float) = 12 bytes

**Arithmetic Intensity:**
```
AI = FLOPs / Bytes = 1 FLOP / 12 bytes = 0.083 FLOP/byte
```

#### Performance Bounds

**1. Memory-Bound Region (Low AI)**

For AI = 0.083 FLOP/byte:

```
Peak Performance = Memory Bandwidth × AI

Example (assuming 1 TB/s memory bandwidth):
Peak = 1000 GB/s × 0.083 FLOP/byte = 83 GFLOP/s
```

**2. Compute-Bound Region (High AI)**

For compute-bound kernels:
```
Peak Performance = Peak Compute Throughput

Example (assuming 25 TFLOPS):
Peak = 25,000 GFLOP/s
```

#### Roofline Plot Interpretation

```
Performance (GFLOP/s)
    |
    |                    /---- Compute Roofline
    |                   /
    |                  /
    |                 /
    |                /
    |               /
    |              /
    |             /
    |            /  Memory Roofline
    |           /  (slope = bandwidth)
    |          /
    |         /
    |        /
    |       /
    |      /
    |     /
    |    /
    |   /
    |  /
    | /
    |/_____________________________
    0.01  0.1   1    10   100   1000
              Arithmetic Intensity (FLOP/byte)
```

**Vector addition (AI = 0.083) falls in the memory-bound region.**

### Performance Expectations

#### Baseline Version (N=1024)

**Expected performance:**
- Very low GFLOP/s due to overhead
- Memory transfer dominates
- Not suitable for roofline analysis (too small)

#### Dynamic Version (Large N)

**Expected performance:**
- Should approach memory bandwidth limit
- Performance ≈ Memory Bandwidth × 0.083 FLOP/byte
- Example: 1000 GB/s × 0.083 = ~83 GFLOP/s

### Measuring Performance

To calculate actual performance from profiling:

```python
# From profiling output
kernel_time_seconds = kernel_duration_ms / 1000.0
array_size = N  # from program output
flops = array_size  # 1 FLOP per element

gflops = (flops / kernel_time_seconds) / 1e9
memory_bandwidth_gb_s = (3 * array_size * 4) / kernel_time_seconds / 1e9
arithmetic_intensity = flops / (3 * array_size * 4)

print(f"Performance: {gflops:.2f} GFLOP/s")
print(f"Memory Bandwidth: {memory_bandwidth_gb_s:.2f} GB/s")
print(f"Arithmetic Intensity: {arithmetic_intensity:.4f} FLOP/byte")
```

### Optimization Opportunities

**1. Increase Arithmetic Intensity**
- Fuse multiple operations (e.g., C = A + B + D)
- Reduce memory accesses through data reuse
- Use shared memory for repeated accesses

**2. Improve Memory Access Patterns**
- Ensure coalesced memory accesses
- Use vectorized loads when possible
- Minimize memory transactions

**3. Reduce Overhead**
- Overlap computation and memory transfers
- Use streams for concurrent operations
- Minimize kernel launch overhead

**4. Optimize Kernel Configuration**
- Tune threads per block
- Ensure sufficient parallelism
- Balance occupancy and register usage

## Example Profiling Workflow

```bash
# 1. Build both versions
make all

# 2. Run dynamic version to see calculated array size
./vector_add_dynamic.exe

# 3. Profile dynamic version
make profile_dynamic

# 4. Examine results
cat profiling_output/dynamic/results.csv

# 5. Calculate performance metrics
# Use the kernel time and array size from output
# Apply the formulas above

# 6. Compare with baseline (if needed)
make profile_both
```

## Troubleshooting

### Profiling Issues

**Problem:** `rocprofv3: command not found`
- **Solution:** Ensure ROCm is installed and in PATH
- Check: `which rocprofv3` or `ls $(HIP_PATH)/bin/rocprofv3`

**Problem:** No profiling output
- **Solution:** Check GPU permissions and ROCm installation
- Verify: `rocminfo` shows GPU information

**Problem:** Profiling shows zero or invalid metrics
- **Solution:** Some metrics require specific GPU architectures
- Try: `make profile_kernel` for basic timing

### Performance Issues

**Problem:** Low GPU utilization
- Check: Array size is large enough
- Verify: Thread block configuration
- Consider: Increasing work per thread

**Problem:** Memory bandwidth not saturated
- Check: Memory access patterns
- Verify: Coalesced memory accesses
- Consider: Increasing array size

## References

- [ROCm Profiling Tools](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [Roofline Model Paper](https://crd.lbl.gov/departments/computer-science/par/research/roofline/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

## Cleanup

```bash
make clean          # Remove executables and object files
make clean_profile  # Remove profiling output
make clean_all      # Remove everything
```

