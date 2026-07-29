# B04 - Precision & quantization. Dual-backend build: AMD (hipcc) and NVIDIA (nvcc).
#
# This Makefile builds the worked examples in hip/ and cuda/, and runs the pure-Python
# quantization demo in tools/. The drills have their own Makefiles (exercises/,
# solutions/) and can be driven from here too.
#
# Quick start:
#   make hip   GPU_ARCH=gfx942     # build all hip/ examples
#   make run-hip                    # run them
#   make cuda  SM_ARCH=sm_90        # build all cuda/ examples
#   make run-cuda                   # run them
#   make python                     # int8 quantization demo (numpy, no GPU needed)
#   make exercises                  # build the exercises/ drills
#   make solutions                  # build the solutions/ reference answers
#   make profile-hip                # rocprofv3 summary + traces (mixed_precision_gemm)
#   make profile-cuda               # nsys timeline (mixed_precision_gemm)
#   make clean                      # clean here + exercises/ + solutions/
#
# Toolchain, arch, and base flags are shared with Track A. Track B has no common.mk
# of its own on purpose - one toolchain definition for the whole repo means an rpath
# or tool-discovery fix lands everywhere at once. From this directory that file is
# two levels up and across: ../../A-gpu-programming/common.mk
include ../../A-gpu-programming/common.mk

# ---- Flags ----
HIP_FLAGS  := $(HIP_BASE_FLAGS) -Ihip
CUDA_FLAGS := $(CUDA_BASE_FLAGS) -Icuda

# Square GEMM dimension for lab 2. Smaller = faster host reference, larger = more
# representative GFLOP/s. The host reference is O(MDIM^3) in double, so it dominates
# start-up time above ~1536.
MDIM ?= 1024

# ---- Programs (one per source file) ----
PROGS     := precision_limits mixed_precision_gemm
HIP_BINS  := $(addprefix hip/,  $(addsuffix .exe, $(PROGS)))
CUDA_BINS := $(addprefix cuda/, $(addsuffix .exe, $(PROGS)))

PYTHON     ?= python3
OUTPUT_DIR := profiling_output

.PHONY: help all hip cuda run-hip run-cuda python exercises solutions \
        run-exercises run-solutions profile-hip profile-cuda clean

help:
	@echo "Targets:"
	@echo "  make hip   GPU_ARCH=$(GPU_ARCH)   - build hip/  examples with hipcc"
	@echo "  make cuda  SM_ARCH=$(SM_ARCH)     - build cuda/ examples with nvcc"
	@echo "  make run-hip / run-cuda           - build + run all examples"
	@echo "  make python                       - int8 quantization demo (no GPU needed)"
	@echo "  make exercises / solutions        - build the drills (own Makefiles)"
	@echo "  make run-exercises / run-solutions- build + run the drills"
	@echo "  make profile-hip / profile-cuda   - profile mixed_precision_gemm"
	@echo "  make clean                        - remove binaries + profiler output"
	@echo "Knobs: MDIM=$(MDIM) (GEMM size), B04_NO_NATIVE_FP8=1 (skip vendor fp8 check)"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: hip cuda

# ---- AMD / HIP ----
hip: $(HIP_BINS)

# MDIM is only meaningful for the GEMM lab, so it is attached to that target rather
# than to every compile in the directory.
hip/mixed_precision_gemm.exe: HIP_FLAGS += -DMDIM=$(MDIM)

hip/%.exe: hip/%.cpp hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run-hip: hip
	@for b in $(HIP_BINS); do echo "== $$b =="; $$b; echo; done

profile-hip: hip/mixed_precision_gemm.exe
	$(ROCPROFV3) --summary --sys-trace --output-format csv -d $(OUTPUT_DIR) -- ./hip/mixed_precision_gemm.exe

# ---- NVIDIA / CUDA ----
cuda: $(CUDA_BINS)

cuda/mixed_precision_gemm.exe: CUDA_FLAGS += -DMDIM=$(MDIM)

cuda/%.exe: cuda/%.cu cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

profile-cuda: cuda/mixed_precision_gemm.exe
	nsys profile --stats=true -o $(OUTPUT_DIR)/mixed_precision_gemm ./cuda/mixed_precision_gemm.exe
	@echo "For per-kernel counters (conversion instructions, LSU): ncu --set full ./cuda/mixed_precision_gemm.exe"

# ---- Quantization demo (CPU only: the lesson is arithmetic, not hardware) ----
python:
	$(PYTHON) tools/quantize_demo.py

# ---- Drills (delegate to their own Makefiles) ----
# Command-line overrides such as GPU_ARCH=gfx950 propagate to the sub-make.
exercises:
	$(MAKE) -C exercises all

solutions:
	$(MAKE) -C solutions all

run-exercises:
	$(MAKE) -C exercises run

run-solutions:
	$(MAKE) -C solutions run

# ---- Housekeeping ----
clean:
	rm -f $(HIP_BINS) $(CUDA_BINS)
	rm -rf $(OUTPUT_DIR) .rocprofv3 *.nsys-rep *.sqlite __pycache__ tools/__pycache__
	$(MAKE) -C exercises clean
	$(MAKE) -C solutions clean
