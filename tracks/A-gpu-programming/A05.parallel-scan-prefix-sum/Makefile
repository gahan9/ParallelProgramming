# A05 - Parallel scan (prefix sum). Dual-backend build: AMD (hipcc) and NVIDIA (nvcc).
#
# This Makefile builds the worked examples in hip/ and cuda/. The drills have
# their own Makefiles (exercises/, solutions/) and can be driven from here too.
#
# Quick start:
#   make hip   GPU_ARCH=gfx942     # build all hip/ examples
#   make run-hip                    # run them
#   make cuda  SM_ARCH=sm_90        # build all cuda/ examples
#   make run-cuda                   # run them
#   make exercises                  # build the exercises/ drills
#   make solutions                  # build the solutions/ reference answers
#   make profile-hip                # rocprofv3 summary + traces (scan)
#   make profile-cuda               # nsys timeline (scan)
#   make clean                      # clean here + exercises/ + solutions/

# Toolchain, arch, and base flags are shared across Track A (see ../common.mk).
include ../common.mk

# ---- Flags ----
HIP_FLAGS  := $(HIP_BASE_FLAGS) -Ihip
CUDA_FLAGS := $(CUDA_BASE_FLAGS) -Icuda

# ---- Programs (one per source file) ----
PROGS       := scan compaction
HIP_BINS    := $(addprefix hip/,  $(addsuffix .exe, $(PROGS)))
CUDA_BINS   := $(addprefix cuda/, $(addsuffix .exe, $(PROGS)))

OUTPUT_DIR := profiling_output

.PHONY: help all hip cuda run-hip run-cuda exercises solutions \
        run-exercises run-solutions profile-hip profile-cuda clean

help:
	@echo "Targets:"
	@echo "  make hip   GPU_ARCH=$(GPU_ARCH)   - build hip/  examples with hipcc"
	@echo "  make cuda  SM_ARCH=$(SM_ARCH)     - build cuda/ examples with nvcc"
	@echo "  make run-hip / run-cuda           - build + run all examples"
	@echo "  make exercises / solutions        - build the drills (own Makefiles)"
	@echo "  make run-exercises / run-solutions- build + run the drills"
	@echo "  make profile-hip / profile-cuda   - profile scan"
	@echo "  make clean                        - remove binaries + profiler output"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: hip cuda

# ---- AMD / HIP ----
hip: $(HIP_BINS)

hip/%.exe: hip/%.cpp hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run-hip: hip
	@for b in $(HIP_BINS); do echo "== $$b =="; $$b; echo; done

profile-hip: hip/scan.exe
	$(ROCPROFV3) --summary --sys-trace --output-format csv -d $(OUTPUT_DIR) -- ./hip/scan.exe

# ---- NVIDIA / CUDA ----
cuda: $(CUDA_BINS)

cuda/%.exe: cuda/%.cu cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

profile-cuda: cuda/scan.exe
	nsys profile --stats=true -o $(OUTPUT_DIR)/scan ./cuda/scan.exe
	@echo "For per-kernel counters + bank-conflict metrics: ncu --set full ./cuda/scan.exe"

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
	rm -rf $(OUTPUT_DIR) .rocprofv3 *.nsys-rep *.sqlite
	$(MAKE) -C exercises clean
	$(MAKE) -C solutions clean
