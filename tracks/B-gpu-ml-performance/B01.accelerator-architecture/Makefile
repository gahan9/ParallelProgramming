# B01 - Accelerator architecture. Dual-backend build: AMD (hipcc) and NVIDIA (nvcc).
#
# This Makefile builds the two introspection labs in hip/ and cuda/. Both labs
# print YOUR machine's architecture, so their output is the input to the roofline
# you will build in B02. The drills have their own Makefiles (exercises/,
# solutions/) and can be driven from here too.
#
# Quick start:
#   make hip   GPU_ARCH=gfx942     # build all hip/ labs
#   make run-hip                    # run them
#   make cuda  SM_ARCH=sm_90        # build all cuda/ labs
#   make run-cuda                   # run them
#   make exercises                  # build the exercises/ drills
#   make solutions                  # build the solutions/ reference answers
#   make clean                      # clean here + exercises/ + solutions/

# Toolchain, arch, and base flags are shared with Track A. Track B has no
# common.mk of its own on purpose: one toolchain definition for the whole repo
# means an rpath or tool-discovery fix lands everywhere at once.
include ../../A-gpu-programming/common.mk

# ---- Flags ----
HIP_FLAGS  := $(HIP_BASE_FLAGS) -Ihip
CUDA_FLAGS := $(CUDA_BASE_FLAGS) -Icuda

# ---- Programs (one per source file) ----
PROGS     := device_query topology
HIP_BINS  := $(addprefix hip/,  $(addsuffix .exe, $(PROGS)))
CUDA_BINS := $(addprefix cuda/, $(addsuffix .exe, $(PROGS)))

.PHONY: help all hip cuda run-hip run-cuda exercises solutions \
        run-exercises run-solutions clean

help:
	@echo "Targets:"
	@echo "  make hip   GPU_ARCH=$(GPU_ARCH)   - build hip/  labs with hipcc"
	@echo "  make cuda  SM_ARCH=$(SM_ARCH)     - build cuda/ labs with nvcc"
	@echo "  make run-hip / run-cuda           - build + run both labs"
	@echo "  make exercises / solutions        - build the drills (own Makefiles)"
	@echo "  make run-exercises / run-solutions- build + run the drills"
	@echo "  make clean                        - remove binaries"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: hip cuda

# ---- AMD / HIP ----
hip: $(HIP_BINS)

hip/%.exe: hip/%.cpp hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run-hip: hip
	@for b in $(HIP_BINS); do echo "== $$b =="; $$b; echo; done

# ---- NVIDIA / CUDA ----
cuda: $(CUDA_BINS)

cuda/%.exe: cuda/%.cu cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

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
	$(MAKE) -C exercises clean
	$(MAKE) -C solutions clean
