# A06 - Tiled matrix multiply. Dual-backend build: AMD (hipcc) and NVIDIA (nvcc).
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
#   make profile-hip                # rocprofv3 summary + traces (matmul)
#   make profile-cuda               # nsys timeline (matmul)
#   make clean                      # clean here + exercises/ + solutions/
#
# Change the problem size without editing the source:
#   make hip MDIM=2048              # square M = N = K = 2048
#   make hip MDIM=1000              # ragged dims exercise the boundary guards

# Toolchain, arch, and base flags are shared across Track A (see ../common.mk).
include ../common.mk

# ---- Problem size (compile-time constant, see hip/matmul.cpp) ----
MDIM ?= 1024

# ---- Flags ----
HIP_FLAGS  := $(HIP_BASE_FLAGS)  -Ihip  -DMDIM=$(MDIM)
CUDA_FLAGS := $(CUDA_BASE_FLAGS) -Icuda -DMDIM=$(MDIM)

# ---- Programs (one per source file) ----
PROGS       := matmul
HIP_BINS    := $(addprefix hip/,  $(addsuffix .exe, $(PROGS)))
CUDA_BINS   := $(addprefix cuda/, $(addsuffix .exe, $(PROGS)))

OUTPUT_DIR := profiling_output

# Make cannot see that -DMDIM changed, so a plain `make hip MDIM=2048` after a
# 1024 build would report "up to date" and silently run the old binary. Record
# the value in a stamp file the binaries depend on. (The rule itself lives below
# `help` so that it never becomes the default goal.)
MDIM_STAMP := .mdim-$(MDIM)

.PHONY: help all hip cuda run-hip run-cuda exercises solutions \
        run-exercises run-solutions profile-hip profile-cuda clean

help:
	@echo "Targets:"
	@echo "  make hip   GPU_ARCH=$(GPU_ARCH)   - build hip/  examples with hipcc"
	@echo "  make cuda  SM_ARCH=$(SM_ARCH)     - build cuda/ examples with nvcc"
	@echo "  make run-hip / run-cuda           - build + run all examples"
	@echo "  make exercises / solutions        - build the drills (own Makefiles)"
	@echo "  make run-exercises / run-solutions- build + run the drills"
	@echo "  make profile-hip / profile-cuda   - profile matmul"
	@echo "  make clean                        - remove binaries + profiler output"
	@echo "Problem size: MDIM=$(MDIM)  (override: make hip MDIM=2048)"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: hip cuda

$(MDIM_STAMP):
	@rm -f .mdim-* $(HIP_BINS) $(CUDA_BINS)
	@touch $@

# ---- AMD / HIP ----
hip: $(HIP_BINS)

hip/%.exe: hip/%.cpp hip/check.h $(MDIM_STAMP)
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run-hip: hip
	@for b in $(HIP_BINS); do echo "== $$b =="; $$b; echo; done

profile-hip: hip/matmul.exe
	$(ROCPROFV3) --summary --sys-trace --output-format csv -d $(OUTPUT_DIR) -- ./hip/matmul.exe

# ---- NVIDIA / CUDA ----
cuda: $(CUDA_BINS)

cuda/%.exe: cuda/%.cu cuda/check.cuh $(MDIM_STAMP)
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

profile-cuda: cuda/matmul.exe
	nsys profile --stats=true -o $(OUTPUT_DIR)/matmul ./cuda/matmul.exe
	@echo "For kernel-level counters (LDS/shared traffic, FMA utilization): ncu --set full ./cuda/matmul.exe"

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
	rm -f $(HIP_BINS) $(CUDA_BINS) .mdim-*
	rm -rf $(OUTPUT_DIR) .rocprofv3 *.nsys-rep *.sqlite
	$(MAKE) -C exercises clean
	$(MAKE) -C solutions clean
