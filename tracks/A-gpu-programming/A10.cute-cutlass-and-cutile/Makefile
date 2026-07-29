# A10 - CuTe / CUTLASS / CuTile. NVIDIA-ONLY, OPTIONAL module.
#
# There is deliberately no hip/ target here: CuTe, CUTLASS, and CuTile are NVIDIA-only
# projects with no AMD backend. AMD readers should skip this module - see README section 1
# for the Composable Kernel / hipBLASLt / MFMA reading list.
#
# Two labs with very different dependency profiles:
#
#   Lab 1  cuda/mma_tensor_core.cu   NO external dependency. Plain nvcc, SM70+ GPU.
#   Lab 2  cuda/cute_layout_demo.cu  REQUIRES a CUTLASS checkout via CUTLASS_DIR.
#
# Lab 2 is gated on purpose. Without CUTLASS_DIR the `cute` target FAILS with an
# explanatory message rather than quietly skipping the file - a green build that silently
# dropped the CuTe code would be worse than no build at all.
#
# Quick start:
#   make                                        # this help, plus the CUTLASS_DIR state
#   make cuda      SM_ARCH=sm_90                # build Lab 1 (no dependencies)
#   make run-cuda                               # build + run Lab 1
#   make cute      CUTLASS_DIR=/path/to/cutlass # build Lab 2
#   make run-cute  CUTLASS_DIR=/path/to/cutlass # build + run Lab 2
#   make exercises / make solutions             # the drills (own Makefiles)
#   make profile-cuda                           # nsys timeline for Lab 1
#   make clean                                  # clean here + exercises/ + solutions/
#
# Get CUTLASS yourself (it is >100 MB; this repo does not vendor it):
#   git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/src/cutlass
#   make run-cute CUTLASS_DIR=~/src/cutlass

# Toolchain, arch, and base flags are shared across Track A (see ../common.mk).
include ../common.mk

# ---- Flags ----
# Lab 1 needs nothing beyond the shared CUDA flags.
CUDA_FLAGS := $(CUDA_BASE_FLAGS) -Icuda

# Lab 2: CUTLASS 4.x headers are C++17 and use constexpr in host/device contexts, so
# -std=c++17 --expt-relaxed-constexpr are both required. CuTe is header-only - there is
# nothing to link.
CUTLASS_DIR ?=
CUTE_FLAGS  := $(CUDA_BASE_FLAGS) -std=c++17 --expt-relaxed-constexpr \
               -Icuda -I$(CUTLASS_DIR)/include

# ---- Programs ----
CUDA_PROGS  := mma_tensor_core
CUDA_BINS   := $(addprefix cuda/, $(addsuffix .exe, $(CUDA_PROGS)))
CUTE_BIN    := cuda/cute_layout_demo.exe

OUTPUT_DIR  := profiling_output

ifeq ($(strip $(CUTLASS_DIR)),)
  # No backticks here: this string is echoed inside double quotes, where the shell would
  # treat them as command substitution.
  CUTLASS_STATE := NOT SET - 'make cute' will fail with instructions
else
  CUTLASS_STATE := $(CUTLASS_DIR)
endif

.PHONY: help all cuda run-cuda cute run-cute check-cutlass \
        exercises solutions run-exercises run-solutions profile-cuda clean

help:
	@echo "A10 - CuTe / CUTLASS / CuTile   (NVIDIA-only, OPTIONAL module)"
	@echo ""
	@echo "  make cuda      SM_ARCH=$(SM_ARCH)   - build Lab 1 (tensor-core GEMM, no deps)"
	@echo "  make run-cuda                       - build + run Lab 1"
	@echo "  make cute      CUTLASS_DIR=<path>   - build Lab 2 (CuTe layout algebra)"
	@echo "  make run-cute  CUTLASS_DIR=<path>   - build + run Lab 2"
	@echo "  make exercises / solutions          - build the drills (own Makefiles)"
	@echo "  make run-exercises / run-solutions  - build + run the drills"
	@echo "  make profile-cuda                   - nsys timeline for Lab 1"
	@echo "  make clean                          - remove binaries + profiler output"
	@echo ""
	@echo "CUTLASS_DIR: $(CUTLASS_STATE)"
	@echo "There is no hip/ target: CUTLASS/CuTe/CuTile are NVIDIA-only. See README section 1."

# `all` builds only the dependency-free labs, so a fresh clone is usable without CUTLASS.
# Build Lab 2 explicitly with `make cute CUTLASS_DIR=...`.
all: cuda

# ---- Lab 1: no external dependency ----
cuda: $(CUDA_BINS)

cuda/%.exe: cuda/%.cu cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

# ---- Lab 2: requires CUTLASS ----
# check-cutlass is the single place the requirement is enforced, so the message cannot
# drift between targets.
check-cutlass:
ifeq ($(strip $(CUTLASS_DIR)),)
	@echo "ERROR: CUTLASS_DIR is not set, so cuda/cute_layout_demo.cu cannot be built."
	@echo ""
	@echo "  CuTe lives inside NVIDIA's CUTLASS repository (>100 MB). This curriculum does"
	@echo "  not vendor it, shim it, or fake a build without it."
	@echo ""
	@echo "  Fix:"
	@echo "    git clone --depth 1 https://github.com/NVIDIA/cutlass.git ~/src/cutlass"
	@echo "    make run-cute CUTLASS_DIR=~/src/cutlass SM_ARCH=$(SM_ARCH)"
	@echo ""
	@echo "  CUTLASS_DIR must point at the repo root - the directory containing include/cute."
	@echo "  Lab 1 (make run-cuda) has no such dependency and works right now."
	@exit 1
else
	@test -d "$(CUTLASS_DIR)/include/cute" || { \
	  echo "ERROR: CUTLASS_DIR=$(CUTLASS_DIR) does not contain include/cute."; \
	  echo "       Point CUTLASS_DIR at the CUTLASS repository root."; \
	  exit 1; }
	@echo "CUTLASS found: $(CUTLASS_DIR)"
endif

cute: check-cutlass
	$(NVCC) $(CUTE_FLAGS) cuda/cute_layout_demo.cu -o $(CUTE_BIN)

run-cute: cute
	@echo "== $(CUTE_BIN) =="
	@$(CUTE_BIN)

# ---- Profiling (Lab 1) ----
profile-cuda: cuda/mma_tensor_core.exe
	nsys profile --stats=true -o $(OUTPUT_DIR)/mma_tensor_core ./cuda/mma_tensor_core.exe
	@echo "For tensor-core pipe utilization: ncu --set full ./cuda/mma_tensor_core.exe"
	@echo "Metric names vary by Nsight version - list yours with: ncu --query-metrics | grep -i tensor"

# ---- Drills (delegate to their own Makefiles) ----
# Command-line overrides such as SM_ARCH=sm_80 propagate to the sub-make.
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
	rm -f $(CUDA_BINS) $(CUTE_BIN)
	rm -rf $(OUTPUT_DIR) *.nsys-rep *.sqlite
	$(MAKE) -C exercises clean
	$(MAKE) -C solutions clean
