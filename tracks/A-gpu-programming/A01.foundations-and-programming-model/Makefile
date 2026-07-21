# A01 - Foundations. Dual-backend build: AMD (hipcc) and NVIDIA (nvcc).
#
# Quick start:
#   make hip   GPU_ARCH=gfx942     # build all hip/ programs
#   make run-hip                    # run them
#   make cuda  SM_ARCH=sm_90        # build all cuda/ programs
#   make run-cuda                   # run them
#   make profile-hip                # rocprofv3 summary + traces
#   make profile-cuda               # nsys timeline
#   make clean

# ---- Toolchains ----
# Call ROCm tools by absolute path: /opt/rocm/bin is often NOT on PATH
# (empty ROCM_PATH), so bare `hipcc`/`rocprofv3` would fail with "No such file".
HIP_PATH  ?= $(wildcard /opt/rocm)
HIPCC     := $(HIP_PATH)/bin/hipcc
ROCPROFV3 := $(HIP_PATH)/bin/rocprofv3
NVCC      := nvcc

# ---- Architectures (override on the command line for your GPU) ----
GPU_ARCH ?= gfx942      # AMD: gfx950 (MI350), gfx942 (MI300), gfx90a (MI200), gfx1100 (RDNA3)...
SM_ARCH  ?= sm_90       # NVIDIA: sm_90 (Hopper), sm_80 (Ampere), sm_89 (Ada)...

# ---- Flags ----
# rpath keeps binaries self-contained: they locate libamdhip64.so under the
# ROCm install even when /opt/rocm/lib is not on the system loader path
# (no ldconfig entry / empty LD_LIBRARY_PATH).
HIP_FLAGS  := -O3 -g --offload-arch=$(GPU_ARCH) -Wl,-rpath,$(HIP_PATH)/lib -Ihip
CUDA_FLAGS := -O3 -g -arch=$(SM_ARCH) -Icuda

# ---- Programs (one per source file) ----
PROGS       := hello vector_add coalescing
HIP_BINS    := $(addprefix hip/,  $(addsuffix .exe, $(PROGS)))
CUDA_BINS   := $(addprefix cuda/, $(addsuffix .exe, $(PROGS)))

OUTPUT_DIR := profiling_output

.PHONY: help all hip cuda run-hip run-cuda profile-hip profile-cuda clean

help:
	@echo "Targets:"
	@echo "  make hip   GPU_ARCH=$(GPU_ARCH)   - build hip/  programs with hipcc"
	@echo "  make cuda  SM_ARCH=$(SM_ARCH)     - build cuda/ programs with nvcc"
	@echo "  make run-hip / run-cuda           - build + run all programs"
	@echo "  make profile-hip / profile-cuda   - profile vector_add"
	@echo "  make clean                        - remove binaries + profiler output"

all: hip cuda

# ---- AMD / HIP ----
hip: $(HIP_BINS)

hip/%.exe: hip/%.cpp hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run-hip: hip
	@for b in $(HIP_BINS); do echo "== $$b =="; $$b; echo; done

profile-hip: hip/vector_add.exe
	$(ROCPROFV3) --summary --sys-trace --output-format csv -d $(OUTPUT_DIR) -- ./hip/vector_add.exe

# ---- NVIDIA / CUDA ----
cuda: $(CUDA_BINS)

cuda/%.exe: cuda/%.cu cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

profile-cuda: cuda/vector_add.exe
	nsys profile --stats=true -o $(OUTPUT_DIR)/vector_add ./cuda/vector_add.exe
	@echo "For kernel-level counters: ncu --set full ./cuda/vector_add.exe"

# ---- Housekeeping ----
clean:
	rm -f $(HIP_BINS) $(CUDA_BINS)
	rm -rf $(OUTPUT_DIR) .rocprofv3 *.nsys-rep *.sqlite
