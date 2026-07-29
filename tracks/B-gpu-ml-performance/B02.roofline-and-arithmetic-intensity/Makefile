# B02 - Roofline model & arithmetic intensity. Dual-backend build: AMD (hipcc) and NVIDIA (nvcc).
#
# Quick start:
#   make hip   GPU_ARCH=gfx942     # build all hip/ examples
#   make run-hip                    # run them (measures both ceilings + sweeps AI)
#   make cuda  SM_ARCH=sm_90        # build all cuda/ examples
#   make run-cuda                   # run them
#   make plot                       # ASCII roofline chart from the plotter's demo ceilings
#   make exercises                  # build the exercises/ drills
#   make solutions                  # build the solutions/ reference answers
#   make clean                      # clean here + exercises/ + solutions/
#
# Order matters when you run by hand: measure_ceilings gives you P and B, then
# roofline_sweep shows a single kernel walking from one roof to the other.

# Track B reuses Track A's toolchain discovery (ROCm location, rpath, arch flags)
# rather than forking a second copy of it. Path: this dir -> tracks/ -> A-gpu-programming.
include ../../A-gpu-programming/common.mk

# ---- Flags ----
HIP_FLAGS  := $(HIP_BASE_FLAGS) -Ihip
CUDA_FLAGS := $(CUDA_BASE_FLAGS) -Icuda

# ---- Programs (ceilings first: the sweep is read against them) ----
PROGS     := measure_ceilings roofline_sweep
HIP_BINS  := $(addprefix hip/,  $(addsuffix .exe, $(PROGS)))
CUDA_BINS := $(addprefix cuda/, $(addsuffix .exe, $(PROGS)))

PYTHON     ?= python3
OUTPUT_DIR := profiling_output

.PHONY: help all hip cuda run-hip run-cuda plot exercises solutions \
        run-exercises run-solutions profile-hip profile-cuda clean

help:
	@echo "Targets:"
	@echo "  make hip   GPU_ARCH=$(GPU_ARCH)   - build hip/  examples with hipcc"
	@echo "  make cuda  SM_ARCH=$(SM_ARCH)     - build cuda/ examples with nvcc"
	@echo "  make run-hip / run-cuda           - build + run: ceilings, then the AI sweep"
	@echo "  make plot                         - ASCII roofline chart (stdlib Python only)"
	@echo "  make exercises / solutions        - build the drills (own Makefiles)"
	@echo "  make run-exercises / run-solutions- build + run the drills"
	@echo "  make profile-hip / profile-cuda   - profile roofline_sweep"
	@echo "  make clean                        - remove binaries + profiler output"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: hip cuda

# ---- AMD / HIP ----
hip: $(HIP_BINS)

hip/%.exe: hip/%.cpp hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run-hip: hip
	@for b in $(HIP_BINS); do echo "== $$b =="; $$b; echo; done

profile-hip: hip/roofline_sweep.exe
	$(ROCPROFV3) --summary --sys-trace --output-format csv -d $(OUTPUT_DIR) -- ./hip/roofline_sweep.exe

# ---- NVIDIA / CUDA ----
cuda: $(CUDA_BINS)

cuda/%.exe: cuda/%.cu cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

profile-cuda: cuda/roofline_sweep.exe
	nsys profile --stats=true -o $(OUTPUT_DIR)/roofline_sweep ./cuda/roofline_sweep.exe
	@echo "For the profiler's own roofline chart: ncu --set full ./cuda/roofline_sweep.exe"

# ---- Roofline chart ----
# Runs with the plotter's illustrative ceilings. Replace them with your measured
# numbers from measure_ceilings:
#   python tools/roofline_plot.py --peak-gflops <P> --peak-gbps <B> --demo-kernels
plot:
	$(PYTHON) tools/roofline_plot.py --demo

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
	rm -rf $(OUTPUT_DIR) .rocprofv3 *.nsys-rep *.sqlite tools/__pycache__
	$(MAKE) -C exercises clean
	$(MAKE) -C solutions clean
