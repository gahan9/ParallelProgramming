# B03 - Profiling & bottleneck analysis. Dual-backend build: AMD (hipcc), NVIDIA (nvcc).
#
# Track B reuses Track A's toolchain file, so a ROCm-discovery fix lands
# everywhere at once. From here that is two levels up into tracks/, then across.
#
# Quick start:
#   make hip   GPU_ARCH=gfx942      # build hip/mystery_kernels
#   make run-hip                     # run it (no diagnosis printed - that's the lab)
#   make cuda  SM_ARCH=sm_90         # build cuda/mystery_kernels
#   make run-cuda
#
#   make profile-hip                 # LEVEL 1 (AMD): timeline - where does wall-clock go?
#   make profile-kernel-hip          # LEVEL 2 (AMD): counters - what limits the kernel?
#   make profile-cuda                # LEVEL 1 (NVIDIA): nsys timeline
#   make profile-ncu                 # LEVEL 2 (NVIDIA): ncu Speed-of-Light + memory
#   make parse                       # summarize the rocprofv3 CSVs (tools/parse_rocprof.py)
#   make clean
#
# Always run LEVEL 1 before LEVEL 2. Counter profiling a kernel that is 2% of the
# run is how engineers lose days - see README section 6.

include ../../A-gpu-programming/common.mk

# ---- Flags ----
HIP_FLAGS  := $(HIP_BASE_FLAGS) -Ihip
CUDA_FLAGS := $(CUDA_BASE_FLAGS) -Icuda

PROGS      := mystery_kernels
HIP_BINS   := $(addprefix hip/,  $(addsuffix .exe, $(PROGS)))
CUDA_BINS  := $(addprefix cuda/, $(addsuffix .exe, $(PROGS)))

OUTPUT_DIR := profiling_output

# Counters for the level-2 AMD pass. These four answer "occupancy, memory, cache,
# divergence" in a single collection; run `$(ROCPROFV3) --list-avail` for the set
# your GPU actually supports (counter availability is architecture-specific).
PMC        ?= MeanOccupancyPerCU VALUUtilization MemUnitStalled L2CacheHit

# Which kernel to counter-profile. The four mystery kernels are independent, so
# profile one at a time: counter collection serializes dispatches and is slow.
KERNEL     ?= kernel_gamma

PYTHON     ?= python3

.PHONY: help all hip cuda run-hip run-cuda \
        profile-hip profile-kernel-hip profile-cuda profile-ncu parse clean

help:
	@echo "Build:"
	@echo "  make hip   GPU_ARCH=$(GPU_ARCH)   - build hip/  with hipcc"
	@echo "  make cuda  SM_ARCH=$(SM_ARCH)     - build cuda/ with nvcc"
	@echo "  make run-hip / run-cuda           - build + run the mystery kernels"
	@echo "Profile (LEVEL 1 = timeline first, LEVEL 2 = counters only after):"
	@echo "  make profile-hip                  - rocprofv3 summary + sys-trace  -> $(OUTPUT_DIR)/"
	@echo "  make profile-kernel-hip           - rocprofv3 --pmc counters, KERNEL=$(KERNEL)"
	@echo "  make profile-cuda                 - nsys profile --stats=true      -> $(OUTPUT_DIR)/"
	@echo "  make profile-ncu                  - ncu SpeedOfLight + memory, KERNEL=$(KERNEL)"
	@echo "  make parse                        - phase/kernel breakdown from the CSVs"
	@echo "Overrides: KERNEL=kernel_beta  PMC='SQ_WAVES GRBM_GUI_ACTIVE'"
	@echo "  make clean                        - remove binaries + profiler output"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: hip cuda

# ---- AMD / HIP ----
hip: $(HIP_BINS)

hip/%.exe: hip/%.cpp hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run-hip: hip
	@for b in $(HIP_BINS); do echo "== $$b =="; $$b; echo; done

# LEVEL 1 - system timeline. --sys-trace covers HIP + HSA API, memory copies,
# allocations and kernel dispatches; --summary aggregates them into the table you
# read first. --output-format csv is explicit because rocprofv3's default output
# format is rocpd (a SQLite3 database) as of ROCm 7.x.
profile-hip: hip/mystery_kernels.exe
	$(ROCPROFV3) --summary --summary-per-domain --sys-trace \
	             --output-format csv -d $(OUTPUT_DIR) -- ./hip/mystery_kernels.exe

# LEVEL 2 - hardware counters for ONE kernel. --kernel-include-regex keeps the
# collection cheap; --kernel-iteration-range skips the warmup dispatches so the
# counters describe steady state. Output lands in $(OUTPUT_DIR)/pmc_1/.
profile-kernel-hip: hip/mystery_kernels.exe
	$(ROCPROFV3) --pmc $(PMC) \
	             --kernel-include-regex '$(KERNEL)' \
	             --kernel-iteration-range 4-8 \
	             --truncate-kernels \
	             --output-format csv -d $(OUTPUT_DIR) -- ./hip/mystery_kernels.exe

# ---- NVIDIA / CUDA ----
cuda: $(CUDA_BINS)

cuda/%.exe: cuda/%.cu cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run-cuda: cuda
	@for b in $(CUDA_BINS); do echo "== $$b =="; $$b; echo; done

# LEVEL 1 - system timeline. --stats=true prints the summary tables (and builds a
# .sqlite alongside the .nsys-rep).
profile-cuda: cuda/mystery_kernels.exe
	mkdir -p $(OUTPUT_DIR)
	nsys profile --stats=true --trace=cuda,osrt \
	             -o $(OUTPUT_DIR)/mystery_kernels --force-overwrite=true \
	             ./cuda/mystery_kernels.exe

# LEVEL 2 - kernel counters. Two sections instead of --set full: SpeedOfLight
# gives the compute-vs-memory verdict and MemoryWorkloadAnalysis explains it,
# at a fraction of the replay cost. --launch-skip drops the warmup launches.
profile-ncu: cuda/mystery_kernels.exe
	mkdir -p $(OUTPUT_DIR)
	ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
	    --kernel-name regex:'$(KERNEL)' --launch-skip 3 --launch-count 3 \
	    -o $(OUTPUT_DIR)/$(KERNEL) --force-overwrite \
	    ./cuda/mystery_kernels.exe
	@echo "Open with: ncu-ui $(OUTPUT_DIR)/$(KERNEL).ncu-rep   (or add --page details for stdout)"

# ---- Analysis ----
parse:
	$(PYTHON) tools/parse_rocprof.py $(OUTPUT_DIR)

# ---- Housekeeping ----
# exercises/ and solutions/ have their own Makefiles that drive the drills
# (cd exercises && make). Nothing to build from here, but their output is ours
# to clean up.
clean:
	rm -f $(HIP_BINS) $(CUDA_BINS)
	rm -rf $(OUTPUT_DIR) .rocprofv3 *.nsys-rep *.sqlite *.ncu-rep
	$(MAKE) -C exercises clean
	$(MAKE) -C solutions clean
