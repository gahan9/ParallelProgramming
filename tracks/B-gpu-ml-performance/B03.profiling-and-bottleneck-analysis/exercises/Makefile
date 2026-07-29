# B03 exercises - drive the profiling runs the drills ask for.
#
# The three B03 drills are written work, not code to fill in: you profile the
# module's mystery_kernels binary (and A02's transpose) and write down the
# evidence. So this Makefile does not compile anything of its own - it builds the
# module binary and runs the exact profiler invocations each drill needs, into
# exercises/profiling_output/.
#
# Quick start:
#   make                            # list targets
#   make timeline    GPU_ARCH=gfx950  # drill 1, level 1: where does wall clock go?
#   make counters    KERNEL=kernel_beta   # drill 1, level 2: what limits ONE kernel?
#   make transpose-timeline           # drill 3: profile A02's transpose ladder
#   make clean

include ../../../A-gpu-programming/common.mk

MODULE_DIR   := ..
TRANSPOSE_DIR := ../../../A-gpu-programming/A02.memory-hierarchy-and-coalescing
OUTPUT_DIR   := profiling_output

# Counters that answer occupancy / memory / cache / divergence in one pass.
# Availability is architecture-specific: check with `rocprofv3 --list-avail`.
PMC    ?= MeanOccupancyPerCU VALUUtilization MemUnitStalled L2CacheHit
KERNEL ?= kernel_gamma
PYTHON ?= python3

.PHONY: help timeline counters parse transpose-timeline transpose-counters clean

help:
	@echo "B03 exercises (GPU_ARCH=$(GPU_ARCH)):"
	@echo "  make timeline            - drill 1 level 1: rocprofv3 --summary --sys-trace"
	@echo "  make counters            - drill 1 level 2: rocprofv3 --pmc, KERNEL=$(KERNEL)"
	@echo "  make parse               - phase/kernel breakdown of whatever is in $(OUTPUT_DIR)/"
	@echo "  make transpose-timeline  - drill 3 level 1: profile A02's transpose"
	@echo "  make transpose-counters  - drill 3 level 2: counters for one transpose kernel"
	@echo "  make clean               - remove $(OUTPUT_DIR)/"
	@echo "Overrides: KERNEL=kernel_beta  PMC='SQ_WAVES GRBM_GUI_ACTIVE'"
	@echo "Drill 2 (02_amdahl.md) is arithmetic - no run needed."
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

# Always run this BEFORE counters. If a kernel is a small share of the run,
# counter-profiling it is wasted effort - that is drill 2's whole point.
timeline:
	$(MAKE) -C $(MODULE_DIR) hip
	$(ROCPROFV3) --summary --summary-per-domain --sys-trace \
	             --output-format csv -d $(OUTPUT_DIR) -- $(MODULE_DIR)/hip/mystery_kernels.exe

counters:
	$(MAKE) -C $(MODULE_DIR) hip
	$(ROCPROFV3) --pmc $(PMC) \
	             --kernel-include-regex '$(KERNEL)' \
	             --kernel-iteration-range 4-8 \
	             --truncate-kernels \
	             --output-format csv -d $(OUTPUT_DIR) -- $(MODULE_DIR)/hip/mystery_kernels.exe

parse:
	$(PYTHON) $(MODULE_DIR)/tools/parse_rocprof.py $(OUTPUT_DIR)

transpose-timeline:
	$(MAKE) -C $(TRANSPOSE_DIR) hip
	$(ROCPROFV3) --summary --summary-per-domain --sys-trace \
	             --output-format csv -d $(OUTPUT_DIR)/transpose \
	             -- $(TRANSPOSE_DIR)/hip/transpose.exe

# TRANSPOSE_KERNEL selects which of A02's three transpose variants to inspect.
TRANSPOSE_KERNEL ?= transpose_tiled

transpose-counters:
	$(MAKE) -C $(TRANSPOSE_DIR) hip
	$(ROCPROFV3) --pmc LDSBankConflict MemUnitStalled L2CacheHit MeanOccupancyPerCU \
	             --kernel-include-regex '$(TRANSPOSE_KERNEL)' \
	             --truncate-kernels \
	             --output-format csv -d $(OUTPUT_DIR)/transpose \
	             -- $(TRANSPOSE_DIR)/hip/transpose.exe

clean:
	rm -rf $(OUTPUT_DIR)
