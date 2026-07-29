# B03 solutions - reproduce every run the answer keys refer to.
#
# The answer keys are written work, so there is nothing to compile here. What
# this Makefile provides is the exact command sequence that produces the evidence
# the keys reason about, so you can check the keys against your own hardware
# rather than taking them on trust.
#
# Quick start:
#   make                       # list targets
#   make reproduce GPU_ARCH=gfx950   # timeline + all four counter runs
#   make parse                 # phase/kernel breakdown of the timeline
#   make clean

include ../../../A-gpu-programming/common.mk

MODULE_DIR := ..
OUTPUT_DIR := profiling_output

PMC     ?= MeanOccupancyPerCU VALUUtilization MemUnitStalled L2CacheHit
KERNELS := kernel_alpha kernel_beta kernel_gamma kernel_delta
PYTHON  ?= python3

.PHONY: help reproduce timeline counters parse clean

help:
	@echo "B03 solutions (GPU_ARCH=$(GPU_ARCH)):"
	@echo "  make reproduce   - timeline + counters for all four kernels"
	@echo "  make timeline    - rocprofv3 --summary --sys-trace only"
	@echo "  make counters    - rocprofv3 --pmc for each of: $(KERNELS)"
	@echo "  make parse       - phase/kernel breakdown of $(OUTPUT_DIR)/"
	@echo "  make clean       - remove $(OUTPUT_DIR)/"
	@echo "PMC set: $(PMC)   (check availability with: $(ROCPROFV3) --list-avail)"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

reproduce: timeline counters parse

timeline:
	$(MAKE) -C $(MODULE_DIR) hip
	$(ROCPROFV3) --summary --summary-per-domain --sys-trace \
	             --output-format csv -d $(OUTPUT_DIR) -- $(MODULE_DIR)/hip/mystery_kernels.exe

# One collection per kernel: counter collection serializes dispatches, so
# profiling all four at once costs four times as much and tells you no more.
# Each kernel gets its own directory so the pmc_1/ outputs do not overwrite.
counters:
	$(MAKE) -C $(MODULE_DIR) hip
	@for k in $(KERNELS); do \
		echo "== counters: $$k =="; \
		$(ROCPROFV3) --pmc $(PMC) \
		             --kernel-include-regex "$$k" \
		             --kernel-iteration-range 4-8 \
		             --truncate-kernels \
		             --output-format csv -d $(OUTPUT_DIR)/$$k \
		             -- $(MODULE_DIR)/hip/mystery_kernels.exe "$$k" || exit 1; \
	done

parse:
	$(PYTHON) $(MODULE_DIR)/tools/parse_rocprof.py $(OUTPUT_DIR)

clean:
	rm -rf $(OUTPUT_DIR)
