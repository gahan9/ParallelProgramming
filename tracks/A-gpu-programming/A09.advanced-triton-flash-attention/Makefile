# A09 - Advanced Triton & FlashAttention. Python/Triton only: there is no hipcc or
# nvcc step here, because the whole point of the module is that one Triton kernel
# serves AMD and NVIDIA. See README section 4 for why no hip/ or cuda/ directory.
#
# Quick start:
#   make                            # list targets
#   make deps                       # install torch (+ triton) - read the caveats first
#   make run                        # run all three labs in order
#   make run-autotune               # lab 1: triton.autotune search + winning config
#   make run-flash                  # lab 2: fused attention, verified vs PyTorch
#   make run-benchmark              # lab 3: naive vs fused across sequence lengths
#   make run-exercises              # the drills (FAIL is expected until solved)
#   make run-solutions              # the reference answers (all PASS)
#   make clean                      # remove __pycache__ and autotune artifacts
#
# Sweep your own sequence lengths:
#   make run-benchmark SEQ="1024 4096 8192"

# Toolchain, arch, and base flags are shared across Track A (see ../common.mk).
# Nothing here compiles C++, but the include keeps the module consistent with the
# rest of the track and gives `make help` the same ROCm banner.
include ../common.mk

PYTHON ?= python3
SEQ    ?=

# Ask Triton to report the winning config and the time it spent tuning. Exported so
# it reaches the interpreter in every recipe below.
TRITON_PRINT_AUTOTUNING ?= 1
export TRITON_PRINT_AUTOTUNING

.PHONY: help deps run run-autotune run-flash run-benchmark \
        run-exercises run-solutions clean

help:
	@echo "Targets:"
	@echo "  make deps                  - pip install -r requirements.txt (READ its comments on ROCm)"
	@echo "  make run                   - run all three labs"
	@echo "  make run-autotune          - lab 1: autotune search space + winner + tuning cost"
	@echo "  make run-flash             - lab 2: fused FlashAttention-style forward pass"
	@echo "  make run-benchmark         - lab 3: naive vs fused sweep  (SEQ=\"1024 4096\")"
	@echo "  make run-exercises         - the drills (FAIL expected until the TODOs are done)"
	@echo "  make run-solutions         - the reference answers (all PASS)"
	@echo "  make clean                 - remove __pycache__ / .triton artifacts"
	@echo "Python: $(PYTHON)   TRITON_PRINT_AUTOTUNING=$(TRITON_PRINT_AUTOTUNING)"
	@echo "ROCm:   $(HIP_TOOLCHAIN_INFO)"

deps:
	@echo "Note: on ROCm, install torch from the ROCm wheel index FIRST - it bundles"
	@echo "a matching Triton. See requirements.txt and ../../../docs/SETUP.md."
	$(PYTHON) -m pip install -r requirements.txt

run: run-autotune run-flash run-benchmark

run-autotune:
	$(PYTHON) triton/autotune_demo.py

run-flash:
	$(PYTHON) triton/flash_attention.py

run-benchmark:
	$(PYTHON) triton/benchmark_attention.py $(SEQ)

# Keep going when a drill fails: an unsolved TODO is expected to FAIL, and stopping
# at the first one would hide the rest.
run-exercises:
	@for f in exercises/*.py; do \
		echo "== $$f =="; \
		$(PYTHON) $$f || echo "   (non-zero exit - expected until the TODOs are done)"; \
		echo; \
	done

run-solutions:
	@for f in solutions/*.py; do \
		echo "== $$f =="; \
		$(PYTHON) $$f; \
		echo; \
	done

clean:
	rm -rf triton/__pycache__ exercises/__pycache__ solutions/__pycache__
	rm -rf .pytest_cache profiling_output
