# A08 - Triton foundations. Python-first: this module has no .cpp / .cu sources.
#
# There is nothing to compile ahead of time - Triton JIT-compiles each kernel on its
# first call - so these targets only RUN things. ../common.mk is still included for
# consistency with the rest of Track A and because `make help` reports the ROCm
# install it finds, which is what supplies the torch/triton wheel on AMD.
#
# Quick start:
#   make help              # targets + detected toolchain
#   make venv              # PRINTS setup instructions only; it creates nothing
#   make run               # run the three triton/ labs (all must PASS)
#   make run-exercises     # run exercises/ (FAIL is expected until solved)
#   make run-solutions     # run solutions/ (every one must PASS)
#   make clean             # remove __pycache__

include ../common.mk

# Override if your interpreter is named differently or lives in a venv:
#   make run PYTHON=.venv/bin/python
PYTHON ?= python3

LABS := triton/vector_add.py \
        triton/fused_softmax.py \
        triton/matmul.py

EXERCISES := exercises/01_fused_relu_add.py \
             exercises/02_masked_mean.py

SOLUTIONS := solutions/01_fused_relu_add_solution.py \
             solutions/02_masked_mean_solution.py

.PHONY: help venv run run-exercises run-solutions clean

help:
	@echo "A08 - Triton foundations (Python-first; nothing to compile)."
	@echo "Targets:"
	@echo "  make venv              - PRINT venv + wheel instructions (creates nothing)"
	@echo "  make run               - run the 3 labs: $(notdir $(LABS))"
	@echo "  make run-exercises     - run exercises/ (FAIL expected until solved)"
	@echo "  make run-solutions     - run solutions/ (must all PASS)"
	@echo "  make clean             - remove __pycache__ directories"
	@echo "Override the interpreter:  make run PYTHON=.venv/bin/python"
	@echo "PYTHON: $(PYTHON)"
	@echo "ROCm:   $(HIP_TOOLCHAIN_INFO)   (supplies the ROCm torch/triton wheel)"
	@echo "Drills 03_triton_vs_cuda are written, not runnable - open the .md files."

# Deliberately does NOT create anything: the correct wheel index depends on your
# vendor and ROCm/CUDA version, and guessing it here would install a broken stack.
venv:
	@echo "This target only prints instructions - it creates nothing."
	@echo
	@echo "  python3 -m venv .venv && source .venv/bin/activate"
	@echo "  # NVIDIA:  pip install torch          (bundles pytorch-triton)"
	@echo "  # AMD:     pip install torch --index-url <ROCm wheel index>"
	@echo
	@echo "The current ROCm wheel index is version-specific; get it from"
	@echo "https://pytorch.org/get-started/locally/ and see ../../../docs/SETUP.md."
	@echo "Then verify:  $(PYTHON) -c 'import torch, triton; print(torch.__version__, torch.cuda.is_available())'"
	@echo "Then:         make run PYTHON=.venv/bin/python"

# Labs must pass, so stop at the first failure.
run:
	@set -e; for f in $(LABS); do \
		echo "== $$f =="; \
		$(PYTHON) $$f; \
		echo; \
	done

# Keep going: an unsolved TODO is expected to FAIL, and stopping at the first one
# would hide the rest.
run-exercises:
	@for f in $(EXERCISES); do \
		echo "== $$f =="; \
		$(PYTHON) $$f || echo "   (non-zero exit or FAILED - expected until the TODOs are done)"; \
		echo; \
	done

run-solutions:
	@set -e; for f in $(SOLUTIONS); do \
		echo "== $$f =="; \
		$(PYTHON) $$f; \
		echo; \
	done

# Only Python bytecode is generated here. The Triton JIT cache is NOT touched: it is
# global (~/.triton, or $TRITON_CACHE_DIR) and shared with every other module.
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
