# B05 - Transformer performance. Python-first: nothing here is compiled.
#
# This module analyzes and measures a transformer rather than writing kernels, so
# unlike the Track A modules it does NOT include ../common.mk - there is no hipcc
# or nvcc in the loop. The only toolchain is Python.
#
# Split of responsibilities:
#   tools/  runs anywhere, no GPU, no third-party packages  -> derived estimates
#   bench/  needs a GPU and torch                          -> measurements
#
# Quick start:
#   make                  # this help
#   make calc             # analytical sizing for every preset (no GPU needed)
#   make run              # calc + both GPU benchmarks
#   make run-exercises    # the drill list, plus a calculator run to check drill 1
#   make clean

PYTHON ?= python3

CALC  := $(PYTHON) tools/model_calculator.py
BENCH := $(PYTHON) bench

# Match to the GPU you are reasoning about; these feed the ceiling calculations.
# mi300x | mi355x | h100 | h200, or pass --bandwidth/--capacity/--peak-tflops.
GPU ?= mi300x

.PHONY: help calc calc-8b calc-70b calc-175b run profile scaling \
        run-exercises exercises solutions check clean

help:
	@echo "B05 - Transformer performance (Python-first, no compile step)"
	@echo ""
	@echo "No GPU required:"
	@echo "  make calc                 - params/FLOPs/KV/tokens-per-sec for all presets"
	@echo "  make calc-8b   GPU=$(GPU)  - Llama-3-8B-shaped sizing"
	@echo "  make calc-70b  GPU=$(GPU)  - Llama-2-70B-shaped sizing, batch 32"
	@echo "  make calc-175b GPU=$(GPU)  - GPT-3-175B-shaped sizing (validates 12 d^2)"
	@echo "  make check                - smoke-test the calculator on every preset"
	@echo ""
	@echo "GPU + torch required (see ../../../docs/SETUP.md):"
	@echo "  make profile              - time a decoder layer: prefill vs decode"
	@echo "  make scaling              - decode throughput vs batch size"
	@echo "  make run                  - calc, then both benchmarks"
	@echo ""
	@echo "Drills:"
	@echo "  make run-exercises        - list the drills + check drill 1's config"
	@echo "  make solutions            - list the reference answers"
	@echo ""
	@echo "  make clean                - remove __pycache__ and stray .pyc"
	@echo ""
	@echo "PYTHON=$(PYTHON)   GPU=$(GPU)"

# ---- Analytical sizing (runs anywhere, stdlib only) ----
calc: calc-8b calc-70b calc-175b

calc-8b:
	$(CALC) --preset llama3-8b --gpu $(GPU)

calc-70b:
	$(CALC) --preset llama2-70b --gpu $(GPU) --batch 32

calc-175b:
	$(CALC) --preset gpt3-175b --gpu $(GPU)

# ---- Measurement (needs a GPU; the scripts refuse to fake a CPU number) ----
profile:
	$(PYTHON) bench/profile_layer.py

scaling:
	$(PYTHON) bench/batch_scaling.py --bandwidth 5300

run: calc profile scaling

# ---- Drills ----
# The drills are paper exercises (markdown), so there is nothing to build. The
# calculator is the answer key for the numeric parts.
run-exercises:
	@echo "B05 drills - all three are paper exercises, see exercises/README.md"
	@echo ""
	@echo "  1. exercises/01_size_a_model.md    size a model end to end"
	@echo "  2. exercises/02_kv_cache_budget.md concurrency under an HBM budget"
	@echo "  3. exercises/03_regime_diagnosis.md read a profile, pick the fix"
	@echo ""
	@echo "Drill 1's configuration, per the calculator (check your arithmetic"
	@echo "against this AFTER working it by hand):"
	@echo ""
	$(CALC) -L 48 -d 6144 --heads 48 --kv-heads 8 --ffn 16384 --glu \
	        --vocab 64000 --seq 8192 --gpu $(GPU)

exercises: run-exercises

solutions:
	@echo "Reference answers with full working:"
	@echo "  solutions/01_size_a_model_solution.md"
	@echo "  solutions/02_kv_cache_budget_solution.md"
	@echo "  solutions/03_regime_diagnosis_solution.md"

# ---- Housekeeping ----
# Smoke test: every preset must produce a report without raising. Cheap enough to
# run in CI, and it catches formula edits that break a code path.
check:
	@for preset in llama3-8b llama2-70b gpt3-175b; do \
	  echo "== $$preset =="; \
	  $(CALC) --preset $$preset > /dev/null || exit 1; \
	done
	@$(PYTHON) tools/model_calculator.py -L 4 -d 512 --heads 8 > /dev/null || exit 1
	@echo "calculator OK on all presets + a hand-specified config"

clean:
	rm -rf tools/__pycache__ bench/__pycache__ __pycache__
	rm -f tools/*.pyc bench/*.pyc
