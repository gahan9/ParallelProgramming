# C07 - Capstone case studies. Design-only module: the "build" is two Python
# calculators, so there is no hipcc/nvcc target here (see README section 4 for
# why this module ships no kernels).
#
# Quick start:
#   make help            # list targets
#   make run             # both calculators with the case-study defaults
#   make run-exercises   # the what-if scenarios the drills ask for
#   make clean           # remove generated caches
#
# Requires Python 3.9+. Standard library only - nothing to install.
#
# Like the rest of this repo, the recipes assume GNU make and a POSIX shell.
# On Windows without them, run the tools directly - there is nothing to build:
#   python tools\llm_sizing.py
#   python tools\funnel_budget.py

PYTHON ?= python
TOOLS  := tools
LLM    := $(TOOLS)/llm_sizing.py
FUNNEL := $(TOOLS)/funnel_budget.py

.PHONY: help run run-llm run-funnel run-exercises check clean

help:
	@echo "Targets:"
	@echo "  make run             - run both calculators with case-study defaults"
	@echo "  make run-llm         - LLM serving sizing only  (case study 1)"
	@echo "  make run-funnel      - recommender funnel budget (case study 2)"
	@echo "  make run-exercises   - the what-if scenarios from the labs and drills"
	@echo "  make check           - byte-compile both tools and exercise a bad-input path"
	@echo "  make clean           - remove __pycache__ and .pyc files"
	@echo ""
	@echo "Both tools are standard-library only. Full flags:"
	@echo "  $(PYTHON) $(LLM) --help"
	@echo "  $(PYTHON) $(FUNNEL) --help"
	@echo ""
	@echo "Using PYTHON=$(PYTHON)  (override with: make run PYTHON=python3)"

run: run-llm run-funnel

run-llm:
	@echo "== case study 1: LLM serving sizing =="
	$(PYTHON) $(LLM)

run-funnel:
	@echo "== case study 2: recommender funnel budget =="
	$(PYTHON) $(FUNNEL)

# Mirrors README section 4 (labs 2, 3, 5): quantization moves you to a different
# roof, long context moves the batching knee, and widening a funnel stage breaks
# the latency budget.
run-exercises:
	@echo "== lab 2: fp8 weights + KV - watch the bound flip to compute =="
	$(PYTHON) $(LLM) --weight-dtype fp8 --kv-dtype fp8
	@echo ""
	@echo "== lab 3: long context - watch the batching knee collapse =="
	$(PYTHON) $(LLM) --avg-ctx 8192 --batch 64
	$(PYTHON) $(LLM) --avg-ctx 32768 --batch 16
	@echo ""
	@echo "== lab 5: widen the ranker - watch the latency budget break =="
	$(PYTHON) $(FUNNEL) --widen rank=2000
	@echo ""
	@echo "== lab 5: widen retrieval by the same factor - 12 ms instead of 90 ms =="
	$(PYTHON) $(FUNNEL) --widen retrieve=3000

# Bad input must produce a one-line message and exit 2, never a traceback.
check:
	$(PYTHON) -m py_compile $(LLM) $(FUNNEL)
	@echo "byte-compile OK"
	@if $(PYTHON) $(LLM) --tp 3 >/dev/null 2>&1; then \
		echo "FAIL: llm_sizing.py accepted a TP that does not divide the node"; exit 1; fi
	@if $(PYTHON) $(FUNNEL) --widen nonesuch=10 >/dev/null 2>&1; then \
		echo "FAIL: funnel_budget.py accepted an unknown stage name"; exit 1; fi
	@echo "bad-input paths exit non-zero without a traceback"

clean:
	rm -rf $(TOOLS)/__pycache__ __pycache__
	rm -f $(TOOLS)/*.pyc *.pyc
