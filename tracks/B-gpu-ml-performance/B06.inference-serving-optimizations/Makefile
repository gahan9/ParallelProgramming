# B06 - Inference serving optimizations. Python-first: no GPU, no compiler,
# no third-party packages. Everything here is stdlib CPython.
#
# Quick start:
#   make help              # list targets
#   make run               # all three models, default parameters
#   make run-exercises     # the exact runs the exercise drills refer to
#   make clean             # remove __pycache__
#
# On Windows the interpreter is usually `python`:  make run PYTHON=python

PYTHON ?= python3

SIM        := sim/serving_sim.py
PLANNER    := sim/kv_cache_planner.py
SPECULATIVE := sim/speculative_model.py

.PHONY: help run run-batching run-kv run-speculative run-sweep run-exercises clean

help:
	@echo "B06 targets (PYTHON=$(PYTHON)):"
	@echo "  make run              - run all three models with default parameters"
	@echo "  make run-batching     - static vs continuous vs chunked-prefill batching"
	@echo "  make run-sweep        - batch-cap sweep: the latency/throughput/goodput knee"
	@echo "  make run-kv           - KV cache capacity plan (paged vs contiguous, GQA, KV quant)"
	@echo "  make run-speculative  - speculative decoding: acceptance-rate math"
	@echo "  make run-exercises    - the exact runs exercises/ 01-03 refer to"
	@echo "  make clean            - remove __pycache__"
	@echo ""
	@echo "Every number these models print is a MODEL OUTPUT, not a hardware measurement."

run: run-batching run-kv run-speculative

run-batching:
	$(PYTHON) $(SIM)

run-sweep:
	$(PYTHON) $(SIM) --policies continuous_chunked --rate 20 \
		--slo-ttft-ms 2000 --slo-tpot-ms 12 --sweep-batch 4,8,16,24,32,48,64,128

run-kv:
	$(PYTHON) $(PLANNER) --preset gqa-70b --gpu mi300x --context 8192 --compute-batch-cap 64

run-speculative:
	$(PYTHON) $(SPECULATIVE)

# Exercise 1 is a capacity plan; exercise 2 is an SLO/batch-size choice; exercise 3
# is a triage drill whose "what would you measure" answers are all visible in the
# batching comparison. Run these, then check yourself against solutions/.
run-exercises:
	@echo "== Exercise 1: capacity planning (70B on one MI300X, 8k context) =="
	$(PYTHON) $(PLANNER) --preset gqa-70b --gpu mi300x --context 8192 \
		--mean-context 2050 --compute-batch-cap 128
	@echo "== Exercise 2: choosing a batch cap under a TTFT/TPOT budget =="
	$(PYTHON) $(SIM) --policies continuous_chunked --rate 20 \
		--slo-ttft-ms 2000 --slo-tpot-ms 12 --sweep-batch 4,8,16,24,32,48,64,128
	@echo "== Exercise 3: symptom triage evidence =="
	$(PYTHON) $(SIM)
	$(PYTHON) $(SPECULATIVE) --alpha 0.6 --k 4

clean:
	rm -rf sim/__pycache__ __pycache__
