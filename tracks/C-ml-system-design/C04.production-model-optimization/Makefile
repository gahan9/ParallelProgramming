# C04 - Production model optimization. Design module: no GPU code, only stdlib Python tools.
#
# Quick start:
#   make help            # list targets
#   make run             # run every tool with its default scenario
#   make run-exercises   # run the exact scenarios the exercises reason about
#   make clean           # remove Python caches
#
# On Windows the interpreter is usually `python`, not `python3`:
#   make run PYTHON=python

PYTHON ?= python3

.PHONY: help run run-cost run-cascade run-cache run-exercises clean

help:
	@echo "Targets:"
	@echo "  make run             - run all three tools with their default scenarios"
	@echo "  make run-cost        - cost_model: \$$/1k requests and optimization break-even"
	@echo "  make run-cascade     - cascade_analyzer: blended accuracy and cost vs escalation"
	@echo "  make run-cache       - cache_simulator: hit rate vs cache size on a Zipf trace"
	@echo "  make run-exercises   - the scenarios exercises/01-03 ask you to analyse"
	@echo "  make clean           - remove __pycache__"
	@echo "Interpreter: $(PYTHON)  (override with 'make run PYTHON=python')"
	@echo "Tools are standard-library only; no pip install is required."

run: run-cost run-cascade run-cache

run-cost:
	@echo "== cost_model: 120 qps, the whole optimization ladder priced =="
	$(PYTHON) tools/cost_model.py

run-cascade:
	@echo "== cascade_analyzer: good confidence signal (gain 3) =="
	$(PYTHON) tools/cascade_analyzer.py
	@echo
	@echo "== cascade_analyzer: escalating at random (gain 1) - the cascade collapses =="
	$(PYTHON) tools/cascade_analyzer.py --confidence-gain 1.0

run-cache:
	@echo "== cache_simulator: skewed traffic (zipf 1.0) =="
	$(PYTHON) tools/cache_simulator.py
	@echo
	@echo "== cache_simulator: uniform traffic (zipf 0) - skew was the whole business case =="
	$(PYTHON) tools/cache_simulator.py --zipf 0

# The exercises state scenarios in prose; these are the same scenarios as tool
# invocations, so you can check a hand calculation against the model.
run-exercises:
	@echo "== exercise 01: the image-captioning service, priced before choosing a rung =="
	$(PYTHON) tools/cost_model.py --qps 40 --tokens-per-request 320 \
	    --per-gpu-tokens-per-second 1800 --gpu-hour-cost 3.20 --headroom 0.45 \
	    --cache-hit-rate 0.05
	@echo
	@echo "== exercise 01: repeat rate of the request log, simulated at low skew =="
	$(PYTHON) tools/cache_simulator.py --zipf 0.3 --cache-size 2000
	@echo
	@echo "== exercise 02 background: what a small quality loss is buying =="
	$(PYTHON) tools/cascade_analyzer.py --small-accuracy 0.938 --large-accuracy 0.961 \
	    --quality-floor 0.955 --confidence-gain 2.5
	@echo
	@echo "== exercise 03: the internal classifier, 18 qps =="
	$(PYTHON) tools/cost_model.py --qps 18 --tokens-per-request 900 \
	    --per-gpu-tokens-per-second 2000 --gpu-hour-cost 2.80 \
	    --cache-hit-rate 0.40 --cache-weeks 3 \
	    --batch-multiplier 1.25 --batch-weeks 2 \
	    --quantize-multiplier 1.5 --quantize-weeks 6 \
	    --distill-multiplier 2.5 --distill-weeks 16

clean:
	rm -rf tools/__pycache__ __pycache__
