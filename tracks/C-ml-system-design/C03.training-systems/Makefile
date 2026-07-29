# C03 - Training systems. Design module: no GPU code, only stdlib Python tools.
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

.PHONY: help run run-checkpoint run-throughput run-hpo run-exercises clean

help:
	@echo "Targets:"
	@echo "  make run             - run all three tools with their default scenarios"
	@echo "  make run-checkpoint  - checkpoint_planner: optimal interval vs MTBF and write cost"
	@echo "  make run-throughput  - throughput_budget: time-to-train and the data-starvation tax"
	@echo "  make run-hpo         - hpo_budget: grid vs random vs successive halving"
	@echo "  make run-exercises   - the scenarios exercises/01-03 ask you to analyse"
	@echo "  make clean           - remove __pycache__"
	@echo "Interpreter: $(PYTHON)  (override with 'make run PYTHON=python')"
	@echo "Tools are standard-library only; no pip install is required."

run: run-checkpoint run-throughput run-hpo

run-checkpoint:
	@echo "== checkpoint_planner: 64-GPU job, MTBF 6h, 180s checkpoint =="
	$(PYTHON) tools/checkpoint_planner.py

run-throughput:
	@echo "== throughput_budget: input pipeline delivering less than the GPUs can eat =="
	$(PYTHON) tools/throughput_budget.py
	@echo
	@echo "== throughput_budget: same job with prefetch disabled =="
	$(PYTHON) tools/throughput_budget.py --no-overlap

run-hpo:
	@echo "== hpo_budget: 2048 GPU-hours, three search strategies =="
	$(PYTHON) tools/hpo_budget.py

# The exercises state scenarios in prose; these are the same scenarios as tool
# invocations, so you can check a hand calculation against the model.
run-exercises:
	@echo "== exercise 01: is the loader or the step the bottleneck? =="
	$(PYTHON) tools/throughput_budget.py --gpus 128 --samples-per-second-per-gpu 95 \
	    --loader-samples-per-second 6800 --dataset-size 40000000 --epochs 3
	@echo
	@echo "== exercise 02, scenario A: on-demand cluster, MTBF 9h, 240s checkpoint =="
	$(PYTHON) tools/checkpoint_planner.py --mtbf-hours 9 --checkpoint-seconds 240 \
	    --restart-minutes 14 --step-seconds 0.9 --shards 16 --gpus 256 \
	    --run-hours 720 --gpu-hour-cost 2.40
	@echo
	@echo "== exercise 02, scenario B: same job on spot, MTBF 45min =="
	$(PYTHON) tools/checkpoint_planner.py --mtbf-hours 0.75 --checkpoint-seconds 240 \
	    --restart-minutes 14 --step-seconds 0.9 --shards 16 --gpus 256 \
	    --run-hours 720 --gpu-hour-cost 0.75
	@echo
	@echo "== exercise 02, part 4: the same spot job with a 4-minute restart =="
	$(PYTHON) tools/checkpoint_planner.py --mtbf-hours 0.75 --checkpoint-seconds 240 \
	    --restart-minutes 4 --step-seconds 0.9 --shards 16 --gpus 256 \
	    --run-hours 720 --gpu-hour-cost 0.75
	@echo
	@echo "== exercise 03 background: how much search does the same budget buy? =="
	$(PYTHON) tools/hpo_budget.py --budget-gpu-hours 4096 --full-run-gpu-hours 128 \
	    --dimensions 5

clean:
	rm -rf tools/__pycache__ __pycache__
