# C01 - Problem framing & metrics. Design module: no GPU code, only stdlib Python tools.
#
# Quick start:
#   make help            # list targets
#   make run             # run every tool in demo mode
#   make run-exercises   # run the scenarios the exercises ask you to reason about
#   make clean           # remove Python caches
#
# On Windows the interpreter is usually `python`, not `python3`:
#   make run PYTHON=python

PYTHON ?= python3
TOOLS  := tools/metric_explorer.py tools/baseline_check.py

.PHONY: help run run-metrics run-baselines run-exercises clean

help:
	@echo "Targets:"
	@echo "  make run             - run both tools in demo mode (no data files needed)"
	@echo "  make run-metrics     - metric_explorer: confusion, sweep, imbalance"
	@echo "  make run-baselines   - baseline_check: what trivial policies already score"
	@echo "  make run-exercises   - the cost/base-rate scenarios from exercises/03"
	@echo "  make clean           - remove __pycache__"
	@echo "Interpreter: $(PYTHON)  (override with 'make run PYTHON=python')"
	@echo "Tools are standard-library only; no pip install is required."

run: run-metrics run-baselines

run-metrics:
	@echo "== metric_explorer: confusion matrix at 1% prevalence =="
	$(PYTHON) tools/metric_explorer.py confusion --tp 40 --fp 10 --fn 60 --tn 9890 \
	    --fp-cost 1 --fn-cost 20
	@echo
	@echo "== metric_explorer: threshold sweep with FN 20x more costly than FP =="
	$(PYTHON) tools/metric_explorer.py sweep --fp-cost 1 --fn-cost 20
	@echo
	@echo "== metric_explorer: ROC-AUC vs PR-AUC as prevalence falls =="
	$(PYTHON) tools/metric_explorer.py imbalance

run-baselines:
	@echo "== baseline_check: 2% prevalence =="
	$(PYTHON) tools/baseline_check.py --positives 200 --negatives 9800 --simulate 200

# Exercise 3 asks for the cost-optimal threshold at two base rates. These are the same
# scenarios, so you can check a hand calculation against the sweep.
run-exercises:
	@echo "== exercise 03, scenario A: base rate 2%, FN costs 20x FP =="
	$(PYTHON) tools/metric_explorer.py sweep --positives 200 --negatives 9800 \
	    --fp-cost 1 --fn-cost 20 --max-rows 12
	@echo
	@echo "== exercise 03, scenario B: base rate 0.2%, same costs =="
	$(PYTHON) tools/metric_explorer.py sweep --positives 200 --negatives 99800 \
	    --fp-cost 1 --fn-cost 20 --max-rows 12
	@echo
	@echo "== exercise 03, baselines for both base rates =="
	$(PYTHON) tools/baseline_check.py --positives 200 --negatives 9800
	$(PYTHON) tools/baseline_check.py --positives 200 --negatives 99800

clean:
	rm -rf tools/__pycache__ __pycache__
