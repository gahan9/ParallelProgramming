# C02 - Data pipelines & features. Design module: no GPU code, only stdlib Python tools.
#
# Quick start:
#   make help            # list targets
#   make run             # run every tool in demo mode
#   make run-exercises   # run the scenarios the exercises ask you to reason about
#   make clean           # remove Python caches and generated demo files
#
# On Windows the interpreter is usually `python`, not `python3`:
#   make run PYTHON=python

PYTHON   ?= python3
DEMO_DIR ?= _demo

.PHONY: help run run-validator run-leakage run-splits run-exercises demo-files clean

help:
	@echo "Targets:"
	@echo "  make run             - run all three tools in demo mode (no data files needed)"
	@echo "  make run-validator   - data_validator: schema + drift gate on a bad batch"
	@echo "  make run-leakage     - leakage_detector: three planted leaks, three checks"
	@echo "  make run-splits      - split_demo: measured optimism from the wrong split"
	@echo "  make run-exercises   - the scenarios exercises/01 and exercises/03 reference"
	@echo "  make demo-files      - write editable sample CSV/JSON into $(DEMO_DIR)/"
	@echo "  make clean           - remove __pycache__ and $(DEMO_DIR)/"
	@echo "Interpreter: $(PYTHON)  (override with 'make run PYTHON=python')"
	@echo "Tools are standard-library only; no pip install is required."
	@echo "Note: data_validator and leakage_detector exit 1 when they find violations."

# The gates exit non-zero by design, so absorb that here and let the report speak.
run: run-validator run-leakage run-splits

run-validator:
	@echo "== data_validator: a batch that should be rejected =="
	-$(PYTHON) tools/data_validator.py --demo

run-leakage:
	@echo "== leakage_detector: three planted leaks, each caught by a different check =="
	-$(PYTHON) tools/leakage_detector.py --demo

run-splits:
	@echo "== split_demo: random vs temporal vs grouped on identical data =="
	$(PYTHON) tools/split_demo.py

# Exercise 01 asks which features leak; exercise 03 asks you to rank causes of an
# offline/online gap. These runs give you the evidence for both.
run-exercises:
	@echo "== exercises/01: audit the feature table for leakage =="
	-$(PYTHON) tools/leakage_detector.py --demo
	@echo
	@echo "== exercises/03: the split artifact that inflates offline AUC =="
	$(PYTHON) tools/split_demo.py --verbose
	@echo
	@echo "== exercises/03: the validation gate that would have caught the pipeline change =="
	-$(PYTHON) tools/data_validator.py --demo

demo-files:
	$(PYTHON) tools/data_validator.py --write-demo $(DEMO_DIR)
	$(PYTHON) tools/leakage_detector.py --write-demo $(DEMO_DIR)

clean:
	rm -rf tools/__pycache__ __pycache__ $(DEMO_DIR)
