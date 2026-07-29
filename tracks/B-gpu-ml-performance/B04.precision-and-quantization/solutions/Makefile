# B04 solutions - reference answers.
#
# Quick start:
#   make            # list targets
#   make all        # syntax-check the Python solution
#   make run        # run it (must complete cleanly - a failure here is a curriculum bug)
#   make clean
#
# As in ../exercises, there is no hipcc/nvcc machinery: the drills are numeric
# reasoning. The module's GPU code is built from the Makefile two levels up.

PYTHON   ?= python3
SOLUTION := 03_quantize_error_solution.py

.PHONY: help all run clean

help:
	@echo "B04 solutions:"
	@echo "  make all              - syntax-check $(SOLUTION)"
	@echo "  make run              - run $(SOLUTION) (must succeed)"
	@echo "  make clean            - remove __pycache__"
	@echo "Written answers: 01_format_selection_solution.md, 02_loss_scaling_solution.md"
	@echo "Python: $(PYTHON) (override with PYTHON=python3.12)"

all:
	$(PYTHON) -m py_compile $(SOLUTION)

run:
	$(PYTHON) $(SOLUTION)

clean:
	rm -rf __pycache__
