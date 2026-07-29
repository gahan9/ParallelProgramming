# C05 - Deployment & serving architecture.
#
# Track C is a DESIGN track: there is no hip/ or cuda/ code here. The runnable
# artefacts are three pure-standard-library Python tools that make the
# module's tradeoffs measurable, plus paper design drills in exercises/.
#
# Quick start:
#   make help              # this list
#   make run               # run all three tools with their demo settings
#   make run-exercises     # list the drills and check every one has a solution
#   make clean             # remove __pycache__ and any generated output
#
# Windows note: this repo's Makefiles assume GNU make and `python3`. In
# PowerShell either run under WSL/Git Bash, or override the interpreter:
#   make run PYTHON=python
# You can also just run any tool directly: python tools/ab_calculator.py -h

PYTHON ?= python3

TOOLS  := tools/ab_calculator.py tools/capacity_autoscale.py tools/degradation_planner.py
DRILLS := 01_design_the_serving_stack 02_rollout_plan 03_ab_test_critique

.PHONY: help run run-tools run-ab run-capacity run-degrade \
        run-exercises run-solutions check clean

help:
	@echo "C05 - Deployment & serving architecture (design module, no GPU code)"
	@echo ""
	@echo "  make run            - run all three analysis tools (demo settings)"
	@echo "  make run-ab         - A/B sample size, significance, peeking simulation"
	@echo "  make run-capacity   - autoscaling / cold-start / warm-pool simulation"
	@echo "  make run-degrade    - availability with and without graceful degradation"
	@echo "  make run-exercises  - list the design drills, verify each has a solution"
	@echo "  make check          - byte-compile the tools and prove they are stdlib-only"
	@echo "  make clean          - remove __pycache__ and generated files"
	@echo ""
	@echo "  interpreter: PYTHON=$(PYTHON)   (override with 'make run PYTHON=python')"
	@echo "  every tool also takes -h for its own options"

run: run-tools

run-tools:
	@for t in $(TOOLS); do \
		echo "================ $$t ================"; \
		$(PYTHON) $$t || exit 1; \
		echo; \
	done

run-ab:
	$(PYTHON) tools/ab_calculator.py

run-capacity:
	$(PYTHON) tools/capacity_autoscale.py

run-degrade:
	$(PYTHON) tools/degradation_planner.py

# The drills are written design documents, not programs. "Running" them means
# confirming the set is complete: a drill with no reference solution is a
# broken exercise, and that is worth catching in the build rather than by a
# learner hitting a dead link.
run-exercises:
	@echo "C05 design drills (write your answer, then compare with solutions/):"
	@echo ""
	@missing=0; \
	for d in $(DRILLS); do \
		if [ ! -f "exercises/$$d.md" ]; then \
			echo "  MISSING exercises/$$d.md"; missing=1; \
		elif [ ! -f "solutions/$${d}_solution.md" ]; then \
			echo "  MISSING solutions/$${d}_solution.md"; missing=1; \
		else \
			echo "  ok  exercises/$$d.md  ->  solutions/$${d}_solution.md"; \
		fi; \
	done; \
	echo ""; \
	if [ $$missing -ne 0 ]; then echo "FAILED: drill set incomplete"; exit 1; fi; \
	echo "PASSED: all $(words $(DRILLS)) drills have reference solutions."; \
	echo "Read exercises/README.md for how each one is graded."

run-solutions: run-exercises

# `python -S` starts without site-packages, so any third-party import fails
# immediately. If these three still run under -S, they are stdlib-only.
check:
	@$(PYTHON) -m compileall -q $(TOOLS) && echo "compile      : ok"
	@for t in $(TOOLS); do \
		$(PYTHON) -S $$t -h > /dev/null || { echo "FAILED under -S: $$t"; exit 1; }; \
	done
	@echo "stdlib-only  : ok (all tools start under 'python -S', no site-packages)"

clean:
	rm -rf tools/__pycache__ __pycache__
	rm -f tools/*.pyc *.pyc
