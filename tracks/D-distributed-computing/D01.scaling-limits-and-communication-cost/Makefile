# D01 - Scaling limits & the cost of talking. Python-first: there is no GPU code here.
#
# All three tools are analytic models on the standard library only, so this module
# builds nothing and needs no toolchain beyond Python 3.8+. Every number they print
# is a model, never a measurement, and each tool says so in its own banner.
#
# Quick start:
#   make help            # this list
#   make run             # run all three tools on the module's worked scenarios
#   make run-exercises   # run the four drill harnesses in exercises/
#   make clean           # remove __pycache__
#
# On Windows/PowerShell the interpreter is usually `python`, not `python3`:
#   make run PYTHON=python
# or just invoke the tools directly: python tools/scaling_model.py --help

PYTHON ?= python3
TOOLS  := tools/scaling_model.py tools/message_cost.py tools/collective_choice.py
DRILLS := exercises/01_efficiency_bisection.py exercises/02_crossover_finder.py
DRILLS += exercises/03_algorithm_selection.py exercises/04_straggler_simulation.py

.DEFAULT_GOAL := help
.PHONY: help run run-scaling run-message run-choice run-exercises smoke clean

help:
	@echo "Targets:"
	@echo "  make run             - run all three analytic tools (module scenarios)"
	@echo "  make run-scaling     - Amdahl/Gustafson curves and the efficiency wall"
	@echo "  make run-message     - alpha-beta sweep and the crossover m*"
	@echo "  make run-choice      - recursive doubling vs ring, and the winner"
	@echo "  make run-exercises   - run the four drill harnesses in exercises/"
	@echo "  make smoke           - assert the tools are standard-library only"
	@echo "  make clean           - remove __pycache__"
	@echo ""
	@echo "Interpreter: $(PYTHON) (override with PYTHON=python on Windows)"
	@echo "Every number these tools print is an analytic model, never a measurement."

run: run-scaling run-message run-choice

run-scaling:
	@echo "== scaling_model: 2% serial, and how little of 1024 ranks it can use =="
	$(PYTHON) tools/scaling_model.py --serial-fraction 0.02 --ranks 1024
	@echo ""
	@echo "== scaling_model: even 99.9% parallel is under 50% efficient at 1024 =="
	$(PYTHON) tools/scaling_model.py --serial-fraction 0.001 --ranks 1024

run-message:
	@echo "== message_cost: where the crossover sits on a 2.5 us / 25 GB/s link =="
	$(PYTHON) tools/message_cost.py --alpha-us 2.5 --bandwidth-gbs 25 --bytes 4096
	@echo ""
	@echo "== message_cost: a 1 GbE-class link, where m* moves DOWN, not up =="
	$(PYTHON) tools/message_cost.py --alpha-us 30 --bandwidth-gbs 0.125 --bytes 65536

run-choice:
	@echo "== collective_choice: 4 KiB over 64 ranks, latency decides =="
	$(PYTHON) tools/collective_choice.py --ranks 64 --bytes 4096
	@echo ""
	@echo "== collective_choice: 1 GiB over 64 ranks, the winner flips =="
	$(PYTHON) tools/collective_choice.py --ranks 64 --bytes 1073741824
	@echo ""
	@echo "== collective_choice: same 4 KiB at 512 ranks, the tie point moves =="
	$(PYTHON) tools/collective_choice.py --ranks 512 --bytes 4096

run-exercises:
	@for d in $(DRILLS); do \
	  echo "== $$d =="; \
	  $(PYTHON) $$d || exit 1; \
	  echo ""; \
	done
	@echo "Fill in the TODOs, then compare against solutions/README.md."

# The point of these tools is that they run anywhere, so guard that property in CI:
# any import outside the allow-list would break a laptop with nothing installed.
STDLIB_CHECK := import ast,sys;t=ast.parse(open(sys.argv[1]).read());\
m={a.name.split('.')[0] for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}\
|{(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n,ast.ImportFrom)};\
bad=m-{'argparse','json','math','random','statistics','sys','__future__'};\
sys.exit('non-stdlib import in %s: %s'%(sys.argv[1],bad)) if bad else print('%s ok'%sys.argv[1])

smoke:
	@for t in $(TOOLS) $(DRILLS); do \
	  $(PYTHON) -c "$(STDLIB_CHECK)" $$t || exit 1; \
	done
	@for t in $(TOOLS); do \
	  $(PYTHON) $$t --help > /dev/null || exit 1; \
	done
	@echo "OK: tools and drills are standard-library only, and every tool has --help"

clean:
	rm -rf tools/__pycache__ exercises/__pycache__ __pycache__
