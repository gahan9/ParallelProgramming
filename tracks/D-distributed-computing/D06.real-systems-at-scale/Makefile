# D06 - Real systems at scale. Analysis capstone: there is no GPU code in this module.
#
# The one tool is an analytic model on the standard library only, so this module
# builds nothing and needs no toolchain beyond Python 3.8+. Everything it prints is
# a MODEL, never a measurement -- the tool says so in its own banner, because it
# predicts behaviour at scales nobody working through this module can measure.
#
# Quick start:
#   make help            # this list
#   make run             # the three built-in presets
#   make run-explain     # the same three with their assumptions spelled out
#   make run-exercises   # each drill's starting command
#   make clean           # remove __pycache__
#
# On Windows/PowerShell the interpreter is usually `python`, not `python3`:
#   make run PYTHON=python
# or invoke the tool directly: python tools/scale_boundary.py --help

PYTHON ?= python3
TOOL   := tools/scale_boundary.py

.PHONY: help run run-dp run-hpc run-llm run-explain run-exercises smoke clean

help:
	@echo "Targets:"
	@echo "  make run             - the three built-in presets through the calculator"
	@echo "  make run-dp          - dp-training: data-parallel gradient all-reduce"
	@echo "  make run-hpc         - hpc-halo:    structured-mesh HPC step"
	@echo "  make run-llm         - llm-16k:     naive flat DP at the 16k scale"
	@echo "  make run-explain     - all three with --explain (states every assumption)"
	@echo "  make run-exercises   - the starting command for each of the four drills"
	@echo "  make smoke           - assert the tool is standard-library only"
	@echo "  make clean           - remove __pycache__"
	@echo ""
	@echo "Interpreter: $(PYTHON) (override with PYTHON=python on Windows)"
	@echo "Every number this tool prints is an analytic model, never a measurement."

run: run-dp run-hpc run-llm

run-dp:
	@echo "== dp-training at 4,096 ranks: compute, then bandwidth, then failure =="
	$(PYTHON) $(TOOL) --preset dp-training --ranks 4096

run-hpc:
	@echo "== hpc-halo at 65,536 ranks: the whole ladder in one table =="
	$(PYTHON) $(TOOL) --preset hpc-halo --ranks 65536

run-llm:
	@echo "== llm-16k at 16,384 ranks: why nobody runs flat data parallelism here =="
	$(PYTHON) $(TOOL) --preset llm-16k --ranks 16384

run-explain:
	@echo "== the same three presets, with every assumption stated =="
	$(PYTHON) $(TOOL) --preset dp-training --ranks 4096 --explain
	@echo ""
	$(PYTHON) $(TOOL) --preset hpc-halo --ranks 65536 --explain
	@echo ""
	$(PYTHON) $(TOOL) --preset llm-16k --ranks 16384 --explain

run-exercises:
	@echo "== drill 1 (scale_boundary.py): the reference the starter must reproduce =="
	$(PYTHON) $(TOOL) --preset dp-training --ranks 4096
	@echo ""
	@echo "== drill 3 (diagnose_the_system.md): check your reasoning against these =="
	$(PYTHON) $(TOOL) --preset hpc-halo --ranks 65536 --explain
	$(PYTHON) $(TOOL) --preset llm-16k --ranks 16384 --explain
	@echo ""
	@echo "== drill 4 (architecture_review.md): the proposal's own numbers =="
	$(PYTHON) $(TOOL) --ranks 4096 --model-bytes 1600000000 --compute-ms 90000 \
	    --serial-fraction 0.002 --alpha-us 4 --bandwidth-gbs 20 --gpus-per-node 8 \
	    --node-mtbf-hours 35000 --job-hours 1008
	@echo ""
	@echo "Drill 2 (read_a_list_entry.md) needs no tool: it needs the current TOP500 edition."
	@echo "Write your answers against exercises/*, then read solutions/."

# The point of this tool is that it runs anywhere, so guard that property in CI:
# any import outside the allow-list would break a laptop with nothing installed.
STDLIB_CHECK := import ast,sys;t=ast.parse(open(sys.argv[1]).read());\
m={a.name.split('.')[0] for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}\
|{(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n,ast.ImportFrom)};\
bad=m-{'argparse','json','math','sys','textwrap','__future__'};\
sys.exit('non-stdlib import in %s: %s'%(sys.argv[1],bad)) if bad else print('%s ok'%sys.argv[1])

smoke:
	@$(PYTHON) -c "$(STDLIB_CHECK)" $(TOOL) || exit 1
	@$(PYTHON) -S -I $(TOOL) --help > /dev/null || exit 1
	@echo "OK: the tool is standard-library only and runs under python -S -I"

clean:
	rm -rf tools/__pycache__ exercises/__pycache__ solutions/__pycache__ __pycache__
