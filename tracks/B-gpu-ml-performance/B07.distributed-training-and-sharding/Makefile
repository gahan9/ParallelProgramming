# B07 - Distributed training & sharding. Python-first: there is no GPU code here.
#
# All three tools are analytic models on the standard library only, so this
# module builds nothing and needs no toolchain beyond Python 3.8+.
#
# Quick start:
#   make help            # this list
#   make run             # run all three tools on the module's worked examples
#   make run-exercises   # print each drill's inputs and its starting command
#   make clean           # remove __pycache__
#
# On Windows/PowerShell the interpreter is usually `python`, not `python3`:
#   make run PYTHON=python
# or just invoke the tools directly: python tools/memory_planner.py --help

PYTHON ?= python3
TOOLS  := tools/memory_planner.py tools/collective_cost.py tools/pipeline_bubble.py

.PHONY: help run run-memory run-collective run-pipeline run-exercises smoke clean

help:
	@echo "Targets:"
	@echo "  make run             - run all three analytic tools (worked examples)"
	@echo "  make run-memory      - per-GPU memory for DDP / ZeRO / TP / PP"
	@echo "  make run-collective  - ring collective cost + bucketing analysis"
	@echo "  make run-pipeline    - pipeline bubble fraction and efficiency"
	@echo "  make run-exercises   - inputs and starting commands for the drills"
	@echo "  make smoke           - assert the tools are standard-library only"
	@echo "  make clean           - remove __pycache__"
	@echo ""
	@echo "Interpreter: $(PYTHON) (override with PYTHON=python on Windows)"
	@echo "Every number these tools print is an analytic model, never a measurement."

run: run-memory run-collective run-pipeline

run-memory:
	@echo "== memory_planner: 70B on 8 GPUs, nothing fits =="
	$(PYTHON) tools/memory_planner.py --preset llama2-70b --gpus 8
	@echo ""
	@echo "== memory_planner: same model on 64 GPUs, ranked fits =="
	$(PYTHON) tools/memory_planner.py --preset llama2-70b --gpus 64 --suggest --top 6

run-collective:
	@echo "== collective_cost: where a ring all-reduce stops being latency-bound =="
	$(PYTHON) tools/collective_cost.py --gpus 8 --bw 50 --latency 8
	@echo ""
	@echo "== collective_cost: 7B DDP gradient all-reduce + bucketing =="
	$(PYTHON) tools/collective_cost.py --gpus 16 --bw 25 --latency 15 --params 7 --step-ms 900

run-pipeline:
	@echo "== pipeline_bubble: 8 stages, 8 micro-batches, and the sweep =="
	$(PYTHON) tools/pipeline_bubble.py --stages 8 --micro-batches 8 --sweep

run-exercises:
	@echo "== 01_fit_the_model: 175B-class model, 8 GPUs x 192 GiB =="
	$(PYTHON) tools/memory_planner.py --preset gpt3-175b --gpus 8 --micro-batch 1
	@echo ""
	@echo "== 02_choose_parallelism: the four scenarios' memory shapes =="
	$(PYTHON) tools/memory_planner.py --preset llama3-8b --gpus 8 --seq 8192
	$(PYTHON) tools/memory_planner.py --preset llama3-8b --gpus 8 --seq 131072 --micro-batch 1
	@echo ""
	@echo "== 03_comm_budget: the link and the model under test =="
	$(PYTHON) tools/collective_cost.py --gpus 32 --bw 25 --latency 15 --params 13 --step-ms 400
	@echo ""
	@echo "Write your answers against exercises/*.md; check solutions/ afterwards."

# The point of these tools is that they run anywhere, so guard that property in CI:
# any import outside the allow-list would break a laptop with no torch installed.
STDLIB_CHECK := import ast,sys;t=ast.parse(open(sys.argv[1]).read());\
m={a.name.split('.')[0] for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}\
|{(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n,ast.ImportFrom)};\
bad=m-{'argparse','json','math','__future__'};\
sys.exit('non-stdlib import in %s: %s'%(sys.argv[1],bad)) if bad else print('%s ok'%sys.argv[1])

smoke:
	@for t in $(TOOLS); do \
	  $(PYTHON) -c "$(STDLIB_CHECK)" $$t || exit 1; \
	  $(PYTHON) $$t --help > /dev/null || exit 1; \
	done
	@echo "OK: all tools are standard-library only and expose --help"

clean:
	rm -rf tools/__pycache__ __pycache__
