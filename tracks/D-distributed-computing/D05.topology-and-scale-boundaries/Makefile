# D05 - Topology & the scale boundary. Python-first: there is no GPU code here.
#
# All four tools are analytic models on the standard library only, so this module
# builds nothing and needs no toolchain beyond Python 3.8+. Every number they print is
# a model, never a measurement, and each tool says so in its own banner.
#
# Quick start:
#   make help            # this list
#   make run             # run all four tools on the module's worked scenarios
#   make run-exercises   # run the four drill harnesses in exercises/
#   make run-solutions   # run the same four, filled in
#   make clean           # remove __pycache__
#
# On Windows/PowerShell the interpreter is usually `python`, not `python3`:
#   make run PYTHON=python
# or just invoke the tools directly: python tools/topology_metrics.py --help

PYTHON ?= python3
TOOLS  := tools/topology_metrics.py tools/collective_hierarchy.py
TOOLS  += tools/oversubscription.py tools/rank_placement.py
DRILLS := exercises/01_topology_metrics.py exercises/02_hierarchical_crossover.py
DRILLS += exercises/03_oversubscription_check.py exercises/04_rank_placement_cost.py
WORKED := solutions/01_topology_metrics.py solutions/02_hierarchical_crossover.py
WORKED += solutions/03_oversubscription_check.py solutions/04_rank_placement_cost.py

.DEFAULT_GOAL := help
.PHONY: help run run-topology run-hierarchy run-oversubscription run-placement
.PHONY: run-exercises run-solutions smoke clean

help:
	@echo "Targets:"
	@echo "  make run                 - run all four analytic tools (module scenarios)"
	@echo "  make run-topology        - diameter, bisection and the all-to-all floor"
	@echo "  make run-hierarchy       - flat ring vs two-level, and the crossover"
	@echo "  make run-oversubscription- the concurrency cliff and implied bisection"
	@echo "  make run-placement       - cross-node and cross-rail cost of a mapping"
	@echo "  make run-exercises       - run the four drill harnesses in exercises/"
	@echo "  make run-solutions       - run the four worked solutions"
	@echo "  make smoke               - assert the tools are standard-library only"
	@echo "  make clean               - remove __pycache__"
	@echo ""
	@echo "Interpreter: $(PYTHON) (override with PYTHON=python on Windows)"
	@echo "Every number these tools print is an analytic model, never a measurement."

run: run-topology run-hierarchy run-oversubscription run-placement

run-topology:
	@echo "== topology_metrics: 4096 endpoints, every shape in one table =="
	$(PYTHON) tools/topology_metrics.py --compare --nodes 4096
	@echo ""
	@echo "== topology_metrics: 2:1 leaves keep the diameter and halve the bisection =="
	$(PYTHON) tools/topology_metrics.py --topology fat-tree --nodes 4096 --radix 64 --oversubscription 2
	@echo ""
	@echo "== topology_metrics: 3D vs 6D torus, the Tofu argument in two commands =="
	$(PYTHON) tools/topology_metrics.py --topology torus --nodes 4096 --dims 3
	$(PYTHON) tools/topology_metrics.py --topology torus --nodes 4096 --dims 6

run-hierarchy:
	@echo "== collective_hierarchy: 1 GiB, and the G-fold slow-tier difference =="
	$(PYTHON) tools/collective_hierarchy.py --gpus-per-node 8 --bytes 1073741824 \
	    --intra-gbs 100 --inter-gbs 25 --alpha-intra-us 1 --alpha-inter-us 3 --max-nodes 512
	@echo ""
	@echo "== collective_hierarchy: 4 KiB, where step count decides instead of bytes =="
	$(PYTHON) tools/collective_hierarchy.py --gpus-per-node 8 --bytes 4096 \
	    --intra-gbs 100 --inter-gbs 25 --alpha-intra-us 1 --alpha-inter-us 3 --max-nodes 512
	@echo ""
	@echo "== collective_hierarchy: the crossover, and what a phase overhead does to it =="
	$(PYTHON) tools/collective_hierarchy.py --find-crossover --gpus-per-node 8 \
	    --bytes 1073741824 --intra-gbs 100 --inter-gbs 25
	$(PYTHON) tools/collective_hierarchy.py --find-crossover --gpus-per-node 8 \
	    --bytes 4096 --intra-gbs 100 --inter-gbs 25 --phase-overhead-us 20000

run-oversubscription:
	@echo "== oversubscription: section 2.3's leaf, and the cliff at 17 senders =="
	$(PYTHON) tools/oversubscription.py --hosts-per-leaf 32 --uplinks 16 \
	    --host-gbs 50 --uplink-gbs 50
	@echo ""
	@echo "== oversubscription: 75% locality tolerates a much worse ratio =="
	$(PYTHON) tools/oversubscription.py --hosts-per-leaf 32 --uplinks 16 \
	    --host-gbs 50 --uplink-gbs 50 --local-fraction 0.75
	@echo ""
	@echo "== oversubscription: half the uplinks, and the cliff moves to 8 =="
	$(PYTHON) tools/oversubscription.py --hosts-per-leaf 32 --uplinks 8 \
	    --host-gbs 50 --uplink-gbs 50 --sweep-senders

run-placement:
	@echo "== rank_placement: block, cyclic and random over 64 nodes x 8 GPUs =="
	$(PYTHON) tools/rank_placement.py --nodes 64 --gpus-per-node 8 --mapping block
	@echo ""
	$(PYTHON) tools/rank_placement.py --nodes 64 --gpus-per-node 8 --mapping cyclic
	@echo ""
	$(PYTHON) tools/rank_placement.py --nodes 64 --gpus-per-node 8 --mapping random --seed 7
	@echo ""
	@echo "== rank_placement: all three side by side, on a rail-optimized fabric =="
	$(PYTHON) tools/rank_placement.py --nodes 64 --gpus-per-node 8 --compare-all \
	    --seed 7 --bytes 1073741824 --rails 8

run-exercises:
	@for d in $(DRILLS); do \
	  echo "== $$d =="; \
	  $(PYTHON) $$d || exit 1; \
	  echo ""; \
	done
	@echo "Fill in the TODOs, then compare against solutions/README.md."

run-solutions:
	@for s in $(WORKED); do \
	  echo "== $$s =="; \
	  $(PYTHON) $$s || exit 1; \
	  echo ""; \
	done
	@echo "Commentary for all four: solutions/README.md"

# The point of these tools is that they run anywhere, so guard that property in CI:
# any import outside the allow-list would break a laptop with nothing installed.
STDLIB_CHECK := import ast,sys;t=ast.parse(open(sys.argv[1]).read());\
m={a.name.split('.')[0] for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}\
|{(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n,ast.ImportFrom)};\
bad=m-{'argparse','json','math','random','sys','__future__'};\
sys.exit('non-stdlib import in %s: %s'%(sys.argv[1],bad)) if bad else print('%s ok'%sys.argv[1])

smoke:
	@for t in $(TOOLS) $(DRILLS) $(WORKED); do \
	  $(PYTHON) -c "$(STDLIB_CHECK)" $$t || exit 1; \
	done
	@for t in $(TOOLS); do \
	  $(PYTHON) $$t --help > /dev/null || exit 1; \
	done
	@echo "OK: tools, drills and solutions are standard-library only, and every tool has --help"

clean:
	rm -rf tools/__pycache__ exercises/__pycache__ solutions/__pycache__ __pycache__
