# B04 exercises - two paper drills and one Python drill.
#
# Quick start:
#   make            # list targets
#   make all        # syntax-check the Python drill
#   make run        # run it (rows print "not implemented" until you fill the TODOs)
#   make clean
#
# There is deliberately no hipcc/nvcc machinery here and no include of Track A's
# common.mk: this module's drills are numeric reasoning, not kernel writing, so a
# toolchain definition would be dead weight. The GPU code for B04 lives in ../hip and
# ../cuda and is built from the module Makefile one level up.

PYTHON ?= python3
DRILL  := 03_quantize_error.py

.PHONY: help all run clean

help:
	@echo "B04 exercises:"
	@echo "  make all              - syntax-check $(DRILL)"
	@echo "  make run              - run $(DRILL) (TODO rows expected until solved)"
	@echo "  make clean            - remove __pycache__"
	@echo "Paper drills (no build): 01_format_selection.md, 02_loss_scaling.md"
	@echo "Python: $(PYTHON) (override with PYTHON=python3.12)"

all:
	$(PYTHON) -m py_compile $(DRILL)

run:
	$(PYTHON) $(DRILL)

clean:
	rm -rf __pycache__
