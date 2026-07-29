# A10 solutions - build the reference answers (CUDA / NVIDIA only).
#
# Quick start:
#   make                       # list targets
#   make all   SM_ARCH=sm_80   # build every solution
#   make run                   # build + run all (every one must PASS)
#   make clean
#
# Unlike exercises/, `make run` here stops on the first failure: a solution that does not
# PASS is a real bug in the curriculum, not a learning exercise.

include ../../common.mk

CUDA_FLAGS := $(CUDA_BASE_FLAGS) -I../cuda

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=.exe)

.PHONY: help all run clean

help:
	@echo "A10 solutions (SM_ARCH=$(SM_ARCH)):"
	@echo "  make all              - build: $(BINS)"
	@echo "  make run              - build + run all (must all PASS)"
	@echo "  make <name>.exe       - build one solution"
	@echo "  make clean            - remove built binaries"
	@echo "Override the arch:  make all SM_ARCH=sm_90"
	@echo "Written drills (no build): 02_layout_algebra_solution.md, 03_library_choice_solution.md"

all: $(BINS)

%.exe: %.cu ../cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

run: all
	@set -e; for b in $(BINS); do \
		echo "== $$b =="; \
		./$$b; \
		echo; \
	done

clean:
	rm -f $(BINS)
