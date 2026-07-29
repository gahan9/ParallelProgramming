# A10 exercises - build the drills you fill in yourself (CUDA / NVIDIA only).
#
# Quick start:
#   make                       # list targets
#   make all   SM_ARCH=sm_80   # build every coding exercise
#   make run                   # build + run all (FAIL is expected until solved)
#   make 01_wmma_gemm.exe      # build just one
#   make clean
#
# This module is NVIDIA-only (see ../README.md section 1), so the drills are .cu built with
# nvcc and include ../cuda/check.cuh for CUDA_CHECK. They compile as-is and FAIL until you
# complete the TODOs - that gap is the lesson.
#
# Exercises 02 and 03 are written drills (.md), so there is nothing to build for them.

include ../../common.mk

CUDA_FLAGS := $(CUDA_BASE_FLAGS) -I../cuda

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=.exe)

.PHONY: help all run clean

help:
	@echo "A10 exercises (SM_ARCH=$(SM_ARCH)):"
	@echo "  make all              - build: $(BINS)"
	@echo "  make run              - build + run all (FAIL expected until solved)"
	@echo "  make <name>.exe       - build one exercise"
	@echo "  make clean            - remove built binaries"
	@echo "Override the arch:  make all SM_ARCH=sm_90"
	@echo "Written drills (no build): 02_layout_algebra.md, 03_library_choice.md"

all: $(BINS)

%.exe: %.cu ../cuda/check.cuh
	$(NVCC) $(CUDA_FLAGS) $< -o $@

# Keep going when an exercise fails: an unsolved TODO is expected to FAIL, and stopping at
# the first one would hide the rest.
run: all
	@for b in $(BINS); do \
		echo "== $$b =="; \
		./$$b || echo "   (non-zero exit - expected until the TODOs are done)"; \
		echo; \
	done

clean:
	rm -f $(BINS)
