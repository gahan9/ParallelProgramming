# B02 exercises - build the drills you fill in yourself (HIP / AMD).
#
# Quick start:
#   make                            # list targets
#   make all   GPU_ARCH=gfx950      # build every exercise
#   make run                        # build + run all (FAIL is expected until solved)
#   make 03_place_your_kernel.exe   # build just one
#   make clean
#
# Exercises 01 and 02 are paper drills (.md) - nothing to build. Exercise 03 is
# HIP-only and includes ../hip/check.h for HIP_CHECK. It compiles as-is and FAILS
# until the TODOs are filled in; that gap is the lesson.

# Path: exercises/ -> module -> B-gpu-ml-performance -> tracks -> A-gpu-programming.
include ../../../A-gpu-programming/common.mk

HIP_FLAGS := $(HIP_BASE_FLAGS) -I../hip

SRCS := $(wildcard *.cpp)
BINS := $(SRCS:.cpp=.exe)

.PHONY: help all run clean

help:
	@echo "B02 exercises (GPU_ARCH=$(GPU_ARCH)):"
	@echo "  make all              - build: $(BINS)"
	@echo "  make run              - build + run all (FAIL expected until solved)"
	@echo "  make <name>.exe       - build one exercise"
	@echo "  make clean            - remove built binaries"
	@echo "Paper drills (no build): 01_classify_kernels.md, 02_ai_of_attention.md"
	@echo "Override the arch:  make all GPU_ARCH=gfx950"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: $(BINS)

%.exe: %.cpp ../hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

# Keep going when an exercise fails: an unsolved TODO is expected to FAIL, and
# stopping at the first one would hide the rest.
run: all
	@for b in $(BINS); do \
		echo "== $$b =="; \
		./$$b || echo "   (non-zero exit - expected until the TODOs are done)"; \
		echo; \
	done

clean:
	rm -f $(BINS)
