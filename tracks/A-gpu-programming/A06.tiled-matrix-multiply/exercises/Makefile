# A06 exercises - build the drills you fill in yourself (HIP / AMD).
#
# Quick start:
#   make                       # list targets
#   make all   GPU_ARCH=gfx950 # build every exercise
#   make run                   # build + run all (FAIL is expected until solved)
#   make 01_tiled_matmul.exe   # build just one
#   make clean
#
# Exercises are HIP-only and include ../hip/check.h for HIP_CHECK.
# They compile as-is; they FAIL until you complete the TODOs. That gap is the
# lesson - fix it, rerun, and compare both correctness and GFLOP/s.

include ../../common.mk

HIP_FLAGS := $(HIP_BASE_FLAGS) -I../hip

SRCS := $(wildcard *.cpp)
BINS := $(SRCS:.cpp=.exe)

.PHONY: help all run clean

help:
	@echo "A06 exercises (GPU_ARCH=$(GPU_ARCH)):"
	@echo "  make all              - build: $(BINS)"
	@echo "  make run              - build + run all (FAIL expected until solved)"
	@echo "  make <name>.exe       - build one exercise"
	@echo "  make clean            - remove built binaries"
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
