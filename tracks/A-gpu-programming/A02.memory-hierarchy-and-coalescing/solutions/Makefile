# A02 solutions - build the reference answers (HIP / AMD).
#
# Quick start:
#   make                       # list targets
#   make all   GPU_ARCH=gfx950 # build every solution
#   make run                   # build + run all (every one must PASS)
#   make clean
#
# Unlike exercises/, `make run` here stops on the first failure: a solution that
# does not PASS is a real bug in the curriculum, not a learning exercise.

include ../../common.mk

HIP_FLAGS := $(HIP_BASE_FLAGS) -I../hip

SRCS := $(wildcard *.cpp)
BINS := $(SRCS:.cpp=.exe)

.PHONY: help all run clean

help:
	@echo "A02 solutions (GPU_ARCH=$(GPU_ARCH)):"
	@echo "  make all              - build: $(BINS)"
	@echo "  make run              - build + run all (must all PASS)"
	@echo "  make <name>.exe       - build one solution"
	@echo "  make clean            - remove built binaries"
	@echo "Override the arch:  make all GPU_ARCH=gfx950"
	@echo "ROCm: $(HIP_TOOLCHAIN_INFO)"

all: $(BINS)

%.exe: %.cpp ../hip/check.h
	$(HIPCC) $(HIP_FLAGS) $< -o $@

run: all
	@set -e; for b in $(BINS); do \
		echo "== $$b =="; \
		./$$b; \
		echo; \
	done

clean:
	rm -f $(BINS)
