# D03 - GPU collectives with NCCL and RCCL. Linux-targeted, dual-backend:
# AMD (hipcc + librccl) and NVIDIA (nvcc + libnccl).
#
# Labs 1, 3 and 4 need ONE node with at least two GPUs. No network, no scheduler,
# no MPI. Lab 2 is the vendor benchmark suite, cloned separately.
#
# Quick start:
#   make             # same as `make hip`: build the hip/ labs with hipcc
#   make cuda        # build the cuda/ labs with nvcc
#   make run         # lab 1, then the lab 3 sweep
#   make trap        # lab 4: the asynchrony trap, three invocations
#   make run-tools   # the Python tools (no GPU, no toolchain needed)
#   make exercises   # build the C++ drills in exercises/
#   make solutions   # build the C++ drills in solutions/
#   make smoke       # assert the tools are stdlib-only and expose --help
#   make clean
#   make help        # this list, plus what was detected on this machine
#
# Overrides:
#   make hip  GPU_ARCH=gfx942        # target one architecture instead of the default
#   make cuda SM_ARCH=sm_90
#   make hip  HIPCC=/opt/rocm/bin/hipcc
#   make run  MIN_BYTES=8 MAX_BYTES=1073741824 FACTOR=2 ITERS=20
#   make run-tools PYTHON=python     # Windows, where python3 is often absent
#
# On measured versus modelled: the labs MEASURE single-node collective bandwidth on
# whatever GPUs you have. tools/busbw.py and tools/parse_collective_report.py consume
# that measured data, so they report what they converted or parsed. Only
# tools/multinode_allreduce_model.py describes hardware this repository does not have,
# and it prints a banner saying its output is a model. Nothing here ships a benchmark
# number: every figure comes from your run.

HIPCC   ?= hipcc
NVCC    ?= nvcc
PYTHON  ?= python3

# Architecture is optional on purpose. With neither variable set, hipcc and nvcc build
# for their own defaults, which is what you want on an unknown machine; set them to cut
# build time or to cross-compile for a node you are not sitting on.
GPU_ARCH ?=
SM_ARCH  ?=
HIP_ARCH_FLAG  := $(if $(GPU_ARCH),--offload-arch=$(GPU_ARCH),)
CUDA_ARCH_FLAG := $(if $(SM_ARCH),-arch=$(SM_ARCH),)

HIP_FLAGS  ?= -O2 -Wall -Wextra $(HIP_ARCH_FLAG)
CUDA_FLAGS ?= -O2 -Xcompiler -Wall $(CUDA_ARCH_FLAG)
HIP_LIBS   ?= -lrccl
CUDA_LIBS  ?= -lnccl

# Sweep defaults match the README's lab 3 invocation.
MIN_BYTES ?= 8
MAX_BYTES ?= 1073741824
FACTOR    ?= 2
ITERS     ?= 20
BIG       ?= 268435456
SMALL     ?= 1024

HIP_BINS  := allreduce_min busbw_sweep stream_trap
CUDA_BINS := allreduce_min_cuda
EX_BINS   := exercises/ex2_allreduce_verify exercises/ex3_crossover_sweep
SOL_BINS  := solutions/ex2_allreduce_verify solutions/ex3_crossover_sweep
TOOLS     := tools/busbw.py tools/parse_collective_report.py \
             tools/multinode_allreduce_model.py

# `make` with no target builds the hip/ labs, as the README's lab section says.
.DEFAULT_GOAL := all

.PHONY: help all hip cuda run run-allreduce run-sweep trap run-tools exercises \
        solutions smoke clean require-hipcc require-nvcc require-python

help:
	@echo "D03 - GPU collectives (NCCL / RCCL). Targets:"
	@echo "  make            - build the hip/ labs with $(HIPCC) (default)"
	@echo "  make cuda       - build the cuda/ labs with $(NVCC)"
	@echo "  make run        - lab 1 (verified all-reduce), then the lab 3 sweep"
	@echo "  make trap       - lab 4: the asynchrony trap at two sizes, plus the fix"
	@echo "  make run-tools  - the Python tools; needs no GPU and no toolchain"
	@echo "  make exercises  - build the C++ drills in exercises/ (they fail on purpose)"
	@echo "  make solutions  - build the reference answers in solutions/"
	@echo "  make smoke      - assert the tools are stdlib-only and expose --help"
	@echo "  make clean      - remove binaries and __pycache__"
	@echo ""
	@echo "Detected on this machine:"
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
	  echo "  HIPCC   $$(command -v $(HIPCC))"; \
	else \
	  echo "  HIPCC   $(HIPCC) NOT FOUND"; \
	fi
	@if command -v $(NVCC) >/dev/null 2>&1; then \
	  echo "  NVCC    $$(command -v $(NVCC))"; \
	else \
	  echo "  NVCC    $(NVCC) NOT FOUND"; \
	fi
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
	  echo "  PYTHON  $$(command -v $(PYTHON))"; \
	else \
	  echo "  PYTHON  $(PYTHON) NOT FOUND - try 'make run-tools PYTHON=python'"; \
	fi
	@if ! command -v $(HIPCC) >/dev/null 2>&1 && ! command -v $(NVCC) >/dev/null 2>&1; then \
	  echo ""; \
	  echo "  Neither GPU toolchain is present, so no lab can be built here."; \
	  echo "  'make run-tools' and 'make smoke' still work: the tools are pure stdlib"; \
	  echo "  Python. Sections 1-3 and 5-9 of the README need no hardware either."; \
	fi
	@echo ""
	@echo "Exercises: exercises/README.md   Solutions: solutions/README.md"

# Guards. Without these a missing toolchain surfaces as a bare "hipcc: command not
# found" from inside a compile rule, which tells a newcomer nothing about the fix. The
# message also has to name the collective library, because ROCm and CUDA both install
# without it and the failure then moves from the compile to the link.
require-hipcc:
	@command -v $(HIPCC) >/dev/null 2>&1 || { \
	  echo "ERROR: '$(HIPCC)' is not on PATH, so the hip/ labs cannot be built."; \
	  echo ""; \
	  echo "  Needed: ROCm (for hipcc) AND RCCL (for -lrccl). RCCL is a separate package."; \
	  echo "    Debian / Ubuntu : sudo apt-get install rocm-hip-sdk rccl rccl-dev"; \
	  echo "    RHEL / Fedora   : sudo dnf install rocm-hip-devel rccl-devel"; \
	  echo "    installed elsewhere: make HIPCC=/opt/rocm/bin/hipcc"; \
	  echo "  On an NVIDIA machine build the other backend instead: make cuda"; \
	  echo ""; \
	  echo "  Nothing was built. 'make run-tools' needs no GPU at all."; \
	  exit 1; }

require-nvcc:
	@command -v $(NVCC) >/dev/null 2>&1 || { \
	  echo "ERROR: '$(NVCC)' is not on PATH, so the cuda/ labs cannot be built."; \
	  echo ""; \
	  echo "  Needed: the CUDA toolkit (for nvcc) AND NCCL (for -lnccl), separate packages."; \
	  echo "    https://developer.nvidia.com/cuda-downloads"; \
	  echo "    https://developer.nvidia.com/nccl"; \
	  echo "    installed elsewhere: make cuda NVCC=/usr/local/cuda/bin/nvcc"; \
	  echo "  On an AMD machine build the other backend instead: make"; \
	  echo ""; \
	  echo "  Nothing was built. 'make run-tools' needs no GPU at all."; \
	  exit 1; }

require-python:
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
	  echo "ERROR: '$(PYTHON)' is not on PATH."; \
	  echo "  The tools need only a stdlib Python 3.8+. On Windows try: make run-tools PYTHON=python"; \
	  exit 1; }

all: hip

hip: require-hipcc $(HIP_BINS)

allreduce_min: hip/allreduce_min.cpp
	$(HIPCC) $(HIP_FLAGS) $< $(HIP_LIBS) -o $@

busbw_sweep: hip/busbw_sweep.cpp
	$(HIPCC) $(HIP_FLAGS) $< $(HIP_LIBS) -o $@

stream_trap: hip/stream_trap.cpp
	$(HIPCC) $(HIP_FLAGS) $< $(HIP_LIBS) -o $@

cuda: require-nvcc $(CUDA_BINS)

# Named with a suffix so an AMD and an NVIDIA build can coexist in one tree. The source
# is the same program: only the runtime wrapper differs (README section 4, lab 1).
allreduce_min_cuda: cuda/allreduce_min.cu
	$(NVCC) $(CUDA_FLAGS) $< $(CUDA_LIBS) -o $@

run: run-allreduce run-sweep

run-allreduce: require-hipcc allreduce_min
	@echo "== Lab 1: verified all-reduce across every visible GPU =="
	@echo "   Needs 2+ GPUs. Every runtime and library call is checked, and the result is"
	@echo "   compared against the closed-form CPU reference N(N+1)/2 with a tolerance."
	./allreduce_min
	@echo ""
	@echo "   Now read the library's own decisions, then reproduce the deadlock:"
	@echo "     NCCL_DEBUG=INFO ./allreduce_min        # rings, channels, algo, protocol"
	@echo "     (delete the ncclGroupStart/End pair and re-run: it hangs)"

run-sweep: require-hipcc busbw_sweep
	@echo ""
	@echo "== Lab 3: bandwidth sweep, $(MIN_BYTES) to $(MAX_BYTES) bytes, x$(FACTOR), $(ITERS) iters =="
	@echo "   These are YOUR numbers, measured on YOUR GPUs and links. Read the busbw"
	@echo "   column for the three regions of README 5.2, and take two things from it:"
	@echo "   the plateau (your effective fabric bandwidth) and the knee (your bucket size)."
	./busbw_sweep --min-bytes $(MIN_BYTES) --max-bytes $(MAX_BYTES) \
	  --factor $(FACTOR) --iters $(ITERS)
	@echo ""
	@echo "   Then force the protocol and watch the small-message end get worse:"
	@echo "     NCCL_PROTO=Simple ./busbw_sweep --min-bytes $(MIN_BYTES) --max-bytes 1048576"

# Lab 4. The first invocation is SUPPOSED to fail, so its line is prefixed with `-`: a
# nonzero exit here is the result, not a build error.
trap: require-hipcc stream_trap
	@echo "== Lab 4: reading a buffer before the collective has completed =="
	@echo ""
	@echo "-- no sync, $(BIG) elements: expect FAIL"
	-./stream_trap --no-sync --count $(BIG)
	@echo ""
	@echo "-- no sync, $(SMALL) elements: often PASSES, and that is the lesson. The same"
	@echo "   defect, invisible at small sizes, is why this reaches production."
	-./stream_trap --no-sync --count $(SMALL)
	@echo ""
	@echo "-- with the stream synchronization: PASS"
	./stream_trap --sync --count $(BIG)
	@echo ""
	@echo "   Worse failure mode, if you want to see it: --free-early releases the send"
	@echo "   buffer while the library may still be reading it. Undefined behaviour, so"
	@echo "   whatever it does on your machine is not a specification."

run-tools: require-python
	@echo "== busbw: convert a MEASURED time into algbw and busbw (README 2.4 factors) =="
	$(PYTHON) tools/busbw.py --collective allreduce --ranks 4 \
	  --bytes 1073741824 --time-ms 40
	@echo ""
	@echo "== busbw: the traffic factor f(N) for every collective at 8 ranks =="
	$(PYTHON) tools/busbw.py --table --ranks 8
	@echo ""
	@echo "== parse_collective_report: audit a report instead of eyeballing it =="
	@echo "   There is deliberately no committed sample log: it would be a number this"
	@echo "   repository cannot reproduce. The example below is SYNTHETIC, generated from"
	@echo "   the alpha-beta model, and is here to exercise the parser - never to quote."
	$(PYTHON) tools/parse_collective_report.py --emit-example --ranks 4 > synthetic_allreduce.log
	$(PYTHON) tools/parse_collective_report.py --input synthetic_allreduce.log \
	  --collective allreduce --ranks 4
	@echo ""
	@echo "   Run it on your own report from lab 2:"
	@echo "     ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 4 -c 1 | tee allreduce.log"
	@echo "     $(PYTHON) tools/parse_collective_report.py --input allreduce.log \\"
	@echo "             --collective allreduce --ranks 4"
	@echo ""
	@echo "== multinode_allreduce_model: 128 GPUs this repository does not have =="
	$(PYTHON) tools/multinode_allreduce_model.py --gpus-per-node 8 --nodes 16 \
	  --intra-gbs 50 --inter-gbs 25 --nics-per-node 8 --bytes 1073741824

# The drills. The exercises build and run but fail their own checks, because the parts
# you are asked to write are marked TODO stubs - a FAIL on the first run means the
# harness works. Only the C++ drills need a compiler; ex1 and ex4 are Python and
# self-test with `$(PYTHON) -S -I exercises/ex1_busbw_convert.py`.
exercises: require-hipcc $(EX_BINS)

solutions: require-hipcc $(SOL_BINS)

exercises/%: exercises/%.cpp
	$(HIPCC) $(HIP_FLAGS) $< $(HIP_LIBS) -o $@

solutions/%: solutions/%.cpp
	$(HIPCC) $(HIP_FLAGS) $< $(HIP_LIBS) -o $@

# The tools must run on a laptop with nothing installed, so guard that property rather
# than trusting it. Any import outside the allow-list breaks the promise, and `-S -I`
# proves it by refusing site-packages and the current directory.
STDLIB_CHECK := import ast,sys;t=ast.parse(open(sys.argv[1]).read());\
m={a.name.split('.')[0] for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}\
|{(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n,ast.ImportFrom)};\
bad=m-{'argparse','json','math','sys','__future__'};\
sys.exit('non-stdlib import in %s: %s'%(sys.argv[1],bad)) if bad else print('%s ok'%sys.argv[1])

smoke: require-python
	@for t in $(TOOLS); do \
	  $(PYTHON) -c "$(STDLIB_CHECK)" $$t || exit 1; \
	  $(PYTHON) -S -I $$t --help > /dev/null || exit 1; \
	done
	@echo "OK: all tools are standard-library only, run under -S -I, and expose --help"

clean:
	rm -f $(HIP_BINS) $(CUDA_BINS) $(EX_BINS) $(SOL_BINS) *.o
	rm -rf tools/__pycache__ exercises/__pycache__ solutions/__pycache__ __pycache__
# The synthetic log is regenerable from the tool in one command, so it is a build
# artifact, not data. Nothing measured is deleted here because nothing measured is
# ever written into this directory by default.
	rm -f synthetic_allreduce.log
