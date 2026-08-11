# D04 - The wire: RDMA, NVLink, xGMI and InfiniBand. Linux-targeted.
#
# Labs 1 and 2 need a node with at least two GPUs. Lab 3 needs an RDMA fabric and the
# userspace verbs tools, and will not exist on most machines - that is a correct result,
# not a failed lab. Lab 4 is analytic and runs anywhere with a stdlib Python 3.
#
# Quick start (these three are the invocations README section 4 documents):
#   make            # build hip/peer_copy_bw.cpp with hipcc
#   make cuda       # build cuda/peer_copy_bw.cu with nvcc
#   make report     # lab 1's topology capture plus the lab 4 models
#
# The rest:
#   make help       # this list
#   make run        # build and run the HIP peer-copy lab on devices 1 -> 0
#   make run-cuda   # the same on the CUDA build
#   make run-tools  # the lab 4 models and the two parsers on the synthetic samples
#   make smoke      # assert the tools are standard-library only and expose --help
#   make clean
#
# Overrides:
#   make GPU_ARCH=gfx950            # your AMD device; find it with offload-arch
#   make cuda SM_ARCH=sm_90         # your NVIDIA device
#   make run SRC=3 DST=2 BYTES=536870912
#   make report PYTHON=python       # Windows, where python3 is often absent
#
# On numbers: tools/staging_cost.py and tools/injection_model.py are analytic models and
# say so in their own banners. tools/topology_report.py and tools/parse_ib_bw.py parse
# output YOU captured, so they name the file instead of printing a disclaimer. The only
# measured numbers this module produces are the ones your own hardware prints.

# Toolchain, arch and base flags are shared with Track A, the same way B01 does it: one
# toolchain definition for the whole repo means a tool-discovery fix lands everywhere at
# once. The one-line hipcc invocation in the source header is the minimal form for
# reading; this build adds -O3 -g and an rpath.
include ../../A-gpu-programming/common.mk

PYTHON ?= python3

HIP_BIN  := peer_copy_bw
CUDA_BIN := peer_copy_bw_cuda

# Defaults for the peer-copy lab: device 1 reading device 0, 256 MiB, which is large
# enough that launch overhead is not what you are measuring.
SRC   ?= 1
DST   ?= 0
BYTES ?= 268435456

TOPO ?= topo.txt

MODELS  := tools/staging_cost.py tools/injection_model.py
PARSERS := tools/topology_report.py tools/parse_ib_bw.py
TOOLS   := $(MODELS) $(PARSERS)
DRILLS  := exercises/01_topology_classifier.py exercises/02_staging_cost.py
DRILLS  += exercises/03_injection_ceiling.py
SOLVED  := solutions/01_topology_classifier.py solutions/02_staging_cost.py
SOLVED  += solutions/03_injection_ceiling.py

.DEFAULT_GOAL := hip
.PHONY: help all hip cuda run run-cuda report run-tools run-exercises smoke clean \
        require-hipcc require-nvcc require-python

help:
	@echo "D04 - Interconnects and RDMA. Targets:"
	@echo "  make            - build hip/peer_copy_bw.cpp with hipcc (default goal)"
	@echo "  make cuda       - build cuda/peer_copy_bw.cu with nvcc"
	@echo "  make report     - lab 1's topology capture plus the lab 4 models"
	@echo "  make run        - build and run the peer-copy lab ($(SRC) -> $(DST), $(BYTES) bytes)"
	@echo "  make run-cuda   - the same on the CUDA build"
	@echo "  make run-tools  - lab 4's models, and both parsers on samples/"
	@echo "  make smoke      - assert the tools are standard-library only"
	@echo "  make clean      - remove binaries, captures and __pycache__"
	@echo ""
	@echo "Detected:"
	@if command -v $(HIPCC) >/dev/null 2>&1; then \
	  echo "  HIPCC   $$(command -v $(HIPCC))  (arch $(GPU_ARCH))"; \
	else \
	  echo "  HIPCC   $(HIPCC) NOT FOUND - ROCm: $(HIP_TOOLCHAIN_INFO)"; \
	fi
	@if command -v $(NVCC) >/dev/null 2>&1; then \
	  echo "  NVCC    $$(command -v $(NVCC))  (arch $(SM_ARCH))"; \
	else \
	  echo "  NVCC    $(NVCC) NOT FOUND"; \
	fi
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
	  echo "  PYTHON  $$(command -v $(PYTHON))"; \
	else \
	  echo "  PYTHON  $(PYTHON) NOT FOUND - try 'make report PYTHON=python'"; \
	fi
	@echo ""
	@echo "No GPU and no fabric? 'make report' and 'make run-tools' still work: the"
	@echo "models are pure Python, and samples/ holds synthetic inputs for the parsers."
	@echo "Exercises: exercises/README.md   Solutions: solutions/README.md"

all: hip cuda

# Guards. Without these a missing toolchain surfaces as a bare "hipcc: command not
# found" from inside a compile rule, which says nothing about what to install or about
# what still works without it.
require-hipcc:
	@command -v $(HIPCC) >/dev/null 2>&1 || { \
	  echo "ERROR: no HIP compiler: '$(HIPCC)' is not on PATH."; \
	  echo "  ROCm search result: $(HIP_TOOLCHAIN_INFO)"; \
	  echo ""; \
	  echo "  Install ROCm, or point this build at an existing install:"; \
	  echo "    make HIP_PATH=/opt/rocm-<version>"; \
	  echo "  Setup instructions: ../../../docs/SETUP.md"; \
	  echo ""; \
	  echo "  Nothing was built. Labs 1 and 4 do not need a compiler:"; \
	  echo "    make report      - capture and normalize your topology, then model"; \
	  echo "    make run-tools   - the analytic models and both parsers"; \
	  exit 1; }

require-nvcc:
	@command -v $(NVCC) >/dev/null 2>&1 || { \
	  echo "ERROR: no CUDA compiler: '$(NVCC)' is not on PATH."; \
	  echo "  Install the CUDA toolkit, or build the AMD path instead: make hip"; \
	  echo "  Setup instructions: ../../../docs/SETUP.md"; \
	  exit 1; }

require-python:
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
	  echo "ERROR: '$(PYTHON)' is not on PATH."; \
	  echo "  The tools need only a stdlib Python 3.8+. On Windows try: PYTHON=python"; \
	  exit 1; }

# ---- AMD / HIP ----
hip: require-hipcc $(HIP_BIN)

$(HIP_BIN): hip/peer_copy_bw.cpp
	$(HIPCC) $(HIP_BASE_FLAGS) $< -o $@

run: hip
	@echo "== Lab 2: unidirectional peer-copy bandwidth, device $(SRC) -> device $(DST) =="
	@echo "   Verification comes first: a nonzero 'wrong bytes' invalidates the GB/s."
	./$(HIP_BIN) $(SRC) $(DST) $(BYTES)
	@echo ""
	@echo "Now run it again on a pair your topology report called anything other than"
	@echo "direct-link. It will still print a number, and the number will be much lower:"
	@echo "that is the silent host-staged fallback of README section 3.5."

# ---- NVIDIA / CUDA ----
cuda: require-nvcc $(CUDA_BIN)

$(CUDA_BIN): cuda/peer_copy_bw.cu
	$(NVCC) $(CUDA_BASE_FLAGS) $< -o $@

run-cuda: cuda
	@echo "== Lab 2 (CUDA): device $(SRC) -> device $(DST) =="
	./$(CUDA_BIN) $(SRC) $(DST) $(BYTES)

# ---- Lab 1 capture + lab 4 models ----
# The capture step is best-effort by design: on a machine with neither vendor tool the
# right answer is to say so and carry on to the models, not to fail the target.
report: require-python
	@echo "== Lab 1: capture this node's topology matrix =="
	@if command -v rocm-smi >/dev/null 2>&1; then \
	  echo "-- rocm-smi --showtopo | tee $(TOPO)"; \
	  rocm-smi --showtopo | tee $(TOPO); \
	  echo ""; \
	  $(PYTHON) tools/topology_report.py --input $(TOPO) --vendor amd; \
	elif command -v nvidia-smi >/dev/null 2>&1; then \
	  echo "-- nvidia-smi topo -m | tee $(TOPO)"; \
	  nvidia-smi topo -m | tee $(TOPO); \
	  echo ""; \
	  $(PYTHON) tools/topology_report.py --input $(TOPO) --vendor nvidia; \
	else \
	  echo "   Neither rocm-smi nor nvidia-smi is on PATH, so there is no topology to"; \
	  echo "   capture here. That is a correct result on a machine with no GPU."; \
	  echo "   To see the shape of the answer, run the parser on a SYNTHETIC sample -"; \
	  echo "   it describes no real machine and says so in its own header:"; \
	  echo "     $(PYTHON) tools/topology_report.py --input samples/nvidia_topo_synthetic.txt"; \
	fi
	@echo ""
	@echo "== Lab 4: the two models. Both label their output as a model, not a measurement =="
	@echo "-- staged versus direct, 1 GiB, with PCIe twice the wire"
	$(PYTHON) tools/staging_cost.py --bytes 1073741824 --pcie-gbs 20 --net-gbs 10
	@echo ""
	@echo "-- injection: eight GPUs sharing ONE adapter"
	$(PYTHON) tools/injection_model.py --gpus-per-node 8 --nics-per-node 1 \
	        --nic-gbs 25 --scaleup-gbs 400
	@echo ""
	@echo "-- injection: the same node with one adapter per GPU"
	$(PYTHON) tools/injection_model.py --gpus-per-node 8 --nics-per-node 8 \
	        --nic-gbs 25 --scaleup-gbs 400
	@echo ""
	@echo "Read the per-GPU share and the ratio in those last two back to back. Then"
	@echo "sweep --nics-per-node yourself and note that the curve is linear: that is the"
	@echo "argument the rail-optimized design in D05 is built on."

run-tools: require-python
	@echo "== the two analytic models (lab 4) =="
	$(PYTHON) tools/staging_cost.py --bytes 1073741824 --pcie-gbs 20 --net-gbs 10
	@echo ""
	$(PYTHON) tools/injection_model.py --gpus-per-node 8 --nics-per-node 1 \
	        --nic-gbs 25 --scaleup-gbs 400
	@echo ""
	@echo "== the two parsers, on SYNTHETIC samples that describe no real hardware =="
	$(PYTHON) tools/topology_report.py --input samples/nvidia_topo_synthetic.txt \
	        --vendor auto --group 0,1,2,3
	@echo ""
	$(PYTHON) tools/parse_ib_bw.py --input samples/ib_write_bw_synthetic.txt \
	        --report gbits
	@echo ""
	@echo "Replace those two inputs with your own captures. The parsers name the file"
	@echo "they read on every run, so a result never loses its provenance."

run-exercises: require-python
	@for d in $(DRILLS); do \
	  echo "== $$d =="; \
	  $(PYTHON) $$d || exit 1; \
	  echo ""; \
	done
	@echo "Fill in the TODOs, then compare against solutions/README.md."

# The point of these tools is that they run on a laptop with nothing installed, so guard
# that property rather than trusting it. Any import outside the allow-list breaks it.
STDLIB_CHECK := import ast,sys;t=ast.parse(open(sys.argv[1]).read());\
m={a.name.split('.')[0] for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}\
|{(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n,ast.ImportFrom)};\
bad=m-{'argparse','json','math','os','re','sys','__future__'};\
sys.exit('non-stdlib import in %s: %s'%(sys.argv[1],bad)) if bad else print('%s ok'%sys.argv[1])

smoke: require-python
	@for t in $(TOOLS) $(DRILLS) $(SOLVED); do \
	  $(PYTHON) -c "$(STDLIB_CHECK)" $$t || exit 1; \
	done
	@for t in $(TOOLS); do \
	  $(PYTHON) -S -I $$t --help > /dev/null || exit 1; \
	done
	@echo "OK: tools and drills are standard-library only, and every tool runs under -S -I"

clean:
	rm -f $(HIP_BIN) $(CUDA_BIN) $(TOPO)
	rm -rf tools/__pycache__ exercises/__pycache__ solutions/__pycache__ __pycache__
