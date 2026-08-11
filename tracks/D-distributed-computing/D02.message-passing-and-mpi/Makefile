# D02 - Message passing with MPI. Linux-targeted; needs an mpicc, not a GPU.
#
# Everything here runs on ONE machine. Four cores are enough to see every effect in the
# module, including deadlock: you need two processes and a message larger than your
# implementation's eager threshold, not a cluster.
#
# Quick start:
#   make help        # this list (default target)
#   make all         # build the four labs in mpi/ with mpicc
#   make run         # labs 1-3
#   make deadlock    # lab 4, each pattern under a timeout so a hang is bounded
#   make run-tools   # the analytic models in tools/ (no MPI needed)
#   make clean
#
# Overrides:
#   make all  MPICC=/opt/openmpi/bin/mpicc     # mpicc not on PATH
#   make run  NP=8 COUNT=16777216              # rank count and all-reduce element count
#   make deadlock BYTES=67108864 TIMEOUT=20    # push past a large eager threshold
#   make run-tools PYTHON=python               # Windows, where python3 is often absent
#
# The tools in tools/ are analytic models and say so in their own output banner. The only
# MEASURED numbers this module produces are the ones you generate yourself in labs 2 and
# 3, on your machine, over a shared-memory transport. tools/fit_alpha_beta.py is the one
# tool that consumes measured data, so it names its input file instead of printing the
# model disclaimer.

MPICC   ?= mpicc
MPIRUN  ?= mpirun
PYTHON  ?= python3
CFLAGS  ?= -O2 -Wall -Wextra
NP      ?= 4
COUNT   ?= 1048576
BYTES   ?= 16777216
SMALL   ?= 1024
TIMEOUT ?= 10

BINS  := hello_ranks ping_pong ring_allreduce deadlock
TOOLS := tools/collective_cost.py tools/allreduce_algorithms.py tools/fit_alpha_beta.py
CSV   ?= pingpong.csv

# `deadlock` names both a binary and the lab-4 run target, so the compile rule is called
# deadlock_bin and writes a file called `deadlock`. Two targets of the same name cannot
# both carry a recipe, and a file rule would let an already-built binary satisfy the run
# target and silently do nothing. The cost is that deadlock.c recompiles on every `make
# all`; one translation unit, so it is not worth more machinery than this comment.
BUILD := hello_ranks ping_pong ring_allreduce deadlock_bin

.PHONY: help all run run-hello run-pingpong run-ring deadlock deadlock_bin run-tools \
        smoke clean require-mpicc require-python

help:
	@echo "D02 - Message passing with MPI. Targets:"
	@echo "  make all         - build $(words $(BINS)) labs in mpi/ with $(MPICC)"
	@echo "  make run         - labs 1-3: ranks/communicators, ping-pong sweep, ring all-reduce"
	@echo "  make deadlock    - lab 4: five exchange patterns, each under a $(TIMEOUT)s timeout"
	@echo "  make run-tools   - the analytic cost models in tools/ (needs no MPI)"
	@echo "  make smoke       - assert the tools are standard-library only and expose --help"
	@echo "  make clean       - remove binaries, objects, __pycache__ and $(CSV)"
	@echo ""
	@echo "Detected:"
	@if command -v $(MPICC) >/dev/null 2>&1; then \
	  echo "  MPICC   $$(command -v $(MPICC))"; \
	else \
	  echo "  MPICC   $(MPICC) NOT FOUND - build targets will explain how to fix this"; \
	fi
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
	  echo "  PYTHON  $$(command -v $(PYTHON))"; \
	else \
	  echo "  PYTHON  $(PYTHON) NOT FOUND - try 'make run-tools PYTHON=python'"; \
	fi
	@echo ""
	@echo "Every number tools/ prints is an analytic model, never a measurement."
	@echo "Exercises: exercises/README.md   Solutions: solutions/README.md"

# Guards. Without these, a missing toolchain surfaces as a bare "mpicc: command not
# found" from inside a compile rule, which says nothing about what to install.
require-mpicc:
	@command -v $(MPICC) >/dev/null 2>&1 || { \
	  echo "ERROR: no MPI C compiler: '$(MPICC)' is not on PATH."; \
	  echo ""; \
	  echo "  This module needs one. Any MPI implementation works; the labs target Open MPI:"; \
	  echo "    Debian / Ubuntu : sudo apt-get install openmpi-bin libopenmpi-dev"; \
	  echo "    RHEL / Fedora   : sudo dnf install openmpi openmpi-devel"; \
	  echo "    HPC module env  : module load openmpi   (or mpi/openmpi-x86_64)"; \
	  echo "    already installed elsewhere: make all MPICC=/path/to/mpicc"; \
	  echo ""; \
	  echo "  Nothing was built. 'make run-tools' still works: the analytic models in"; \
	  echo "  tools/ are pure Python and need no MPI at all."; \
	  exit 1; }

require-python:
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
	  echo "ERROR: '$(PYTHON)' is not on PATH."; \
	  echo "  The tools need only a stdlib Python 3.8+. On Windows try: make run-tools PYTHON=python"; \
	  exit 1; }

all: require-mpicc $(BUILD)

hello_ranks: mpi/hello_ranks.c mpi/mpi_check.h
	$(MPICC) $(CFLAGS) -Impi -o $@ $<

ping_pong: mpi/ping_pong.c mpi/mpi_check.h
	$(MPICC) $(CFLAGS) -Impi -o $@ $<

ring_allreduce: mpi/ring_allreduce.c mpi/mpi_check.h
	$(MPICC) $(CFLAGS) -Impi -o $@ $<

deadlock_bin: mpi/deadlock.c mpi/mpi_check.h
	$(MPICC) $(CFLAGS) -Impi -o deadlock $<

run: run-hello run-pingpong run-ring

run-hello: require-mpicc hello_ranks
	@echo "== Lab 1: ranks, node-local communicators, and a dup =="
	@echo "   Line order differs between runs: there is no global ordering of independent events."
	$(MPIRUN) -np $(NP) ./hello_ranks

run-pingpong: require-mpicc ping_pong
	@echo ""
	@echo "== Lab 2: half round-trip latency across a size sweep (exactly 2 ranks) =="
	@echo "   This writes $(CSV). It is YOUR measurement of YOUR transport - on one node"
	@echo "   that is shared memory, not a network. Label it before you quote it."
	$(MPIRUN) -np 2 ./ping_pong | tee $(CSV)
	@echo ""
	@if command -v $(PYTHON) >/dev/null 2>&1; then \
	  echo "== Fitting alpha and beta from $(CSV) =="; \
	  $(PYTHON) tools/fit_alpha_beta.py --input $(CSV); \
	else \
	  echo "($(PYTHON) not found: fit it yourself with"; \
	  echo " make run-tools PYTHON=python, or python tools/fit_alpha_beta.py --input $(CSV))"; \
	fi

run-ring: require-mpicc ring_allreduce
	@echo ""
	@echo "== Lab 3: hand-written ring all-reduce vs MPI_Allreduce =="
	@echo "-- small message: your ring pays 2(N-1) latencies for a nearly free payload"
	$(MPIRUN) -np $(NP) ./ring_allreduce $(SMALL)
	@echo "-- large message: same 2m(N-1)/N bytes as the library, so the gap narrows"
	$(MPIRUN) -np $(NP) ./ring_allreduce $(COUNT)
	@echo "-- count not divisible by rank count, and count smaller than rank count"
	$(MPIRUN) -np $(NP) ./ring_allreduce 1000003
	$(MPIRUN) -np $(NP) ./ring_allreduce 3

# Lab 4. Two of these patterns are SUPPOSED to hang, so every line is prefixed with `-`
# and wrapped in `timeout`: a nonzero exit is data, not a build failure. Exit 124 is
# timeout's way of saying it killed the job, which here means "deadlocked as predicted".
deadlock: require-mpicc deadlock_bin
	@echo "== Lab 4: five exchange patterns, 2 ranks, $(TIMEOUT)s timeout each =="
	@echo "   Exit 124 means timeout killed it: that IS the deadlock."
	@echo ""
	@echo "-- deadlock pattern, $(SMALL) bytes: below the eager threshold, expect it to COMPLETE"
	-timeout $(TIMEOUT) $(MPIRUN) -np 2 ./deadlock deadlock $(SMALL)
	@echo ""
	@echo "-- deadlock pattern, $(BYTES) bytes: rendezvous, expect it to HANG"
	-timeout $(TIMEOUT) $(MPIRUN) -np 2 ./deadlock deadlock $(BYTES)
	@echo ""
	@echo "-- ssend, $(SMALL) bytes: a synchronous send cannot be hidden by an eager buffer,"
	@echo "   so identical logic HANGS at a size where MPI_Send succeeded. That is the"
	@echo "   diagnostic: put MPI_Ssend in a debug build and latent deadlock stops being"
	@echo "   size-dependent."
	-timeout $(TIMEOUT) $(MPIRUN) -np 2 ./deadlock ssend $(SMALL)
	@echo ""
	@echo "-- the three fixes at $(BYTES) bytes, all expected to complete"
	-timeout $(TIMEOUT) $(MPIRUN) -np 2 ./deadlock ordered $(BYTES)
	-timeout $(TIMEOUT) $(MPIRUN) -np 2 ./deadlock sendrecv $(BYTES)
	-timeout $(TIMEOUT) $(MPIRUN) -np 2 ./deadlock nonblocking $(BYTES)
	@echo ""
	@echo "Now bisect BYTES in the deadlock pattern to find your own eager threshold:"
	@echo "  make deadlock BYTES=<n>    - it is a tuning knob, not a constant, and not portable."

run-tools: require-python
	@echo "== allreduce_algorithms: the three families under the module's illustrative parameters =="
	@echo "   (reproduces the comparison table in README section 3.3)"
	$(PYTHON) tools/allreduce_algorithms.py --crossover
	@echo ""
	@echo "== allreduce_algorithms: the same at 1024 ranks, where the ring's O(N) steps bite =="
	$(PYTHON) tools/allreduce_algorithms.py --ranks 1024 --sizes 1KiB,1GiB --crossover
	@echo ""
	@echo "== collective_cost: the section 3.2 catalogue at 64 ranks =="
	$(PYTHON) tools/collective_cost.py --ranks 64 --sizes 4KiB,4MiB
	@echo ""
	@echo "== fit_alpha_beta works on MEASURED data, so it has nothing to model =="
	@if [ -f $(CSV) ]; then \
	  $(PYTHON) tools/fit_alpha_beta.py --input $(CSV) --classify 1KiB,64KiB,16MiB; \
	else \
	  echo "   No $(CSV) yet. Produce one first, then fit it:"; \
	  echo "     make run-pingpong"; \
	  echo "     $(PYTHON) tools/fit_alpha_beta.py --input $(CSV) --classify 1KiB,64KiB,16MiB"; \
	  echo "   There is deliberately no sample CSV in this repository: a committed"; \
	  echo "   performance file would be a number the repo cannot reproduce."; \
	fi

# The tools must run on a laptop with nothing installed, so guard that property rather
# than trusting it. Any import outside the allow-list breaks the promise.
STDLIB_CHECK := import ast,sys;t=ast.parse(open(sys.argv[1]).read());\
m={a.name.split('.')[0] for n in ast.walk(t) if isinstance(n,ast.Import) for a in n.names}\
|{(n.module or '').split('.')[0] for n in ast.walk(t) if isinstance(n,ast.ImportFrom)};\
bad=m-{'argparse','csv','json','math','sys','__future__'};\
sys.exit('non-stdlib import in %s: %s'%(sys.argv[1],bad)) if bad else print('%s ok'%sys.argv[1])

smoke: require-python
	@for t in $(TOOLS); do \
	  $(PYTHON) -c "$(STDLIB_CHECK)" $$t || exit 1; \
	  $(PYTHON) -S -I $$t --help > /dev/null || exit 1; \
	done
	@echo "OK: all tools are standard-library only, run under -S -I, and expose --help"

clean:
	rm -f $(BINS) *.o
	rm -rf tools/__pycache__ exercises/__pycache__ solutions/__pycache__ __pycache__
# $(CSV) too: it is a measurement of one machine's transport, and this module's own
# policy is that no such file belongs in the repository. Keep it by copying it out
# first, or by running with CSV=/somewhere/else.csv.
	rm -f $(CSV)
