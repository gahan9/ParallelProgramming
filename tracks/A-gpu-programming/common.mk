# Shared GPU toolchain settings for every Track A module.
#
# Included by:
#   a module root         ->  include ../common.mk
#   exercises/ solutions/ ->  include ../../common.mk
#
# Keeping the toolchain in one file means a fix (rpath, tool discovery) lands for
# every module and every drill at once, instead of drifting module by module.
#
# This file defines VARIABLES ONLY - no targets. It is included at the top of
# each Makefile, so a target here would silently become the default goal.
#
# Comments sit on their own line on purpose: in make, a trailing comment after
# `VAR ?= value   # note` keeps the whitespace before the `#` inside the value.

# ---- Locate ROCm ----
# Search order, first hit wins:
#   1. HIP_PATH   set on the command line or in the environment (explicit override)
#   2. ROCM_PATH  the ROCm convention, when the environment already sets it
#   3. /opt/rocm            the versionless symlink, when the install provides one
#   4. /opt/rocm-<version>  a versioned-only install (common on lab machines)
# Step 4 matters: many systems ship /opt/rocm-7.15.0 with no /opt/rocm symlink,
# and it sorts textually, so with several versioned installs side by side pass
# HIP_PATH explicitly rather than trusting the pick.
ROCM_AUTODETECT := $(firstword $(ROCM_PATH) \
                               $(wildcard /opt/rocm) \
                               $(lastword $(sort $(wildcard /opt/rocm-*))))
HIP_PATH ?= $(ROCM_AUTODETECT)

# ---- Toolchains ----
# Prefer absolute tool paths: <rocm>/bin is frequently NOT on PATH, so a bare
# `hipcc` / `rocprofv3` fails with "No such file or directory". Fall back to PATH
# only when no install was found, so the error message names the real tool.
ifeq ($(strip $(HIP_PATH)),)
  HIPCC     := hipcc
  ROCPROFV3 := rocprofv3
  HIP_RPATH :=
  HIP_TOOLCHAIN_INFO := ROCm not found - pass HIP_PATH=/opt/rocm-<version>
else
  HIPCC     := $(HIP_PATH)/bin/hipcc
  ROCPROFV3 := $(HIP_PATH)/bin/rocprofv3
  # rpath keeps binaries self-contained: they locate libamdhip64.so inside the
  # ROCm install even when its lib/ is not on the loader path (no ldconfig
  # entry, empty LD_LIBRARY_PATH).
  HIP_RPATH := -Wl,-rpath,$(HIP_PATH)/lib
  HIP_TOOLCHAIN_INFO := $(HIP_PATH)
endif

NVCC := nvcc

# ---- Architectures (override on the command line for your GPU) ----
# AMD:    gfx950 (MI350), gfx942 (MI300), gfx90a (MI200), gfx1100 (RDNA3)
# NVIDIA: sm_90 (Hopper), sm_80 (Ampere), sm_89 (Ada)
# Find yours with: $(HIP_PATH)/bin/offload-arch   (or: rocminfo | grep gfx)
GPU_ARCH ?= gfx942
SM_ARCH  ?= sm_90

# ---- Base flags (each Makefile appends its own -I include path) ----
HIP_BASE_FLAGS  := -O3 -g --offload-arch=$(GPU_ARCH) $(HIP_RPATH)
CUDA_BASE_FLAGS := -O3 -g -arch=$(SM_ARCH)
