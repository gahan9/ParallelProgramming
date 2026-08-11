#!/usr/bin/env bash
# Collect sanitized lab output on an AMD GPU system (gfx950 / ROCm 7).
# Usage (on the GPU host, after cloning the repo):
#   git clone https://github.com/gahan9/ParallelProgramming
#   cd ParallelProgramming
#   bash scripts/collect_outputs.sh
#
# Writes raw captures under outputs/_raw/ and sanitized copies under outputs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_ARCH="${GPU_ARCH:-gfx950}"
HIP_PATH="${HIP_PATH:-/opt/rocm}"
export GPU_ARCH HIP_PATH
MAKE="make -s"
RAW_DIR="$ROOT/outputs/_raw"
OUT_DIR="$ROOT/outputs"
MANIFEST="$OUT_DIR/manifest.json"

mkdir -p "$RAW_DIR" "$OUT_DIR"

run_capture() {
  local id="$1"
  shift
  local raw="$RAW_DIR/${id}.txt"
  local clean="$OUT_DIR/${id}.txt"
  mkdir -p "$(dirname "$raw")" "$(dirname "$clean")"
  echo "== capture: $id =="
  {
    echo "# cmd: $*"
    echo "# gpu_arch: $GPU_ARCH"
    echo "# captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "---"
    "$@" 2>&1 || true
  } | tee "$raw"
  python3 "$ROOT/scripts/sanitize_output.py" "$raw" > "$clean"
}

# ---- Track A (HIP labs) ----
A_MODULES=(
  A01.foundations-and-programming-model
  A02.memory-hierarchy-and-coalescing
  A03.execution-model-and-occupancy
  A04.parallel-reduction
  A05.parallel-scan-prefix-sum
  A06.tiled-matrix-multiply
  A07.softmax-and-fused-kernels
  A08.triton-foundations
  A09.advanced-triton-flash-attention
)

for mod in "${A_MODULES[@]}"; do
  base="tracks/A-gpu-programming/$mod"
  [[ -f "$base/Makefile" ]] || continue
  if grep -q '^run-hip:' "$base/Makefile" 2>/dev/null; then
    run_capture "track-a/${mod}/run-hip" bash -lc "cd '$base' && $MAKE run-hip GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH"
  elif grep -qE '^run:' "$base/Makefile" 2>/dev/null; then
    run_capture "track-a/${mod}/run" bash -lc "cd '$base' && $MAKE run GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH"
  fi
  if [[ -d "$base/exercises" && -f "$base/exercises/Makefile" ]]; then
    run_capture "track-a/${mod}/exercises-run" bash -lc "cd '$base/exercises' && $MAKE run GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH"
  fi
  if [[ -d "$base/solutions" && -f "$base/solutions/Makefile" ]]; then
    run_capture "track-a/${mod}/solutions-run" bash -lc "cd '$base/solutions' && $MAKE run GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH"
  fi
done

# A10 optional CUDA/CuTe — HIP-only modules skip if no hip target
if [[ -f tracks/A-gpu-programming/A10.cute-cutlass-and-cutile/Makefile ]]; then
  run_capture "track-a/A10.cute-cutlass-and-cutile/run-cuda" bash -lc \
    "cd tracks/A-gpu-programming/A10.cute-cutlass-and-cutile && $MAKE run-cuda 2>&1" || true
fi

# ---- Track B (HIP labs + stdlib Python tools) ----
B_MODULES=(
  B01.accelerator-architecture
  B02.roofline-and-arithmetic-intensity
  B03.profiling-and-bottleneck-analysis
  B04.precision-and-quantization
  B05.transformer-performance
  B06.inference-serving-optimizations
  B07.distributed-training-and-sharding
)

for mod in "${B_MODULES[@]}"; do
  base="tracks/B-gpu-ml-performance/$mod"
  [[ -f "$base/Makefile" ]] || continue
  if grep -q 'run-hip' "$base/Makefile" 2>/dev/null; then
    run_capture "track-b/${mod}/run-hip" bash -lc "cd '$base' && $MAKE run-hip GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH"
  elif grep -q '^run:' "$base/Makefile" 2>/dev/null; then
    run_capture "track-b/${mod}/run" bash -lc "cd '$base' && $MAKE run"
  fi
  if [[ -d "$base/exercises" && -f "$base/exercises/Makefile" ]]; then
    run_capture "track-b/${mod}/exercises-run" bash -lc "cd '$base/exercises' && $MAKE run GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH 2>&1 || $MAKE run 2>&1"
  fi
  if [[ -d "$base/solutions" && -f "$base/solutions/Makefile" ]]; then
    run_capture "track-b/${mod}/solutions-run" bash -lc "cd '$base/solutions' && $MAKE run GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH 2>&1 || $MAKE run 2>&1"
  fi
done

# ---- Track C (stdlib Python — host-only) ----
C_MODULES=(
  C01.problem-framing-and-metrics
  C02.data-pipelines-and-features
  C03.training-systems
  C04.production-model-optimization
  C05.deployment-and-serving-architecture
  C06.monitoring-drift-and-ethics
)

for mod in "${C_MODULES[@]}"; do
  base="tracks/C-ml-system-design/$mod"
  [[ -f "$base/Makefile" ]] || continue
  if grep -q '^run:' "$base/Makefile" 2>/dev/null; then
    run_capture "track-c/${mod}/run" bash -lc "cd '$base' && $MAKE run"
  fi
done

# ---- Track D (stdlib Python models; D02/D03/D04 also have MPI/HIP labs) ----
D_MODULES=(
  D01.scaling-limits-and-communication-cost
  D02.message-passing-and-mpi
  D03.gpu-collectives-nccl-and-rccl
  D04.interconnects-and-rdma
  D05.topology-and-scale-boundaries
  D06.real-systems-at-scale
)

for mod in "${D_MODULES[@]}"; do
  base="tracks/D-distributed-computing/$mod"
  [[ -f "$base/Makefile" ]] || continue
  if grep -q '^run-hip:' "$base/Makefile" 2>/dev/null; then
    run_capture "track-d/${mod}/run-hip" bash -lc "cd '$base' && $MAKE run-hip GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH"
  elif grep -q '^run:' "$base/Makefile" 2>/dev/null; then
    run_capture "track-d/${mod}/run" bash -lc "cd '$base' && $MAKE run"
  fi
done

# ---- Profile sample (A01 vector_add) ----
run_capture "track-a/A01.foundations-and-programming-model/profile-hip" bash -lc \
  "cd tracks/A-gpu-programming/A01.foundations-and-programming-model && $MAKE profile-hip GPU_ARCH=$GPU_ARCH HIP_PATH=$HIP_PATH"

# Manifest (public metadata only)
python3 - <<'PY'
import json, os, datetime
from pathlib import Path
root = Path("outputs")
entries = sorted(p.relative_to(root).as_posix() for p in root.glob("**/*.txt") if "_raw" not in p.parts)
manifest = {
    "gpu_arch": "gfx950",
    "rocm_major": 7,
    "note": "Sanitized captures; no hostname, username, or detailed toolchain version.",
    "reproduce": "git clone https://github.com/gahan9/ParallelProgramming && cd ParallelProgramming && bash scripts/collect_outputs.sh",
    "captured_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    "artifacts": entries,
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"wrote {len(entries)} artifacts")
PY

echo "Done. Sanitized outputs in $OUT_DIR"
