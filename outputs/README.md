# Cached run outputs

Sanitized stdout/stderr from running curriculum labs on **AMD Instinct MI350 series (`gfx950`)**
with **ROCm 7**. These files back the ` ```text ` sample-output blocks in module READMEs and
exercise guides.

## What is (and is not) in these files

| Included | Stripped |
|----------|----------|
| `gfx950` architecture label | Hostnames, IPs, usernames |
| `ROCm 7` major version | Detailed ROCm/HIP/HSA patch versions |
| PASS/FAIL states, bandwidth shapes, profiler table structure | Internal paths, PIDs, timestamps |
| Reproducible command transcripts | Conductor / SSH login banners |

## Reproduce locally

```bash
git clone https://github.com/gahan9/ParallelProgramming
cd ParallelProgramming
export GPU_ARCH=gfx950          # MI350 series
export HIP_PATH=/opt/rocm       # ROCm 7 install
bash scripts/collect_outputs.sh
```

Sanitized artifacts land in `outputs/`; raw captures (pre-strip) in `outputs/_raw/` (gitignored).

### Coverage notes

| Category | Cached on `gfx950` / ROCm 7 | Gap |
|----------|----------------------------|-----|
| Track A HIP labs + drills | A01–A07 | A08/A09 Triton need a `torch` wheel (`make venv` prints setup) |
| Track B HIP labs + drills | B01–B04 | — |
| Track B Python tools | B05–B07 | Host-only stdlib tools |
| Track C Python tools | C01–C06 | Host-only stdlib tools |
| NVIDIA CUDA / CuTe (A10) | — | Requires `nvcc` (not on AMD hosts) |

Re-run `bash scripts/collect_outputs.sh` after installing missing wheels or on an NVIDIA box for CUDA targets.

## Refresh a single module

```bash
cd tracks/A-gpu-programming/A01.foundations-and-programming-model
make run-hip GPU_ARCH=gfx950
# pipe through the sanitizer before publishing:
make run-hip GPU_ARCH=gfx950 2>&1 | python3 ../../../scripts/sanitize_output.py
```

See [`manifest.json`](manifest.json) for the artifact index and capture date.
