#!/usr/bin/env python3
"""Sanitize captured command output for public curriculum docs.

Strips hostnames, usernames, IPs, detailed toolchain versions, and internal
paths while preserving gfx950 and ROCm major version (ROCm 7) labels.
Stdlib only — run with: python -S -I scripts/sanitize_output.py < raw.txt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Conductor / SSH login banners and ASCII art blocks.
_BANNER_START = re.compile(
    r"^\s*\$\$\$\$\$\$\\.*|^Checking authorization now\.\.\.|"
    r"^Warning: Permanently added .* to the list of known hosts\.",
    re.MULTILINE,
)
_BANNER_BLOCK = re.compile(
    r"(?:^\s*\$\$\$\$\$\$\\.*\n)+|"
    r"^.*is protected by Conductor SUT Authentication.*\n"
    r"(?:^.*\n){0,20}?^Checking authorization now\.\.\.\n",
    re.MULTILINE,
)

# Host / user / network identifiers.
_HOST = re.compile(
    r"\b[a-z0-9][a-z0-9\-]*\.(?:dcgpu|png-odc\.dcgpu|ctr\.dcgpu|"
    r"pla\.dcgpu|cs-clt\.dcgpu|dh170\.dcgpu|mkm\.dcgpu)\b",
    re.IGNORECASE,
)
_USER_AT_HOST = re.compile(r"\b[a-z][a-z0-9_\-]{0,31}@[a-z0-9\.\-]+\b", re.IGNORECASE)
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOME = re.compile(r"/(?:home|users)/[a-zA-Z0-9_\-]+")
_TMP_USER = re.compile(r"/tmp/[a-zA-Z0-9_\-]+/")

# Toolchain detail — keep major only.
_ROCM_PATH = re.compile(r"/opt/rocm(?:-\d+\.\d+\.\d+)?")
_ROCM_VER = re.compile(r"\bROCm\s+\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)
_HIP_VER = re.compile(
    r"HIP\s*\((?:compiler|runtime)\)\s+version\s+\d+\.\d+(?:\.\d+)?",
    re.IGNORECASE,
)
_HSA_VER = re.compile(r"HSA version\s+\d+\.\d+(?:\.\d+)?", re.IGNORECASE)
_GENERIC_VER = re.compile(
    r"\b(?:version|v)\s+\d+\.\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE
)

# Profiler / log noise.
_PID_LINE = re.compile(r"^[WE]\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+\s+\d+\s+", re.MULTILINE)
_LOG_PREFIX = re.compile(r"^[WE]\d{4}\s+\d+\s+", re.MULTILINE)
_NODE_ID = re.compile(r"\bon node \d+\b")
_UUID = re.compile(
    r"\(UUID=[0-9a-f\-]+\)", re.IGNORECASE
)
_AGENT_ID = re.compile(r"\(agent \d+\)")
_PCI = re.compile(r"\b[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\b", re.IGNORECASE)
_HIP_RT = re.compile(r"HIP runtime \d+, driver \d+")
_VISIBLE_DEVS = re.compile(r"Visible devices: \d+")
_AGENT_NUM = re.compile(r"\bagent \d{6,}\b")
_TOPO_SECTION = re.compile(
    r"== hip/topology\.exe ==.*\Z",
    re.DOTALL,
)
_MULTI_DEVICE = re.compile(
    r"(?m)^={64,}\nDevice [1-9]\d*:.*?(?=^={64,}\nDevice |\Z)",
    re.DOTALL,
)
_DEVICE_MARKETING = re.compile(r"AMD Instinct MI\d+[A-Z]?")
_PROF_PATH = re.compile(
    r"profiling_output/[^/\s]+/(\d+)_", re.IGNORECASE
)


def sanitize(text: str) -> str:
    out = text.replace("\r\n", "\n")
    out = _BANNER_BLOCK.sub("", out)
    out = _BANNER_START.sub("", out)
    out = _HOST.sub("<host>", out)
    out = _USER_AT_HOST.sub("<user>@<host>", out)
    out = _IP.sub("<ip>", out)
    out = _HOME.sub("/home/<user>", out)
    out = _TMP_USER.sub("/tmp/<workdir>/", out)
    out = _ROCM_PATH.sub("/opt/rocm", out)
    out = _ROCM_VER.sub("ROCm 7", out)
    out = _HIP_VER.sub(lambda m: m.group(0).split("version")[0] + "version ROCm 7", out)
    out = _HSA_VER.sub("HSA version (ROCm 7 runtime)", out)
    out = _GENERIC_VER.sub("version (ROCm 7)", out)
    out = _PROF_PATH.sub("profiling_output/<host>/<pid>_", out)
    out = _PID_LINE.sub("W<ts> <pid> ", out)
    out = _LOG_PREFIX.sub("", out)
    out = _NODE_ID.sub("on node <host>", out)
    out = _UUID.sub("(UUID=<redacted>)", out)
    out = _AGENT_ID.sub("(agent <id>)", out)
    out = _PCI.sub("<pci>", out)
    out = _HIP_RT.sub("HIP runtime (ROCm 7)", out)
    out = _VISIBLE_DEVS.sub("Visible devices: <n>", out)
    out = _AGENT_NUM.sub("agent <id>", out)
    out = _MULTI_DEVICE.sub(
        "[additional devices omitted — same arch (gfx950), run locally for a full listing]\n",
        out,
    )
    out = _TOPO_SECTION.sub(
        "== hip/topology.exe ==\n"
        "[topology matrices omitted — run `make run-hip` locally to inspect peer access on your node]\n",
        out,
    )
    out = _DEVICE_MARKETING.sub("AMD Instinct GPU (gfx950)", out)
    out = re.sub(
        r"(?:\[additional devices omitted[^\]]+\]\n)+",
        "[additional devices omitted — same arch (gfx950), run locally for a full listing]\n",
        out,
    )

    # Collapse repeated blank lines introduced by stripping.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


def main() -> int:
    raw = sys.stdin.read() if len(sys.argv) <= 1 else Path(sys.argv[1]).read_text(encoding="utf-8")
    sys.stdout.write(sanitize(raw))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
