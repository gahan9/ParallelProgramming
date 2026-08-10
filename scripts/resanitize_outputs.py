#!/usr/bin/env python3
"""Re-sanitize every file under outputs/_raw/ into outputs/. Stdlib only."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sanitize_output import sanitize  # noqa: E402


def main() -> int:
    raw_root = ROOT / "outputs" / "_raw"
    out_root = ROOT / "outputs"
    if not raw_root.exists():
        print("no outputs/_raw/ — run collect_outputs.sh on a GPU host first", file=sys.stderr)
        return 1
    count = 0
    for raw in raw_root.rglob("*.txt"):
        rel = raw.relative_to(raw_root)
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(sanitize(raw.read_text(encoding="utf-8")), encoding="utf-8")
        count += 1
    print(f"re-sanitized {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
