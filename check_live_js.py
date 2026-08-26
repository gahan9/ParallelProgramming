#!/usr/bin/env python3
"""Syntax-check the inline JavaScript in every self-contained page under live/.

Why this exists
---------------
The live pages carry their script inline, with no build step and no bundler, so nothing
between the editor and the browser will tell you about a stray comma. A syntax error
there does not fail any build — the page just renders its static shell and quietly does
nothing, which looks like a content bug rather than a broken script.

``node --check`` catches it in a second. This wraps the extraction so the check is one
command on any shell, rather than a quoting exercise.

Usage
-----
    python check_live_js.py           # exits non-zero if any page fails
    python check_live_js.py --list    # just show what would be checked
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
LIVE = REPO / "live"
SCRIPT_BLOCK = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S)


def pages() -> list[Path]:
    return sorted(LIVE.rglob("*.html"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list the pages and exit")
    args = parser.parse_args()

    found = pages()
    if args.list:
        for p in found:
            print(p.relative_to(REPO))
        return 0

    if not shutil.which("node"):
        print("node is not on PATH; cannot syntax-check the inline scripts", file=sys.stderr)
        return 1

    failures = 0
    for page in found:
        blocks = SCRIPT_BLOCK.findall(page.read_text(encoding="utf-8"))
        rel = page.relative_to(REPO)
        if not blocks:
            print(f"  --   {rel} (no inline script)")
            continue

        # Concatenating the blocks matches how the browser sees them: one shared global
        # scope, so a redeclaration across two blocks is a real error worth catching.
        source = "\n".join(blocks)
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(source)
            tmp = Path(handle.name)
        try:
            result = subprocess.run(["node", "--check", str(tmp)],
                                    capture_output=True, text=True)
        finally:
            tmp.unlink(missing_ok=True)

        if result.returncode == 0:
            print(f"  ok   {rel} ({len(blocks)} block(s), {len(source)} chars)")
        else:
            failures += 1
            print(f"  FAIL {rel}", file=sys.stderr)
            print(result.stderr.rstrip(), file=sys.stderr)

    if failures:
        print(f"\n{failures} page(s) with invalid inline JavaScript", file=sys.stderr)
        return 1
    print(f"\n{len(found)} page(s) checked, all parse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
