#!/usr/bin/env python3
"""Regenerate the "Where each source is cited" table in ``docs/REFERENCES.md``.

Why this is generated
---------------------
``AGENTS.md`` records that module data is duplicated across five surfaces and that a
generator is "not yet earned" for them — five is few enough to update by hand. This table
is different in kind: it is a source-by-module matrix that grows as the product of two
lists, and a hand-maintained version would be wrong within one editing session. A stale
index is worse than no index, because it makes a claim the repository cannot support.

So it is derived from the links modules actually contain. If a module does not link to a
source, that source is not listed as cited by it — the table cannot flatter the prose.

Usage
-----
    python build_reference_index.py            # rewrite the generated block
    python build_reference_index.py --check    # exit 1 if the block is out of date (CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
REFERENCES = REPO / "docs" / "REFERENCES.md"
BEGIN = "<!-- BEGIN GENERATED: reference-index -->"
END = "<!-- END GENERATED: reference-index -->"

# Only entries under these headings are sources; the page's own prose sections are not.
NON_SOURCE_SECTIONS = {"How to cite into this page", "What gets a date, and what does not",
                       "Where each source is cited"}

MODULE_DIR = re.compile(r"^(?P<id>[A-D]\d{2})\.")
LINK_TO_REFS = re.compile(r"\]\(([^)\s]*REFERENCES\.md)#([^)\s]+)\)")


def module_of(path: Path) -> str | None:
    """The module id (``A02``) that owns *path*, or None for shared docs."""
    parts = path.relative_to(REPO).parts
    if len(parts) < 3 or parts[0] != "tracks":
        return None
    match = MODULE_DIR.match(parts[2])
    return match.group("id") if match else None


def sources() -> list[tuple[str, str, str]]:
    """(section, title, anchor) for every ``###`` source entry, in page order."""
    sys.path.insert(0, str(REPO))
    from check_links import slugify  # same slug rule as the checker and the site

    found: list[tuple[str, str, str]] = []
    section = ""
    for line in REFERENCES.read_text(encoding="utf-8").splitlines():
        if line.startswith("## ") and not line.startswith("###"):
            section = line[3:].strip()
        elif line.startswith("### "):
            title = line[4:].strip()
            if section not in NON_SOURCE_SECTIONS:
                found.append((section, title, slugify(title)))
    return found


def citations() -> dict[str, set[str]]:
    """anchor -> set of module ids that link to it."""
    sys.path.insert(0, str(REPO))
    from check_links import iter_markdown, strip_code

    cited: dict[str, set[str]] = defaultdict(set)
    for path in iter_markdown(REPO):
        module = module_of(path)
        if module is None:
            continue
        body = strip_code(path.read_text(encoding="utf-8"))
        for _target, anchor in LINK_TO_REFS.findall(body):
            cited[anchor].add(module)
    return cited


def render() -> str:
    entries = sources()
    cited = citations()
    lines = ["", "| Source | Cited by |", "|---|---|"]
    uncited: list[str] = []
    for _section, title, anchor in entries:
        modules = sorted(cited.get(anchor, ()))
        if not modules:
            uncited.append(title)
            continue
        lines.append(f"| [{title}](#{anchor}) | {', '.join(modules)} |")

    if len(lines) == 3:
        lines = ["", "*No module cites a source on this page yet.*"]
    if uncited:
        lines += ["",
                  f"**Not yet cited from any module ({len(uncited)}):** " + "; ".join(uncited) +
                  ". These are here as background reading, not as support for a claim."]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 if the generated block is stale")
    args = parser.parse_args(argv)

    try:
        text = REFERENCES.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read {REFERENCES.name}: {exc}", file=sys.stderr)
        return 2

    start, stop = text.find(BEGIN), text.find(END)
    if start == -1 or stop == -1 or stop < start:
        print(f"error: {REFERENCES.name} is missing the generated block markers", file=sys.stderr)
        return 2

    updated = text[:start + len(BEGIN)] + render() + text[stop:]
    if updated == text:
        print("reference index is up to date")
        return 0
    if args.check:
        print("error: the reference index is stale; run 'python build_reference_index.py'",
              file=sys.stderr)
        return 1

    REFERENCES.write_text(updated, encoding="utf-8", newline="\n")
    print(f"rewrote the reference index in {REFERENCES.relative_to(REPO).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
