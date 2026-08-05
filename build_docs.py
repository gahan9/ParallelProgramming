#!/usr/bin/env python3
"""Stage the publishable files into a docs root that MkDocs will accept.

Why this exists
---------------
The curriculum's Markdown deliberately lives at the repository root (``README.md``,
``CURRICULUM.md``, ``tracks/``, ``docs/``, ``live/``) so that GitHub renders it in place
and every relative link works when browsing the repo. The obvious way to publish that
is ``docs_dir: .`` — but MkDocs rejects a ``docs_dir`` that is the parent of
``mkdocs.yml``, and it also rejects a ``site_dir`` nested inside ``docs_dir``:

    ERROR - Config value 'docs_dir': The 'docs_dir' should not be the parent
            directory of the config file.

So instead of contorting the repository layout for the benefit of the site generator,
this script copies just the publishable files into ``_docs/``, preserving their relative
paths, and ``mkdocs.yml`` points at that. Because the layout is preserved, every path in
the ``nav`` and every relative link inside a page keeps working unchanged.

Usage
-----
    python build_docs.py          # stage into _docs/
    mkdocs build --clean          # then build normally

Both the staging directory and the built site are gitignored.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
STAGE = REPO / "_docs"

# Extensions that make up the site: prose, the standalone planner/deck, the theme
# override, and the hand-authored infographics in assets/. Everything else in the repo
# (sources, Makefiles, requirements) is intentionally not published — the READMEs link
# to it on GitHub instead.
#
# .svg and .js are load-bearing: without them the infographics and the read-aloud script
# resolve locally but 404 on the published site.
PUBLISH_SUFFIXES = {".md", ".html", ".css", ".svg", ".js"}

# Directories never worth walking. `site` and `_docs` are build output; the rest are
# tooling or caches that would only slow the walk down.
SKIP_DIRS = {
    ".git", ".github", ".cursor", "_docs", "site",
    "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache",
    "profiling_output", "node_modules",
}


def iter_publishable(root: Path):
    """Yield every file that belongs in the site, as a path relative to *root*."""
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.suffix.lower() in PUBLISH_SUFFIXES:
            yield path.relative_to(root)


def stage(clean: bool = True) -> int:
    if clean and STAGE.exists():
        shutil.rmtree(STAGE)

    count = 0
    for rel in iter_publishable(REPO):
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dest)
        count += 1

    # A missing index is not a crash, but it does mean a broken landing page, so say so
    # loudly rather than letting MkDocs emit a confusing nav warning later.
    if not (STAGE / "README.md").exists():
        print("warning: README.md was not staged; the site will have no home page",
              file=sys.stderr)

    print(f"staged {count} files into {STAGE.relative_to(REPO)}/")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-clean", action="store_true",
                        help="add to an existing _docs/ instead of recreating it")
    args = parser.parse_args()
    return 0 if stage(clean=not args.no_clean) else 1


if __name__ == "__main__":
    raise SystemExit(main())
