#!/usr/bin/env python3
"""Validate live/data/reference.json: every fact cited, every link resolving.

Why this exists
---------------
``check_links.py`` reads Markdown. It cannot see inside JSON, so the hrefs behind the
reference sheets are invisible to it and a renamed heading breaks them silently — the
page keeps rendering, the citation just goes nowhere. This closes that blind spot.

It enforces the one contract that file has: **every bullet carries a cite**, and every
cite points at something that actually exists. It also models GitHub's heading slugs the
same way ``check_links.py`` does, so an anchor that works here works on both surfaces.

Usage
-----
    python check_reference_data.py          # exits non-zero on any problem
    python check_reference_data.py --quiet  # only print problems
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "live" / "data" / "reference.json"
# hrefs in the file are written relative to live/cheatsheets.html, which is where the
# fetch happens, so they resolve against live/ rather than the repo root.
BASE = REPO / "live"

SECTIONS = ("tracks", "clusters", "modules", "analogies")
LINK_KEYS = ("href", "svg", "concept")


def slugify(heading: str) -> str:
    """Match GitHub's anchor generation closely enough for this repo's headings."""
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading.strip().lower()))


def headings(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {slugify(m[1]) for m in re.findall(r"^(#{1,6})\s+(.*)$", text, re.M)}


def collect_links(doc: dict) -> set[str]:
    found: set[str] = set()
    for section in SECTIONS:
        for entry in doc.get(section, []):
            found.update(entry[k] for k in LINK_KEYS if entry.get(k))
            if entry.get("cite", {}).get("href"):
                found.add(entry["cite"]["href"])
            for bullet in entry.get("bullets", []):
                if bullet.get("cite", {}).get("href"):
                    found.add(bullet["cite"]["href"])
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="only print problems")
    args = parser.parse_args()

    try:
        doc = json.loads(DATA.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"reference.json is not valid JSON: {exc}", file=sys.stderr)
        return 1

    problems: list[str] = []

    # 1. Every bullet must carry a cite. This is the contract, not a nicety: an
    #    uncited fact on a reference sheet is indistinguishable from one someone
    #    remembered wrong.
    for section in SECTIONS:
        for entry in doc.get(section, []):
            for bullet in entry.get("bullets", []):
                if not bullet.get("cite", {}).get("href"):
                    problems.append(f"UNCITED  {section}/{entry.get('id')}: {bullet.get('text', '')[:60]}")
            if section == "analogies" and not entry.get("cite", {}).get("href"):
                problems.append(f"UNCITED  analogies/{entry.get('id')}")

    # 2. Every link must resolve, anchor included.
    links = collect_links(doc)
    for href in sorted(links):
        path, _, anchor = href.partition("#")
        target = (BASE / path).resolve()
        if not target.exists():
            problems.append(f"MISSING FILE    {href}")
        elif anchor and target.suffix == ".md" and anchor not in headings(target):
            problems.append(f"MISSING ANCHOR  {href}")

    # 3. Cross-section consistency: a module may not point at a cluster that is gone,
    #    and an analogy may not name a module that does not exist.
    cluster_ids = {c["id"] for c in doc.get("clusters", [])}
    module_ids = {m["id"] for m in doc.get("modules", [])}
    for module in doc.get("modules", []):
        if module.get("cluster") and module["cluster"] not in cluster_ids:
            problems.append(f"DANGLING CLUSTER {module['id']} -> {module['cluster']}")
    for cluster in doc.get("clusters", []):
        for mid in cluster.get("modules", []):
            if mid not in module_ids:
                problems.append(f"DANGLING MODULE  {cluster['id']} -> {mid}")
    for analogy in doc.get("analogies", []):
        if analogy["id"] not in module_ids:
            problems.append(f"DANGLING MODULE  analogies/{analogy['id']}")

    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(f"\n{len(problems)} problem(s) in {DATA.relative_to(REPO)}", file=sys.stderr)
        return 1

    if not args.quiet:
        counts = ", ".join(f"{len(doc.get(s, []))} {s}" for s in SECTIONS)
        missing_break = [a["id"] for a in doc.get("analogies", []) if not a.get("breaks")]
        print(f"{counts}; {len(links)} links resolve; every bullet cited")
        if missing_break:
            # Not a failure: the module template asks Section 1 to say where the analogy
            # breaks, and these modules currently do not. Surfacing it beats hiding it.
            print(f"note: no break stated upstream for {', '.join(missing_break)} "
                  f"(see docs/MODULE_TEMPLATE.md section 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
