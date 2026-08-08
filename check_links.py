#!/usr/bin/env python3
"""Resolve every cross-reference in the repository and fail on the ones that are broken.

Why this exists
---------------
``mkdocs.yml`` sets ``validation.links.not_found: ignore`` on purpose: module READMEs link
to their own sources (``hip/vector_add.cpp``) which are deliberately never published, and
leaving that category loud would bury every warning that matters under hundreds that do
not. The cost is that MkDocs no longer reports *any* broken relative link, so a typo in a
cross-reference ships silently. This script is the replacement guard, and it checks two
things MkDocs cannot:

* **Links to unpublished files.** ``hip/transpose.cpp`` is not in the site, but it is on
  disk, so a typo in its path is still detectable here.
* **Anchors under the rule both surfaces actually use.** GitHub emits one hyphen per
  space; stock Python-Markdown collapses a run of spaces into one. ``mkdocs.yml`` pins
  ``pymdownx.slugs.slugify``, which does not collapse, so the site and GitHub now agree —
  and :func:`slugify` below reproduces that shared rule. Keep the three in step.

Usage
-----
    python check_links.py                 # relative links + anchors (what CI runs)
    python check_links.py --external      # also HTTP-check every external URL
    python check_links.py --external-only # skip local checks, only reach out

Exits 0 when clean, 1 when something is broken, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

REPO = Path(__file__).resolve().parent

# Build output and tooling. `_docs` and `site` are copies of what we are already
# checking, so walking them would double every finding.
SKIP_DIRS = {
    ".git", ".github", "_docs", "site", "__pycache__",
    ".venv", "venv", ".mypy_cache", ".pytest_cache", "node_modules",
}

# Fenced blocks and inline code hold example markdown that must not be resolved.
FENCE = re.compile(r"^(?P<f>```|~~~).*?^(?P=f)\s*$", re.S | re.M)
INLINE_CODE = re.compile(r"`[^`\n]*`")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

MD_LINK = re.compile(r"!?\[(?P<text>(?:[^\]\[]|\[[^\]]*\])*)\]\((?P<href><[^>]+>|[^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
HTML_HREF = re.compile(r"<a\b[^>]*\bhref=[\"'](?P<href>[^\"']+)[\"']", re.I)
HTML_SRC = re.compile(r"<img\b[^>]*\bsrc=[\"'](?P<href>[^\"']+)[\"']", re.I)
REF_DEF = re.compile(r"^\s{0,3}\[(?P<label>[^\]]+)\]:\s*(?P<href>\S+)", re.M)

ATX_HEADING = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*#*$", re.M)
SETEXT_HEADING = re.compile(r"^(?P<text>[^\n]+)\n(?:=+|-{2,})\s*$", re.M)
EXPLICIT_ID = re.compile(r"(?:\bid|\bname)=[\"'](?P<id>[^\"']+)[\"']")
ATTR_LIST_ID = re.compile(r"\{[^}\n]*#(?P<id>[-\w]+)[^}\n]*\}\s*$")

# Mirrors pymdownx.slugs.slugify(case='lower'), which is also GitHub's rule.
INVALID_SLUG_CHAR = re.compile(r"[^\w\- ]", re.UNICODE)
STRIP_TAGS = re.compile(r"</?[^>]*>")
# `_` is deliberately absent: it is a word character that both slugifiers keep, so
# `tools/capacity_autoscale.py` anchors as `toolscapacity_autoscalepy`. No heading in the
# repository uses `_underscore_` emphasis, so nothing is lost by leaving it alone.
STRIP_FORMATTING = re.compile(r"[*~`]")
STRIP_MD_LINK = re.compile(r"!?\[(?P<text>[^\]]*)\]\([^)]*\)")


def slugify(text: str) -> str:
    """Heading text -> anchor id, matching GitHub and the site's pinned slugifier."""
    slug = unicodedata.normalize("NFC", text)
    slug = STRIP_MD_LINK.sub(lambda m: m.group("text"), slug)
    slug = STRIP_TAGS.sub("", slug)
    slug = STRIP_FORMATTING.sub("", slug).strip().lower()
    return INVALID_SLUG_CHAR.sub("", slug).replace(" ", "-")


def _blank(match: re.Match[str]) -> str:
    """Replace a span with spaces so line and column numbers survive."""
    return re.sub(r"[^\n]", " ", match.group(0))


def strip_fences(body: str) -> str:
    """Blank fenced blocks and HTML comments. Inline code survives, because a heading
    like ``### Reading the `rocprofv3` output`` slugs with the code text included."""
    return HTML_COMMENT.sub(_blank, FENCE.sub(_blank, body))


def strip_code(body: str) -> str:
    """As :func:`strip_fences`, and also blank inline code, so that a link written as an
    example inside backticks is not mistaken for one the reader can follow."""
    return INLINE_CODE.sub(_blank, strip_fences(body))


def read(path: Path) -> str:
    """Read as UTF-8, refusing to guess. A mangled file is a finding, not a fallback."""
    return path.read_text(encoding="utf-8")


def anchors_of(path: Path) -> set[str]:
    """Every fragment that resolves on *path*: headings plus explicit ids."""
    if path.suffix.lower() not in {".md", ".markdown", ".html", ".htm"}:
        return set()
    try:
        raw = read(path)
    except (OSError, UnicodeDecodeError):
        return set()

    if path.suffix.lower() in {".html", ".htm"}:
        return set(EXPLICIT_ID.findall(raw))

    body = strip_fences(raw)
    found: set[str] = set(EXPLICIT_ID.findall(raw))
    for match in ATX_HEADING.finditer(body):
        text = match.group("text")
        explicit = ATTR_LIST_ID.search(text)
        if explicit:
            found.add(explicit.group("id"))
            text = ATTR_LIST_ID.sub("", text)
        found.add(slugify(text))
    for match in SETEXT_HEADING.finditer(body):
        found.add(slugify(match.group("text")))
    found.discard("")
    return found


def iter_markdown(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.md")):
        if not any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            yield path


MD_ESCAPE = re.compile(r"\\([!-/:-@\[-`{-~])")


def iter_hrefs(body: str) -> Iterator[tuple[int, str, str]]:
    """Yield (line number, link text, href) for every link on the page."""
    for pattern, group in ((MD_LINK, "text"), (HTML_HREF, None), (HTML_SRC, None), (REF_DEF, "label")):
        for match in pattern.finditer(body):
            href = match.group("href").strip()
            if href.startswith("<") and href.endswith(">"):
                href = href[1:-1].strip()
            # A destination may escape the parentheses that would otherwise close it,
            # as the Hot Chips deck's `...ASmith\(MI300X\).pdf` does. Renderers unescape
            # before fetching, so resolve against the unescaped form.
            href = MD_ESCAPE.sub(r"\1", href)
            line = body.count("\n", 0, match.start()) + 1
            text = match.group(group).strip() if group else ""
            yield line, text, href


class Problem:
    __slots__ = ("source", "line", "href", "reason", "hint")

    def __init__(self, source: Path, line: int, href: str, reason: str, hint: str = "") -> None:
        self.source, self.line, self.href, self.reason, self.hint = source, line, href, reason, hint

    def __str__(self) -> str:
        where = f"{self.source.relative_to(REPO).as_posix()}:{self.line}"
        tail = f"  (did you mean '{self.hint}'?)" if self.hint else ""
        return f"{where}: {self.reason}: {self.href}{tail}"


def resolve(source: Path, path_part: str) -> Path | None:
    """Resolve a relative link target, following a directory to its index page."""
    target = (source.parent / path_part).resolve()
    if target.is_dir():
        for index in ("README.md", "index.md", "index.html"):
            if (target / index).exists():
                return target / index
        return target
    return target if target.exists() else None


def check_local(paths: Iterable[Path]) -> tuple[list[Problem], int]:
    problems: list[Problem] = []
    anchor_cache: dict[Path, set[str]] = {}
    checked = 0

    for source in paths:
        try:
            body = strip_code(read(source))
        except UnicodeDecodeError as exc:
            problems.append(Problem(source, 1, source.name, f"not valid UTF-8 ({exc.reason})"))
            continue

        for line, _text, href in iter_hrefs(body):
            split = urlsplit(href)
            if split.scheme or href.startswith("//"):
                continue  # external or mailto; handled by --external
            if href.startswith("#"):
                path_part, fragment = "", href[1:]
            else:
                path_part, _, fragment = href.partition("#")
            path_part = unquote(path_part)
            checked += 1

            target = source
            if path_part:
                if path_part.startswith("/"):
                    problems.append(Problem(source, line, href, "absolute path will not resolve on GitHub"))
                    continue
                resolved = resolve(source, path_part)
                if resolved is None:
                    problems.append(Problem(source, line, href, "target does not exist"))
                    continue
                target = resolved

            if not fragment:
                continue
            if target not in anchor_cache:
                anchor_cache[target] = anchors_of(target)
            available = anchor_cache[target]
            wanted = unquote(fragment)
            if wanted in available:
                continue
            close = difflib.get_close_matches(wanted, available, n=1, cutoff=0.7)
            problems.append(Problem(source, line, href, "anchor not found", close[0] if close else ""))

    return problems, checked


def check_external(paths: Iterable[Path], timeout: float) -> tuple[list[Problem], int]:
    """HTTP-check each distinct external URL once. Opt-in: hosts go down for their own reasons."""
    import urllib.error
    import urllib.request

    seen: dict[str, list[tuple[Path, int]]] = {}
    for source in paths:
        try:
            body = strip_code(read(source))
        except UnicodeDecodeError:
            continue
        for line, _text, href in iter_hrefs(body):
            if urlsplit(href).scheme in {"http", "https"}:
                seen.setdefault(href.rstrip(">"), []).append((source, line))
        for match in re.finditer(r"<(https?://[^>\s]+)>", body):
            seen.setdefault(match.group(1), []).append((source, body.count("\n", 0, match.start()) + 1))

    problems: list[Problem] = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ParallelProgramming-link-check/1.0)"}
    for url in sorted(seen):
        status = ""
        for method in ("HEAD", "GET"):
            request = urllib.request.Request(url, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = "" if response.status < 400 else f"HTTP {response.status}"
                break
            except urllib.error.HTTPError as exc:
                status = f"HTTP {exc.code}"
                if exc.code in {403, 405, 501} and method == "HEAD":
                    continue  # some hosts refuse HEAD; retry as GET
                break
            except Exception as exc:  # noqa: BLE001 - any transport failure is a finding
                status = f"{type(exc).__name__}: {exc}"
                break
        if status:
            source, line = seen[url][0]
            extra = f" (+{len(seen[url]) - 1} more)" if len(seen[url]) > 1 else ""
            problems.append(Problem(source, line, url, f"unreachable{extra} - {status}"))
        print(f"  {'ok  ' if not status else 'FAIL'} {url}", file=sys.stderr)

    return problems, len(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check every relative link, anchor and (optionally) external URL.")
    parser.add_argument("--external", action="store_true", help="also HTTP-check external URLs")
    parser.add_argument("--external-only", action="store_true", help="skip the local checks")
    parser.add_argument("--timeout", type=float, default=20.0, help="external request timeout (s)")
    args = parser.parse_args(argv)

    if args.timeout <= 0:
        print("error: --timeout must be positive", file=sys.stderr)
        return 2

    paths = list(iter_markdown(REPO))
    if not paths:
        print("error: no markdown found; run this from the repository", file=sys.stderr)
        return 2

    problems: list[Problem] = []
    if not args.external_only:
        local, checked = check_local(paths)
        problems += local
        print(f"checked {checked} relative links across {len(paths)} files")
    if args.external or args.external_only:
        external, count = check_external(paths, args.timeout)
        problems += external
        print(f"checked {count} distinct external URLs")

    if not problems:
        print("all links resolve")
        return 0

    print(f"\n{len(problems)} broken link(s):", file=sys.stderr)
    for problem in sorted(problems, key=lambda p: (str(p.source), p.line)):
        print(f"  {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
