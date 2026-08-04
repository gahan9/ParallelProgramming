# Agent notes

Continuity notes for an AI agent picking this repo up. Read this before touching anything; it
records the state, the non-negotiables, and the traps that have already cost real time.

Human-facing docs: [README.md](README.md) (catalog), [CURRICULUM.md](CURRICULUM.md) (module index and
progress tracker), [docs/MODULE_TEMPLATE.md](docs/MODULE_TEMPLATE.md) (the content contract).

---

## What this is

A GPU + ML teaching curriculum: 24 modules across three tracks, dual-vendor (AMD ROCm/HIP and
NVIDIA CUDA) with a portable Triton path. Every module explains a concept twice - a plain-language
analogy, then expert depth - with runnable labs, cited sources, and an honest tradeoffs section.

## Current state (as of 2026-08-04)

| Thing | State |
|---|---|
| Modules | **24 / 24 authored.** A01-A10, B01-B07, C01-C07. All 9 sections, labs, exercises, solutions |
| Docs site | Published, `--strict`, warning-free: <https://gahan9.github.io/ParallelProgramming/> |
| Live pages | `live/` - study planner, 49 cited recall cards, 14-slide deck. Self-contained HTML |
| Infographics | `assets/` - 6 hand-authored SVGs for what Mermaid cannot express |
| Content studio | `content/` - video/Shorts scripts. Packs exist for A01 and A02 only |
| Cursor canvas | `canvases/curriculum-dashboard.canvas.tsx` (outside the repo, in the Cursor project dir) |
| Git | `main` clean and pushed; every workflow run green |

## Layout

```
README.md CURRICULUM.md          catalog + module index/tracker
AGENTS.md                        this file
build_docs.py                    stages the publishable files into _docs/ (see Traps)
mkdocs.yml                       site config; nav lists all 24 modules
docs/                            SETUP, MODULE_TEMPLATE, BRAND, GLOSSARY, REFERENCES
tracks/A-gpu-programming/        A01-A10 (+ legacy 01/02/03 raw examples, superseded by A01)
tracks/B-gpu-ml-performance/     B01-B07
tracks/C-ml-system-design/       C01-C07 (design track: no GPU code, stdlib-only Python tools)
assets/                          SVG infographics + their system doc
live/                            dashboard.html, flashcards.html, slides.html
content/                         video scripts + brand overlay spec
.cursor/skills/parallel-programming-brand/   brand skill (load when authoring visuals)
```

---

## Non-negotiables

1. **Never publish a number the repo cannot reproduce.** This is the spine of the whole project.
   Benchmark tables are omitted in favour of expected *ordering* and traffic ratios (which follow
   from the algorithm, not the silicon). Analytic tools print a banner marking output as a model, not
   a measurement. Published figures are attributed to their paper *and its conditions*. Hardware
   specs carry a source link and a retrieval date.
2. **Nine sections, exact headings**, per `docs/MODULE_TEMPLATE.md`. Section 1 opens with a
   load-bearing layman analogy and says where the analogy breaks. Section 9 puts conceptual answers
   inside a `<details><summary>Answers</summary>` block.
3. **Cite or hedge.** Verify version-sensitive claims (hardware specs, tool flags, library APIs) with
   a web search and stamp the retrieval date. If you cannot verify it, say so in the text rather than
   asserting it. Never invent a citation, a DOI, or a benchmark.
4. **Dual-vendor.** Where AMD and NVIDIA differ, show both (wavefront 64 vs warp 32). NVIDIA-only
   material is marked optional and names its AMD counterpart.
5. **Error checking in every GPU sample** - `HIP_CHECK` / `CUDA_CHECK` on every runtime call, a
   launch-error check after every launch, bounds guards, and verification against a CPU reference.
6. **Brand tokens only**, from `docs/BRAND.md`. Every Mermaid diagram starts with the init block
   from that file. Do not introduce new colours.
7. **Python tools in Tracks B and C are standard-library only** - no numpy, no torch. Verify with
   `python -S -I <tool>`. They must exit non-zero with a one-line message on bad input, never a
   traceback.

---

## Verify before you claim

```powershell
# Docs site (this is what CI runs; --strict means a warning fails the deploy)
python build_docs.py
python -m mkdocs build --clean --strict --site-dir ./site

# Inline JS in the live pages - a syntax error leaves a page silently dead
node --check <extracted script>

# SVGs must be well-formed XML and pure ASCII
# (XmlDocument.LoadXml + a scan for [^\x09\x0A\x0D\x20-\x7E])

# Any file you wrote with typography: confirm it is really UTF-8 (see Traps #6)
```

---

## Traps that have already cost time

1. **Auth.** `GITHUB_TOKEN` is set in the environment to an AMD enterprise token
   (`gsaraiya_amdeng`) which has **pull-only** access to this repo, and it *masks* the `gahan9`
   keyring account. Any `git push` or `gh` call fails with 403 / "Not Found" until you clear it:
   ```powershell
   $env:GITHUB_TOKEN = ""      # then gh resolves to gahan9 (admin, push)
   ```
2. **`docs_dir` cannot be `.`** - MkDocs 1.6 hard-rejects a docs_dir that is the parent of
   `mkdocs.yml`, and rejects a `site_dir` nested inside it. The curriculum's Markdown must stay at
   the repo root so GitHub renders it with working relative links, so `build_docs.py` stages the
   publishable files into a gitignored `_docs/`. **Always run it before `mkdocs build`.**
3. **`validation:` uses the nested schema** (`links:` / `nav:`). The flat pre-1.5 form is silently
   ignored, which once buried hundreds of real warnings under link noise.
4. **New asset types need two edits** or they resolve on GitHub and 404 on the site: add the suffix
   to `PUBLISH_SUFFIXES` in `build_docs.py`, *and* add a path glob to
   `.github/workflows/pages.yml` (it only triggers on listed paths).
5. **PowerShell, not bash.** No heredocs, no `&&`. For commit messages, write the message to a file
   with `[System.IO.File]::WriteAllText(path, msg, (New-Object System.Text.UTF8Encoding $false))`
   then `git commit -F`. `Set-Content -Encoding utf8` adds a BOM that ends up in the commit subject.
   Git operations here take 15-40s; budget for it.
6. **Writing non-ASCII can silently mangle it.** Em dashes and arrows have landed as Windows-1252 or
   U+FFFD, which made six SVGs invalid XML and broke a `--strict` build on one Markdown file. After
   writing any file with typography, verify it decodes as UTF-8. **SVGs are pure ASCII by policy** -
   they use XML numeric entities (`&#183;`, `&#8594;`, `&#215;`). Keep them that way.
7. **Module data is duplicated in five places.** Update all of them together: `CURRICULUM.md`
   (tables + totals), `live/dashboard.html` (`M`, `PATHS`), `live/flashcards.html` (`CARDS`),
   `live/slides.html` (headline counts), and the Cursor canvas. Each surface must stand alone with no
   shared runtime; five is few enough to do by hand and a generator is not yet earned.

---

## Do not claim this is verified

**No GPU code in this repo has ever been compiled or run.** There is no ROCm or CUDA toolchain on
this machine. Every `.cpp`, `.cu`, and Triton kernel is hand-reviewed only. The Python tools and the
docs build *are* verified. Be explicit about that boundary in any summary.

Known likely first-build friction:
- HIP's `warpSize` stopped being `constexpr` in ROCm 7.0, and `__AMDGCN_WAVEFRONT_SIZE__` is
  deprecated (A04 documents this).
- B04's optional native-fp8 cross-check is behind `__has_include` with a `-DB04_NO_NATIVE_FP8`
  escape hatch; it is the most likely place a build breaks.
- Makefiles are Linux-targeted (`rm -rf`); `make` is not installed here, so no recipe has been run.

## Domain facts worth not re-deriving

- `rocprofv3` now defaults to **`rocpd` SQLite**, not CSV. Every `profile-hip` target passes
  `--output-format csv` explicitly for this reason.
- CDNA 3 (`gfx942`) implements **FNUZ** fp8 (bias 8, max 240); Hopper implements **OCP** (bias 7,
  max 448). The same bits decode 2x apart, and `0x80` is NaN under FNUZ but -0.0 under OCP.
- AMD's own CDNA 3 whitepaper (17.2 TB/s) and Hot Chips 2024 slides (14.7 TB/s) **disagree** on
  Infinity Cache bandwidth. B01 reports both rather than picking one; do not "fix" this.
- Omniperf/Omnitrace were renamed in ROCm 6.3 (`rocprof-compute`, `rocprof-sys-*`).

---

## Open work, roughly by value

1. **Build-and-run pass on a real GPU.** The highest-value next step: it converts every
   expected-shape table into a measured one and unblocks item 2.
2. **Content packs** for A04, A06, A07, B02 - each needs a real measured number to lead with, which
   is why they are blocked on item 1. `content/README.md` names them.
3. **More infographics.** `assets/README.md` lists the four strongest candidates: the reduction tree
   collapsing over log2(n) rounds (A04), tile reuse (A06), naive-vs-fused attention traffic (A09),
   and the prefill/decode intensity split (B05). Rule: only if Mermaid genuinely cannot express it.
4. **Bump Node 20 actions.** CI warns that `checkout@v4`, `configure-pages@v5`, `setup-python@v5`,
   and `upload-artifact@v4` are being forced onto Node 24. Nothing is broken; it is upkeep.
5. **Legacy cleanup.** `tracks/A-gpu-programming/01.hello_world/Makefile` still calls bare
   `rocprofv3` instead of `$(ROCPROFV3)`, so it fails when ROCm is not on `PATH` - the bug fixed
   everywhere else. Kept as raw reference material, superseded by A01.

---

## Session log

Rough effort record, kept because the operator asks for usage tracking.

| Date | Work |
|---|---|
| 2026-07-18 | Scaffold, shared docs, Module A01 as the gold standard; legacy folders migrated under Track A; brand system |
| 2026-07-19 | A02; A01 depth pass |
| 2026-07-29 | Modules A03-A10, B01-B07, C01-C07 authored via ~19 parallel subagents (~68k lines); content studio; canvas planner; wire-up. One subagent died of resource exhaustion at 19 concurrent - keep waves smaller |
| 2026-08-02 | Fixed the Pages deploy: enabled Pages (never switched on - the real cause of the 404), staged docs root, repaired `validation:`, turned on `--strict`. First successful publish |
| 2026-08-04 | `assets/` infographic system (6 SVGs) + encoding repairs; `live/flashcards.html` (49 cited cards, three-box Leitner) and shared nav across the three live pages |
