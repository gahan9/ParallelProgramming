# Infographics — the visual system

Hand-authored SVG for the things a flowchart cannot say. The repo has 238 Mermaid diagrams and they
carry most of the explanatory load; these assets exist for the four jobs Mermaid genuinely cannot do:

| Job | Why Mermaid can't | Asset |
|---|---|---|
| Show a **proportion** honestly | No scale — every box is the same size | [`where-time-goes.svg`](where-time-goes.svg) |
| Draw a **curve** | No plotting, no log axes | [`roofline.svg`](roofline.svg) |
| Show something happening **in sequence** | Static; a graph of two branches can't say "one after the other" | [`divergence.svg`](divergence.svg) |
| Carry a visual **identity** | Themeable, but not designed | [`hero-lanes.svg`](hero-lanes.svg) |

If a diagram is boxes and arrows, use Mermaid — it is editable in place, diffs cleanly, and the brand
init block in [`../docs/BRAND.md`](../docs/BRAND.md) already themes it. Reach for SVG only when you
hit one of the four rows above.

There is a third surface, above these: if the reader needs to **change an input and watch the answer
move**, that is a concept player in [`../live/concepts/`](../live/README.md), not an asset. The test
is whether the geometry genuinely changes with the input. It does for access stride, which is why
`coalescing.svg` has an interactive companion; it does not for the divergence panels, which is why
that one does not. The SVG stays canonical either way — it is what the module embeds and what
survives JavaScript being off.

[`sources/canva.md`](sources/canva.md) covers laying an asset out in Canva and the checklist an
export must clear before it can be committed.

---

## Why SVG and not GIF

The brief was "editable". A GIF is a binary blob: you cannot diff it, you cannot fix a typo in it, and
regenerating it requires whatever tool made it. These files are **plain text you can edit in the same
editor as the code**, they version-control properly, and the animation is CSS — so changing a timing
means changing one number.

They also stay sharp at any zoom, weigh a few kilobytes, and need no build step.

---

## The signature: the lane grid

Every asset shares one device — **a square is a lane**. A block of them is a wavefront (64 lanes on
AMD CDNA) or a warp (32 on NVIDIA). The fill state always encodes something true, never decoration:

| Fill | Means | Used in |
|---|---|---|
| Solid cyan | Lane is active this cycle | hero, coalescing |
| Dim slate with a hairline | Lane is **masked off** — present but not executing | divergence |
| Dashed outline, no fill | Path **not executed at all** (branched over) | divergence |
| Amber | Bytes fetched and **discarded** | coalescing |

Because the device recurs, a reader who learns it in the hero already understands the divergence
graphic. That is the point of having a system rather than a folder of pictures. If you add an asset and
the lane grid does not fit the content, don't force it — but don't invent a second unrelated device
either.

## Ground, not theme

Every asset carries its **own dark slate ground** (`#0F172A`) rather than trying to sit transparently
on the page. This is deliberate:

- GitHub's light/dark setting is a *page* preference, but `prefers-color-scheme` inside an SVG follows
  the *operating system*. Those disagree constantly, so a "theme-aware" asset inverts wrongly for a
  large share of readers.
- A self-contained dark card reads as an instrument panel, which suits the subject and matches the
  dark deck variant already specified in `BRAND.md`.

So: no `<picture>` duplication, no `#gh-dark-mode-only` hacks, and identical rendering everywhere.

## Tokens

Straight from [`../docs/BRAND.md`](../docs/BRAND.md) — do not introduce new colours.

| Token | Hex | Role in these assets |
|---|---|---|
| `brand-primary` | `#0891B2` | Active lanes, data movement, Track A |
| `brand-accent` | `#6366F1` | Roofline ceilings, compute, Track B |
| `brand-highlight` | `#F59E0B` | Waste, cost, attention, Track C |
| `brand-success` | `#10B981` | Verified, free, "start here" |
| `brand-deep` | `#0F172A` | Ground |
| `brand-surface` | `#F8FAFC` | Headings |
| `brand-muted` | `#64748B` | Captions, ticks |
| `vendor-amd` | `#ED1C24` | AMD pill only |
| `vendor-nvidia` | `#76B900` | NVIDIA pill only |

Type: **Inter / Segoe UI / system-ui** for prose, **JetBrains Mono / Consolas** for every number.
Numbers get the monospace treatment because measured numbers are this curriculum's whole credibility
argument — they should look like readings, not like copy.

---

## Rules for adding one

1. **It must say something a Mermaid diagram can't.** Otherwise use Mermaid.
2. **Animate only what teaches.** Motion earns its place when sequence or waste *is* the lesson
   (`divergence`, `coalescing`). It is decoration anywhere else. Every animation ships with:
   ```css
   @media (prefers-reduced-motion: reduce) { .thing { animation: none } }
   ```
   and the static state must still be readable — a reader who never sees the motion must not lose
   information.
3. **Never distort scale silently.** `where-time-goes.svg` draws the kernel tick four times wider than
   true scale so it is visible at all, and *says so on the graphic*. A chart that quietly exaggerates
   the segment it is arguing about is worthless.
4. **Label the source of every number.** Measured, derived, or representative — say which, on the
   asset. `roofline.svg` states that its ceilings are representative and points at B02's lab for real ones.
5. **Accessibility:** `role="img"` plus `<title>` and `<desc>`, referenced by `aria-labelledby`. The
   `<desc>` should let someone who cannot see the image reach the same conclusion.
6. **Text tolerance.** Inter and JetBrains Mono will not be present on every machine, so fallback
   metrics shift. Leave slack around text; never fit a label tightly to a box.

## Inventory

| Asset | Module | Motion | Notes |
|---|---|---|---|
| `hero-lanes.svg` | — | Lockstep sweep | Wordmark + signature device |
| `curriculum-map.svg` | — | Start beacon | Row length encodes module count |
| `where-time-goes.svg` | A01 | Kernel tick | Real `rocprofv3` run, gfx950 |
| `coalescing.svg` | A02 | Discarded bytes | 8 lanes ? 1 line vs 8 lines |
| `divergence.svg` | A03 | THEN/ELSE antiphase | Cost is the sum, not the max |
| `reduction-tree.svg` | A04 | Four rounds in turn | Lane grid collapsing; 15 of 64 lane-slots do work |
| `tile-reuse.svg` | A06 | Eliminated span | 549.8 GB vs 17.2 GB, drawn to scale |
| `attention-traffic.svg` | A09 | Four HBM crossings | 132 MiB vs 4 MiB per head; the skipped span is dashed |
| `roofline.svg` | B02 | Vector-add pulse | Log-log, 5 kernels plotted |
| `prefill-decode-split.svg` | B05 | Decode point pulse | Log-log; intensity is batch size until it saturates |
| `scaling-walls.svg` | D01 | Turning-point pulse | Log-log; Amdahl ceilings vs the communication downturn |
| `ring-allreduce.svg` | D03, B07 | Three-panel phase cue | Start, after reduce-scatter, after all-gather |
| `fabric-tiers.svg` | D04 | HBM breathe | Log bandwidth axis; the on-package-to-wire cliff |
| `scale-boundary.svg` | D06 | Failure-band creep | Which constraint binds at each decade of rank count |

Each asset carries a `brief:` block in its opening comment — id, module, which of the four jobs it
does, what the motion claims, and whether its numbers are measured, derived or representative. It
lives in the file rather than in a sidecar because the brief is only ever read while editing the
asset, and a second file would be one more thing to leave stale.

Candidates worth adding, all with something Mermaid can't express: the KV-cache growth curve against
batch and context (B06), and the pipeline-bubble timeline that shows why interleaved schedules beat
naive ones (B07).

## If you need a GIF

You do not, and this repo will not commit one — see [Why SVG and not GIF](#why-svg-and-not-gif)
above. These files are already animated, so an animated SVG *is* the editable GIF.

When an external surface genuinely cannot take SVG (PowerPoint, some social embeds), **derive one,
never commit it**. The SVG stays the single source of truth, and
[`../scripts/export_visual.sh`](../scripts/export_visual.sh) does the derivation:

```bash
bash scripts/export_visual.sh coalescing            # 16:9 GIF into exports/
bash scripts/export_visual.sh divergence mp4        # MP4 for a video editor
bash scripts/export_visual.sh reduction-tree gif 9x16   # vertical, for a Short
```

`exports/` is gitignored. The script needs Chrome and ffmpeg, neither of which is a repo
dependency; it checks for both and exits with a one-line message rather than failing halfway. It is
not part of any build, and nothing depends on its output.

For a talk or a screen recording you usually do not need it at all: open the SVG in a browser tab
and capture that. The export exists for the cases where a file has to be handed over.

Committing generated GIFs would put unreviewable binaries in the tree and add a second artifact to
regenerate on every content edit. If that ever changes, it needs **two** edits or it resolves on
GitHub and 404s on the site: add `.gif` to `PUBLISH_SUFFIXES` in `../build_docs.py` *and* add a glob
to `.github/workflows/pages.yml`.

## Publishing notes

`../build_docs.py` stages `.svg` into the docs root, so these resolve on the published site. Markdown
references use paths relative to the referencing file, which keeps them working in three places at
once: the GitHub repo view, the MkDocs site, and a local clone.
