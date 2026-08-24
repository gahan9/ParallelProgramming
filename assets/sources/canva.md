# Canva as an authoring layer

Canva is useful here for exactly one thing: **laying out a static composition quickly** when you
would otherwise be guessing at coordinates in a text editor. It is not a publishing surface for this
repo, and nothing in `assets/` is a Canva export as-shipped.

The committed artifact is always a hand-tuned SVG in [`../`](../README.md). If a Canva file and an
SVG ever disagree, the SVG wins, because the SVG is the one that is reviewed, diffed and published.

---

## Why not just commit the Canva export

Three reasons, all of which have already cost time somewhere in this repo:

1. **Canva cannot express the animation.** Every asset here animates in CSS, inside the SVG. Canva
   has no way to author that, so a Canva export is at best the static frame of a finished asset.
2. **Canva exports are not ASCII.** They arrive full of real `×`, `–` and `→` characters, plus
   `<tspan>` soup and embedded font subsets. SVGs in this repo are pure ASCII by policy — the
   encoding trap in [`../../AGENTS.md`](../../AGENTS.md) is not theoretical, it has produced invalid
   XML here before.
3. **A `.canva` link is not a source.** It lives in someone's account, behind a login, and cannot be
   reviewed in a pull request. So the link is documented and the SVG is committed, never the reverse.

---

## Brand setup

Canva's brand kit takes the tokens straight from [`../../docs/BRAND.md`](../../docs/BRAND.md).
Enter these and nothing else — the "do not invent new colours" rule applies to Canva too:

| Token | Hex | Role |
|---|---|---|
| `brand-primary` | `#0891B2` | Active lanes, data movement, Track A |
| `brand-accent` | `#6366F1` | Compute ceilings, Track B |
| `brand-highlight` | `#F59E0B` | Waste, cost, attention, Track C |
| `brand-trackd` | `#DB2777` | Track D |
| `brand-success` | `#10B981` | Verified, free, "start here" |
| `brand-deep` | `#0F172A` | Ground |
| `brand-surface` | `#F8FAFC` | Headings |
| `brand-muted` | `#64748B` | Captions, ticks |
| `vendor-amd` | `#ED1C24` | AMD pill only |
| `vendor-nvidia` | `#76B900` | NVIDIA pill only |

Fonts: **Inter** for prose, **JetBrains Mono** for every number. Both are on Canva's font list; if a
brand plan is not available, Canva's default sans will substitute and the export will need its font
stack rewritten by hand anyway.

## The lane-grid master

The signature device is described in [`../README.md`](../README.md#the-signature-the-lane-grid): a
square is a lane, and the fill state always encodes something true. To rebuild it in Canva:

- Square, 20 x 20 px, corner radius 2, no stroke.
- Horizontal pitch 28 px (so an 8 px gutter).
- A row is a wavefront: **64 squares for AMD CDNA, 32 for an NVIDIA warp**. Do not draw a 16-lane
  row and call it a wavefront; use 16 only when the graphic says it is showing a subset.
- Fill states, and their fixed meanings:
  - solid `#0891B2` — lane is active this cycle
  - `#172033` with a `#223049` hairline — lane is present but masked off
  - no fill, `#334155` dashed outline — this path does not execute at all
  - `#7C3A0B` to `#F59E0B` — bytes fetched and discarded

Keeping those meanings fixed is the entire value of the device: a reader who learns it on one asset
reads the next one for free.

**Template link:** _not yet created._ When one exists, put the share URL here with the date it was
made, and note which assets were laid out with it. A link with no provenance is worse than no link.

---

## Export checklist

Run through all of this before an export becomes a committed asset. It is the same bar every other
file in `assets/` had to clear.

- [ ] **Flatten the text.** Replace Canva's `<tspan>` stacks with single `<text>` elements and a
      `class`. The stylesheet at the top of each existing asset is the pattern to copy.
- [ ] **Strip embedded fonts.** Replace with the repo's stacks:
      `Inter, "Segoe UI", system-ui, sans-serif` and
      `"JetBrains Mono", Consolas, ui-monospace, monospace`.
- [ ] **Make it ASCII.** Convert every non-ASCII character to a numeric entity: `&#183;` for the
      middot, `&#215;` for the multiplication sign, `&#8594;` for the arrow, `&#8212;` for the em
      dash. Then verify — do not assume:
      ```powershell
      $b = [System.IO.File]::ReadAllBytes("assets/new-asset.svg")
      $b | Where-Object { $_ -notin 9,10,13 -and ($_ -lt 32 -or $_ -gt 126) }
      ```
      That must print nothing.
- [ ] **Confirm it is well-formed XML.** `(New-Object System.Xml.XmlDocument).Load(<path>)`.
- [ ] **Add the accessibility block.** `role="img"`, `aria-labelledby="t d"`, and a `<title>` plus a
      `<desc>` that lets someone who cannot see the image reach the same conclusion.
- [ ] **Add the `brief:` comment header.** id, module, which of the four jobs it does, what the
      motion claims, and whether the numbers are measured, derived or representative.
- [ ] **Author the animation by hand,** in a `<style>` block, with a
      `@media (prefers-reduced-motion: reduce)` escape and a static state that still teaches.
- [ ] **Give it its own dark ground** (`#0F172A`) rather than relying on the page behind it.
- [ ] **Label the source of every number** on the graphic itself.
- [ ] **Leave slack around text.** Inter and JetBrains Mono are not on every machine, so fallback
      metrics shift; never fit a label tightly to a box.
- [ ] Run `python check_links.py` and `python build_docs.py` after wiring it into a module.

If that list looks long relative to the time Canva saved, that is the honest signal: for most assets
here, writing the SVG directly is simply faster.
