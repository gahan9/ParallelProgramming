# Medium starter plan — Parallel Spectrum

How the curriculum becomes a Medium publication. Source packs live in
[`../social/`](../social/) (feed-length) and [`../A0x-*/`](../) (video studio); this plan is the
**long-form text** channel of the same measured stories.

Brand: [`../../docs/BRAND.md`](../../docs/BRAND.md) · Social chain: [`../social/HISTORY.md`](../social/HISTORY.md)

---

## 1. Positioning

**Publication name:** Parallel Spectrum
**Tagline:** *Measured GPU engineering — kernels, bandwidth, and the honest path between them.*
**Angle that differentiates:** every post leads with a **real profiled number** and ends with an
**honest assessment** (what broke, when *not* to use it, what to verify). No hype, no vanity benchmarks.

One publication (not scattered profile posts) so the series accumulates followers and a table of contents.

## 2. Setup checklist (one-time)

- [ ] Create Medium **publication** named `Parallel Spectrum` (not just profile posts)
- [ ] Avatar/header: slate `#0F172A` + cyan `#0891B2` bar (brand, no rainbow, no vendor logos)
- [ ] Homepage layout: one **featured** story (post 001) + a "Series" section per track
- [ ] About page: 3 lines — what you measure, who it's for, link to GitHub repo
- [ ] Set publication SEO description: "GPU programming and ML performance, taught with real profiles — HIP, CUDA, Triton"
- [ ] Connect social accounts; enable newsletter on the publication

## 3. Editorial calendar — first 8 posts

Cadence: **1 post/week**. LinkedIn + Short echo within 48 h of each Medium post (same hook number).

| Wk | Social # | Module | Working title (names the outcome, not the topic) | Hook number | Source status |
|----|----------|--------|---------------------------------------------------|-------------|----------------|
| 1 | 001 | A01 | *Your GPU kernel took 44 µs. The program took 72 ms.* | kernel = 0.06% of wall clock | ✅ blog.md ready — re-run first |
| 2 | 002 | A02 | *Your GPU kernel passed. It still wasted ~85% of bandwidth* | ~700 → ~3,400 GB/s | ✅ `content/social/002-a02-coalescing-bandwidth/blog.md` ready — re-run first |
| 3 | 003 | A01 | *This kernel corrupts memory and still prints the right answer* | N=1000, block=256 → 24 threads OOB | ✅ shorts script exists; needs blog.md |
| 4 | 004 | A04 | *Nine ways to reduce: the reduction ladder's stage-1-to-stage-9 spread* | TBD — re-run ladder | ⬜ needs measurement |
| 5 | 005 | A06 | *Naive vs tiled GEMM: what shared memory actually buys* | TBD — re-run GEMM | ⬜ needs measurement |
| 6 | 006 | A07 | *The poisoned row: one `nan` and your softmax lies* | poisoned-row repro | ⬜ needs measurement |
| 7 | 007 | B02 | *The roofline knee your sweep finds by itself* | ridge-point crossover | ⬜ needs measurement |
| 8 | 008 | B03 | *Profile first, optimize second: a bottleneck triage walkthrough* | TBD | ⬜ needs measurement |

**Hard rule from the content studio:** re-run the lab on your own GPU before publishing; quote
*your* numbers and name the GPU. Unattributed numbers are worse than none.

Track mix: first 3 are Track A (cyan `#0891B2`). From week 4 alternate tracks so the publication
doesn't read as only-kernels: A → A → A → A → A → A → B → B → C (C01 *Problem framing* is the
natural week-9 palate cleanser).

## 4. Medium post template (per post)

Source = `content/social/NNN-<slug>/blog.md` **expanded** to ~800–1,500 words (7-min read).
The social stub is the skeleton; add the module's Section 2 first-principles excerpt in the middle.

```
Title       — outcome + number ("...44 µs... 72 ms"), not a topic label
Subtitle    — one line: what the reader will be able to assess after
Hero image  — Parallel Spectrum visual from the pack (visual.png); alt-text required
Kicker      — "Parallel Spectrum · Post NNN · Track X"
Sections:
  1. TL;DR (3 sentences)
  2. What we measured — phase/bandwidth table
  3. Why (first principles, excerpted from module README S2)
  4. The snippet (short fenced block; gist embed if >30 lines)
  5. Reproduce (one command block)
  6. Honest assessment — watch for / when not to / I'd verify
  7. Series links — prev/next post + module folder + repo
```

### Medium-specific mechanics

- **Mermaid does not render on Medium.** Export diagrams as PNG (brand init block intact) and
  embed as images with alt-text. GitHub versions keep live Mermaid.
- **Code:** fenced blocks with language tag for ≤30 lines; GitHub **gist embeds** for longer so
  readers can fork. Never screenshots of code.
- **Tags (max 5):** pick from `GPU Programming` `CUDA` `ROCm` `Parallel Computing` `Machine Learning`
  `Performance Engineering` — keep 2 stable across the series for follow-through.
- **Canonical:** if a version first appears on a personal blog/GitHub Pages, use Medium's
  **Import story** tool (`medium.com/p/import`) so canonical points to the original. If Medium is
  primary, publish native and link out — no import needed.
- **SEO:** set custom SEO title/description in story settings; first paragraph must work as the
  search snippet (that's where the hook number goes).
- **End card:** fixed footer — "Parallel Spectrum · Post NNN · Prev: … · Next: … · Code: …"

## 5. Cross-channel loop (each post)

```
Medium (long-form text, week N)
  ├─▶ LinkedIn post (≤48 h later) — hook number + "full write-up" link
  ├─▶ Short/Reel (same week) — 55 s cut from content/<module>/shorts.md
  └─▶ GitHub — module folder already linked; add Medium URL to social pack blog.md
```

Then: **update [`../social/HISTORY.md`](../social/HISTORY.md)** — Publish log row per channel with
real URL and date. The chain is the single source of "what went out when."

## 6. Publish-day checklist

- [ ] Lab re-run on my GPU; numbers in post match my terminal
- [ ] GPU + ROCm/CUDA version named in the post
- [ ] Hero image alt-text filled; Mermaid exported as PNG
- [ ] Title names the outcome; subtitle names the reader's takeaway
- [ ] Kicker + footer series links correct (prev/next match HISTORY)
- [ ] 5 tags set; SEO title/description set
- [ ] Code runs from the reproduce block verbatim
- [ ] After publish: HISTORY publish log + LinkedIn/Short echo scheduled

## 7. Immediate next actions

1. Re-run A01 (`make run-hip` + `make profile-hip`), confirm/adjust the 44 µs / 72 ms figures in
   [`../social/001-a01-kernel-vs-path/blog.md`](../social/001-a01-kernel-vs-path/blog.md)
2. Expand that blog.md to ~1,000 words using module A01 Section 2 (first principles)
3. Create the publication (Section 2 checklist) and import/publish post 001
4. Log the Medium URL in HISTORY; then commission pack 002 (A02) with `linkedin-shorts-maker`
