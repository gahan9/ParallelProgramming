---
name: linkedin-shorts-maker
description: Drafts short, crisp Parallel Spectrum social packs — LinkedIn posts, YouTube Shorts scripts, Medium/GitHub companion blogs, code-snippet CTAs, human assessment-style copy, and branded multi-panel infographic/GIF visuals. Use when the user asks for a LinkedIn post, YouTube Short, social draft, engagement post, Medium write-up, or shareable curriculum teaser.
---

# LinkedIn / Shorts Content Maker — Parallel Spectrum

Ship one **content pack** per topic: scroll-stopping social copy, a pursuable code path, and a brand-true visual. Prefer clarity over hype.

## Before you write

1. **Load brand** — read and apply [parallel-programming-brand](../parallel-programming-brand/SKILL.md). Use hex tokens from [docs/BRAND.md](../../../docs/BRAND.md) verbatim.
2. **Ground the claim** — pick one module, lab result, or pitfall from the curriculum. Prefer measured deltas over adjectives.
3. **Pick a track accent** — A `#0891B2` · B `#6366F1` · C `#F59E0B`. One track color dominates the visual; vendors only as small pills.

## Voice (human assessment, not marketing)

Write like a principal engineer reviewing a lab notebook — direct, slightly reflective, zero fluff.

| Do | Don't |
|----|-------|
| "Here's what I'd check before calling it done." | "Game-changing!! 🚀🚀" |
| "On my run: X → Y. Your numbers may differ — that's the point." | Fake exact benchmarks without a source |
| "Tradeoff: faster kernel, harder to debug." | "No downsides" |
| Short clauses. One idea per line on LinkedIn. | Essay paragraphs in the post body |
| Invite judgment: "Would you ship this?" | Engagement bait with no substance |

Assessment word bank (use sparingly, rotate): *check*, *verify*, *measure*, *I'd expect*, *watch for*, *before you ship*, *honest read*, *what broke*, *what held*.

## Deliverable pack (always produce all five)

```
Task Progress:
- [ ] 1. Hook + LinkedIn post (≤1,300 chars; prefer ~150–250 words)
- [ ] 2. YouTube Shorts script (45–60s spoken)
- [ ] 3. Companion blog stub (Medium + GitHub README-ready)
- [ ] 4. Code pursue path (repo path + 8–20 line snippet)
- [ ] 5. Infographic / multi-result visual brief (+ generate if asked)
```

### 1. LinkedIn post

Structure:

```
[HOOK — one line, concrete tension or result]

[2–4 short lines — what you did / saw]

[Assessment beat — what to verify, what broke, or the tradeoff]

[CTA — link to module / Gist / Medium + one pursuable next step]

[Hashtags — 3–5 max, track-relevant]
```

Rules: no wall of hashtags; no emoji spam (0–2 max); first line must stand alone in the feed preview.

### 2. YouTube Shorts script

```
0–3s   Visual hook (result number or wrong-vs-right frame)
3–15s  Problem in plain words
15–40s Demo / code / before→after
40–55s Assessment line (what to measure)
55–60s CTA ("full write-up in comments / description")
```

On-screen text ≤6 words per beat. Speak conversationally; no "hey guys welcome back".

### 3. Companion blog (Medium + GitHub)

Short, postable stub (~400–700 words) with:

1. **TL;DR** (3 sentences)
2. **What we measured / built**
3. **Snippet** (same as the post CTA)
4. **How to reproduce** (clone path + one command)
5. **Honest assessment** (pitfall + when *not* to use it)
6. **Links** — module README, code path, related modules

Save drafts under `content/social/NNN-<slug>/` when writing to disk (preferred for this repo):

```
content/social/
├── HISTORY.md              # chronological chain — UPDATE EVERY PACK
├── README.md
└── NNN-<slug>/
    ├── linkedin.md
    ├── shorts-script.md
    ├── blog.md
    ├── snippet.<ext>
    └── visual-brief.md
```

### Historical chain (recurrent posts — mandatory)

Every pack is a node in [`content/social/HISTORY.md`](../../../content/social/HISTORY.md):

1. Read HISTORY — note last `NNN`, `Prev`/`Next`, and series narrative.
2. Create `NNN = last + 1` folder; do not reuse numbers.
3. **Update HISTORY in the same turn:** new Chain row (newest first), set prior pack `Next`, set this pack `Prev`, extend Publish log placeholders.
4. If continuity helps, add one callback line to the prior hook in the LinkedIn post.
5. After the user publishes, fill Publish log URLs/dates (do not invent URLs).

### 4. Code pursue path

Always give:

- Repo-relative path (e.g. `tracks/A-gpu-programming/A01.../hip/`)
- One **run** command
- A **tight** snippet (teaching core only — not the whole file)
- One "stretch" exercise line for readers who want to fork

### 5. Infographic / multi-result visual

**Always** write a `visual-brief.md` (or inline brief) the designer/agent can render. Prefer **multi-panel** so the post can show progression as a carousel, GIF, or stacked image.

#### Multi-result panel recipe (3–4 frames)

| Frame | Purpose | Brand |
|-------|---------|-------|
| 1 · Setup | Problem or naive path | `brand-surface` bg, `brand-deep` text |
| 2 · Change | The one insight / fix | Track accent bar `#0891B2` / `#6366F1` / `#F59E0B` |
| 3 · Result | Before → after numbers or pass/fail | `brand-success` for pass; `brand-highlight` for watch-outs |
| 4 · Assess | One human check question | muted caption `#64748B` |

Layout rules (from brand skill):

- Dark panel: `#0F172A` → `#1E293B`; light panel: `#F8FAFC`
- Accent bar 4px in track color
- Footer: `Parallel Spectrum · Track X · Module Y` in `#64748B`
- Typography cue: Inter / Segoe UI; code in JetBrains Mono on `#0F172A`
- ≤40 words of body text per panel
- Vendor AMD/NVIDIA only as small pills (`#ED1C24` / `#76B900`)

When the user asks to **generate** the image: use Parallel Spectrum colors; one idea per panel; no purple glow, no rainbow gradients, no emoji stickers on the art.

GIF / carousel note: same art direction across frames; only the result region animates or changes (numbers, checkmarks, bars) so the loop reads as assessment, not decoration.

ASCII fallback (when no image yet) — keep widths aligned:

```
  naive                 optimized
 ┌──────────┐          ┌──────────┐
 │ 12.4 ms  │  ──▶     │  3.1 ms  │
 │ bandwidth│          │ compute  │
 └──────────┘          └──────────┘
  assess: is the copy hidden yet?
```

## Robust post design checklist

Before handing off:

- [ ] Hook is specific (number, failure, or surprising tradeoff)
- [ ] Assessment wording sounds human, not brochure
- [ ] Brand tokens only — no invented palette
- [ ] Code path is real (or clearly marked as illustrative)
- [ ] Visual has ≥2 result states (before/after or multi-metric)
- [ ] Blog + Shorts + LinkedIn tell the **same** story at three lengths
- [ ] CTA points somewhere pursuable (module, snippet, Medium)
- [ ] `HISTORY.md` chain updated (NNN, Prev/Next, series narrative)

## Output format (chat)

Return the pack in this order, with clear headings:

1. **LinkedIn** (copy-paste ready)
2. **Shorts script** (timed beats)
3. **Blog stub** (Medium/GitHub)
4. **Code pursue** (path + snippet + command)
5. **Visual brief** (panel table + generation prompt)
6. Optional: **Hashtag / alt-text** line for accessibility

## Additional resources

- Full pack templates: [templates.md](templates.md)
- Worked example (vector add / foundations tone): [examples.md](examples.md)
- Brand tokens & Mermaid: [docs/BRAND.md](../../../docs/BRAND.md)
