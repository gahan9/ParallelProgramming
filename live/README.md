# Live pages — plan, test, present

Three self-contained pages published with the docs site, sharing one nav so they read as a single
tool. All are plain HTML with no framework, no CDN, and no build step, so they work on GitHub Pages,
from a local checkout, and on a locked-down corporate network alike.

| Page | What it does | Open it |
|---|---|---|
| **Study planner** | Pick a goal, get an ordered module ladder, tick modules off, watch remaining hours drop | [dashboard.html](dashboard.html) |
| **Recall cards** | Retrieval practice on the curriculum's own Q&A, sorted into three Leitner boxes, every card cited | [flashcards.html](flashcards.html) |
| **Slide deck** | Presentable overview of the curriculum and the findings from building it | [slides.html](slides.html) |

They form a loop: **plan** what to study, **test** whether it stuck, **present** it to find out what
you only half-know.

> **On GitHub these render as source, not as pages.** GitHub does not serve HTML from a repository
> view. Use the published site (the `Deploy docs to GitHub Pages` workflow builds it on every push to
> `main`), or clone the repo and open the files directly in a browser — both work.

---

## Study planner

The planner is the in-repo counterpart to the Cursor canvas dashboard, rebuilt so it works anywhere
a browser does.

- **Five goal-based paths** — write fast kernels, make models fast, design ML systems, interview
  soon, or everything in dependency order. Choosing one reorders the ladder and recomputes the hours.
- **Progress that persists.** Ticks are stored in `localStorage` under `ppc.progress.v1`, so closing
  the tab does not lose your place. Nothing is uploaded and there is no account.
- **A "next up" recommendation** — the first module on your path you have not finished, with its hour
  estimate, prerequisite depth, and a direct link.
- **Per-track bars** showing completed versus remaining hours for Tracks A, B, and C.
- **Reset** clears your ticks after a confirmation.

It degrades honestly: with JavaScript disabled it says so and points at
[CURRICULUM.md](../CURRICULUM.md), which needs nothing but a browser.

## Recall cards

Every module's Section 9 already holds conceptual Q&A, but it sits inside `<details>` blocks in a
900-line README — so it is read passively, if at all. This page turns that material into retrieval
practice, which is a measurably different activity: Roediger & Karpicke found re-testing produced
**61%** recall after a week against **40%** for re-reading the same material the same number of
times, *while the re-readers felt more confident*
([Test-Enhanced Learning](https://doi.org/10.1111/j.1467-9280.2006.01693.x), Psychological Science,
2006). Confidence is not retention, which is the same argument the curriculum makes about benchmarks.

- **Three Leitner boxes, not a streak.** Cards start in Box 1. *Solid* promotes, *Shaky* holds,
  *Missed* drops to Box 1. Drawing is weighted 3:2:1 toward Box 1, so the deck keeps returning to
  what you get wrong. The box display *is* the progress meter — the useful question is "what is still
  in Box 1", not "how many did I click".
- **The answer reveals in place; the question stays.** Deliberately not a flip card, which hides the
  question at exactly the moment you want to compare it against the answer.
- **Every card cites a source** — a paper, a vendor document, or the module it came from. If a card
  and its source disagree, the source wins.
- **Keyboard:** `space` reveals, `1`/`2`/`3` grade, `→` skips.
- Filter by track; progress persists in `localStorage` under `ppc.cards.v1`.

Deliberately a three-box Leitner system rather than a full SM-2 scheduler: interval tuning is not
what stands between a learner and this material, and a simpler mechanic you can see is worth more
than a better one you can't.

## Slide deck

Fourteen slides following the slide template in [docs/BRAND.md](../docs/BRAND.md) — 4px cyan accent
bar, Inter and JetBrains Mono, dark dividers between parts, one idea per slide.

- **Navigate** with arrow keys, space, `PageUp`/`PageDown`, `Home`/`End`, or just scroll.
- **Export to PDF** with `Ctrl-P` / `Cmd-P`. Print CSS gives one slide per landscape page, so the
  deck doubles as a handout without a separate export step.
- **Content**: the design intent, the three tracks, the nine-section contract, the dual-vendor
  approach, the evidence standard, and five concrete findings from building the curriculum (the
  kernel that was 0.06% of its runtime, the 4.8× transpose, the two incompatible fp8 formats,
  why pipelining does not save activation memory, and the KV cache that was not the binding
  constraint).

Use it as-is for a talk, or fork a slide. The source is one file with the content inline as HTML —
edit the `<section class="slide">` blocks and it is still a deck.

---

## Keeping them accurate

The planner and deck hardcode the module list, hour estimates, and depths. The source of truth is
[CURRICULUM.md](../CURRICULUM.md); when a module is added or an estimate changes, update:

1. `CURRICULUM.md` — the status tables and the totals.
2. `live/dashboard.html` — the `M` array near the top of the script, and `PATHS` if the new module
   belongs on a learning path.
3. `live/flashcards.html` — add cards to `CARDS`, reusing an entry from the `S` source table or adding
   one. A card without a real source does not belong in the deck.
4. `live/slides.html` — only if the headline counts (24 modules, ~164 hours) change.
5. The Cursor canvas at `canvases/curriculum-dashboard.canvas.tsx`, which carries the same data for
   use inside the editor.

The duplication is deliberate: each surface has to stand alone with no shared runtime. Four places is
few enough to keep in step by hand, and a build step to generate them would be more machinery than
the problem deserves.
