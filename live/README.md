# Live pages — plan, look up, test, present

Five self-contained pages published with the docs site, sharing one nav so they read as a single
tool. All are plain HTML with no framework, no CDN, and no build step, so they work on GitHub Pages,
from a local checkout, and on a locked-down corporate network alike.

| Page | What it does | Open it |
|---|---|---|
| **Study planner** | Pick a goal, get an ordered module ladder, tick modules off, watch remaining hours drop | [dashboard.html](dashboard.html) |
| **Reference sheets** | Track, concept and module cheatsheets — searchable, printable, every fact cited | [cheatsheets.html](cheatsheets.html) |
| **Recall cards** | Retrieval practice on the curriculum's own Q&A, sorted into three Leitner boxes, every card cited | [flashcards.html](flashcards.html) |
| **Slide deck** | Presentable overview of the curriculum and the findings from building it | [slides.html](slides.html) |
| **Concept players** | Twelve interactive versions of the infographics, one per concept cluster, where a control is what teaches | [concepts/index.html](concepts/index.html) |

They form a loop: **plan** what to study, **look up** what you forgot mid-lab, **test** whether it
stuck, **play** with the one control that makes a concept click, and **present** it to find out
what you only half-know. Every surface links to the other three: a concept player's footer points
at its reference sheet and its recall cards; a reference sheet's cluster card points at its concept
player and its recall cards; a recall card's track filter matches the same track colours
everywhere.

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
- **Listen to a card.** `L` (or the button) reads it aloud using the browser's own speech
  synthesis — the question only, until you reveal, so it never spoils a card you have not
  attempted. Useful for testing yourself away from the screen.
- **Keyboard:** `space` reveals, `1`/`2`/`3` grade, `→` skips, `L` listens.
- Filter by track; progress persists in `localStorage` under `ppc.cards.v1`.

Deliberately a three-box Leitner system rather than a full SM-2 scheduler: interval tuning is not
what stands between a learner and this material, and a simpler mechanic you can see is worth more
than a better one you can't.

## Reference sheets

Recall and reference are different activities, and merging them makes both worse. The cards ask you
a question; these answer one you already have, mid-lab, without making you re-read a 900-line module.

- **Four tiers, four questions.** A *track* sheet says what that part of the curriculum owns and
  what you need before starting it. A *concept* sheet crosses modules, because the concepts do —
  twelve of them, from memory hierarchy to fabric tiers. A *module* card is the "what was the one
  idea in A04 again" answer, one per module. An *analogy* card gives the everyday version and then
  the machine's, set in two different typefaces so you can see at a glance which world you are in.
- **Every analogy comes with its own limits.** The card states where the module says the picture
  stops being true, because that is the half people skip and the half the engineering lives in. Two
  modules, A01 and A02, do not state one yet; those cards say so rather than implying the analogy is
  airtight.
- **Every bullet is cited** to the module section or glossary entry that derives it, and the numbers
  are the module's numbers. Nothing here is restated from memory.
- **Search and filter.** Type to search every sheet at once; `/` focuses the box and `Escape` clears
  it. Track pills narrow to one track.
- **The rail is load-bearing.** Thirty lanes across the top, one per module, coloured by track —
  the same "a square is a unit" device the [infographics](../assets/README.md) use. Filtering dims
  the lanes that drop out, so it always answers "how much of the curriculum am I looking at". Click
  a lane to jump to its card.
- **Printable.** `Ctrl-P` gives one concept sheet per landscape page with the citation URLs spelled
  out, so the printout stands on its own.

The facts live in [`data/reference.json`](data/reference.json), not inside the HTML, so they have one
home and can be checked mechanically — every bullet must carry a `cite`. The cost of that choice:
browsers refuse to `fetch()` a sibling file from a page opened with `file://`, so this one page needs
an HTTP server locally (`python -m http.server` from the repo root). It says so on screen, with the
command, rather than rendering blank. The published site is unaffected.

## Concept players

One page per idea where a **control** teaches better than a fixed drawing —
[`concepts/index.html`](concepts/index.html) is the hub, one tile per concept cluster, filterable by
track and cross-linked to that cluster's reference sheet and recall cards. Twelve clusters, twelve
players, all now built:

| Concept | Modules | Player |
|---|---|---|
| Memory hierarchy & coalescing | A02, A06, B01 | [coalescing.html](concepts/coalescing.html) |
| Divergence & occupancy | A03, A04, A05 | [divergence.html](concepts/divergence.html) |
| Fusion & FlashAttention traffic | A07, A08, A09 | [attention-traffic.html](concepts/attention-traffic.html) |
| Roofline & profiling | B02, B03 | [roofline.html](concepts/roofline.html) |
| Precision, numerics & fp8 | B04 | [precision-formats.html](concepts/precision-formats.html) |
| Prefill, decode & serving | B05, B06 | [prefill-decode.html](concepts/prefill-decode.html) |
| Decision thresholds & cost | C01, C02 | [decision-threshold.html](concepts/decision-threshold.html) |
| Optimization ladder | C04, C05, C07 | [optimization-ladder.html](concepts/optimization-ladder.html) |
| Drift & monitoring | C03, C06 | [drift-monitor.html](concepts/drift-monitor.html) |
| Scaling limits & cluster shape | D01, D05 | [scaling-limits.html](concepts/scaling-limits.html) |
| Ring all-reduce & collectives | D02, D03, B07 | [ring-allreduce.html](concepts/ring-allreduce.html) |
| Interconnect & fabric tiers | D04, D06 | [fabric-tiers.html](concepts/fabric-tiers.html) |

[`concepts/coalescing.html`](concepts/coalescing.html) was the first: change the access stride and
the lockstep width, and watch the cache lines touched, the bytes fetched and the bytes wasted move
with it. The efficiency turns out to be `1/stride` on both a 32-lane warp and a 64-lane wavefront,
which is a more convincing way to learn that than being told it. The other eleven follow the same
rule and the same visual system (dark chrome, brand tokens from
[docs/BRAND.md](../docs/BRAND.md), track-coloured accent bars): every reading is exact arithmetic
on a formula the linked module already states and cites — never a new measurement, never a new
claim — and every page says so plainly in its own disclosure note, the same convention the module's
own analytic labs use.

The bar for adding one is that **the geometry has to genuinely change with the input**. If the
picture is the same shape at every setting, it should stay an SVG in [`../assets/`](../assets/README.md),
which remains the canonical version that the module embeds and that survives JavaScript being off —
every concept player links back to its own canonical SVG for exactly that reason. The two clusters
this repository had already flagged as earning a page, the ring all-reduce phases (D03) and the
roofline intensity slider (B02), are both now built.

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
4. `live/data/reference.json` — add a `modules` entry, and a `clusters` entry if the module brings a
   genuinely new concept. Every bullet needs a `cite`. If the new cluster earns a concept player
   (the geometry genuinely changes with an input), add a `concept` field pointing at it —
   `cheatsheets.html`'s cluster card wires the link automatically.
5. `live/concepts/index.html` — add a tile to the inline `CLUSTERS` array so the hub lists it. This
   mirrors `reference.json`'s `clusters` array on purpose (AGENTS.md trap #7): each live page stands
   alone with no shared runtime, so the two lists are kept in step by hand.
6. `live/slides.html` — only if the headline counts (30 modules, ~203 hours) change.
7. The Cursor canvas at `canvases/curriculum-dashboard.canvas.tsx`, which carries the same data for
   use inside the editor.

The duplication is deliberate: each surface has to stand alone with no shared runtime. Five places is
few enough to keep in step by hand, and a build step to generate them would be more machinery than
the problem deserves.

Two things are deliberately *not* duplicated, and should stay that way. The reference sheets derive
their per-track module counts and hour totals from the `modules` array rather than storing them
again, and the recall deck builds its track filter row from the cards that actually exist. Both used
to be hand-written lists; both were quietly wrong the moment Track D landed.

**Known gap:** the recall deck has 49 cards covering Tracks A, B and C, and none for Track D — it
was built before that track existed. The reference sheets link to it per track
(`flashcards.html?track=D`), and the deck says plainly that the track has no cards yet rather than
showing an empty screen. Writing them is the fix; each needs a real source.
