# Social post history — Parallel Spectrum



Living chain for LinkedIn / Shorts / Medium packs. **Update this file whenever a pack is created or published.**



Status legend: `draft` · `ready` · `posted` · `archived`



---



## Chain (newest first)



| # | Date | Slug | Track | Module | Hook (one line) | Status | Prev | Next | Folder |

|---|------|------|-------|--------|-----------------|--------|------|------|--------|

| 003 | 2026-08-31 | `a01-bounds-silent-corruption` | A | A01 | PASSED · 24 threads still wrote past N (1000 / 256) | ready | `a02-coalescing-bandwidth` | *TBD-004* | [`003-a01-bounds-silent-corruption/`](003-a01-bounds-silent-corruption/) |

| 002 | 2026-08-10 | `a02-coalescing-bandwidth` | A | A02 | PASSED at ~15% BW · ~700 → ~3,400 GB/s (~5×) | ready | `a01-kernel-vs-path` | `a01-bounds-silent-corruption` | [`002-a02-coalescing-bandwidth/`](002-a02-coalescing-bandwidth/) |

| 001 | 2026-08-02 | `a01-kernel-vs-path` | A | A01 | Kernel 44 µs · program ~72 ms (0.06%) | posted | — | `a02-coalescing-bandwidth` | [`001-a01-kernel-vs-path/`](001-a01-kernel-vs-path/) |



---



## Series narrative (recurrent)



Keep the through-line honest across posts:



1. **001 — Measure the path, not just the kernel** (A01) ← *posted*

2. **002 — Coalescing: same math, different bandwidth** (A02 · ~700 → ~3400 GB/s) ← *ready, unpublished*

3. **003 — Bounds guard / silent corruption** (A01 · 24 extra threads) ← *you are here*

4. *Planned 004 — Reduction ladder spread* (A04) once re-run on your GPU



When posting #N, open with a one-line callback to #N−1 when it strengthens continuity

(e.g. "Last time the kernel was 0.06% of the run. This time the *reads* decide…").



---



## Engagement learnings (update after each post)



| # | Channel | Impressions | Reactions | Comments | What to keep / change |

|---|---------|------------:|----------:|---------:|------------------------|

| 001 | LinkedIn | ~1,455 | 9 | — | Keep concrete numbers + assessment voice. **Change:** no outbound URL in body (first comment); attach carousel; plain-language gloss; end with a judgment question; bridge to a familiar ML API when honest. |

| 002 | LinkedIn | — | — | — | Applying 001 fixes above. Fill after publish. |



---



## Publish log



Append a row when something goes live (URL optional until you have it).



| # | Channel | Posted at | URL | Notes |

|---|---------|-----------|-----|-------|

| 001 | Medium | 2026-08-03 | [Medium — post 001](https://gahan9.medium.com/your-gpu-kernel-took-44-microseconds-the-program-took-72-milliseconds-09c5e3b920c1) | Published (gahan9) |

| 001 | LinkedIn | ~2026-08-04 | [LinkedIn — post 001](https://www.linkedin.com/posts/gahan-saraiya_gpuprogramming-parallelcomputing-hip-share-7490001593329762305-NG4U/) | ~1455 impressions / 9 reactions — see engagement learnings |

| 001 | YouTube Shorts | — | — | Aligns with `content/A01-foundations/shorts.md` Short 1 |

| 002 | Medium | — | — | Draft ready in pack `blog.md` — re-run lab before publish |

| 002 | LinkedIn | — | — | Copy in `linkedin.md`; attach `visual.png`; Medium URL in first comment |

| 002 | YouTube Shorts | — | — | Aligns with `content/A02-memory-coalescing/shorts.md` Short 2 |

| 003 | Medium | — | — | Draft ready in pack `blog.md` — arithmetic post; no GPU re-run required |

| 003 | LinkedIn | — | — | Copy in `linkedin.md`; attach `visual.png`; Medium URL in first comment; use 001-callback variant if 002 still unpublished |

| 003 | YouTube Shorts | — | — | Aligns with `content/A01-foundations/shorts.md` Short 2 |



**Fix-on-Medium notes (001):** duplicate "The measurement" heading (merge intro + table under one);

empty `GPU_ARCH=` in the reproduce block — set your arch or keep `<your-arch>`; published version

dropped "Honest assessment" — consider re-adding, it's the series signature.



---



## Agent checklist (recurrent update)



When creating the **next** pack with `linkedin-shorts-maker`:



- [ ] Assign `NNN = last + 1`

- [ ] Insert a new row at top of **Chain**

- [ ] Set previous pack's `Next` from `TBD` → this slug

- [ ] Set this pack's `Prev` → previous slug

- [ ] Add planned callback line in LinkedIn if series continuity helps

- [ ] Apply **Engagement learnings** from prior posts

- [ ] Leave Publish log rows blank until actually posted; fill URLs after

