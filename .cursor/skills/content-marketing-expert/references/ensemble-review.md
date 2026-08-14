<!-- SPDX-License-Identifier: MIT -->

# Ensemble Review

A structured review checklist, run by one agent adopting five positions in turn.
It is not a set of live multi-model API calls. The value is in the conflicts: any
single reviewer optimizes its own axis into the ground, and the disagreements are
where the real edits live.

## The five players

| Player | Maximizes | Fails by |
|--------|-----------|----------|
| Strategist | Brand coherence, long-term trust, message consistency | Producing safe content nobody remembers |
| Educator | Layman clarity, accessibility across age bands | Over-explaining until the pace dies |
| Formatter | Platform specs, editability, file structure | Treating a checklist as the goal |
| Optimizer | Keyword placement, metadata completeness, discoverability | Stuffing keywords until the copy reads like a machine |
| Amplifier | Hook strength, early engagement, native formatting | Drifting into engagement-bait |

Each player scores the draft 1-5 on its own axis and names one specific change it
wants. Vague notes ("make it punchier") are not admissible; the note must point
at a line.

## Resolving conflicts

The recurring conflicts are predictable:

| Conflict | Resolution rule |
|----------|-----------------|
| Optimizer keyword density vs Educator clarity | Clarity wins. Move the keyword to a heading or the alt text where it reads naturally. |
| Amplifier hook strength vs Strategist brand trust | Trust wins. A hook that overpromises costs more than the view it buys. |
| Formatter platform cap vs Educator completeness | Cap wins. Split into two pieces rather than overrunning a limit. |
| Strategist consistency vs Amplifier novelty | Novelty in form, consistency in claim. Change the packaging, not the promise. |
| Optimizer metadata vs Formatter editability | Both. Metadata belongs in frontmatter, which is editable by definition. |

## Reaching an equilibrium

Iterate toward a **Pareto improvement**: keep making changes that raise at least
one player's score without dropping another below its tolerance floor. Stop when
no such change remains.

Tolerance floors — a draft below any of these does not ship regardless of how
well it scores elsewhere:

| Player | Floor |
|--------|-------|
| Strategist | 3 — no claim the brand cannot defend |
| Educator | 4 — a stranger understands it on the first read |
| Formatter | 4 — meets platform specs, source is editable |
| Optimizer | 3 — primary keyword and metadata present |
| Amplifier | 3 — the hook works cold |

The Educator and Formatter floors are the highest deliberately. Unclear content
and unusable files are unrecoverable failures; a weaker hook is not.

## Output

Close the review with the score table and short equilibrium notes: what was
traded away, for what, and what the next iteration should test.

```markdown
## Ensemble review
| Player | Score | Note |
|--------|-------|------|
| Strategist | 4 | Claim is defensible; opening line drifts off-brand |
| Educator | 5 | Reads clean cold |
| Formatter | 4 | Shot list has timecodes; alt text missing on slide 3 |
| Optimizer | 3 | Primary keyword only in the title |
| Amplifier | 4 | Hook B is strongest |

Equilibrium notes: dropped one secondary keyword from the opening to keep the
first two lines plain; moved it into the H2. Next iteration should test hook A
against hook B on the same thumbnail.
```

## Boundary

This review improves the draft in front of it. It does not validate factual
claims — unverified numbers stay labeled `TBD-verify` until a human confirms
them, and no score compensates for a fabricated statistic.
