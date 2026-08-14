<!-- SPDX-License-Identifier: MIT -->

# Memory Shots

A memory shot is a small JSON file that lets a later session resume a campaign
without re-deriving the brief, the hooks, the keywords, and the asset prompts.

## Ask-before-save protocol

**Never write a memory shot without explicit approval.** After delivering, ask
the user once:

> Save a memory shot so a later session can continue this campaign?

Options: save to the default path, save to a path the user names, or skip. On
skip, keep the state in the session only and say so in the output. Do not ask
again in the same turn, and do not re-ask on every subsequent deliverable — a
decline holds for the session unless the user reopens it.

When updating an existing memory shot, the same rule applies: confirm before
overwriting, and append to `iteration_log` rather than replacing it.

## Location

Default: `.content-campaign/<slug>/memory-shot.json`

`.content-campaign/` is gitignored. Campaign state is the user's working
material, not repository content. If the user wants it version-controlled, they
choose an explicit path outside that folder.

## Schema

```json
{
  "campaign_id": "spring-launch-2026",
  "created": "2026-08-09",
  "updated": "2026-08-09",
  "goal": "saves",
  "platforms": ["reels", "shorts", "blog"],
  "mode": "hybrid",
  "audience": {
    "age_band": "25-40",
    "familiarity": "cold",
    "reading_level_target": "grade-8"
  },
  "core_promise": "Cut your weekly reporting from four hours to twenty minutes.",
  "hooks": [
    {"variant": "curiosity", "text": "...", "selected": false},
    {"variant": "pain-point", "text": "...", "selected": true}
  ],
  "seo": {
    "primary_keyword": "automated weekly reporting",
    "secondary_keywords": ["reporting template", "report automation"],
    "title_variants": ["...", "..."],
    "meta_description": "...",
    "unverified": ["search volume for primary keyword"]
  },
  "assets": [
    {
      "path": "spring-launch-2026/assets/reel-cover.png",
      "surface": "reel cover",
      "prompt": "...",
      "backend": "local image generator",
      "seed": 42,
      "wired_into": "shot-list.md line 3"
    }
  ],
  "deliverables": [
    {"path": "spring-launch-2026/reel-shot-list.md", "surface": "instagram reels"}
  ],
  "iteration_log": [
    {"loop": 1, "pass": "hook", "reason": "opening needed context to parse"},
    {"loop": 2, "pass": "seo", "reason": "keyword moved to H2 to protect clarity"}
  ],
  "ensemble_scores": {
    "strategist": 4, "educator": 5, "formatter": 4,
    "optimizer": 3, "amplifier": 4
  },
  "pending_todos": ["verify the four-hour baseline before publishing"],
  "brand": {
    "tone": "direct, no hype",
    "banned_phrases": ["game-changer", "revolutionary"],
    "standing_cta": "save this"
  }
}
```

All fields are optional except `campaign_id`, `goal`, `platforms`, and `mode`.
`mode` is required because it drives the asset-routing branch on resume. Omit
what does not apply rather than filling in placeholders.

## Resuming from a memory shot

When a user references an existing campaign:

1. Read the memory shot and restate the brief in two lines so they can correct it.
2. Check `pending_todos` and surface anything unverified before building on it.
3. Confirm the asset paths still exist; regenerate from the stored prompt and
   seed if any are missing.
4. Continue from the last `iteration_log` entry rather than starting the loop
   over.

## What not to store

- Credentials, API keys, or tokens of any kind.
- Personal data about audience members.
- Full copies of deliverables — store paths, not content. The memory shot is an
  index, and it stays small enough to read in full at the start of a session.
