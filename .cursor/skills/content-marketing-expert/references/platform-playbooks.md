<!-- SPDX-License-Identifier: MIT -->

# Platform Playbooks

Per-surface specs, caps, and native conventions. Numbers are directional targets,
not platform guarantees — verify current limits before a paid campaign.

## Instagram Reels

| Property | Target |
|----------|--------|
| Length | 15-90s; 20-35s for a single-idea reel |
| Aspect | 9:16, 1080x1920 |
| Safe area | Keep text out of the top 250px and bottom 400px (UI overlays) |
| Hook window | First 1 second — motion or a claim, never a logo sting |
| Captions | Burned-in, always. Most views are muted. |
| Hashtags | 3-5, specific over broad |
| CTA | Save or share. Comment-keyword works; "link in bio" leaks reach. |

Structure: hook (0-1s) → promise (1-3s) → three beats of value → payoff → loop or
CTA. Give each beat a visual change; a static talking head loses retention around
the four-second mark.

## YouTube Shorts

| Property | Target |
|----------|--------|
| Length | Under 60s; 25-40s is the reliable band |
| Aspect | 9:16, 1080x1920 |
| Title | Carries real search weight — front-load the keyword |
| Ending | Loop-friendly: the last frame should flow into the first |
| CTA | Pinned comment or subscribe prompt; avoid mid-video interruptions |

Shorts behave more like search than Reels do. Write the title as a query someone
would actually type, and say the keyword out loud in the first five seconds so
auto-captions carry it.

## TikTok

Same 9:16 grammar as Reels, with a faster cut rhythm and a stronger tolerance for
raw, unpolished footage. Native text-on-screen beats an imported lower-third.
Trends decay fast: only reference one if the content ships within days.

## Blog

| Property | Target |
|----------|--------|
| Title | Under 60 chars, primary keyword near the front |
| Meta description | Under 155 chars, written as a promise not a summary |
| Structure | H2 per section, one idea each; short paragraphs |
| Internal links | 2-4 to related posts |
| Images | Every image gets descriptive alt text |
| Schema | Article or HowTo JSON-LD where it fits |

Delegate research and long-form drafting to `content-research-writer`. This
playbook covers the packaging: frontmatter, headings, metadata, image wiring, and
the social derivatives cut from the finished post.

## LinkedIn

Longer tolerance for text, lower tolerance for hype. Lead with a specific
observation, not a rhetorical question. Line breaks every one or two sentences.
Native documents (carousels) outperform link posts. Keep external links in the
first comment when reach matters.

## X / Twitter

Write the draft here, then hand the final tuning pass to
`twitter-algorithm-optimizer`. Thread rule: post one carries the strongest claim,
and every later post stands alone well enough to be quoted on its own. Keep
hashtags to two at most, put any external link in a reply rather than the primary
post, and post when the audience is active — early engagement decides whether the
post reaches beyond existing followers.

## Email newsletter

| Property | Target |
|----------|--------|
| Subject | Under 45 chars so mobile does not truncate it |
| Preheader | Extends the subject, never repeats it |
| Body | One primary CTA; scannable subheads |
| Plain text | Always include a plain-text alternative |

## Cross-posting rules

- Re-cut, do not re-upload. Visible watermarks from another platform suppress
  distribution.
- Rewrite the hook per surface even when the body is identical.
- Match aspect ratio per surface rather than letterboxing one master.
- Stagger posting times; identical simultaneous posts read as automation.
