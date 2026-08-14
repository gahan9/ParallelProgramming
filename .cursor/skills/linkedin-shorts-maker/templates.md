# Templates — LinkedIn / Shorts pack

Copy and fill. Keep brand tokens from `docs/BRAND.md`.

---

## LinkedIn skeleton

```
[Result or tension in one line.]

I [built / profiled / broke] [thing] on [HIP|CUDA|Triton].

What showed up:
→ [observation 1]
→ [observation 2]

Honest check before I'd ship it:
[one verification or tradeoff]

Pursue it here: [module path or Medium]
Snippet + run command in the comments / link below.

#[TrackTag] #GPU #HIP #CUDA   ← pick 3–5
```

---

## Shorts beat sheet

| t | On-screen (≤6 words) | Voiceover |
|---|----------------------|-----------|
| 0–3s | `[big number or FAIL→PASS]` | Hook |
| 3–15s | `The naive path` | Problem |
| 15–40s | `One change` | Demo |
| 40–55s | `Measure this` | Assessment |
| 55–60s | `Link in description` | CTA |

---

## Blog stub outline

```markdown
# [Title — concrete, not clickbait]

**TL;DR:** [3 sentences]

## What we built / measured
[1 short section]

## The snippet
\`\`\`[lang]
[8–20 lines]
\`\`\`

## Reproduce
\`\`\`bash
[clone / cd / make or python command]
\`\`\`

## Honest assessment
- **Watch for:** …
- **When not to:** …
- **I'd verify:** …

## Links
- Module: …
- Code: …
```

---

## Visual brief template

```markdown
# Visual brief — [slug]
Track: [A|B|C]  Accent: [#0891B2|#6366F1|#F59E0B]
Format: [carousel 4 panels | GIF 3 frames | single infographic]
Background: dark #0F172A → #1E293B  (or light #F8FAFC)

## Panels
1. Setup — …
2. Change — …
3. Results — before: … / after: …  (success #10B981)
4. Assess — question: "…"

## Footer
Parallel Spectrum · Track [X] · [Module ID]

## Generation prompt (for image tool)
[One paragraph: layout, colors hex, text to render exactly, no extra logos]
```

---

## Assessment micro-lines (rotate)

- "I'd verify this on your device before trusting the slide number."
- "Watch for: the win disappears when the working set fits in cache."
- "Honest read: clearer code lost ~X% — still the right call for [reason]."
- "Before you ship: does the profiler agree with the story?"
- "What held: correctness. What broke: my assumption about occupancy."
