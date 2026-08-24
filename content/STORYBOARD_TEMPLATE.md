# Storyboard template — copy per module

Copy this file into `content/<module-id>-<slug>/long-form.md` and `shorts.md`, then fill it in.
Keep it editable: this file *is* the deliverable, not a rendered video.

Delete the guidance in italics as you write. If a row has no real content, cut the row — an empty
beat on screen reads as padding.

---

## Pre-flight (fill before scripting)

| Field | Value |
|---|---|
| Module | *e.g. A04 Parallel reduction* |
| The ONE idea | *one sentence a viewer repeats afterwards* |
| The measured number | *e.g. "stage 7 is 12× stage 1" — from an actual run, name the GPU* |
| GPU used | *e.g. MI300X (gfx942), ROCm 6.x* |
| Prerequisite the viewer needs | *e.g. knows what a block is* |
| What I will NOT cover | *scope guard, prevents rambling* |

If you cannot fill "the measured number" from a run you did, stop and run the lab first.

---

## Long-form script (16:9, 8–12 min)

Write in **shot rows**. `VISUAL` is what is on screen; `SAY` is spoken, conversational, contraction-friendly.
Aim for ~140 words per minute — 12 minutes is roughly 1,600 words.

**Naming a visual.** Where a shot uses one of the [infographics](../assets/README.md), name the asset
id rather than describing it — `asset: coalescing` — so the shot list and the file stay tied
together. Capture it by opening the SVG in a browser tab, which is enough for most shots. If an
editor needs a file, derive one with
`bash scripts/export_visual.sh <asset> gif 9x16` and point at `exports/`; that render is
gitignored on purpose, so re-derive it rather than checking it in. Where a shot is better served by
a control than by a fixed drawing, use the [concept players](../live/README.md) instead: the
stride dial in `live/concepts/coalescing.html` reads well on camera.

| # | Time | VISUAL | SAY |
|---|------|--------|-----|
| 1 | 0:00 | *Terminal, final result already printed* | *Cold-open: state the payoff number out loud.* |
| 2 | 0:30 | *Analogy illustration or whiteboard* | *The layman analogy. Then: "here's where that analogy breaks."* |
| 3 | 1:30 | *Diagram of the naive approach* | *Derive why the obvious way is slow, from first principles.* |
| 4 | 3:30 | *Editor, typing the fix* | *Live-code the real version. Say what each decision buys.* |
| 5 | 6:30 | *Terminal, running it* | *Run. Read the number. Compare to peak / roofline.* |
| 6 | 8:30 | *Slide: tradeoffs* | *When this is the wrong tool. What it costs.* |
| 7 | 10:00 | *Slide: real systems* | *Where this exact idea appears in PyTorch / vLLM / training.* |
| 8 | 11:00 | *Repo on screen* | *The drill for the viewer. Next module teaser.* |

### Chapters (paste into the YouTube description)

```
0:00 The result first
0:30 The intuition
1:30 Why the naive version is slow
3:30 Writing the fast version
6:30 Measuring it
8:30 The tradeoffs nobody mentions
10:00 Where this shows up in real ML
11:00 Your turn
```

---

## Shorts / Reels (9:16, 30–60 s each)

Three Shorts per module. Each stands alone — assume zero context and no prior video.

### Short 1 — the counterintuitive number
*Lead with the result that sounds impossible.*

| Beat | Time | VISUAL | SAY / ON-SCREEN TEXT |
|---|---|---|---|
| Hook | 0–3 s | *Big number, brand cyan on slate* | *"X made this Y times faster."* |
| Stakes | 3–8 s | *The two code versions side by side* | *"Same math. One difference."* |
| Reveal | 8–35 s | *Zoom the diff / diagram* | *The idea, in one breath.* |
| Proof | 35–50 s | *Terminal output* | *"Here it is on a real \<GPU\>."* |
| CTA | 50–60 s | *Repo URL* | *"Runnable code in the description."* |

### Short 2 — the bug/gotcha
*Show a failure, then the fix. Failures out-perform explanations.*

| Beat | Time | VISUAL | SAY / ON-SCREEN TEXT |
|---|---|---|---|
| Hook | 0–3 s | *Wrong output / NaN / garbage* | *"This kernel looks right. It isn't."* |
| Stakes | 3–8 s | *Highlight the innocent-looking line* | *"This line corrupts memory silently."* |
| Reveal | 8–35 s | *The fix typed in* | *Why it breaks, in plain words.* |
| Proof | 35–50 s | *PASSED output* | *"Now it's correct."* |
| CTA | 50–60 s | *Repo URL* | *"There's a drill for this."* |

### Short 3 — the mental model
*Pure intuition, no code. This is the one that gets shared.*

| Beat | Time | VISUAL | SAY / ON-SCREEN TEXT |
|---|---|---|---|
| Hook | 0–3 s | *Analogy visual* | *"Here's how a GPU actually thinks."* |
| Reveal | 3–40 s | *Animated diagram* | *The analogy, then where it breaks.* |
| Payoff | 40–55 s | *Map analogy onto real terms* | *"That's a wavefront. 64 lanes on AMD, 32 on NVIDIA."* |
| CTA | 55–60 s | *Repo URL* | *"Full module linked below."* |

---

## Live-coding run sheet (optional, 45–90 min)

| Segment | Minutes | What happens |
|---|---|---|
| Setup + hardware check | 5 | `rocm-smi` / `nvidia-smi`, show the toolchain, state the goal |
| Build the naive version | 15 | Type it live, including the mistakes — do not hide them |
| Break it deliberately | 10 | Remove the bounds guard / the sync; show the symptom |
| Optimize | 25 | One change at a time, re-measure after each |
| Profile | 15 | `rocprofv3` / `ncu`, read the counters on stream |
| Q&A | 10+ | Answer chat; log unanswered questions into the module's exercises |

Rule for live: **re-measure after every single change.** The value of a live stream is showing that
performance work is empirical, not a recipe you memorized.

---

## Accuracy guardrails

- Never state a benchmark number you did not run. Name the GPU, the toolkit version, and the size.
- If AMD and NVIDIA differ (warp 32 vs wavefront 64), say both — the audience is mixed.
- If you are unsure on stream, say "I'd have to check the guide" and link it in the description.
  Confidently wrong is the fastest way to lose a technical audience.
