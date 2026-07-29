# A01 Foundations — Shorts / Reels (9:16, 3 × ~50 s)

Source module: [`../../tracks/A-gpu-programming/A01.foundations-and-programming-model/`](../../tracks/A-gpu-programming/A01.foundations-and-programming-model/)

All three stand alone. Burned-in captions, ≥28pt code, no title cards. Numbers come from the
`rocprofv3` run documented in [`long-form.md`](long-form.md) — restate your own GPU on screen.

---

## Short 1 — "Your kernel isn't the slow part"

**Hook value:** the kernel is 0.06% of the runtime.

| Beat | Time | VISUAL | SAY / ON-SCREEN |
|---|---|---|---|
| Hook | 0–3 s | Full-frame: `44 µs` in green, then `72,373 µs` in amber below it | "Your GPU kernel: forty-four microseconds. Your program: seventy-two thousand." |
| Stakes | 3–9 s | Profiler summary, kernel row boxed amber | "The math was never the problem." |
| Reveal | 9–32 s | Horizontal bar chart of the 5 phases, kernel bar a sliver | "Allocating memory: sixty-eight milliseconds. Copying data in: two. The actual computation: a rounding error. On a GPU, moving the data costs more than doing the math — almost always." |
| Proof | 32–45 s | Terminal scrolling the real `rocprofv3` output | "That's a real profile. `rocprofv3`, sixteen million elements, on \<GPU\>." |
| CTA | 45–52 s | Repo path on slate | "If you only measure kernel time, you'll optimize the wrong thing. Full module linked." |

**On-screen text plan:** `44 µs` → `72 ms` → `0.06%` → `move data, not math`.

---

## Short 2 — "This kernel corrupts memory and still prints the right answer"

**Hook value:** a missing one-line guard.

| Beat | Time | VISUAL | SAY / ON-SCREEN |
|---|---|---|---|
| Hook | 0–3 s | Kernel on screen *without* the bounds guard | "This kernel looks correct. It's writing past the end of your array." |
| Stakes | 3–10 s | Grid diagram: 4 blocks × 256 threads over N = 1000, last 24 threads flashing amber | "N is a thousand. Block size is two fifty-six. So you launch four blocks — a thousand and twenty-four threads. Twenty-four of them have no data to work on." |
| Reveal | 10–30 s | Type in `if (idx < n)`, highlight it | "Those twenty-four write to memory that isn't yours. Sometimes it crashes. Usually it silently corrupts something you'll debug next week. One line fixes it." |
| Proof | 30–44 s | Terminal: PASSED | "Guard added. Verified against a CPU reference." |
| CTA | 44–52 s | `exercises/` folder | "There's a drill for this — plus a histogram with a real data race. Linked below." |

**On-screen text plan:** `1000 / 256 = 4 blocks` → `1024 threads` → `24 out of bounds` → `if (idx < n)`.

---

## Short 3 — "A GPU is a stadium, not a computer"

**Hook value:** the mental model. No code — this is the shareable one.

| Beat | Time | VISUAL | SAY / ON-SCREEN |
|---|---|---|---|
| Hook | 0–3 s | Split illustration: 4 desks vs a packed stadium | "A CPU is a few geniuses. A GPU is a stadium of fifty thousand students." |
| Reveal | 3–24 s | Animate: slips of paper handed out, everyone adds at once | "Ask the geniuses to add two huge lists — fast, but one pair at a time. Give every student in the stadium one pair and shout 'go' — done in a single moment." |
| Twist | 24–40 s | Highlight rows of 64 seats moving together | "But they're not individuals. They move in locked rows — all doing the same step at the same instant. Sixty-four seats per row on AMD, thirty-two on NVIDIA." |
| Payoff | 40–52 s | Overlay real terms on the illustration | "That row is a wavefront, or a warp. And if you tell half a row to do something different, the row does both things one after the other — that's why branching hurts on a GPU." |
| CTA | 52–58 s | Repo path | "Full module, runnable on both vendors. Linked." |

**On-screen text plan:** `CPU = few geniuses` → `GPU = 50,000 students` → `they move in locked rows` → `wavefront 64 · warp 32`.

---

## Publishing notes

- Short 3 is the audience-growth one (no code, pure intuition) — publish it first.
- Short 1 is the credibility one (real profiler data) — publish second, it converts to the long-form.
- Short 2 is the practitioner one — publish third; it performs best with people already writing GPU code.
- Pin a comment on each linking the module folder.
