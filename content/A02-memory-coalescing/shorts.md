# A02 Memory Coalescing — Shorts / Reels (9:16, 3 × ~50 s)

Source module: [`../../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/`](../../tracks/A-gpu-programming/A02.memory-hierarchy-and-coalescing/)

Numbers from the module's reference run (see [`long-form.md`](long-form.md)). Re-run on your GPU and
quote your own figures on screen. Burned-in captions, ≥28pt code, no title cards.

---

## Short 1 — "One character made this GPU kernel 30% faster"

**Hook value:** `tile[32][32]` → `tile[32][33]`, ~2,600 → ~3,400 GB/s.

| Beat | Time | VISUAL | SAY / ON-SCREEN |
|---|---|---|---|
| Hook | 0–3 s | Code with `[32][32]`, cursor changes the second `32` to `33` | "One character. Thirty percent faster." |
| Stakes | 3–10 s | Two GB/s numbers side by side | "Same kernel. Same answer. I changed an array dimension by one." |
| Reveal | 10–34 s | Bank diagram: 32 lanes → one bank, then padded → 32 banks | "Shared memory has thirty-two banks. My tile was thirty-two wide, so reading a column put every lane on the *same* bank — completely serialized. Make the row thirty-three wide, and thirty-three mod thirty-two is one, so the lanes spread across all thirty-two banks." |
| Proof | 34–46 s | Terminal: both runs, PASSED on both | "Twenty-six hundred to thirty-four hundred gigabytes a second, on \<GPU\>." |
| CTA | 46–53 s | Repo path | "Bank conflicts, explained properly in the module. Linked." |

**On-screen text:** `tile[32][32]` → `32-way conflict` → `tile[32][33]` → `conflict-free`.

---

## Short 2 — "Your kernel is correct and still wasting 85% of your GPU"

**Hook value:** naive transpose at ~15% of the copy ceiling.

| Beat | Time | VISUAL | SAY / ON-SCREEN |
|---|---|---|---|
| Hook | 0–3 s | Terminal: `PASSED` in green, `~700 GB/s` in amber, `ceiling ~4800` above it | "This kernel passes every test and wastes eighty-five percent of your GPU." |
| Stakes | 3–10 s | Transpose kernel, write index highlighted | "It's a matrix transpose. Reading rows is fine. Writing columns is the problem." |
| Reveal | 10–32 s | Animation: 32 lanes reading adjacent (1 truck) vs writing scattered (32 trucks) | "The GPU moves memory in whole cache lines. Adjacent threads reading adjacent addresses is one fetch. Adjacent threads writing a *column* are a whole row apart — so every lane triggers its own fetch and throws most of it away." |
| Proof | 32–45 s | Side-by-side GB/s: 700 vs 3400 after tiling+padding | "Stage the tile through shared memory and it goes to thirty-four hundred. Nearly five times, no math changed." |
| CTA | 45–53 s | Repo path | "Correctness tests will never catch this. Module linked." |

**On-screen text:** `PASSED ✓` → `15% of peak` → `strided writes` → `4.8x after tiling`.

---

## Short 3 — "Why PyTorch has .contiguous()"

**Hook value:** connects a familiar API call to hardware. Most shareable.

| Beat | Time | VISUAL | SAY / ON-SCREEN |
|---|---|---|---|
| Hook | 0–3 s | `x.contiguous()` in a PyTorch snippet | "Ever wondered why PyTorch makes you call this?" |
| Stakes | 3–12 s | Memory layout diagram: logical view vs actual strides | "A transposed tensor doesn't move any data — it just changes the strides. Logically it's transposed. Physically, the numbers are still in the old order." |
| Reveal | 12–34 s | 32 lanes hitting scattered addresses because of stride | "So when a kernel reads it, adjacent threads land far apart in memory. The GPU fetches whole cache lines and throws most of each one away. That's not a small penalty — it can be several times slower." |
| Payoff | 34–48 s | `.contiguous()` → coalesced diagram → GB/s jump | "`.contiguous()` physically reorders the data so adjacent threads read adjacent bytes. You pay one copy to stop paying on every access." |
| CTA | 48–55 s | Repo path | "The hardware reason, measured, in the module. Linked." |

**On-screen text:** `strides ≠ layout` → `scattered reads` → `wasted bandwidth` → `pay once, not every access`.

---

## Publishing notes

- Short 3 first — it hooks the large PyTorch audience and needs no GPU background.
- Short 1 second — the one-character payoff is the most "wait, what?" moment.
- Short 2 third — targets people already writing kernels, converts best to the long-form.
- All three should pin a comment linking the module and mention it runs on AMD *and* NVIDIA.
