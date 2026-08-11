# Drill 2 — Read a list entry like an architect

**Clock: 45 minutes.** Reference answer:
[`../solutions/read_a_list_entry.md`](../solutions/read_a_list_entry.md).

This is the module's Lab 2, as a worksheet you fill in. The skill is extracting *design
decisions* from a public specification, and the failure it trains you out of is reading the
rank and stopping.

---

## Before you start: why there are no numbers in this file

Every figure you are about to write down is an **edition-scoped fact**. A system's rank, and
often its reported configuration, is a property of one specific list edition, published twice
a year. This worksheet therefore bakes in **no** node counts, no FLOPS, no link rates and no
positions, and neither does the module. You go and read the current edition:

- [TOP500 and Green500](../../../../docs/REFERENCES.md#top500-and-green500) — the ranked
  lists and the structured system specifications behind them. The same page links the
  HPCG list, which is published alongside.
- The **operating site's own documentation** for the system you pick. This is usually a
  better source than the list row for node composition, and it is the only source for
  anything about the software stack.

Two rules for everything you record, and they are the point of the drill as much as the
answers are:

1. **Every cell gets a source and a retrieval date.** A number without an edition attached
   is not a fact you can defend in a review three months from now.
2. **If you cannot find it, write "not published".** That is a finding, not a gap. Section
   6.2 of [the module](../README.md#6-challenges-drawbacks--tradeoffs) lists what the public
   record systematically does not contain, and half of what you will want here is on it.

---

## The drill

Open the current list. Pick **one system that is not in
[section 3.8](../README.md#3-deep-dive)** — so not Frontier, El Capitan, Aurora, Fugaku,
Summit or LUMI. Prefer one you have never heard of over one you have; the whole exercise is
reading a machine cold.

Then fill in the worksheet below.

### Part 1 — Provenance

| Field | Your answer |
|---|---|
| System name |  |
| Operating site and country |  |
| List edition you are reading (month and year) |  |
| Retrieval date |  |
| Site's own documentation URL |  |

### Part 2 — The six items

Fill each row with the value, where you found it, and whether it was published at all.

| # | What to find | Value | Source | Published? |
|---|---|---|---|---|
| 1a | Accelerator family (or "none", which is a real answer) |  |  |  |
| 1b | CPU-to-accelerator link: coherent, high-bandwidth, or PCIe |  |  |  |
| 2a | Interconnect family |  |  |  |
| 2b | [Topology class](../../../../docs/GLOSSARY.md#fat-tree-dragonfly-and-torus), if published |  |  |  |
| 3a | Accelerators per node |  |  |  |
| 3b | NICs or adapters per node |  |  |  |
| 4a | Its Green500 position in the **same** edition |  |  |  |
| 4b | Does the Green500 ordering differ from TOP500 around it? |  |  |  |
| 5a | Does an HPCG result exist for this system? |  |  |  |
| 5b | If so, the ratio of HPCG to its HPL figure |  |  |  |

Item **3b** is the one people skip and the one that decides your experience of the machine.
It is the injection-gap term `g` from
[D01 section 3.2](../../D01.scaling-limits-and-communication-cost/README.md#3-deep-dive): a node
with one adapter and eight accelerators has an injection ceiling that all eight must share,
and no amount of fabric fixes it. If it is not published, say so and say what you would ask
the site.

### Part 3 — The design lesson

Write **one sentence** in the style of section 3.8's table: name the constraint this design
was built to defeat.

> **Design lesson:**

Then one sentence on what you could **not** determine from public sources, and why it
matters:

> **What is missing:**

### Part 4 — Two questions to answer in prose

**(a) What does the HPL-to-HPCG ratio license you to say** about this machine, and what does
it not? If no HPCG result exists, say what you would run instead and on how large an
allocation.

**(b) Suppose this system and one architecturally similar system sit several positions apart
on TOP500.** Name a workload for which that gap predicts nothing at all, and a workload for
which it predicts something. One sentence each.

---

## What "done" looks like

- Every filled cell carries a source and a retrieval date, and every empty one says
  "not published" rather than being blank.
- Items 1a to 3b are answered before item 4a is looked at. If you wrote the rank down first,
  start over; the ordering is the habit being trained.
- The design lesson names a **constraint**, not a component. "Fast GPUs" is not a lesson.
  "Bought bisection bandwidth because its flagship workload is all-to-all" is.
- You noticed at least one place where the list and the site's own documentation disagree, or
  you checked and can say they agree. Both are findings.

## The trap this drill exists to defuse

The rank is the easiest thing on the page to find and the least useful thing on it to know.
It orders systems by HPL, a dense LU factorization with very high
[arithmetic intensity](../../../../docs/GLOSSARY.md#arithmetic-intensity), regular access and
almost no global communication — close to a best case for any machine. If your workload is
sparse, latency-bound, collective-heavy or memory-capacity-bound, the ranking carries no
information about it, and the HPCG list exists precisely because that is true. Use HPL to
**bound**, never to predict, and see the
[Gordon Bell Prize](../../../../docs/REFERENCES.md#acm-gordon-bell-prize-acm) record when you
want to know what real applications had to do to run at that scale.
