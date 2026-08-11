# D06 Exercises — Real Systems at Scale

Four drills. One is code; **three are analyses**, because this module is Track D's reading
capstone and the skill it trains is judgement about somebody else's system, not a kernel.

| # | File | Clock | What it tests |
|---|---|------:|---|
| 1 | [`scale_boundary.py`](scale_boundary.py) | 40 min | Assemble the five constraints into one prediction: dominant term, named binding constraint, knee, expected interruptions |
| 2 | [`read_a_list_entry.md`](read_a_list_entry.md) | 45 min | Extract *design decisions* from a public specification instead of reading a ranking |
| 3 | [`diagnose_the_system.md`](diagnose_the_system.md) | 60 min | Symptom to binding constraint to **discriminating measurement**, across eight scenarios |
| 4 | [`architecture_review.md`](architecture_review.md) | 45 min | Review a plausible, flawed cluster proposal against the six-question checklist and write the memo |

Work them **in order**. Drill 1 builds the arithmetic the other three lean on, and drill 4 is
the one that most resembles a Tuesday afternoon: a document arrives, it is not wrong anywhere
you can point to easily, and you have to say something useful about it.

---

## How to work these

**1. Estimate before you compute.** Every formula is derived in
[`../README.md` section 2](../README.md#2-first-principles) and in
[D01 section 3](../../D01.scaling-limits-and-communication-cost/README.md#3-deep-dive). If you
reach for the calculator before you have a rough number in your head, you are using it as a
crutch rather than as a check, and you will not notice when you have typed a wrong flag.

**2. Then check with the tool.** It is an analytic model, so a disagreement between your
arithmetic and its output means one of you has a bug. Find out which; that is the whole value
of the step.

```bash
# from the module directory (one level up from here)
python3 tools/scale_boundary.py --help
python3 tools/scale_boundary.py --preset dp-training --ranks 4096 --explain
make run                     # all three presets
```

On Windows PowerShell the interpreter is usually `python`, not `python3`.

**3. Name a measurement, not a suspicion.** Drills 3 and 4 are graded on the *discriminating*
observation — the one that would confirm your diagnosis **and rule out the next most likely
explanation**. A diagnosis you cannot falsify is a guess with a confident tone.

**4. Write it down in prose.** These are the artifacts you would actually send: a filled
worksheet, eight short diagnoses, a memo. Bullet fragments let you skip the sentence you have
not thought through.

---

## The honesty rules, which are graded

This module is where a curriculum most easily starts lying, so three rules apply to your
answers as much as to the module text:

1. **Every figure you write down carries a source and a retrieval date.** List positions and
   reported configurations are scoped to one list edition, published twice a year. A number
   without its edition is not defensible three months later.
2. **"Not published" is an answer.** Section
   [6.2](../README.md#6-challenges-drawbacks--tradeoffs) enumerates what the public record
   systematically omits, and roughly half of what drill 2 asks for is on that list. Recording
   the absence is a finding.
3. **A model is not a measurement.** The calculator predicts things nobody working through
   this module can measure. When you quote its output, say so in those words. It says so in
   its own banner for exactly this reason.

---

## Shared assumptions

The calculator's presets are this repo's assumptions, not datasheet quotes and not any
system's measured behaviour. Pass `--explain` and it prints the justification for every input
it used.

| Preset | Shape | What it is for |
|---|---|---|
| `dp-training` | One gradient all-reduce on the critical path, 8 ranks per node | The default. Shows compute, then collective bandwidth, then failure taking over |
| `hpc-halo` | Small per-step exchange plus a global reduction, 4 ranks per node | Walks the whole ladder: compute, variance, latency, failure |
| `llm-16k` | A deliberately naive flat all-reduce at the 16k-accelerator scale | Shows *why* nobody runs flat data parallelism there |

Two properties of the model to keep in front of you while you use it:

- The four printed terms **sum to the step time**, so you can always see which one you would
  have to remove to matter.
- The variance *factor* grows with rank count while the variance *term* usually shrinks,
  because the step it multiplies is shrinking faster. That is not a bug and it is a favourite
  interview question; drill 1 asks you to explain it.

`llm-16k` takes its rank count from the 16k-accelerator scale documented in
[the Llama 3 paper](../../../../docs/REFERENCES.md#the-llama-3-herd-of-models-grattafiori-et-al-2024)
and **nothing else**. Its bytes, MTBF and job length are this repo's illustrative inputs. That
paper's own interruption statistics belong to its cluster and its snapshot window; borrowing
them as planning constants is precisely the error section 2.4 of the module warns about. Read
the paper for its figures and supply your own MTBF here.

---

## What "done" looks like

- **Drill 1**: the tool reproduces the module's numbers, rejects `ranks < 1` and a
  `serial_fraction` outside `[0, 1]` with a one-line message, and you can say which ladder row
  your own workload sits on and which single mitigation belongs to that row.
- **Drill 2**: every cell sourced and dated, every gap marked "not published", and a design
  lesson that names a **constraint** rather than a component.
- **Drill 3**: eight diagnoses, at least three marked ambiguous with a named alternative, and
  every mitigation drawn from the row you named rather than the row above it.
- **Drill 4**: a specific compliment, one plainly-stated workload characteristic that breaks
  the design, and findings ordered by expected damage.

If you find yourself writing "this should scale fine", stop and put a number there. In this
track that sentence is the most reliable marker of an analysis that has not been done.

Reference answers with full working are in [`../solutions/`](../solutions/). Read them after
you have written yours down — several of them point out where a different answer would have
been equally defensible, which is information you only get if you have your own answer to
compare.

Useful while you work: the [glossary](../../../../docs/GLOSSARY.md) for any term used without
definition, and the module's [section 8](../README.md#8-cited-references) for the sources
behind every claim it makes.
