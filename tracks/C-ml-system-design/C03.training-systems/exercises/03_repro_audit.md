# Exercise 3 — Reproducibility audit

> Reference answer: [`../solutions/03_repro_audit_solution.md`](../solutions/03_repro_audit_solution.md)
> — write yours first.

## The situation

A research team reports a **+1.3 point** improvement on the primary eval metric and wants to ship
the model. Before it goes to production you are asked to sign off that the result is real and the
run can be reproduced.

Here is the setup, exactly as it exists today:

- Training is launched from a **Jupyter notebook** on a shared box. The notebook calls a training
  function with keyword arguments typed into the cell.
- `torch.manual_seed(42)` is called at the top of the notebook. Nothing else is seeded.
- The data loader uses **8 workers**. Worker seeding is left at the framework default and nobody has
  looked at it.
- The dataset lives at `/mnt/data/corpus/latest/`. It is a **symlink**. It was last repointed
  "a couple of weeks ago, I think".
- Augmentation is on-the-fly: random crop, horizontal flip, and a colour jitter whose probability is
  read from a config file that is **not** in version control.
- Training is data-parallel across 8 GPUs using standard collectives, in **mixed precision**.
- Two checkpoints exist. The team is not certain which one produced the reported number.
- The eval was run **once**, on a holdout set that "was regenerated at some point to fix a leak".
- The baseline number being compared against was produced **five weeks ago** by a different person,
  on a different container image.
- Nobody recorded the driver or framework version for either run.
- The git working tree had uncommitted changes when the run started. The team believes they were
  "just print statements".

## Your task

**1. Enumerate every obstacle to an exact re-run.** Go through the setup and list them. For each,
state (a) whether it blocks **bit-exact** reproducibility, **statistical** reproducibility, or
**both**, and (b) whether it is *silent* — meaning nothing errors and you would never know.

**2. Rank by damage.** Which three obstacles most threaten the **validity of the +1.3 claim** — as
opposed to merely making the run hard to repeat? Justify the ranking. This is the distinction between
"we can't reproduce it" and "it may not be true", and conflating them is the most common mistake here.

**3. Is +1.3 even a result?** State the minimum evidence you would require before believing a
+1.3 point improvement. Include what you would ask the team to run and roughly what it costs them.
Then answer the uncomfortable version of the question: what would make you say the claim is
*unsupported* rather than merely *unverified*?

**4. Two fix lists.**
   - **Minimum bar to ship** — the smallest set of changes you would require before this model goes
     to production. Justify why each is non-negotiable.
   - **Full bit-exact reproducibility** — what it would additionally take, and what it costs. Be
     specific about the performance cost of deterministic kernels and pinned reduction order, and
     name at least one thing that may not be achievable at all.

**5. Recommend a target.** For *this* team and *this* model, which kind of reproducibility should be
the standing requirement? Write the one-sentence claim you would put in the model card — the sentence
must be both honest and defensible under questioning.

**6. Mixed precision and collectives.** Explain, in terms a non-specialist manager would follow, why
two runs of this job with identical seeds can still produce different weights. Then give the
one-sentence expert version, naming the underlying property of floating-point arithmetic.

## Constraints on your answer

- Do not recommend "make everything deterministic". State what you would *not* fix, and why paying
  for it would be the wrong call.
- Separate **process** fixes (what the team must do differently) from **infrastructure** fixes (what
  the platform should make impossible to get wrong). The second list is the more valuable one.
- One page. This is a sign-off memo, not a manifesto.
