# Solution 3 — Reproducibility audit

Reference answer for [`../exercises/03_repro_audit.md`](../exercises/03_repro_audit.md).

**Headline: the +1.3 point claim is *unsupported*, not merely unverified.** The eval set changed
between the baseline and the candidate, and there is one seed. Those two facts alone mean the number
does not measure what the team thinks it measures. Determinism is a real problem here too, but it is
the *second* problem, and fixing it first would be wasted work.

That ordering — validity before reproducibility — is the whole exercise.

---

## 1. Every obstacle, classified

| # | Obstacle | Blocks | Silent? |
|---|---|---|---|
| 1 | Launched from a notebook; hyperparameters typed into a cell | both | **yes** — cell execution order is unrecorded and undiffable |
| 2 | Only `torch.manual_seed(42)`; Python `random` and NumPy unseeded | bit-exact | **yes** |
| 3 | 8 loader workers with default seeding | bit-exact (and possibly *quality*, if workers duplicate augmentation) | **yes** |
| 4 | Dataset path is a **symlink**, repointed at an unknown time | **both** | **yes** — the most dangerous single item |
| 5 | Augmentation probability in an unversioned config file | both | **yes** |
| 6 | Mixed precision + collectives ⇒ non-deterministic reduction order | bit-exact | **yes** |
| 7 | Two checkpoints, provenance unknown | both | no — you know you don't know |
| 8 | Eval run **once**, no seed replicates | statistical | **yes** — a single sample masquerading as a measurement |
| 9 | Holdout set "regenerated at some point to fix a leak" | **both, and validity** | **yes** |
| 10 | Baseline from 5 weeks ago, different person, different container | **both, and validity** | **yes** |
| 11 | Driver / framework versions unrecorded for both runs | bit-exact | no |
| 12 | Uncommitted working-tree changes ("just print statements") | both | **yes** — nobody has verified that claim |

Eleven of twelve are silent. That is the characteristic property of reproducibility debt: **nothing
ever fails**, so nothing forces the issue until a result matters.

## 2. The three that threaten *validity*

Reproducibility failures make a result hard to repeat. **Validity** failures make it possibly false.
Ranked:

1. **#9 — the holdout set was regenerated.** If the baseline was scored on the old holdout and the
   candidate on the new one, +1.3 measures *the difference between two eval sets*, not between two
   models. This is not a reproducibility problem; it is a broken comparison, and it can produce a
   difference of any sign or size. Nothing else matters until this is resolved.
2. **#8 — a single seed, single eval.** Seed-to-seed variance on a fine-tuning run is routinely on
   the order of the improvement being claimed. Without replicates you cannot distinguish +1.3 from
   noise. Note that this is not solved by determinism: a bit-exact run reproduces the *same* sample,
   which tells you nothing about whether the difference is real. Determinism and statistical evidence
   are orthogonal, and people confuse them constantly.
3. **#4 — the dataset symlink moved.** If the training data changed between baseline and candidate,
   the comparison confounds a model change with a data change. Worse: if the repoint pulled in data
   overlapping the holdout, the +1.3 is **leakage**, and the model will look wonderful right up until
   it ships.

Note what is *not* in this top three: mixed-precision non-determinism (#6). It is real, it blocks
bit-exactness, and it is almost certainly irrelevant to whether the +1.3 is true.

## 3. Is +1.3 a result?

**Minimum evidence I would require:**

1. **One eval set, frozen, versioned by content hash**, and a re-score of *both* baseline and
   candidate checkpoints on it. This is cheap — inference only, no retraining — and it either
   preserves the +1.3 or dissolves it. **Do this first**; it may end the discussion.
2. **Three seeds** of the candidate and, if affordable, three of the baseline, reporting mean and
   spread. If +1.3 is inside the seed spread, it is not a result. The cost is 3× training, which is
   the honest price of a shippable claim; if that is unaffordable, say so explicitly and label the
   result "single-seed, indicative" rather than shipping it as established.
3. **Confirmed provenance**: which checkpoint, which data snapshot, which commit, for both runs.
4. **Per-slice metrics**, not just the aggregate. A +1.3 average that is +4 on the majority slice and
   −2 on a minority slice is a different decision entirely — and the aggregate hides it.
5. **A leakage check** between the training snapshot and the holdout, given #4 and #9.

**When I would call it unsupported rather than unverified:** if the two numbers were produced on
different eval sets and cannot be re-scored on a common one, the claim has no meaning and should be
withdrawn, not caveated. Same if a leakage check comes back positive. "Unverified" means *we believe
it and haven't checked*; "unsupported" means *the measurement does not measure the thing*. The
regenerated holdout puts this claim in the second category until proven otherwise.

## 4. Two fix lists

### Minimum bar to ship

| Fix | Why non-negotiable |
|---|---|
| Freeze the eval set; content-hash it; re-score both models on it | without this there is no comparison at all |
| Pin the dataset to an **immutable snapshot ID**, never a mutable symlink | a moving symlink silently invalidates every past and future result |
| Move the config into version control and launch from a committed script | the config *is* the experiment; if it isn't diffable, nothing is |
| Record commit SHA + dirty flag, container digest, framework and driver versions, automatically | the "just print statements" claim must be checkable, not trusted |
| Seed-replicate the headline claim (≥3 seeds) and report the spread | a single run is a sample, not a measurement |
| Report per-slice metrics alongside the aggregate | catches harm concentrated in a subgroup before users do |
| Record which checkpoint produced the reported number | otherwise you ship a model nobody can identify |

None of those requires deterministic kernels, and none costs throughput.

### Full bit-exact reproducibility (additional)

| Fix | Cost |
|---|---|
| Seed Python `random`, NumPy, framework CPU and per-device generators; derive a **distinct** seed per loader worker | ~free; the per-worker seeding may also fix a real augmentation-diversity bug |
| Enable deterministic algorithm selection | measurable throughput loss; the size depends entirely on which ops are hot, so **measure it on this model rather than quoting a number** |
| Pin reduction order in collectives / avoid atomics in reductions | further throughput loss; may require giving up the fastest communication path |
| Pin framework, communication library, compiler, **driver**, and container digest | operational cost: you cannot take security patches without re-validating |
| Persist and restore all RNG streams in the checkpoint | small |

**What may not be achievable at all:** some operations have no deterministic implementation, and
enabling strict determinism makes them raise an error rather than silently vary — see PyTorch's
[reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html). If one of those
is central to the model, bit-exactness is off the table and you should say so rather than promising
it. Determinism also does not survive a hardware change: the same code on a different GPU
architecture may select different kernels.

## 5. Recommended target, and the model-card sentence

**Statistical reproducibility, as the standing requirement.** This is a product model being compared
against a baseline; what the business needs is confidence that the improvement is real and will
recur, not that the weights are byte-identical. Bit-exactness should be switched on *temporarily and
deliberately* — when debugging a divergence, or when an audit demands it — and switched back off.

The sentence for the model card:

> *Trained from commit `<sha>` on data snapshot `<id>` in container `<digest>`. The reported
> +X.X point improvement is the mean over 3 seeds (spread ±Y.Y) on frozen eval set `<hash>`, with the
> baseline re-scored on the same set. Runs are not bit-exact: mixed-precision collectives use a
> non-deterministic reduction order, which we accept for throughput.*

That sentence is defensible under questioning because every clause is checkable, and it pre-empts the
"is it reproducible?" question by answering the version of it that matters.

## 6. Mixed precision and collectives, two ways

**For a manager:** when many computers add up their results, they don't always add them in the same
order — whoever finishes first goes first. Computer arithmetic on fractional numbers gives very
slightly different answers depending on the order you add them in. Those tiny differences get
amplified over hundreds of thousands of training steps, so two runs that started identically end up
in slightly different places. It is not a bug, and it doesn't mean either run is wrong.

**Expert version, one sentence:** floating-point addition is **not associative**, so an all-reduce
(or any atomic-based reduction) that combines partial sums in a hardware-scheduled, run-dependent
order produces run-dependent rounding, and low-precision formats have fewer mantissa bits so each
rounding is larger — the same tree-shape dependence derived in
[A04 Parallel Reduction](../../../A-gpu-programming/A04.parallel-reduction/README.md), operating at
cluster scale.

## What I would *not* fix, and why

- **I would not enable deterministic kernels for production training.** It costs throughput on every
  run forever to buy a property this team does not need. Enable it per-incident.
- **I would not chase bit-exactness against the 5-week-old baseline.** That run's environment is
  gone. Re-run the baseline in the current environment instead — cheaper, and it produces a
  comparison that is actually valid.
- **I would not ban notebooks.** They are the right tool for looking at data. The rule is narrower
  and enforceable: *a number that leaves the team must come from a committed script.*

## Process fixes vs infrastructure fixes

**Process** (the team must do differently): freeze eval sets, run seed replicates, report slices,
commit before launching, record which checkpoint produced which number.

**Infrastructure** (the platform should make it impossible to get wrong) — the more valuable list,
because process discipline decays and infrastructure does not:

- The launcher **refuses to start** on a dirty working tree unless explicitly overridden, and stamps
  the commit SHA, container digest, and resolved config into the run record automatically.
- Dataset paths resolve to **immutable snapshot IDs**; mutable symlinks are not accepted as a data
  source.
- Eval sets are **content-addressed**; the eval harness records the hash with every score and refuses
  to compare two scores computed on different hashes.
- The checkpoint writer persists config, commit, and environment **by construction**, so an
  unidentifiable checkpoint cannot exist.
- The experiment tracker is written by the training loop, not by a human remembering to log.

Every item on that list converts a silent failure into a loud one at the moment it happens, which is
the only reliable defence against debt whose defining property is that nothing ever errors.
