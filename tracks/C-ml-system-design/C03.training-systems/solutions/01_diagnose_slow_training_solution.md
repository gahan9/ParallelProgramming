# Solution 1 — Diagnose a slow training run

Reference answer for [`../exercises/01_diagnose_slow_training.md`](../exercises/01_diagnose_slow_training.md).

**Headline: there are three separate problems in this report, and the team is working on none of
them.** The learning rate is fine — the loss is decreasing normally, which rules out the hypothesis
they have spent two days on. That is the first thing to say, gently, out loud.

---

## 1. Ordered checks, and why that order

The ordering principle: **liveness → shape → distribution → change history**. Cheap and
discriminating first; anything requiring a code change or a restart last.

| # | Check | Confirms | Eliminates |
|---|---|---|---|
| 1 | Is the **step counter** advancing at a steady rate? | the job is slow, not hung | a collective deadlock (§3.3) — must be ruled out before anything else, because it has a completely different fix |
| 2 | **Utilization shape**, not mean. Is it sawtooth or flat? | sawtooth ⇒ input starvation | a kernel/step problem; a flat 61% would point at Track B instead |
| 3 | **Step-time breakdown**: time in data-wait vs forward/backward vs collective | which stage owns the 1.4 s | guesswork; this is the single most informative measurement and it is one profiler run |
| 4 | **Per-rank** step-time distribution — is one rank consistently slowest? | a straggler | uniform slowness (which would mean a communication or global problem) |
| 5 | **Host CPU per process**: are loader workers saturating cores, and how many workers are configured? | CPU-bound decode/augment | an I/O-bound loader, which needs a different fix |
| 6 | **What the re-export changed**: file count, file size distribution, compression, container format | a storage-layout regression | a coincidence — but the timing makes this the prime suspect |
| 7 | Diff the config and container digest against **last month's** working run | any silent environment change | "nothing changed" (something always changed) |
| 8 | **Checkpoint contents**: does it include optimizer, scheduler, RNG, loader position? | the restart bump | the belief that the bump is harmless |

Checks 1–4 are read-only dashboard/profiler work and take minutes. Only 6 and 8 require touching
anything.

## 2. Root causes

**Most likely: input-pipeline starvation caused by the dataset re-export.** The evidence is
specific and mutually reinforcing:

- Utilization **sawtooth** 0→100→0 is the textbook starvation signature — compute finishes, then
  everything waits for the next batch.
- **Host CPU at ~95% across all cores** says the bottleneck is CPU-side work: decode, augment,
  collate. The GPUs are idle *because the CPUs cannot keep up*.
- The regression **coincides with a re-export** that "added a field". The overwhelmingly common
  version of this: the export changed the on-disk layout — larger shards, a different compression
  codec, uncompressed images that now need more decode, or worst of all a shard format replaced by
  millions of small files. Any of those multiplies per-sample CPU or metadata cost.
- GPU memory flat at 45% and loss healthy: nothing about the *model* changed.

**Second most likely: a straggler, or a tail in the loader, that scales with rank count.**
Mean step time 1.4 s against **p99 6.8 s** is a 4.9× tail, and it *got worse with more ranks*. That
is a distribution problem living underneath the starvation problem, and fixing the loader will not
remove it.

**Third (separate) problem: the checkpoint is incomplete** — see §4.

## 3. Why p99 grows with scale

Because a synchronous step is a **barrier**: every rank waits for the slowest one, so the step time
is the **maximum** over ranks, not the mean.

If each rank independently has probability `p` of a slow step, the probability that *at least one*
rank is slow in a given step is `1 − (1 − p)^N`. With `p = 1%`, going from 64 to 128 ranks takes
that from `1 − 0.99^64 ≈ 47%` to `1 − 0.99^128 ≈ 72%`. **The per-rank distribution did not change at
all; only the number of draws did.** This is why tail latency, not average latency, is the quantity
that matters at scale — and why "our average GPU utilization is fine" is not a defence.

Doubling the ranks also doubles the concurrent load on any *shared* resource (the filesystem, the
object store, the metadata server), so the per-rank distribution usually gets worse too. Both effects
push the same direction.

## 4. The loss bump at restart is not harmless

Twice-daily restarts each followed by "a little bump that recovers" is the classic signature of an
**incomplete checkpoint**. In order of likelihood:

1. **LR-scheduler state not saved** — the schedule restarts (often back into warmup), so the model
   takes several hundred steps at the wrong learning rate. "Recovers within a few hundred steps" is
   almost a fingerprint for this.
2. **Optimizer state not saved** — Adam's moment estimates reset to zero and must re-accumulate,
   which produces exactly this bump shape.
3. **Data-loader position not saved** — the epoch restarts from sample 0, so the model re-sees the
   head of the dataset. This one does *not* produce a bump, which is precisely why it is the
   dangerous one: at two restarts a day, the tail of the dataset may never be trained on.

**What to ask for:** the keys in the saved checkpoint dict, and a plot of learning rate against step
across a restart boundary. If the LR jumps, you have your answer in one chart. Also ask what fraction
of the dataset the run has actually visited — if the loader restarts each time, that number is much
lower than they think, and it is a **quality** bug masquerading as a performance annoyance.

## 5. Fix list, by expected improvement ÷ effort

| Fix | Expected gain | Cost | Notes |
|---|---|---|---|
| **Increase loader workers / prefetch depth** | possibly large, possibly zero | ~an hour | Try first because it is free — but host CPU is already at 95%, so this may be *already saturated*. If more workers do not help, that itself is diagnostic: you are CPU-bound, not concurrency-bound. |
| **Revert or fix the export format** | likely the whole 2.1× regression | days | The regression coincides with it. Sharded sequential format (WebDataset/TFRecord-style), sensible shard size, and decode-cheap encoding. Highest expected value. |
| **Move augmentation off the hot path** | large if decode/augment is the CPU cost | days | Precompute deterministic transforms (resize, tokenize) offline. Keep random crop/flip on-the-fly — precomputing those destroys the regularisation you wanted. |
| **Fix the checkpoint** (optimizer + scheduler + RNG + loader position) | no throughput gain; fixes a **correctness** bug | hours | Do it anyway. It is cheap and it is the only *quality* item on this list. |
| **Pinned memory + non-blocking H2D** | small here | hours | Worth doing, but the bottleneck is upstream of the transfer. Same physics as [A01 Lab 4](../../../A-gpu-programming/A01.foundations-and-programming-model/README.md#lab-4-optional--profile-it), one level up. |
| **Find and evict the straggler** | attacks the 4.9× tail | ongoing | Needs per-rank p99 monitoring, which they do not currently have. Build the monitoring even before you find the culprit. |
| **DECLINE: add more GPUs** | negative | large | See below. |
| **DECLINE: keep tuning the learning rate** | zero | two days already spent | The loss is decreasing normally. This hypothesis is already falsified by their own dashboard. |

**Why more GPUs is the wrong move — and the sentence to say in the meeting:** the GPUs they have are
idle 39% of the time. Adding more does not add throughput, because the bottleneck is the host-side
input pipeline and a shared storage tier; it adds *load* to the very resource that is saturated. It
also makes the tail worse — see §3, where more ranks means more chances for one to be slow. You would
be paying more money to lower utilization.

**Sanity check with the model** (`tools/throughput_budget.py`, an analytic model, not a measurement):
at 128 GPUs × 95 samples/s with a loader capping out at 6,800 samples/s, utilization lands around
62% — consistent with the 61% observed. That agreement does not *prove* the diagnosis, but it does
mean the starvation hypothesis is quantitatively sufficient to explain the whole gap, so you do not
need to posit a second cause for the *mean*. The tail still needs its own explanation.

## 6. The "it depends on X" pivot

**X is what the re-export actually changed on disk.** Concretely: the file-count and file-size
distribution before and after.

- If the export replaced large shards with **many small files**, the bottleneck is storage metadata
  and latency, not CPU — and the fix is re-sharding, not more workers. Host CPU at 95% would then be
  workers spinning on I/O wait, which some monitoring reports as busy.
- If the export switched to a **more expensive codec** or dropped a pre-resize step, it is genuinely
  CPU-bound, and precomputation is the fix.
- If the export changed nothing structural and the regression is coincidental, my top recommendation
  is wrong, and I would move the straggler investigation to the top — the p99 evidence stands on its
  own regardless.

**What I would measure next:** a single profiled step with the data-wait time broken out, plus
per-rank step times for 1,000 steps. Those two artifacts settle every open question above, and both
are one run.
