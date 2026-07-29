# Solution 2 — Design a checkpoint strategy

Reference answer for [`../exercises/02_checkpoint_strategy.md`](../exercises/02_checkpoint_strategy.md).

**Headline: sharded + async checkpointing takes overhead from 14.8% to 3.6% on-demand, and spot is
about 54% cheaper per useful GPU-hour — but the variable that decides the spot question is
`restart time`, not the checkpoint interval.** That inversion is the point of the exercise.

All arithmetic below uses `waste(T) = C/T + (T/2 + R)/M` and `T* = sqrt(2·C·M)`. Every figure is
reproducible with `tools/checkpoint_planner.py`; the tool is an **analytic model**, not a benchmark.

---

## 1. On-demand baseline

Given `M = 9 h = 32,400 s`, `C = 240 s`, `R = 14 min = 840 s`, step = 0.9 s.

```
T* = sqrt(2 · 240 · 32,400) = sqrt(15,552,000) = 3,943.6 s = 65.7 min = 4,382 steps

write tax = 240 / 3,943.6                    = 6.09%
lost work = (3,943.6/2 + 840) / 32,400       = 8.68%
total                                        = 14.76%   ->  goodput 85.2%
```

A 720-hour run therefore takes **844.7 h** of wall clock; the 124.7 wasted hours across 256 GPUs at
$2.40 cost **~$76,600**.

Note already that `lost work` (8.68%) exceeds `write tax` (6.09%), and that **the restart term alone
is 840/32,400 = 2.6%** — a fixed cost the interval cannot touch. Keep an eye on that term; it becomes
the whole story in §3.

## 2. Making the checkpoint cheaper

| Strategy | Blocking `C` | `T*` | Write | Lost | Total | Goodput | Wasted $ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Synchronous, single writer | 240 s | 65.7 min | 6.09% | 8.68% | **14.76%** | 85.2% | ~$76,600 |
| (a) Sharded × 16 | 15 s | 16.4 min | 1.52% | 4.11% | **5.64%** | 94.4% | ~$26,400 |
| (b) Sharded + async (10% stall) | 1.5 s | 5.2 min | 0.48% | 3.07% | **3.55%** | 96.5% | ~$16,300 |

**The pattern to internalise:** a 160× reduction in blocking cost bought a 4.2× reduction in total
overhead — not 160×. That is the square root at work, plus the irreducible restart term. It is still
the best money on the table, and it is pure infrastructure: **nothing about the model changes.**

**New constraints each strategy introduces:**

- **(a) Sharded.** The checkpoint is now a *set* of files that must be written and read atomically as
  a unit. You need a manifest and a commit protocol — a half-written checkpoint set that looks
  complete is worse than no checkpoint. You also need resharding on load, or you can never resume on
  a different world size (which §3's elastic story requires). The 16-writer limit is a property of
  the filesystem, so it is also a *shared* resource you do not control alone (see §7).
- **(b) Async.** Two hard constraints:
  1. **The background write must complete before the next checkpoint begins.** Here the write takes
     15 s and `T*` is 312 s, so there is 20× headroom — comfortable. If `T*` ever drops near 15 s,
     writes queue, the "async" saving evaporates, and you silently degrade to synchronous. The
     planner tool warns when this is violated.
  2. **The snapshot must be a true copy.** If the background thread reads tensors the training loop
     is concurrently mutating, you persist a *torn* checkpoint: it loads without error and is subtly
     wrong. This is a months-later bug. Require an explicit staging copy (or copy-on-write) and a
     test that resumes from an async checkpoint and compares against a synchronous one.

## 3. The spot decision

At `M = 45 min = 2,700 s`, everything changes character. Using sharded + async (`C = 1.5 s`):

```
T* = sqrt(2 · 1.5 · 2,700) = sqrt(8,100) = 90 s = 100 steps

write tax = 1.5 / 90                    =  1.67%
lost work = (45 + 840) / 2,700          = 32.78%
total                                   = 34.44%  ->  goodput 65.6%
```

**Look at where the 32.78% comes from: 45 s of re-done work and 840 s of restart.** The restart term
alone is 31.1%. The checkpoint interval is now almost irrelevant — you could set it anywhere from 60
to 300 seconds and barely move the total. **On spot, restart time is the design variable.**

Now the business question, in the only unit that answers it — **dollars per hour of *useful*
compute**:

```
on-demand (sharded+async):  $2.40 / 0.9645  = $2.49 per useful GPU-hour
spot      (sharded+async):  $0.75 / 0.6556  = $1.14 per useful GPU-hour
```

**Spot wins by ~54%.** Total job cost: ~$210,900 on spot against ~$458,600 on-demand.

The break-even is worth computing because it is what a reviewer will push on: spot stops winning when
its goodput falls below `0.75 / 2.49 = 30.1%`. Given that the restart term alone is 31%, you reach
that cliff if restart time roughly doubles to ~28 minutes, or if the interruption rate roughly
doubles. **Both are plausible.** So the recommendation is "yes, with monitoring on those two numbers"
— not "yes".

And the corollary that reframes the whole exercise: cut restart from 14 minutes to **4 minutes**
(warm standby capacity, pre-pulled container image, cached checkpoint on local NVMe, fast rendezvous)
and spot goodput rises to ~87.8%, i.e. **$0.85 per useful GPU-hour**. Ten minutes of restart latency
is worth more than every checkpoint optimization in §2 combined. That is the non-obvious conclusion,
and it is the one to lead with.

## 4. Using the 120-second warning

**What the job does in 120 s:** on receiving the interruption notice — (1) finish the current step,
(2) trigger an immediate synchronous-to-host snapshot and flush shards (15 s of I/O, well inside the
budget), (3) write a "clean shutdown at step N" marker, (4) exit non-zero so the supervisor requeues
rather than treating it as success.

**Does it change the answer?** Only slightly, and understanding *why* is the point. The warning
eliminates the `T/2` re-done-work term (45 s of the 885 s lost per interruption, ~5%). It does
nothing about the 840 s restart. Goodput moves from ~65.6% to roughly `1 − 840/2700 ≈ 68.9%`. Real,
but small — **because you were never losing much work; you were losing much time coming back.**

A secondary benefit worth mentioning: if most interruptions are announced, you can *lengthen* the
interval and drop the write tax toward zero, reducing filesystem load. The periodic checkpoint then
exists only for the unannounced failures.

**How the graceful path fails exactly when you need it:**
- The notice is delivered **per instance**, not per job. Your rank 3 gets 120 s; the other 255 ranks
  learn about it only when rank 3 disappears. If your save path is a *collective* operation, it
  deadlocks — the departing rank is trying to participate in a barrier while being terminated.
  The save must be initiated by a broadcast to all ranks, with a timeout, or be rank-local.
- The 120 s is a **maximum, not a guarantee**; capacity can be reclaimed faster.
- If the interruption hits during an existing background write, you now have two writes contending
  and may finish neither.
- The path is exercised only during interruptions, so it is the least-tested code in the system.
  **Test it deliberately** — kill a rank on a schedule in staging — or it will not work the first
  time it matters.

## 5. What must be in the checkpoint

| State | Missing ⇒ symptom | Keep for statistical resume too? |
|---|---|---|
| Model weights | nothing works | yes |
| Optimizer state (Adam moments — roughly 2× the weights, most of the 1.4 TB) | loss spike at resume; adaptation history lost | **yes** — this is a real quality effect, not just reproducibility |
| LR-scheduler state / step count | resumes at the wrong LR, silently | **yes** — same reason |
| RNG states (Python, NumPy, framework CPU, per-device, **per loader worker**) | different noise stream | no — statistically equivalent; drop it if it is expensive |
| Data-loader position + shuffle permutation | re-trains the head of the epoch, may never see the tail | **yes** — this is a data-coverage bug, not a reproducibility nicety |
| Global step / epoch counters | off-by-one in logging, eval, scheduling | yes |
| Config + code commit + container digest | checkpoint is unloadable later | yes — costs bytes, saves months |

**The distinction that matters:** only RNG state is purely a *reproducibility* concern. Everything
else on that list affects the resulting model even if you never care about bit-exactness. Teams that
say "we only need statistical reproducibility, so we skip the extra state" usually skip the wrong
things.

## 6. Retention

Three tiers, because the three purposes are genuinely different:

| Tier | Frequency | Location | Retain | Purpose |
|---|---|---|---|---|
| Hot | every `T*` (~5 min) | node-local NVMe + one replica | last 2 | crash recovery only; deleted within the hour |
| Warm | every ~2 hours | shared cluster filesystem | last 6 | recover from a bad node *set*, or a corrupted hot copy |
| Cold | every ~12 hours, plus every eval milestone | object storage, immutable, tagged with config + commit | full run | provenance, restart-from-a-known-good after a divergence, model release |

Justification: hot checkpoints are for the failure that happens every 45 minutes and are worthless
after the next one is written — paying object-storage latency for them is pure waste. Cold
checkpoints are the ones you will want in three months when someone asks how the model was produced,
and they must be immutable and tagged or they are worthless for that purpose.

Storage arithmetic: 1.4 TB × (2 hot + 6 warm) = 11.2 TB steady-state, plus ~60 cold copies for a
30-day run = 84 TB. **Price that before you propose it** — at typical object-storage rates it is a
real line item, and if it is unacceptable the honest lever is fewer cold copies, not fewer hot ones.

## 7. The shared-filesystem thundering herd

**What happens:** 200 jobs each independently apply Young/Daly, each concludes it should checkpoint
every ~5–15 minutes, and their writes drift into phase alignment (they all start after the same
maintenance window, or all restart after the same incident). The filesystem sees 200 × 1.4 TB of
burst writes in the same minute. `C` — which every job's model treats as a **constant** — rises by an
order of magnitude. Every job's model then says "checkpoint even more often", which is exactly wrong.
This is a feedback loop that can take down a storage tier.

**Prevention:**
1. **Randomise the phase.** Each job offsets its first checkpoint by `uniform(0, T)`. One line of
   code, removes the correlation, costs nothing.
2. **Measure `C` continuously and feed it back.** If the observed write time rises, the interval
   should *lengthen*, not shorten. Use the measured `C`, not the design-time one.
3. **Write hot checkpoints to node-local storage** (§6). This removes most of the shared-tier load
   entirely and is the structural fix.
4. **Rate-limit at the storage tier** so one job's bad configuration cannot starve the others, and
   alert on aggregate checkpoint bandwidth as a cluster-level metric.

## Recommendation (top of the design doc)

> Run this job on **spot capacity** with **sharded, asynchronous checkpointing** every ~90 seconds
> (100 steps), writing hot checkpoints to node-local NVMe and promoting to object storage every 12
> hours. Modelled goodput is ~66%, giving ~$1.14 per useful GPU-hour against ~$2.49 on-demand — about
> a **54% saving**, roughly $248,000 on this run.
>
> The decision rests on two numbers, both of which must be monitored: **restart latency (14 min)** and
> **interruption rate (45 min MTBF)**. Spot stops being cheaper if goodput falls below ~30%, which
> happens if restart latency roughly doubles or interruptions roughly double. The single highest-value
> follow-up is **cutting restart latency to under 5 minutes** — worth more than every checkpoint
> optimization combined, and it would raise the saving to roughly 66%.
>
> Revisit if: the run must hit a hard deadline (spot's variance in wall-clock time is the risk that
> the cost model does not price), or if the model grows enough that 16 parallel writers no longer
> keep the write under a few seconds.

**The "it depends on X" pivot:** X is *whether the deadline or the budget binds*. Everything above
optimises dollars. If there is a hard delivery date, spot's wall-clock variance — not its cost — is
the governing risk, and the correct answer flips to on-demand with the same checkpoint strategy.
Say which one you are optimising before you present the number.
