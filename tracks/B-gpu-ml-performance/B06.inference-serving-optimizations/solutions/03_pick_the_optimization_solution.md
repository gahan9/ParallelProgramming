# Solution 3 — Triage: five symptoms, five levers

Each answer gives the mechanism, the fix, the confirming measurement, and — most importantly — the
falsifier.

---

## 3.1 — "TTFT fine, TPOT terrible at high load"

**Mechanism.** TTFT is dominated by prefill, which is compute-bound and roughly independent of how
many *other* sequences are decoding. TPOT is dominated by the decode step, whose cost is
`fixed_weight_pass + marginal_per_sequence × batch`. As load rises the scheduler admits more
sequences, the marginal term grows, and every step gets slower for everyone. TPOT rising with load
while TTFT holds steady is the *signature* of a batch that has grown past its useful point — the
`t_step(B) = 5.00 + 0.086·B` curve from exercise 2, walked to the right.

**Primary fix.** Cap the admission size (`--max-num-seqs`) at the value that maximizes goodput, not
throughput. In the exercise-2 sweep that was 32, where raising the cap to 128 bought 1.6% throughput
and cost 12% of SLO-conforming traffic. Expect TPOT to drop back inside budget immediately and total
throughput to fall a few percent — that is the trade you are deliberately making.

**Confirming measurement.** Plot p95 TPOT against *concurrent sequences* (not against wall-clock time).
If the diagnosis is right, the relationship is monotone and close to affine, and you can read the
right cap straight off the x-axis where it crosses the budget.

**Falsifier.** If p95 TPOT is high but shows *no* correlation with concurrency, the batch is not the
cause. Look at the **inter-token-latency tail** instead: you are probably looking at symptom 3.4
(prefill interference), where mean TPOT is inflated by a handful of long stalls rather than by
uniformly slower steps.

---

## 3.2 — "GPU at 30% utilization with a full queue"

**Two mechanisms produce this, and they need opposite fixes.**

**(a) Static / request-level batching.** The engine formed a batch and will not admit anyone new
until the *last* sequence in it finishes. Because output lengths are long-tailed, most slots are
occupied by sequences that finished long ago, and the GPU is computing on padding. The simulator
makes this exact: under static batching **71% of computed slot-time is spent on already-finished
sequences**, throughput is 816 tok/s against continuous batching's 1541 tok/s on the identical trace,
and goodput is **8%** against 100%.

**(b) Launch-bound tiny kernels.** Decode kernels are small and numerous; if per-step launch and
scheduling overhead is a large share of step time, the GPU spends its life idle between kernels. The
simulator's `launch_overhead_pct` shows ~4–6% at these batch sizes, but at batch 1–2 with an
under-optimized stack it can be far larger.

**The one panel that separates them:** a **kernel-level timeline** (`rocprofv3 --sys-trace`, or Nsight
Systems). Under (a) the GPU is executing large kernels back-to-back at high occupancy — it is *busy*
doing useless work, and the coarse "utilization" counter is simply lying to you, because it reports
"a kernel is resident," not "useful work is happening." Under (b) you see visible gaps *between*
kernels and short kernel durations.

**Fixes.** (a) Switch to continuous / in-flight batching — iteration-level scheduling
([Orca, OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu)). This is the single
largest throughput win available in serving and it is a scheduler change, not a kernel change.
(b) Capture the decode step in a **HIP/CUDA graph** so one launch replays the whole step, and fuse
small elementwise kernels.

**Falsifier.** If you switch to continuous batching and utilization stays at 30% with the queue still
full, it was (b) — or the queue is blocked on something outside the GPU entirely (tokenizer on a
single CPU thread, KV-block exhaustion causing constant preemption, or a synchronous logit
post-processing step). Check whether the scheduler is *able* to admit: a full queue with free KV
blocks and an idle GPU means the bottleneck is on the host.

---

## 3.3 — "OOM only with long contexts"

**Mechanism.** The admission controller is sizing sequences by the *advertised maximum* context rather
than by what they actually hold, or the allocator cannot satisfy a large contiguous request even though
enough total memory is free (external fragmentation). Exercise 1 quantifies how violent this is: with
contiguous reservation, raising max context from 8k to 128k drops concurrency from **16 to 1**, and a
30k-token prompt in a `gqa-8b`-shaped model costs `128 KiB × 30000 ≈ 3.7 GiB` of KV for **one**
request. A burst of those and you are out of memory regardless of the average load.

**Primary fix.** Paged KV allocation with a block table
([PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)): allocate 16-token blocks on demand, so a
sequence costs what it holds. Add explicit **preemption** (swap or recompute the lowest-priority
sequence when blocks run out) so that memory pressure degrades latency instead of crashing the
process. Exercise 1 shows 67 vs 1 concurrent sequences at 128k advertised context — a 67× difference
from allocation policy alone.

**Confirming measurement.** Instrument **KV blocks allocated vs total** over time, and correlate the
peak against prompt length. If the crash follows a block-utilization spike driven by a few long
prompts, this is it. Also check the allocator: total free bytes at the moment of failure. If there was
plenty free and the allocation still failed, it is external fragmentation and paging fixes it directly.

**Why "halve the admission cap" is the wrong fix.** It buys stability by taxing the 99% of traffic
that never sends a long prompt — you halve throughput for everyone to survive a rare event. The right
shape of fix is either per-request cost accounting (admit by *tokens*, not by *requests*) or paging.

**Falsifier.** If OOM persists with paging enabled and block accounting looks healthy, the memory is
going somewhere else — activation peaks during a large prefill (a 30k-token prefill chunk has a large
transient footprint), or an allocator that never returns freed blocks. Chunked prefill also bounds
*that* peak, which is a second reason to enable it.

---

## 3.4 — "Throughput fine but p99 awful"

**Mechanism.** This is the one that is **invisible in mean TPOT**. A new request's prefill occupies a
whole decode step, so every sequence currently generating sees one long inter-token gap. Averaged over
a 165-token response, a single 200 ms stall adds ~1.2 ms to that request's mean TPOT — nothing. But
the user sees the freeze. The simulator makes the split explicit on an identical trace:

| policy | mean TPOT | p99 ITL | max ITL |
|---|---:|---:|---:|
| continuous (whole-prompt prefill) | 7.79 ms | 69.9 ms | **293.1 ms** |
| continuous + chunked prefill | 7.76 ms | 29.6 ms | **29.8 ms** |

Same throughput, same mean TPOT, **10× difference in the worst gap**. "Nobody can reproduce it on
demand" is expected: it requires another user submitting a long prompt at the wrong moment.

**Primary fix.** **Chunked prefill** with a token budget
([Sarathi-Serve](https://arxiv.org/abs/2403.02310)): split the prompt and piggyback it onto decode
steps so ongoing decodes never stall for more than one chunk's worth of work. Expect the ITL tail to
collapse to roughly `fixed_step + per_token × chunk_size` and TTFT to worsen slightly.

**Confirming measurement.** Stop looking at mean TPOT. Record **every** inter-token interval and plot
its p99/max, then correlate the spikes against prefill events. If the spikes line up with admissions
of long prompts, you have it. (If you cannot easily measure per-token intervals in your stack,
instrumenting that is the first task — you cannot manage a tail you do not record.)

**Falsifier.** If ITL spikes do *not* correlate with prefill events, look for preemption/swapping
(a sequence evicted for KV blocks and later recomputed shows exactly this shape), a periodic host-side
stall (garbage collection, metrics flush, logit post-processing), or network jitter in the streaming
path — which is not a GPU problem at all and no amount of chunking will fix it.

---

## 3.5 — "Cost per token too high"

**Mechanism.** Every request re-computes the prefill for an identical 1800-token system prompt. With
60-token answers, the *prompt* is 30× the size of the response: you are paying for 1800 tokens of
compute to produce 60. At 47 µs/token that is ~85 ms of GPU time per request thrown away on work whose
answer is byte-identical to the previous request's.

**Primary fix.** **Prefix / prompt caching.** Keep the KV blocks of the shared prefix and let every
request that starts with the same tokens point at them — the block table makes this a reference count,
not a copy. RadixAttention in SGLang generalizes it to a radix tree with LRU eviction and cache-aware
scheduling, so *conversation* prefixes hit too, not just the system prompt
([Zheng et al.](https://arxiv.org/abs/2312.07104)). Expect prefill work per request to fall by roughly
the shared fraction of the prompt, which here is most of it — and prefill was ~52% of GPU time in
exercise 2's load.

**Why this is usually the biggest win in a real product** and simultaneously the least discussed: it
is pure deduplication. There is no accuracy tradeoff (the KV values are *exactly* what recomputing
would produce), no new failure mode in the model, and it attacks work rather than making work faster.

**Confirming measurement.** **Prefix cache hit rate**, plus prefill tokens computed per request versus
prompt tokens received. If prompts share 1800 of ~1900 tokens and your hit rate is near zero, the
saving is sitting on the table. Then re-measure cost per output token, which is the metric finance
actually asked about.

**Falsifier.** If the hit rate is already high, or prompts are genuinely unique (per-user RAG context
injected at the *front* of the prompt), prefix caching cannot help — and the fix moves to reducing
cost per token directly: weight/KV quantization (B04), a smaller model for this task, or better
batching. Note the design trap this reveals: **put the variable part of the prompt last**. A prefix
cache can only reuse a *prefix*, so injecting per-user context ahead of the shared system prompt
destroys the hit rate for free.

---

## Ranking by expected value

| Rank | Symptom | Fix | Why here |
|---|---|---|---|
| 1 | 3.2(a) | Continuous batching | Largest single win, and it is a solved problem — every modern engine ships it. If you are still on static batching, nothing else matters yet. |
| 2 | 3.5 | Prefix caching | Removes work instead of accelerating it. No accuracy risk. Days of work, not months. |
| 3 | 3.3 | Paged KV + preemption | Turns a crash into a latency degradation. Also a solved problem; the cost is adopting an engine that has it. |
| 4 | 3.4 | Chunked prefill | Usually a configuration flag. Fixes a real user-visible defect that no aggregate dashboard shows. |
| 5 | 3.1 | Tune the batch cap | Cheapest of all to *try*, ranked last because it only redistributes a fixed capacity — it cannot create any. |

**The one I would not do: introduce speculative decoding.** It is the wrong answer to at least three
of these five, and here is why:

- **3.1 and 3.2** are *high-load, large-batch* situations. Speculative decoding buys its speedup from
  spare compute that exists only while decode is bandwidth-bound. At a full batch the spare compute is
  gone; the model in [`../sim/speculative_model.py`](../sim/speculative_model.py) shows the speedup at
  α=0.7, k=4 falling from 2.31× to 0.87× — an actual *slowdown* — as the marginal verification cost
  per position rises from 0 to 0.5. Adding speculation to an overloaded server makes it slower.
- **3.5** is a duplicated-work problem. Speculation makes each token cheaper in latency terms but adds
  a second model's weights to HBM and discards a majority of drafted tokens (55.7% at α=0.7, k=4).
  For cost per token it is close to the *worst* available lever.

Speculative decoding is a **latency** optimization for **low-to-moderate concurrency** — an
interactive single-user or few-user endpoint where the GPU is bandwidth-bound and idle compute is
genuinely free. It is lossless, so it costs no accuracy; but it costs memory, a second model to train,
version, evaluate and deploy, and an acceptance rate that silently decays as traffic shifts away from
the draft model's distribution. That operational bill is why it is last, not the math.
