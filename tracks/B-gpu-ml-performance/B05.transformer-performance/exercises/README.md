# B05 Exercises

Three paper drills. There is nothing to compile and nothing to run — the skill being tested is
producing correct performance arithmetic **on a whiteboard, under time pressure**, which is exactly
the interview format these questions come from.

Work each one by hand first. Only then check yourself with the calculator:

```bash
cd ..
python tools/model_calculator.py --help          # options
make run-exercises                               # prints drill 1's config, calculated
```

Using the calculator before you have your own answer defeats the purpose. The point is not the
number; it is being able to *produce* the number when there is no tooling in the room.

| # | File | Skill being tested |
|---|------|--------------------|
| 1 | [`01_size_a_model.md`](01_size_a_model.md) | Size a model end to end from its config: parameters, weight memory across dtypes, prefill vs decode FLOPs, and the predicted decode tokens/s on named hardware. |
| 2 | [`02_kv_cache_budget.md`](02_kv_cache_budget.md) | Turn an HBM budget into a concurrency number under MHA vs GQA, then propose and *quantify* a change that doubles it. |
| 3 | [`03_regime_diagnosis.md`](03_regime_diagnosis.md) | Read profiler-style evidence, name the regime, and pick the optimization that follows — including cases where the obvious answer is wrong. |

Reference answers with full working: [`../solutions/`](../solutions/).

---

## The formulas you should be able to reproduce from memory

If you cannot write these down cold, reread [§2 of the module README](../README.md#2-first-principles)
before starting. Everything in these drills is an application of one of them.

```
parameters per layer        (4 + 2m) d²                 ~ 12 d² at m = 4, MHA
                            attention 4d², FFN 2md²
                            GQA: replace 4d² with 2d² + 2·d·d_kv

forward FLOPs per token     2 × params
train FLOPs per token       6 × params                  (fwd 2 + bwd 4)

attention FLOPs per layer   4 N² d                      per sequence, prefill
attention/FFN crossover     N = (2 + m) d               ~ 6d at m = 4

KV cache bytes per seq      2 × layers × N × d_kv × dtype_bytes
                            d_kv = kv_heads × d_head

arithmetic intensity of a   T d / (d + 2T)  FLOP/byte   in fp16; ~= T for small T
d × d projection, T tokens

decode tokens/s ceiling     HBM bandwidth / bytes read per step
                            bytes per step = weights + batch × KV_per_seq

machine ridge point         peak FLOP/s / peak bandwidth
```

---

## Conventions used in every drill

- **Decimal units**: 1 GB = 10⁹ bytes, matching how vendors quote both capacity and bandwidth. This
  keeps `tokens/s = GB/s ÷ GB` an exact division.
- **fp16/bf16 = 2 bytes**, int8/fp8 = 1 byte, int4 = 0.5 bytes.
- **A "decode step"** advances every sequence in the batch by exactly one token.
- Ignore norm gains, biases, and positional embeddings — they are `O(d)` and never change an answer.
- Hardware figures, when a drill names a GPU (vendor-published dense peaks):

| | HBM bandwidth | Capacity | Dense bf16 peak | Ridge point |
|---|---:|---:|---:|---:|
| AMD MI300X | 5.3 TB/s | 192 GB | 1307.4 TFLOP/s | 247 FLOP/byte |
| NVIDIA H100 SXM | 3.35 TB/s | 80 GB | 989.4 TFLOP/s | 295 FLOP/byte |

---

## How these are graded (and what an interviewer is watching)

Getting the digits right is the easy half. What distinguishes a strong answer:

1. **You state the formula before you plug in numbers.** It shows the answer is derived, not recalled.
2. **You carry units.** `GB/s ÷ GB = 1/s` is what makes the tokens/s result obviously right.
3. **You say what you ignored and why it is safe.** "I'm dropping the norm gains, they're `O(d)`
   against `O(d²)`" is worth more than three extra significant figures.
4. **You sanity-check the magnitude.** "38 tokens/s for a 70 B model on one GPU — that matches what
   single-stream serving actually feels like" catches an order-of-magnitude slip instantly.
5. **You name the regime before recommending anything.** Every optimization in this subject is
   correct in one regime and useless in the other.
