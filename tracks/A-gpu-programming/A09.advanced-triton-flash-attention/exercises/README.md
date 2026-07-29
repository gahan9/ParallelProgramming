# A09 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real,
running, edge-case-correct code.

These drills are **Triton/Python only** — no `hipcc`/`nvcc` step, so there is no
`Makefile` here. Run them directly, or drive them from the module root:

```bash
python 01_causal_mask.py            # from this directory
python 02_online_rescale.py

cd .. && make run-exercises         # from the module root: runs all of them
```

Every drill guards on `torch.cuda.is_available()` and points at
[`../../../../docs/SETUP.md`](../../../../docs/SETUP.md) if no GPU is visible. Each
one **compiles and runs as shipped** and prints `FAILED` until you finish the TODOs
— that gap is the lesson.

| # | File | Skill |
|---|------|-------|
| 1 | `01_causal_mask.py` | Add causal masking to a working non-causal tiled kernel: skip fully-future K/V blocks (the ~2x win) *and* mask inside the diagonal block (the correctness fix). |
| 2 | `02_online_rescale.py` | Implement the `(max, sum, acc)` online rescaling update from a skeleton. This is the heart of FlashAttention. |
| 3 | `03_flash_math.md` | Paper drill: HBM bytes for naive vs fused at N = 1k/4k/16k, the sequence length where the score matrix breaks a cache/HBM budget, and why fusing changes the asymptotics. |

Each maps to an interview competency: masking and index reasoning (1), numerically
safe streaming reductions (2), and quantitative IO analysis (3).

---

## Sample sandbox output

The blocks below show the **structure** of what the scripts print and the expected
**order of magnitude** of the errors. They are illustrative, not a captured run on any
particular GPU — milliseconds depend on your device, dtype, and Triton/PyTorch
versions, so measure your own and do not compare across machines.

### Exercise 1 — Causal masking (`01_causal_mask.py`)

As shipped, the kernel is non-causal, so it disagrees with the causal reference and
`causal` is no faster than `full`:

```text
shape         : batch=2 heads=8 seq=1000 head_dim=64
full   (ms)   : <x.xxx>
causal (ms)   : <x.xxx>   (1.00x vs full)
max abs err   : far above the tolerance of 0.02
result        : FAILED

Expected until TODO 2 is done: ...
Causal is not meaningfully faster than full - TODO 1 is still open.
```

After TODO 2 (the diagonal mask) the error collapses to fp16 rounding; after TODO 1
(the loop bound) the causal run visits roughly half the K/V blocks:

```text
full   (ms)   : <x.xxx>
causal (ms)   : <y.yyy>   (clearly above 1x, approaching but never reaching 2x)
max abs err   : order 1e-3   (tolerance 0.02)
result        : PASSED
```

The ideal is 2x — half the blocks visited — and `01_causal_mask_solution.py` prints the
exact visited fraction for your shape plus the three reasons the measurement lands
below it.

> [!TIP]
> If you do TODO 2 but not TODO 1, you get `PASSED` with **no speedup** — a correct
> kernel that threw away the entire performance reason for causal masking. That is a
> real review finding, not a hypothetical.

### Exercise 2 — Online rescaling (`02_online_rescale.py`)

The shipped skeleton normalizes each block to its own max and never rescales:

```text
max abs err   : orders of magnitude above the tolerance
result        : FAILED
```

Note the failure *signature*: finite, plausible-looking numbers in roughly the right
range. A missing rescale does not crash — it silently over-weights whichever block
held the largest scores. After the correct update:

```text
max abs err   : order 1e-3   (tolerance 0.02)
result        : PASSED
```

### Exercise 3 — Byte arithmetic (no program)

Paper drill. Check your table against
[`../solutions/03_flash_math_solution.md`](../solutions/03_flash_math_solution.md).
The headline answers to aim for: the naive-to-fused traffic ratio is exactly
**`1 + N/d`** (so **129x** at `N = 16384, d = 128`), and the naive path's arithmetic
intensity saturates at **`d/2`** flops per byte no matter how long the sequence gets.
