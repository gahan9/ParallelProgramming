# B04 Exercises

Do these yourself before opening [`../solutions/`](../solutions/). Two are paper drills
and one is code; all three are the kind of question a review or an interview actually
asks, because "which format?" is a decision you will defend far more often than you will
write a conversion intrinsic.

```bash
make            # list targets
make all        # syntax-check the Python drill
make run        # run it (TODO rows are expected until you solve it)
make clean
```

No GPU and no ROCm/CUDA toolchain needed for any of these — `numpy` is the only
dependency, and the Python drill guards the import with an actionable message if it is
missing.

| # | File | Skill |
|---|------|-------|
| 1 | `01_format_selection.md` | Choose fp32/tf32/bf16/fp16/fp8/int8 per scenario and justify it with range, precision, and byte numbers — including a KV-cache sizing calculation and the fp8 FNUZ-vs-OCP portability trap. |
| 2 | `02_loss_scaling.md` | Prove gradients underflow in fp16 from a magnitude histogram, size a power-of-two loss scale, order clipping vs unscaling correctly, and total the real memory bill of mixed precision. |
| 3 | `03_quantize_error.py` | Implement per-channel and group-wise int8 quantization, account for scale bytes honestly, and explain why group-wise contains outlier damage rather than removing it. |

Each drill maps to an interview competency: **format reasoning** / **numerical debugging
of a training run** / **quantization engineering with an honest cost model**. The
solutions include the reasoning, not just the answer — the reasoning is the transferable
part.

---

## What "done" looks like

### Exercise 1 — no program

You should end with a table whose every cell is filled: format · the range number ·
the precision number · the throughput argument · the risk accepted. If you cannot fill
the risk column, you have chosen a format without deciding what would make you revert.

### Exercise 2 — no program

Expect these landmarks (derive them, do not trust this list): roughly **two thirds** of
the gradients are subnormal in fp16 and about **3%** round to exactly zero; the largest
usable power-of-two loss scale is **2^16**; and mixed precision with Adam costs the *same*
bytes per parameter as pure fp32 training, which surprises most people the first time
they add it up.

### Exercise 3 — runs, but incomplete until you fill the TODOs

Before (three of four TODOs unfilled — this is the expected starting state):

```text
scheme                      rel err W rel err Wx rest max err rest rms err     bytes   vs fp32
----------------------------------------------------------------------------------------------
per-tensor                    0.04044    0.03993     0.049083     0.028496         0      nanx
per-channel                not implemented yet (fill in the TODO)
per-group g=128            not implemented yet (fill in the TODO)
...
```

After (reference run on the same seed, so your numbers should match to the last digit —
this drill is deterministic on purpose):

```text
scheme                      rel err W rel err Wx rest max err rest rms err     bytes   vs fp32
----------------------------------------------------------------------------------------------
per-tensor                    0.04044    0.03993     0.049083     0.028496     16386     4.00x
per-channel                   0.02341    0.02228     0.048995     0.016522     16512     3.97x
per-group g=128               0.01654    0.01623     0.048446     0.011509     16640     3.94x
per-group g=64                0.01171    0.01133     0.048446     0.007919     16896     3.88x
per-group g=32                0.00819    0.00836     0.047309     0.005176     17408     3.76x
per-group g=16                0.00565    0.00569     0.042899     0.002966     18432     3.56x
per-group g=8                 0.00472    0.00486     0.042247     0.002510     20480     3.20x
```

> [!TIP]
> The interesting column is `rest max err`: it barely moves as `g` shrinks, while
> `rest rms err` improves by more than 6x. Understanding *why* is the whole exercise, and
> it is the reason real quantizers do more than shrink the group size.

### The module labs, for comparison

`make run-hip` in the module directory prints real numbers for the format table, the
fp16 `exp()` overflow point, the vanishing update, the fp16-vs-fp32-vs-fp64 dot product,
the summation-order spread, and the mixed-precision GEMM/axpy tables. Those are measured
on your GPU, so they are not reproduced here — run them.
