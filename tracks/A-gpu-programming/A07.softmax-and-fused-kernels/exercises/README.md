# A07 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Write real, compiling,
edge-case-correct code — softmax is the one kernel where "it runs" and "it is correct" are
very different claims.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                              # list targets
make all GPU_ARCH=gfx950          # build every exercise for your GPU
make 01_safe_softmax.exe          # build just one
make run                          # build + run all (FAIL is expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200); find it with
`/opt/rocm/bin/offload-arch`. The equivalent raw command, if you prefer to see it spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 01_safe_softmax.cpp -o 01_safe_softmax.exe && ./01_safe_softmax.exe
# NVIDIA: rename to .cu, replace hip* with cuda*, __shfl_xor -> __shfl_xor_sync(0xffffffffu, ...),
#         and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercises reuse the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_safe_softmax.cpp` | Implement a block max reduction and the max-subtraction trick; make an overflowing row finite. |
| 2 | `02_masked_softmax.cpp` | Apply an additive `-inf` mask in *every* pass, and stop a fully-masked row from emitting NaN. |
| 3 | `03_online_softmax_math.md` | Derive the online rescaling update, prove it equals the 3-pass result, and count HBM bytes. |

Each drill maps to an interview competency (numerical-stability reasoning / edge-case
correctness under masking / algorithmic derivation + bandwidth accounting). Solutions include
the reasoning, not just the answer key.

> [!TIP]
> Do them in order. Exercise 1 builds the `block_reduce_max` that exercise 2 hands you, and
> exercise 3 explains why the labs' online kernel is the one that generalises to attention.

---

## Sample sandbox output

The **PASS/FAIL column and the non-finite counts are the contract** — reproduce those exactly.
Timings and GB/s are hardware-dependent and shown as `<hw>`; fill in your own and compare
against your copy-bandwidth ceiling from A02 Lab 1. Error magnitudes marked `~` are
deterministic (the inputs come from a fixed LCG) but their exact digits are not worth
memorising — the order of magnitude is the signal. NVIDIA output is equivalent
(`CUDA error …` instead of `HIP error …`).

### Exercise 1 — Safe softmax (`01_safe_softmax.exe`)

**As shipped** (`block_reduce_max` returns `0.0f`, pass 3 unshifted). Row 0 holds `x = 100`, so
`expf(x)` overflows: the normalizer becomes `inf`, the big score becomes `inf/inf = nan`, and
every other entry in that row becomes `finite/inf = 0`.

```text
$ ./01_safe_softmax.exe
rows = 4096, cols = 512, block = 256
time / iter    : <hw> ms
effective BW   : <hw> GB/s   (3 reads + 1 write)
non-finite     : 1
max |err|      : ~6e-08
row 0 sum      : nan  (must be 1.0)
result         : FAILED
```

Note what this teaches about *testing*: `max |err|` looks perfect, because it is computed only
over finite outputs and rows 1+ happen to be correct (passes 2 and 3 agree on the same wrong
shift of `0`). **A max-absolute-error check alone would have passed this kernel.** You need the
finiteness count and the row sum too.

**After TODO 1 only** (real row max, but pass 3 still forgets the shift). Every row now
normalizes to `exp(max)` instead of `1`, and the poisoned row overflows one pass later — this
time in the numerator:

```text
non-finite     : 1
max |err|      : ~1e-01
row 0 sum      : inf  (must be 1.0)
result         : FAILED
```

**After TODO 1 + TODO 2:**

```text
non-finite     : 0
max |err|      : ~6e-08
row 0 sum      : 1.000000  (must be 1.0)
result         : PASSED
```

> [!TIP]
> The input is deliberately **all negative** apart from the poison. If you use `0.0f` as the
> identity of the max reduction instead of `-INFINITY`, every row max clamps to 0, the shift is
> wrong, and the result is quietly incorrect rather than loudly `nan`. That identity is the
> single most commonly botched line in a hand-written softmax.

### Exercise 2 — Masked softmax (`02_masked_softmax.exe`)

**As shipped** the mask is applied only in the output pass, so the distribution is normalized
over positions that were supposed to be dropped. Each row's probabilities are therefore too
small by roughly the factor `valid / cols`, which is exactly what the row sums show. Row 0
(nothing masked) and row 3 (everything masked, which collapses to zeros for the wrong reason)
pass by accident:

```text
$ ./02_masked_softmax.exe
row   valid   row sum       max |err|     non-fin   result
----- ------- ------------- ------------- --------- ------
0     256     1.000000      ~6e-08        0         PASSED
1     200     ~0.78         ~1e-03        0         FAILED
2     128     ~0.50         ~4e-03        0         FAILED
3     0       0.000000      0.000e+00     0         PASSED
4     64      ~0.25         ~1e-02        0         FAILED
5     1       ~0.004        ~1e+00        0         FAILED
6     250     ~0.98         ~3e-04        0         FAILED
7     17      ~0.07         ~5e-02        0         FAILED

overall: FAILED
```

Row 5 is the loudest: with one valid position the answer must be exactly `1.0`, and the
unmasked normalizer turns it into `~0.004` — a max error of ~1.0.

**After TODO 1** the mask reaches the max and normalizer passes — and row 3's max becomes
`-inf`, so `exp(-inf - -inf) = exp(nan)` and the entire row goes non-finite:

```text
row   valid   row sum       max |err|     non-fin   result
----- ------- ------------- ------------- --------- ------
0     256     1.000000      ~6e-08        0         PASSED
1     200     1.000000      ~6e-08        0         PASSED
2     128     1.000000      ~6e-08        0         PASSED
3     0       0.000000      0.000e+00     256       FAILED
4     64      1.000000      ~6e-08        0         PASSED
...
overall: FAILED
```

**After TODO 1 + TODO 2** every row passes and row 3 sums to exactly `0`:

```text
3     0       0.000000      0.000e+00     0         PASSED

overall: PASSED
```

> [!WARNING]
> The `non-fin 256` line on row 3 is the whole exercise. In a real training run that NaN does
> not stay local: it flows into `P · V`, into the residual stream, into the gradients, and one
> step later every parameter in the model is `nan`. Fully-masked rows are produced by ordinary
> things — a truncated sequence, a padded batch tail, a sliding window that starts before token 0.

### Exercise 3 — Online-softmax derivation (no program)

Paper drill. Check your algebra and byte counts against
[`../solutions/03_online_softmax_math_solution.md`](../solutions/03_online_softmax_math_solution.md).
Expected headlines: the rescale factor is `exp(m_k − m_{k+1}) ∈ (0, 1]`; the merge operator is a
monoid with identity `(−∞, 0)`; and at `R = 8192`, `C = 4096`, fp32 the tensor is **128 MiB**, so
the safe 3-pass moves **512 MiB**, the online 2-pass **384 MiB**, and the cached 1-pass
**256 MiB** — a **2×** span with identical arithmetic.

### Module labs, for comparison

```text
$ make run-hip
== hip/softmax.exe ==
device            : <your GPU> (warpSize = 64)
softmax over rows : rows = 8192, cols = 1024 (32.0 MiB per tensor), block = 256
poison            : x[0][7] = 100.0  (expf overflows fp32 above ~88.7)

kernel                           passes  ms        GB/s        max |err|   nonfin result
-------------------------------- ------- --------- ----------- ----------- ------ ------
1 naive unsafe (no max sub)      3       <hw>      <hw>        inf         1      FAILED
2 safe 3-pass (max trick)        4       <hw>      <hw>        ~6e-08      0      PASSED
3 online 2-pass (running max)    3       <hw>      <hw>        ~6e-08      0      PASSED
4 online 1-pass (row in regs)    2       <hw>      <hw>        ~6e-08      0      PASSED

row 0 (the poisoned row): sum over the row, then the first 6 outputs.
Expected: sum = 1.0, and ~1.0 at col 3 with ~0 everywhere else.
  1 naive unsafe (no max sub)      sum=       nan |   0.000000   0.000000   0.000000        nan ...
  2 safe 3-pass (max trick)        sum=  1.000000 |   0.000000   0.000000   0.000000   1.000000 ...
```

Kernel 1 failing is the intended outcome: a demonstrated failure teaches the max trick far
better than a paragraph about it.
