# B02 Exercises

Do these yourself before peeking at [`../solutions/`](../solutions/). Exercises 1 and 2 are
paper drills — the whole point is that you can do them without a GPU, in an interview, on a
whiteboard. Exercise 3 is the code drill that makes it real.

This directory has its own `Makefile`, so you can build and run without leaving it:

```bash
make                              # list targets
make all GPU_ARCH=gfx950          # build every exercise for your GPU
make 03_place_your_kernel.exe     # build just one
make run                          # build + run all (FAIL is expected until solved)
make clean
```

Set `GPU_ARCH` to your device (`gfx950` MI350, `gfx942` MI300, `gfx90a` MI200); find it with
`/opt/rocm/bin/offload-arch`. The equivalent raw command, spelled out:

```bash
hipcc -O3 --offload-arch=gfx942 -I../hip 03_place_your_kernel.cpp -o 03_place_your_kernel.exe
./03_place_your_kernel.exe 78500        # 78500 = your measured P, in GFLOP/s
# NVIDIA: rename to .cu, replace hip* with cuda*, and: nvcc -O3 -arch=sm_90 -I../cuda ...
```

(The exercise reuses the module's `HIP_CHECK` macro via `-I../hip`.)

| # | File | Skill |
|---|------|-------|
| 1 | `01_classify_kernels.md` | Compute AI for nine real operators, compare to two different ridge points, pick the right lever for each. |
| 2 | `02_ai_of_attention.md` | Derive `I(s, d)` for fused and unfused attention; find the sequence length where fusion changes the classification. |
| 3 | `03_place_your_kernel.cpp` | Instrument a reduction and an FMA-heavy kernel to report their own FLOPs and bytes, then place both on the ceilings you measured. |

Each drill maps to an interview competency: operator-level intensity reasoning / algebraic
derivation under questioning / turning a measurement into a decision. Exercise 1 Part C and
Exercise 2 Part F are the parts interviewers actually probe, because they test whether you know
where your model stops working.

---

## Do these in order

```
hip/measure_ceilings.exe   ->  gives you P and B for YOUR machine
        |
        v
01_classify_kernels.md     ->  paper: AI and verdicts on Device R's round numbers
02_ai_of_attention.md      ->  paper: the derivation that justifies FlashAttention
        |
        v
03_place_your_kernel.exe P ->  code: your own kernels on your own measured roofline
```

Exercises 1 and 2 use the fictional "Device R" with deliberately round ceilings so the algebra
stays legible. Exercise 3 uses your real hardware, where the numbers are ugly and the reasoning
is identical.

---

## Expected output shape

Values marked `<measured>` come from your GPU and will differ from anyone else's. The
**arithmetic intensities are analytic** and must match exactly — if your `I` for `sum_reduce` is
not ~0.25 FLOP/byte, the accounting is wrong, not the hardware.

### Exercise 3 — before solving

The kernels are already correct; only the instrumentation is missing, so the verification
column passes while the instrumentation line fails:

```text
$ ./03_place_your_kernel.exe 78500
device      : <name> (<arch>), <N> CUs
n           : 67108864 floats, <B> blocks x 256 threads

ceilings
  memory roof  B : <measured> GB/s     (measured here, coalesced copy)
  compute roof P : 78500.0 GFLOP/s  (from measure_ceilings)
  ridge point    : <computed> FLOP/byte

kernel              ms       FLOPs       bytes   I (F/B)     GFLOP/s   roof GF/s   % roof  verdict         verify
------------ --------- ----------- ----------- --------- ----------- ----------- -------- --------------- ------
sum_reduce   <measured>  0.000e+00   0.000e+00     0.000         0.0         0.0     0.0% TODO              PASS
fma_heavy    <measured>  0.000e+00   0.000e+00     0.000         0.0         0.0     0.0% TODO              PASS

kernels verified : PASSED
instrumentation  : FAILED (TODOs still return 0)
```

### Exercise 3 — after solving

```text
kernel              ms       FLOPs       bytes   I (F/B)     GFLOP/s   roof GF/s   % roof  verdict         verify
------------ --------- ----------- ----------- --------- ----------- ----------- -------- --------------- ------
sum_reduce   <measured>  6.773e+07   2.684e+08     0.252  <measured>  <computed> <comp.>% memory-bound      PASS
fma_heavy    <measured>  8.590e+09   5.369e+08    16.000  <measured>  <computed> <comp.>% <depends on P/B>  PASS

kernels verified : PASSED
instrumentation  : PASSED (TODOs filled in)
```

The `fma_heavy` verdict is genuinely machine-dependent: `I = 16 FLOP/byte` sits right on top of
the fp32 ridge point of current data-center GPUs (~15–30 FLOP/byte). On a bandwidth-rich part it
reads memory-bound; on a compute-rich one, compute-bound. A drill whose answer depends on the
hardware is the correct drill — that is what "measure, do not assume" means in practice.

### Paper drills

No program output. Check your algebra against
[`../solutions/01_classify_kernels_solution.md`](../solutions/01_classify_kernels_solution.md)
and
[`../solutions/02_ai_of_attention_solution.md`](../solutions/02_ai_of_attention_solution.md).
Headline answers to check yourself against, in case you only want the punchlines:

- Exercise 1: the fp32 ridge is 15 FLOP/byte and the bf16 matrix-core ridge is 300; the thin-K
  bf16 GEMM (row 8) is **memory-bound** despite `I = 124`, because it must be judged against the
  matrix-core ceiling.
- Exercise 2: `I_unfused → (4d+5)/8 ≈ 64.6` FLOP/byte and never crosses the ridge at any
  sequence length; `I_fused ≈ 0.505·s` crosses it at `s ≈ 595`.
