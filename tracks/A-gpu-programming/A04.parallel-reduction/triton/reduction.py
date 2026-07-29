"""Lab (Triton) - Sum reduction, and what the compiler does for you.

The C++ labs walk the whole optimization ladder by hand: you choose the
addressing pattern, place every barrier, unroll the last warp, and decide how
many elements each thread consumes. Triton removes the first four rungs. You
describe a *block* of work and call `tl.sum`; the compiler emits the
divergence-free, conflict-free, shuffle-based intra-block tree for you.

What Triton does NOT decide for you is the part that actually matters at the
top of the ladder (README S5):

  * how many elements each program consumes  -> BLOCK, and whether you cascade
  * how the per-block partials are combined  -> second launch vs atomics

So this file benchmarks exactly those two choices:

  1. `block`   - one program per BLOCK-sized chunk, then a second launch to sum
                 the partials. The direct translation of ladder stages 3-6.
  2. `cascade` - a fixed, machine-sized grid where each program grid-strides
                 over many tiles. The translation of ladder stage 7, and the
                 one that should reach bandwidth peak.
  3. `torch.sum` - the library baseline you have to beat to justify a custom
                 kernel (usually you cannot; that is a useful thing to learn).

Every result is verified against a float64 reference, because a float32 tree and
a float32 sequential sum do not agree bit for bit - see README S6.

Run: python reduction.py
Requires: torch + triton (see ../../../docs/SETUP.md). Works on CUDA or ROCm torch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sum_block_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """One program reduces one BLOCK-sized chunk to a single partial.

    `other=0.0` on the masked load is the identity element for addition - the
    same trick as padding the shared tile in the C++ stages. Get the identity
    wrong (e.g. for a max reduction) and the tail block silently poisons the
    answer; that is exercise 1.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
    tl.store(out_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def sum_cascade_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Fixed grid; each program grid-strides over as many tiles as it takes.

    `acc` is a full BLOCK-wide vector living in registers, so the serial
    accumulation costs one vector add per tile and no shared memory at all. Only
    after the loop does the (logarithmic, compiler-generated) tree run - exactly
    the algorithm cascade of ladder stage 7.
    """
    pid = tl.program_id(0)
    step = BLOCK * tl.num_programs(0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for base in range(pid * BLOCK, n, step):
        offs = base + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0)

    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def sum_block(x: torch.Tensor, block: int = 4096) -> torch.Tensor:
    """Two-level reduction: n -> ceil(n/BLOCK) partials -> 1."""
    n = x.numel()
    partials = torch.empty(triton.cdiv(n, block), device=x.device, dtype=torch.float32)
    sum_block_kernel[(partials.numel(),)](x, partials, n, BLOCK=block)

    # The partial array must fit one program's BLOCK on the second pass. With
    # n = 16.7M and BLOCK = 4096 there are 4096 partials, so one pass suffices;
    # a larger n would need another level (the same recursion the C++ driver does).
    assert partials.numel() <= block, "add another reduction level for this n"
    total = torch.empty(1, device=x.device, dtype=torch.float32)
    sum_block_kernel[(1,)](partials, total, partials.numel(), BLOCK=block)
    return total


def sum_cascade(x: torch.Tensor, block: int = 1024, grid: int = 1024) -> torch.Tensor:
    """Machine-sized grid, grid-stride loop, then one small second pass."""
    n = x.numel()
    programs = min(grid, triton.cdiv(n, block))
    partials = torch.empty(programs, device=x.device, dtype=torch.float32)
    sum_cascade_kernel[(programs,)](x, partials, n, BLOCK=block)

    total = torch.empty(1, device=x.device, dtype=torch.float32)
    sum_block_kernel[(1,)](partials, total, programs, BLOCK=triton.next_power_of_2(programs))
    return total


def benchmark(fn, iters: int = 50) -> float:
    """Milliseconds per call, measured with CUDA/HIP events after a warmup."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to torch. See ../../../docs/SETUP.md")

    n = 1 << 24  # 16,777,216 floats = 64 MiB, same as the C++ labs
    torch.manual_seed(0)
    x = torch.rand(n, device="cuda", dtype=torch.float32)

    # float64 reference. Comparing against a float32 CPU loop would be comparing
    # two differently-wrong answers; the double sum is the one to trust.
    ref = x.double().sum().item()
    read_bytes = x.element_size() * n

    cases = [
        ("triton block  (stage 3-6 shape)", lambda: sum_block(x)),
        ("triton cascade (stage 7 shape)", lambda: sum_cascade(x)),
        ("torch.sum      (library)", lambda: torch.sum(x)),
    ]

    print(f"n = {n} floats ({read_bytes / 2**20:.1f} MiB)")
    print(f"reference (fp64): {ref:.6f}\n")
    print(f"{'implementation':<34} {'ms':>8} {'GB/s':>9} {'rel err':>10}  result")
    print(f"{'-' * 34} {'-' * 8} {'-' * 9} {'-' * 10}  ------")

    for name, fn in cases:
        got = fn().item()
        rel = abs(got - ref) / abs(ref)
        ms = benchmark(fn)
        gbps = read_bytes / (ms / 1e3) / 1e9
        print(f"{name:<34} {ms:8.4f} {gbps:9.1f} {rel:10.2e}  "
              f"{'PASSED' if rel < 1e-4 else 'FAILED'}")

    print("\nExpect the cascade to beat the plain block version and to land close\n"
          "to torch.sum, which is itself a tuned cascade. All three are\n"
          "memory-bound: judge them against your GPU's peak HBM bandwidth,\n"
          "not against each other's FLOPs.")


if __name__ == "__main__":
    main()
