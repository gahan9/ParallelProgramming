#!/usr/bin/env python3
"""Plot an ASCII roofline and classify kernels against it.

The roofline model in one line:

    attainable FLOP/s = min(P, I * B)

where P is peak compute throughput, B is peak memory bandwidth, and I is the
kernel's arithmetic intensity (FLOPs per byte moved from the memory level being
modelled). The ridge point I_ridge = P / B separates the memory-bound regime
(left) from the compute-bound regime (right).

Standard library only - no numpy, no matplotlib - so it runs anywhere Python 3
runs, including a bare container on a cluster login node.

Typical use, with ceilings from ``measure_ceilings``:

    python roofline_plot.py --peak-gflops 78500 --peak-gbps 4650 --demo-kernels

    python roofline_plot.py --peak-gflops 78500 --peak-gbps 4650 \
        --kernel "my_softmax, 5*2**24, 8*2**24, 410"

Kernel spec is ``name, flops, bytes[, achieved_gflops]``. The numeric fields
accept arithmetic expressions (``2*4096**3``) so you can paste the formula you
derived rather than a pre-multiplied constant.
"""

from __future__ import annotations

import argparse
import ast
import math
import operator
import sys
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Expression parsing
# --------------------------------------------------------------------------- #

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def eval_number(text: str) -> float:
    """Evaluate a pure-arithmetic expression such as ``2*4096**3``.

    Walks the AST and rejects anything that is not a numeric literal or a
    basic arithmetic operator, so a spec string can never execute code.
    """

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"not a number: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](visit(node.operand))
        raise ValueError("only numeric literals and + - * / // % ** are allowed")

    stripped = text.strip()
    if not stripped:
        raise ValueError("empty numeric field")
    try:
        tree = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not an arithmetic expression: {stripped!r}") from exc
    return visit(tree)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Kernel:
    """One point on the roofline."""

    def __init__(self, name: str, flops: float, byts: float,
                 achieved_gflops: Optional[float] = None) -> None:
        if byts <= 0:
            raise ValueError(f"{name}: bytes must be > 0 (got {byts})")
        if flops < 0:
            raise ValueError(f"{name}: FLOPs must be >= 0 (got {flops})")
        self.name = name
        self.flops = flops
        self.bytes = byts
        self.achieved_gflops = achieved_gflops

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes

    def roof_gflops(self, peak_gflops: float, peak_gbps: float) -> float:
        """attainable = min(P, I * B); units work out because GB/s * FLOP/byte = GFLOP/s."""
        return min(peak_gflops, self.intensity * peak_gbps)


def parse_kernel(spec: str) -> Kernel:
    fields = [f.strip() for f in spec.split(",")]
    if len(fields) not in (3, 4):
        raise ValueError(
            f"expected 'name, flops, bytes[, achieved_gflops]', got {spec!r}")
    name = fields[0]
    if not name:
        raise ValueError(f"kernel name must not be empty in {spec!r}")
    flops = eval_number(fields[1])
    byts = eval_number(fields[2])
    achieved = eval_number(fields[3]) if len(fields) == 4 else None
    return Kernel(name, flops, byts, achieved)


def demo_kernels() -> List[Kernel]:
    """Analytic FLOP and byte counts for the operators worked out in the README.

    These are derived counts, not measurements: nothing here claims to be a
    timing from any machine. Every entry is fp32 on purpose, so the single
    compute ceiling passed on the command line applies to all of them. Mixing
    dtypes on one roofline is a mistake - see README Section 6.
    """
    n = 2 ** 24            # ~16.8M elements for the elementwise/reduction cases
    f4 = 4                 # fp32, bytes per element
    gemm_n = 4096          # square GEMM
    seq, head = 4096, 128  # attention, one head, forward only

    # exp() is charged as 1 FLOP throughout, the usual accounting convention.
    attn_flops = 4 * seq**2 * head + 5 * seq**2
    # Unfused: the S x S score matrix is written and read back twice through HBM.
    attn_bytes_unfused = f4 * (4 * seq * head + 4 * seq**2)
    # Fused: Q, K, V read once and O written once; the S x S matrix never lands.
    attn_bytes_fused = f4 * (4 * seq * head)

    return [
        Kernel("vector_add fp32", 1 * n, 3 * f4 * n),
        Kernel("relu fp32", 1 * n, 2 * f4 * n),
        Kernel("sum_reduce fp32", 1 * n, 1 * f4 * n),
        Kernel("softmax fp32 (3-pass)", 5 * n, 5 * f4 * n),
        Kernel("softmax fp32 (fused)", 5 * n, 2 * f4 * n),
        Kernel("gemm fp32 N=4096", 2 * gemm_n**3, 3 * f4 * gemm_n**2),
        Kernel("attention fp32 unfused S=4096", attn_flops, attn_bytes_unfused),
        Kernel("attention fp32 fused S=4096", attn_flops, attn_bytes_fused),
    ]


# --------------------------------------------------------------------------- #
# ASCII chart
# --------------------------------------------------------------------------- #

_ROWS_PER_DECADE = 4
_COLS_PER_DECADE = 9
_Y_DECADES = 5
_MARKERS = "abcdefghijklmnopqrstuvwxyz"


def _decade_span(values: Sequence[float], pad_low: int, pad_high: int) -> Tuple[int, int]:
    positive = [v for v in values if v > 0]
    if not positive:
        return 0, 1
    lo = math.floor(math.log10(min(positive))) - pad_low
    hi = math.ceil(math.log10(max(positive))) + pad_high
    return lo, max(hi, lo + 1)


def render_chart(peak_gflops: float, peak_gbps: float, kernels: Sequence[Kernel]) -> str:
    """Log-log roofline as a character grid.

    Mermaid and markdown cannot draw a log-log plot, and matplotlib may not be
    installed, so the chart is built by hand: one column per intensity step,
    ``#`` on the roof, a letter where each kernel lands.
    """
    ridge = peak_gflops / peak_gbps
    plotted = [k for k in kernels if k.intensity > 0]
    skipped = [k for k in kernels if k.intensity <= 0]

    x_values = [ridge] + [k.intensity for k in plotted]
    x_lo, x_hi = _decade_span(x_values, pad_low=1, pad_high=1)
    x_decades = x_hi - x_lo
    width = x_decades * _COLS_PER_DECADE + 1

    y_hi = math.ceil(math.log10(peak_gflops))
    y_lo = y_hi - _Y_DECADES
    y_decades = _Y_DECADES
    height = y_decades * _ROWS_PER_DECADE + 1

    def col_of(intensity: float) -> int:
        frac = (math.log10(intensity) - x_lo) / x_decades
        return min(width - 1, max(0, round(frac * (width - 1))))

    def row_of(gflops: float) -> Optional[int]:
        if gflops <= 0:
            return None
        frac = (math.log10(gflops) - y_lo) / y_decades
        row = round((1.0 - frac) * (height - 1))
        return row if 0 <= row < height else None

    grid = [[" "] * width for _ in range(height)]

    ridge_col = col_of(ridge)
    for r in range(height):
        grid[r][ridge_col] = ":"

    for c in range(width):
        intensity = 10 ** (x_lo + x_decades * c / (width - 1))
        row = row_of(min(peak_gflops, intensity * peak_gbps))
        if row is not None:
            grid[row][c] = "#"

    legend: List[str] = []
    for idx, k in enumerate(plotted):
        marker = _MARKERS[idx % len(_MARKERS)]
        value = k.achieved_gflops if k.achieved_gflops else k.roof_gflops(peak_gflops, peak_gbps)
        row, col = row_of(value), col_of(k.intensity)
        if row is None:
            placed = "off the plotted range"
        elif grid[row][col] in _MARKERS:
            # Two kernels with the same intensity and rate land on one cell. Say so
            # rather than silently overwriting the earlier marker.
            placed = f"same cell as '{grid[row][col]}'"
        else:
            grid[row][col] = marker
            placed = "on chart"
        kind = "achieved" if k.achieved_gflops else "at its roof"
        legend.append(f"  {marker}  {k.name} ({kind}, {placed})")

    gutter = 10
    lines = [
        f"  attainable GFLOP/s   (P = {peak_gflops:,.0f} GFLOP/s, "
        f"B = {peak_gbps:,.0f} GB/s, ridge = {ridge:.2f} FLOP/byte)",
        "",
    ]
    for r in range(height):
        exponent = y_hi - r / _ROWS_PER_DECADE
        label = f"1e{int(round(exponent)):+03d}" if abs(exponent - round(exponent)) < 1e-9 else ""
        lines.append(f"{label:>{gutter - 2}} |{''.join(grid[r])}")
    lines.append(" " * (gutter - 1) + "+" + "-" * width)

    ticks = [" "] * width
    for d in range(x_decades + 1):
        c = min(width - 1, round(d * (width - 1) / x_decades))
        text = _format_intensity(10 ** (x_lo + d))
        start = min(max(0, c - len(text) // 2), width - len(text))
        for i, ch in enumerate(text):
            ticks[start + i] = ch
    lines.append(" " * gutter + "".join(ticks).rstrip())
    lines.append(" " * gutter + "arithmetic intensity (FLOP/byte, log scale)")
    lines.append("")
    lines.append("  # roof = min(P, I x B)    : ridge point    letters = kernels")
    lines.extend(legend)
    for k in skipped:
        lines.append(f"  -  {k.name}: 0 FLOPs, intensity 0 - roofline degenerates to the")
        lines.append("     bandwidth ceiling alone; judge it as % of achievable GB/s.")
    return "\n".join(lines)


def _format_intensity(value: float) -> str:
    if value >= 1000 or value < 0.01:
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
    if value >= 1:
        return f"{value:g}"
    return f"{value:g}"


# --------------------------------------------------------------------------- #
# Classification table
# --------------------------------------------------------------------------- #

_MEMORY_ACTION = "cut bytes: fuse ops, raise reuse (tile/shared), coalesce, smaller dtype"
_COMPUTE_ACTION = "raise FLOP rate: matrix cores, better instruction mix, more ILP/occupancy"
_RIDGE_ACTION = "both roofs bite: overlap load and math, then re-measure"
_UNDER_ROOF_ACTION = "far under its own roof: suspect latency-bound (low occupancy, tiny launch)"
_NO_FLOPS_ACTION = "no arithmetic at all: score it as % of achievable GB/s, not GFLOP/s"


def classify(kernel: Kernel, peak_gflops: float, peak_gbps: float) -> Tuple[str, str]:
    ridge = peak_gflops / peak_gbps
    intensity = kernel.intensity
    if intensity == 0:
        return "pure data movement", _NO_FLOPS_ACTION
    if intensity < ridge / 2:
        regime, action = "memory-bound", _MEMORY_ACTION
    elif intensity > ridge * 2:
        regime, action = "compute-bound", _COMPUTE_ACTION
    else:
        regime, action = "near ridge", _RIDGE_ACTION

    if kernel.achieved_gflops:
        roof = kernel.roof_gflops(peak_gflops, peak_gbps)
        if roof > 0 and kernel.achieved_gflops / roof < 0.5:
            action = _UNDER_ROOF_ACTION
    return regime, action


def render_table(peak_gflops: float, peak_gbps: float, kernels: Sequence[Kernel]) -> str:
    header = (f"{'kernel':<30} {'FLOPs':>11} {'bytes':>11} {'I (F/B)':>10} "
              f"{'roof GF/s':>11} {'achieved':>10} {'% roof':>7}  {'regime':<18} next step")
    lines = [header, "-" * len(header)]
    for k in kernels:
        roof = k.roof_gflops(peak_gflops, peak_gbps)
        regime, action = classify(k, peak_gflops, peak_gbps)
        achieved = f"{k.achieved_gflops:,.1f}" if k.achieved_gflops else "-"
        pct = (f"{100.0 * k.achieved_gflops / roof:.0f}%"
               if k.achieved_gflops and roof > 0 else "-")
        lines.append(
            f"{k.name:<30} {k.flops:>11.3e} {k.bytes:>11.3e} {k.intensity:>10.3f} "
            f"{roof:>11.1f} {achieved:>10} {pct:>7}  {regime:<18} {action}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_DEMO_GFLOPS = 80000.0
_DEMO_GBPS = 4500.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roofline_plot.py",
        description="ASCII roofline chart + memory/compute classification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Get real ceilings first:\n"
                "  make run-hip   # or run-cuda; read off the memory and compute roofs\n"
                "  python tools/roofline_plot.py --peak-gflops <P> --peak-gbps <B> --demo-kernels"),
    )
    p.add_argument("--peak-gflops", type=float,
                   help="compute ceiling P in GFLOP/s (from measure_ceilings)")
    p.add_argument("--peak-gbps", type=float,
                   help="memory ceiling B in GB/s (from measure_ceilings)")
    p.add_argument("--kernel", action="append", default=[], metavar="SPEC",
                   help="'name, flops, bytes[, achieved_gflops]'; repeatable")
    p.add_argument("--demo-kernels", action="store_true",
                   help="add the README's analytic operator catalogue as points")
    p.add_argument("--demo", action="store_true",
                   help="illustrative ceilings AND the operator catalogue (no measurement)")
    p.add_argument("--no-chart", action="store_true", help="print the table only")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    peak_gflops, peak_gbps = args.peak_gflops, args.peak_gbps
    if args.demo:
        peak_gflops = peak_gflops or _DEMO_GFLOPS
        peak_gbps = peak_gbps or _DEMO_GBPS
    if peak_gflops is None or peak_gbps is None:
        parser.error("--peak-gflops and --peak-gbps are required (or pass --demo)")
    if peak_gflops <= 0 or peak_gbps <= 0:
        parser.error("ceilings must be positive")

    kernels: List[Kernel] = []
    for spec in args.kernel:
        try:
            kernels.append(parse_kernel(spec))
        except ValueError as exc:
            parser.error(f"bad --kernel {spec!r}: {exc}")
    if args.demo or args.demo_kernels:
        kernels.extend(demo_kernels())
    if not kernels:
        parser.error("no kernels given: use --kernel or --demo-kernels")

    if args.demo and (args.peak_gflops is None or args.peak_gbps is None):
        print("NOTE: --demo ceilings are illustrative round numbers, NOT a measurement.")
        print("      Run measure_ceilings on your GPU and pass the real P and B.\n")

    if not args.no_chart:
        print(render_chart(peak_gflops, peak_gbps, kernels))
        print()
    print(render_table(peak_gflops, peak_gbps, kernels))
    print()
    print(f"ridge point I_ridge = P / B = {peak_gflops / peak_gbps:.2f} FLOP/byte")
    print("Reminder: a point sitting far below its own roof is not bound by that roof -")
    print("it is latency- or occupancy-bound, and the roofline cannot tell you that.")
    print("One chart = one dtype. A bf16 matrix-core kernel needs the matrix-core peak")
    print("as --peak-gflops, not the fp32 peak; see README Section 6.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
