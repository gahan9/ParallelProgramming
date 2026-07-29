#!/usr/bin/env python3
"""Analytic model of speculative decoding: when does it pay, and when does it stop?

A small draft model proposes `k` tokens; the large target model verifies all of
them in ONE forward pass. Under the standard rejection-sampling scheme the output
distribution is identical to plain sampling from the target -- speculative decoding
is *lossless*, so the only question is arithmetic.

Model, in units of one target forward pass:

    expected tokens per verification step   E(alpha, k) = (1 - alpha^(k+1)) / (1 - alpha)
    cost per verification step              C(k)        = 1 + k*c + k*v
    speedup vs one-token-at-a-time decode   S           = E / C

`alpha` is the per-token acceptance rate (modelled as i.i.d., which is the
standard simplifying assumption and an optimistic one -- see the caveats below).
The `+1` in the exponent is the free bonus token: even if all k drafts are
rejected, the target's own verification pass still yields one correct token, so
speculative decoding can never produce fewer than 1 token per step.

`c` is the draft cost ratio (one draft forward / one target forward). `v` is the
marginal cost of each extra token *position* inside the verification pass. `v` is
near zero exactly while decode is bandwidth-bound -- the weights are already
streamed, so scoring 5 positions costs almost what scoring 1 costs. That is the
entire reason speculative decoding works, and it is also its expiry condition:
push the batch large enough that decode becomes compute-bound and `v` grows, and
the free lunch ends.

Stdlib only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class SpecConfig:
    draft_cost: float      # c: draft forward / target forward
    verify_overhead: float # v: marginal target cost per extra draft position


def expected_tokens(alpha: float, k: int) -> float:
    """Expected accepted tokens per verification step, including the bonus token."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def step_cost(k: int, cfg: SpecConfig) -> float:
    """Cost of one draft-and-verify cycle, in units of one target forward pass."""
    return 1.0 + k * cfg.draft_cost + k * cfg.verify_overhead


def speedup(alpha: float, k: int, cfg: SpecConfig) -> float:
    return expected_tokens(alpha, k) / step_cost(k, cfg)


def best_k(alpha: float, cfg: SpecConfig, k_max: int) -> int:
    return max(range(1, k_max + 1), key=lambda k: speedup(alpha, k, cfg))


def break_even_alpha(k: int, cfg: SpecConfig, tol: float = 1e-6) -> Optional[float]:
    """Smallest alpha at which drafting k tokens beats not drafting at all.

    Speedup is monotonically increasing in alpha for fixed k, so a bisection is
    exact enough. Returns None when even perfect acceptance cannot pay for the
    draft -- which happens when the draft model is too expensive relative to k.
    """
    if speedup(1.0, k, cfg) <= 1.0:
        return None
    if speedup(0.0, k, cfg) > 1.0:
        return 0.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if speedup(mid, k, cfg) > 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def print_table(title: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    cells = [[f"{c}" for c in row] for row in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in cells)) for i in range(len(header))]
    print(title)
    print("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print("  ".join(c.rjust(w) for c, w in zip(row, widths)))
    print()


def report_point(alpha: float, k: int, cfg: SpecConfig) -> None:
    tokens = expected_tokens(alpha, k)
    cost = step_cost(k, cfg)
    rows = [
        ["expected tokens / step", f"{tokens:.3f}"],
        ["cost / step (target passes)", f"{cost:.3f}"],
        ["speedup vs plain decode", f"{speedup(alpha, k, cfg):.3f}x"],
        ["drafted tokens discarded", f"{100 * (1 - (tokens - 1) / k):.1f}%"],
        ["draft share of step cost", f"{100 * k * cfg.draft_cost / cost:.1f}%"],
    ]
    print_table(f"Operating point: alpha={alpha}, k={k}", ["quantity", "value"], rows)


def report_sweep(alphas: Sequence[float], ks: Sequence[int], cfg: SpecConfig) -> None:
    header = ["alpha"] + [f"k={k}" for k in ks] + ["best k", "best speedup"]
    rows: List[List[object]] = []
    for alpha in alphas:
        speedups = [speedup(alpha, k, cfg) for k in ks]
        bk = best_k(alpha, cfg, max(ks))
        rows.append(
            [f"{alpha:.2f}"]
            + [f"{s:.2f}x" for s in speedups]
            + [bk, f"{speedup(alpha, bk, cfg):.2f}x"]
        )
    print_table("Speedup vs acceptance rate and draft length", header, rows)

    rows = []
    for k in ks:
        be = break_even_alpha(k, cfg)
        rows.append([k, "never pays" if be is None else f"{be:.3f}"])
    print_table("Break-even acceptance rate (below this, drafting loses)", ["k", "min alpha"], rows)


def report_verify_pressure(alpha: float, k: int, cfg: SpecConfig) -> None:
    """What happens as the batch grows and verification stops being free."""
    rows = []
    for v in (0.0, 0.02, 0.05, 0.10, 0.20, 0.50):
        alt = SpecConfig(draft_cost=cfg.draft_cost, verify_overhead=v)
        s = speedup(alpha, k, alt)
        rows.append([
            f"{v:.2f}",
            f"{step_cost(k, alt):.3f}",
            f"{s:.2f}x",
            "pays" if s > 1.0 else "LOSES",
        ])
    print_table(
        f"Sensitivity to verification cost (alpha={alpha}, k={k})",
        ["v (per extra position)", "cost/step", "speedup", "verdict"],
        rows,
    )
    print(
        "v rises as the batch fills: with enough concurrent sequences the decode step is no longer"
        " bandwidth-bound, the spare compute speculation was spending is gone, and the same draft"
        " model that helped at batch 1 now costs throughput. Speculative decoding is a *latency*"
        " optimization; measure it at your real concurrency, not at batch 1.\n"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--alpha", type=float, default=0.7, help="per-token acceptance rate")
    p.add_argument("--k", type=int, default=4, help="draft length (tokens proposed per step)")
    p.add_argument("--draft-cost", type=float, default=0.05,
                   help="one draft forward / one target forward")
    p.add_argument("--verify-overhead", type=float, default=0.0,
                   help="marginal target cost per extra draft position")
    p.add_argument("--alphas", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    p.add_argument("--ks", default="1,2,3,4,5,6,8,12")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = SpecConfig(draft_cost=args.draft_cost, verify_overhead=args.verify_overhead)

    print("=== Speculative decoding: analytic model (arithmetic, not measured) ===")
    print(
        f"draft cost c={cfg.draft_cost} target passes, verify overhead v={cfg.verify_overhead}"
        " per extra position\n"
    )
    report_point(args.alpha, args.k, cfg)
    report_sweep(
        [float(a) for a in args.alphas.split(",")],
        [int(k) for k in args.ks.split(",")],
        cfg,
    )
    report_verify_pressure(args.alpha, args.k, cfg)
    print(
        "Read the sweep left to right: for a weak draft model (low alpha) the best k is small,"
        " because every extra draft token is paid for and rarely accepted. Raising k only helps"
        " while alpha^k stays meaningful -- the expected-token curve saturates at 1/(1-alpha) while"
        " the cost keeps growing linearly in k. That is the whole design rule."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
