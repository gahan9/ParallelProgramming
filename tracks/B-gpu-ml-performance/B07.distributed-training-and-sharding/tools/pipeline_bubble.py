#!/usr/bin/env python3
"""Pipeline-parallel bubble arithmetic: how much of the pipeline is idle, and why.

Everything here is an ANALYTIC MODEL. It assumes perfectly balanced stages, free
point-to-point communication, and a synchronous flush at the end of each batch
(GPipe / 1F1B semantics). Real schedules are worse; the model tells you the
ceiling you are working against.

Derivation for S stages and M micro-batches, each stage taking t per micro-batch:
  fill        the first micro-batch must traverse all S stages : (S-1)*t idle per stage
  steady      M micro-batches stream through                   :  M*t
  drain       symmetric to fill, folded into the same (S-1)*t
  total       (M + S - 1) * t
  ideal       M * t
  bubble      (S-1) / (M + S - 1)
  efficiency  M / (M + S - 1)
Interleaved 1F1B with v model chunks per device divides the bubble by v
(arXiv:2104.04473, sec. 2.2.2):
  bubble      ((S-1)/v) / (M + (S-1)/v)

Stdlib only: runs on any Python 3.8+ with no torch, no numpy.
"""

from __future__ import annotations

import argparse
import json


def bubble_fraction(stages: int, micro_batches: int, chunks: int = 1) -> float:
    if stages <= 1:
        return 0.0
    depth = (stages - 1) / float(chunks)
    return depth / (micro_batches + depth)


def efficiency(stages: int, micro_batches: int, chunks: int = 1) -> float:
    return 1.0 - bubble_fraction(stages, micro_batches, chunks)


def micro_batches_for(stages: int, target: float, chunks: int = 1) -> int:
    """Smallest M with efficiency >= target.

    From E = M / (M + d) with d = (S-1)/v:  M >= E*d / (1 - E).
    """
    if stages <= 1:
        return 1
    if target >= 1.0:
        raise ValueError("target efficiency must be < 1: the bubble never reaches zero")
    depth = (stages - 1) / float(chunks)
    exact = target * depth / (1.0 - target)
    m = max(1, int(exact))
    while efficiency(stages, m, chunks) < target:
        m += 1
    return m


def inflight_activation_multiplier(stages: int) -> int:
    """Micro-batches whose activations the first stage must hold under 1F1B.

    This is the cost of raising M: the bubble shrinks, but the deepest-loaded
    stage keeps ~S micro-batches of activations alive. Pipeline depth, not M,
    sets that peak -- which is why 1F1B replaced all-forward-all-backward.
    """
    return max(1, stages)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Pipeline bubble fraction, efficiency, and the micro-batch count you need.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--stages", type=int, default=4, help="pipeline stages S (pp degree)")
    ap.add_argument("--micro-batches", type=int, default=8, help="micro-batches per step M")
    ap.add_argument("--chunks", type=int, default=1,
                    help="interleaved model chunks per device v (1 = plain 1F1B)")
    ap.add_argument("--target-efficiency", type=float, default=0.9)
    ap.add_argument("--sweep", action="store_true", help="table of efficiency vs S and M")
    ap.add_argument("--stage-ms", type=float, default=0.0,
                    help="per-stage per-micro-batch time in ms; prints wall-clock step time")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not 0.0 < args.target_efficiency < 1.0:
        ap.error("--target-efficiency must be in (0, 1): the bubble never reaches zero")

    s, m, v = args.stages, args.micro_batches, args.chunks
    b = bubble_fraction(s, m, v)
    e = efficiency(s, m, v)
    need = micro_batches_for(s, args.target_efficiency, v)

    result = dict(
        stages=s,
        micro_batches=m,
        chunks=v,
        bubble_fraction=b,
        efficiency=e,
        target_efficiency=args.target_efficiency,
        micro_batches_needed=need,
        inflight_activation_multiplier=inflight_activation_multiplier(s),
    )

    if args.stage_ms:
        result["step_ms"] = (m + (s - 1) / v) * args.stage_ms
        result["ideal_ms"] = m * args.stage_ms

    if args.sweep:
        table = []
        for stages in (2, 4, 8, 16, 32):
            row = dict(stages=stages)
            for mb in (1, 2, 4, 8, 16, 32, 64, 128):
                row[str(mb)] = efficiency(stages, mb, v)
            table.append(row)
        result["sweep"] = table

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"stages S={s}  micro-batches M={m}  chunks v={v}")
    print(f"bubble fraction (S-1)/v / (M + (S-1)/v) = {b*100:.1f}%")
    print(f"pipeline efficiency                     = {e*100:.1f}%")
    if args.stage_ms:
        print(f"step time  (M + (S-1)/v) * t = {result['step_ms']:.1f} ms "
              f"vs ideal {result['ideal_ms']:.1f} ms")
    print(f"to reach {args.target_efficiency*100:.0f}% efficiency you need M >= {need} "
          f"(={need/max(1,s):.1f}x the stage count)")
    print(f"cost of that: the first stage keeps ~{inflight_activation_multiplier(s)} "
          f"micro-batches of activations alive under 1F1B")

    if args.sweep:
        print()
        header = f"{'S \\ M':>7}" + "".join(f"{mb:>8}" for mb in (1, 2, 4, 8, 16, 32, 64, 128))
        print(header)
        print("-" * len(header))
        for row in result["sweep"]:
            line = f"{row['stages']:>7}"
            for mb in (1, 2, 4, 8, 16, 32, 64, 128):
                line += f"{row[str(mb)]*100:>7.0f}%"
            print(line)
        print("-" * len(header))
        print(f"pipeline efficiency %, analytic model with v={v} (balanced stages, free P2P)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
