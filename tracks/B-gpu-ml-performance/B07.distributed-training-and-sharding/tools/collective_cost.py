#!/usr/bin/env python3
"""Ring-collective cost models: when is a collective latency-bound vs bandwidth-bound?

Everything here is an ANALYTIC MODEL of the classic ring algorithms (Baidu /
Horovod ring all-reduce, arXiv:1802.05799; NCCL and RCCL implement these plus
tree and topology-aware variants). No GPU is touched; nothing is measured.

Cost model for P ranks, S bytes, per-link bandwidth B, per-step latency alpha:
  all-reduce      T = 2*(P-1)*alpha + (2*(P-1)/P) * S/B
  reduce-scatter  T =   (P-1)*alpha +   ((P-1)/P) * S/B
  all-gather      T =   (P-1)*alpha +   ((P-1)/P) * S/B
  all-to-all      T =         alpha +   ((P-1)/P) * S/B     (full-bisection fabric)
  broadcast       T = log2(P)*alpha +               S/B     (pipelined tree)

Stdlib only: runs on any Python 3.8+ with no torch, no numpy.
"""

from __future__ import annotations

import argparse
import json
import math

MIB = 1024.0**2
US = 1e-6


def ring_time(op: str, nbytes: float, ranks: int, bw_bytes_per_s: float, alpha_s: float):
    """Return (seconds, latency_seconds, transfer_seconds) for one collective."""
    p = max(1, ranks)
    if p == 1:
        return 0.0, 0.0, 0.0
    if op == "allreduce":
        steps, volume = 2 * (p - 1), 2.0 * (p - 1) / p * nbytes
    elif op in ("reducescatter", "allgather"):
        steps, volume = (p - 1), (p - 1) / p * nbytes
    elif op == "alltoall":
        steps, volume = 1, (p - 1) / p * nbytes
    elif op == "broadcast":
        steps, volume = max(1, int(math.log2(p))), float(nbytes)
    else:
        raise ValueError(f"unknown op: {op}")
    lat = steps * alpha_s
    xfer = volume / bw_bytes_per_s
    return lat + xfer, lat, xfer


def crossover_bytes(op: str, ranks: int, bw_bytes_per_s: float, alpha_s: float) -> float:
    """Message size at which the transfer term equals the latency term.

    For all-reduce: 2(P-1)*alpha = 2(P-1)/P * S/B  =>  S* = P * alpha * B.
    Reduce-scatter and all-gather give the identical crossover; the ring depth
    cancels out. Below S*, the collective is latency/launch-bound and merging
    messages (gradient bucketing) is nearly free. Above S*, you are on the wire
    and only more bandwidth helps.
    """
    p = max(2, ranks)
    if op in ("allreduce", "reducescatter", "allgather"):
        return p * alpha_s * bw_bytes_per_s
    if op == "alltoall":
        return alpha_s * bw_bytes_per_s * p / (p - 1)
    if op == "broadcast":
        return math.log2(p) * alpha_s * bw_bytes_per_s
    raise ValueError(op)


def bus_bandwidth(op: str, nbytes: float, ranks: int, seconds: float) -> float:
    """NCCL/RCCL 'busbw': algorithm bandwidth scaled by the ring's link factor.

    algbw = S / T; busbw = algbw * (ring volume factor). Reporting busbw is what
    lets you compare a collective against the raw link speed. See the NCCL tests
    performance notes: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    """
    if seconds <= 0:
        return float("inf")
    p = max(2, ranks)
    algbw = nbytes / seconds
    factor = {
        "allreduce": 2.0 * (p - 1) / p,
        "reducescatter": (p - 1) / p,
        "allgather": (p - 1) / p,
        "alltoall": (p - 1) / p,
        "broadcast": 1.0,
    }[op]
    return algbw * factor


def sweep(op, ranks, bw, alpha, sizes_mib):
    rows = []
    star = crossover_bytes(op, ranks, bw, alpha)
    for mib in sizes_mib:
        nbytes = mib * MIB
        t, lat, xfer = ring_time(op, nbytes, ranks, bw, alpha)
        rows.append(
            dict(
                mib=mib,
                seconds=t,
                latency_share=(lat / t if t else 0.0),
                busbw_gbs=bus_bandwidth(op, nbytes, ranks, t) / 1e9,
                regime="latency-bound" if nbytes < star else "bandwidth-bound",
            )
        )
    return rows, star


def bucket_analysis(total_bytes, ranks, bw, alpha, buckets_mib):
    """Gradient bucketing: one big collective is cheapest, but blocks overlap.

    Fusing all gradients into one message pays the ring latency once; splitting
    into k buckets pays it k times but lets bucket i reduce while layer i-1 is
    still computing its backward pass. The useful rule that falls out: pick the
    SMALLEST bucket that is still bandwidth-bound, i.e. bucket >= P*alpha*B.
    """
    rows = []
    star = crossover_bytes("allreduce", ranks, bw, alpha)
    _, ring_latency, wire = ring_time("allreduce", total_bytes, ranks, bw, alpha)
    for mib in buckets_mib:
        bucket = mib * MIB
        n = max(1, math.ceil(total_bytes / bucket))
        # Each bucket pays the ring latency again; total wire time is unchanged.
        seconds = wire + n * ring_latency
        rows.append(
            dict(
                bucket_mib=mib,
                buckets=n,
                seconds=seconds,
                latency_share=(n * ring_latency) / seconds if seconds else 0.0,
                overlappable=n > 1,
                bandwidth_bound=bucket >= star,
            )
        )
    return rows, star


def layer_ratio(args, bw, alpha):
    """Communication-to-compute ratio for one transformer layer.

    Compute: a transformer layer's forward+backward costs about 6*P_layer FLOPs
    per token for the weight matmuls (2 forward, 4 backward), plus attention
    terms -- see arXiv:2104.04473 App. A. We use the weight term.
    Communication: DDP all-reduces the layer's gradients once per OPTIMIZER step,
    so gradient accumulation divides the ratio: --grad-accum is the honest lever.
    """
    params = args.layer_params
    if params <= 0:
        h, ff = args.hidden, 4 * args.hidden
        params = 4 * h * h + 2 * h * ff  # attention proj + MLP, standard shape
    tokens = args.micro_batch * args.seq * max(1, args.grad_accum)
    flops = 6.0 * params * tokens
    compute_s = flops / (args.tflops * 1e12)
    nbytes = params * args.dtype_bytes
    comm_s, lat, xfer = ring_time("allreduce", nbytes, args.gpus, bw, alpha)
    return dict(
        layer_params=params,
        grad_bytes=nbytes,
        flops=flops,
        compute_s=compute_s,
        comm_s=comm_s,
        ratio=comm_s / compute_s if compute_s else float("inf"),
        latency_share=lat / comm_s if comm_s else 0.0,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Analytic ring-collective cost model (latency vs bandwidth regimes).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--op", default="allreduce",
                    choices=("allreduce", "reducescatter", "allgather", "alltoall", "broadcast"))
    ap.add_argument("--gpus", type=int, default=8, help="ranks in the collective")
    ap.add_argument("--bw", type=float, default=50.0,
                    help="achievable per-link bandwidth in GB/s (1 GB = 1e9 bytes)")
    ap.add_argument("--latency", type=float, default=8.0,
                    help="per-ring-step latency alpha in microseconds")
    ap.add_argument("--bytes", type=float, default=0.0, help="single message size in bytes")
    ap.add_argument("--sizes", default="0.25,1,4,16,64,256,1024",
                    help="comma-separated sweep sizes in MiB")
    ap.add_argument("--params", type=float, default=0.0,
                    help="model parameters in billions; models a full DDP gradient all-reduce")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="gradient element size")
    ap.add_argument("--buckets", default="1,4,25,100,400",
                    help="comma-separated bucket sizes in MiB for the bucketing table")
    ap.add_argument("--step-ms", type=float, default=0.0,
                    help="measured step time in ms; prints the comm fraction of the step")
    ap.add_argument("--layer-ratio", action="store_true",
                    help="communication-to-compute ratio for one transformer layer")
    ap.add_argument("--layer-params", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=8192)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step")
    ap.add_argument("--tflops", type=float, default=400.0,
                    help="achieved dense TFLOP/s per GPU (not peak)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    bw = args.bw * 1e9
    alpha = args.latency * US
    sizes = [float(x) for x in args.sizes.split(",") if x]
    out = {}

    rows, star = sweep(args.op, args.gpus, bw, alpha, sizes)
    out["sweep"] = rows
    out["crossover_bytes"] = star

    if not args.json:
        print(f"op={args.op}  ranks={args.gpus}  link={args.bw:g} GB/s  "
              f"alpha={args.latency:g} us")
        print(f"latency/bandwidth crossover S* = P*alpha*B = {star/MIB:.2f} MiB "
              f"(below this, fuse your messages)")
        print()
        print(f"{'size MiB':>10}{'time ms':>10}{'latency %':>11}{'busbw GB/s':>12}  regime")
        print("-" * 62)
        for r in rows:
            print(f"{r['mib']:>10.2f}{r['seconds']*1e3:>10.3f}"
                  f"{r['latency_share']*100:>11.1f}{r['busbw_gbs']:>12.1f}  {r['regime']}")
        print("-" * 62)
        print("analytic ring model, not measured")

    if args.bytes:
        t, lat, xfer = ring_time(args.op, args.bytes, args.gpus, bw, alpha)
        out["single"] = dict(bytes=args.bytes, seconds=t, latency_s=lat, transfer_s=xfer)
        if not args.json:
            print(f"\nsingle message {args.bytes/MIB:.2f} MiB: {t*1e3:.3f} ms "
                  f"(latency {lat*1e6:.1f} us + wire {xfer*1e3:.3f} ms)")

    if args.params:
        total = args.params * 1e9 * args.dtype_bytes
        t, lat, xfer = ring_time("allreduce", total, args.gpus, bw, alpha)
        out["gradient_allreduce"] = dict(bytes=total, seconds=t)
        if not args.json:
            print(f"\nDDP gradient all-reduce for {args.params:g}B params "
                  f"@ {args.dtype_bytes}B: {total/MIB:.0f} MiB -> {t*1e3:.1f} ms")
            if args.step_ms:
                frac = t / (args.step_ms * 1e-3)
                verdict = "COMMUNICATION-BOUND" if frac > 0.5 else (
                    "comm-sensitive" if frac > 0.2 else "compute-bound")
                print(f"  fraction of a {args.step_ms:g} ms step: {frac*100:.1f}%  -> {verdict}")
                out["gradient_allreduce"]["step_fraction"] = frac
        brows, bstar = bucket_analysis(total, args.gpus, bw, alpha,
                                       [float(x) for x in args.buckets.split(",") if x])
        out["buckets"] = brows
        if not args.json:
            print(f"\nbucketing (total {total/MIB:.0f} MiB, "
                  f"bandwidth-bound above {bstar/MIB:.2f} MiB):")
            print(f"{'bucket MiB':>11}{'count':>7}{'time ms':>10}{'latency %':>11}"
                  f"  overlap?  bandwidth-bound?")
            print("-" * 66)
            for r in brows:
                print(f"{r['bucket_mib']:>11.0f}{r['buckets']:>7}{r['seconds']*1e3:>10.1f}"
                      f"{r['latency_share']*100:>11.1f}"
                      f"  {'yes' if r['overlappable'] else 'no':<9}"
                      f" {'yes' if r['bandwidth_bound'] else 'no'}")

    if args.layer_ratio:
        lr = layer_ratio(args, bw, alpha)
        out["layer_ratio"] = lr
        if not args.json:
            print(f"\none layer: {lr['layer_params']/1e6:.1f}M params, "
                  f"{lr['flops']/1e12:.2f} TFLOP fwd+bwd "
                  f"(grad-accum {max(1, args.grad_accum)})")
            print(f"  compute {lr['compute_s']*1e3:.3f} ms @ {args.tflops:g} TFLOP/s")
            print(f"  gradient all-reduce {lr['grad_bytes']/MIB:.1f} MiB -> "
                  f"{lr['comm_s']*1e3:.3f} ms ({lr['latency_share']*100:.0f}% latency)")
            print(f"  comm/compute ratio  {lr['ratio']:.3f}"
                  f"  -> {'cannot hide' if lr['ratio'] > 1 else 'hideable behind backward'}")

    if args.json:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
