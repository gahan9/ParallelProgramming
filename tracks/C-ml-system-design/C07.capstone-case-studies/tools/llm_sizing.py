#!/usr/bin/env python3
"""LLM serving sizing calculator for module C07, case study 1.

Derives, from first principles and explicitly stated assumptions:

  * model weight bytes at a chosen dtype
  * KV-cache bytes per token  =  2 * layers * d_kv * kv_dtype_bytes
  * the KV pool left in node HBM, and therefore max concurrency
  * decode step time as max(memory, compute) + collective + fixed overhead
  * prefill time including the quadratic attention term
  * the scheduler's KV-token budget implied by a TPOT SLO
  * cost per million input / output tokens

Every number is a derived estimate from the assumptions printed in the header,
not a measurement. Re-run with your own numbers; treat the output as the
target a real deployment should be measured against (expect 50-70% of it on
day one -- that is what --efficiency encodes).

Standard library only. Run `python llm_sizing.py --help` for all knobs.
"""

from __future__ import annotations

import argparse
import sys

# Bytes per element. int4 is stored packed, hence the fractional entry.
DTYPE_BYTES = {
    "fp32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

GB = 1e9
TB = 1e12


class InputError(Exception):
    """Raised for user-supplied values that cannot produce a valid design."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="llm_sizing.py",
        description="Size an LLM serving node from first principles (C07 case study 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model = p.add_argument_group("model shape (defaults: a 70B-class GQA decoder)")
    model.add_argument("--params-b", type=float, default=70.0,
                       help="parameter count in billions")
    model.add_argument("--layers", type=int, default=80,
                       help="number of transformer layers")
    model.add_argument("--d-model", type=int, default=8192,
                       help="hidden size")
    model.add_argument("--kv-heads", type=int, default=8,
                       help="key/value heads (GQA); equals query heads for MHA")
    model.add_argument("--head-dim", type=int, default=128,
                       help="dimension per attention head")
    model.add_argument("--weight-dtype", default="bf16", choices=sorted(DTYPE_BYTES),
                       help="dtype the weights are stored in")
    model.add_argument("--kv-dtype", default="bf16", choices=sorted(DTYPE_BYTES),
                       help="dtype the KV cache is stored in")
    model.add_argument("--act-dtype", default="bf16", choices=sorted(DTYPE_BYTES),
                       help="dtype of the activations moved by the all-reduce")

    hw = p.add_argument_group("hardware (defaults: an 8-GPU 192 GB / 5.3 TB/s node)")
    hw.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    hw.add_argument("--hbm-gb", type=float, default=192.0, help="HBM per GPU, GB")
    hw.add_argument("--bw-tbs", type=float, default=5.3,
                    help="peak HBM bandwidth per GPU, TB/s")
    hw.add_argument("--tflops", type=float, default=1300.0,
                    help="peak dense matmul throughput per GPU at the weight dtype, TFLOP/s")
    hw.add_argument("--link-gbs", type=float, default=300.0,
                    help="achievable per-GPU all-reduce bandwidth on the fabric, GB/s")
    hw.add_argument("--link-latency-us", type=float, default=8.0,
                    help="per-collective launch + sync latency, microseconds")

    eff = p.add_argument_group("achievable fractions (assumptions -- measure yours)")
    eff.add_argument("--bw-eff", type=float, default=0.70,
                     help="fraction of peak HBM bandwidth a decode kernel reaches")
    eff.add_argument("--flops-eff", type=float, default=0.40,
                     help="fraction of peak FLOP/s a prefill GEMM reaches")
    eff.add_argument("--efficiency", type=float, default=0.60,
                     help="end-to-end system efficiency vs the roofline model "
                          "(scheduler gaps, ragged batches, preemption, sampling)")
    eff.add_argument("--fixed-ms", type=float, default=1.5,
                     help="per-step non-model overhead: schedule, sample, detokenize")

    work = p.add_argument_group("workload")
    work.add_argument("--tp", type=int, default=8,
                      help="tensor-parallel degree; replicas per node = gpus / tp")
    work.add_argument("--batch", type=int, default=192,
                      help="concurrent decode streams per replica")
    work.add_argument("--avg-ctx", type=int, default=2048,
                      help="average live context length per stream, tokens")
    work.add_argument("--prompt", type=int, default=2048,
                      help="prompt length used for the prefill estimate, tokens")
    work.add_argument("--reserve-frac", type=float, default=0.15,
                      help="fraction of HBM reserved for activations, workspace, "
                           "comms buffers and fragmentation")
    work.add_argument("--tpot-slo-ms", type=float, default=30.0,
                      help="inter-token latency SLO used to solve for the KV-token budget")

    cost = p.add_argument_group("cost")
    cost.add_argument("--gpu-hour-usd", type=float, default=3.00,
                      help="fully-loaded cost per GPU-hour (hardware amortization, "
                           "power, datacenter, ops)")
    cost.add_argument("--utilization", type=float, default=0.45,
                      help="average fleet load as a fraction of provisioned peak")

    return p.parse_args(argv)


def validate(a: argparse.Namespace) -> None:
    positive = {
        "--params-b": a.params_b, "--layers": a.layers, "--d-model": a.d_model,
        "--kv-heads": a.kv_heads, "--head-dim": a.head_dim, "--gpus": a.gpus,
        "--hbm-gb": a.hbm_gb, "--bw-tbs": a.bw_tbs, "--tflops": a.tflops,
        "--link-gbs": a.link_gbs, "--batch": a.batch, "--avg-ctx": a.avg_ctx,
        "--prompt": a.prompt, "--tp": a.tp, "--gpu-hour-usd": a.gpu_hour_usd,
        "--tpot-slo-ms": a.tpot_slo_ms,
    }
    for flag, value in positive.items():
        if value <= 0:
            raise InputError(f"{flag} must be greater than zero (got {value}).")

    fractions = {
        "--bw-eff": a.bw_eff, "--flops-eff": a.flops_eff,
        "--efficiency": a.efficiency, "--utilization": a.utilization,
    }
    for flag, value in fractions.items():
        if not 0.0 < value <= 1.0:
            raise InputError(f"{flag} must be in the interval (0, 1] (got {value}).")

    if not 0.0 <= a.reserve_frac < 1.0:
        raise InputError(f"--reserve-frac must be in [0, 1) (got {a.reserve_frac}).")

    if a.gpus % a.tp:
        raise InputError(
            f"--tp {a.tp} does not divide --gpus {a.gpus}; a replica must fit a whole "
            f"number of times into a node."
        )
    if a.kv_heads * a.head_dim > a.d_model:
        raise InputError(
            f"kv_heads * head_dim = {a.kv_heads * a.head_dim} exceeds --d-model "
            f"{a.d_model}; check the model shape."
        )


def fmt_gb(byte_count: float) -> str:
    return f"{byte_count / GB:,.1f} GB"


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        validate(a)
    except InputError as exc:
        print(f"llm_sizing.py: {exc}", file=sys.stderr)
        return 2

    w_bytes = DTYPE_BYTES[a.weight_dtype]
    kv_bytes = DTYPE_BYTES[a.kv_dtype]
    act_bytes = DTYPE_BYTES[a.act_dtype]

    params = a.params_b * 1e9
    d_kv = a.kv_heads * a.head_dim
    replicas = a.gpus // a.tp

    node_hbm = a.gpus * a.hbm_gb * GB
    bw_per_gpu = a.bw_tbs * TB * a.bw_eff              # achievable bytes/s per GPU
    flops_per_replica = a.tp * a.tflops * 1e12 * a.flops_eff

    # ---- Memory -----------------------------------------------------------
    weight_bytes = params * w_bytes
    node_weight_bytes = weight_bytes * replicas        # one copy per replica
    reserve_bytes = node_hbm * a.reserve_frac
    kv_pool = node_hbm - node_weight_bytes - reserve_bytes

    if kv_pool <= 0:
        print(
            "llm_sizing.py: no HBM left for the KV cache.\n"
            f"  node HBM        {fmt_gb(node_hbm)}\n"
            f"  weights x{replicas} replica(s)  {fmt_gb(node_weight_bytes)}\n"
            f"  runtime reserve {fmt_gb(reserve_bytes)}\n"
            "  Fix: quantize the weights (--weight-dtype fp8), raise --tp so one "
            "replica spans the node, or use a node with more HBM.",
            file=sys.stderr,
        )
        return 2

    kv_per_token = 2 * a.layers * d_kv * kv_bytes      # K and V, all layers, one token
    kv_tokens = kv_pool / kv_per_token
    max_seqs = kv_tokens / a.avg_ctx

    # ---- Decode step ------------------------------------------------------
    def step_ms(batch: int) -> dict[str, float]:
        """Roofline decode step for one replica, broken into its terms."""
        weights_read = weight_bytes / a.tp
        kv_read = batch * a.avg_ctx * kv_per_token / a.tp
        t_mem = (weights_read + kv_read) / bw_per_gpu

        # Every weight is multiplied by every token in the batch, twice (MAC).
        t_compute = 2 * params * batch / flops_per_replica

        if a.tp > 1:
            msg = batch * a.d_model * act_bytes
            per_gpu = 2 * (a.tp - 1) / a.tp * msg      # ring all-reduce volume
            n_collectives = 2 * a.layers               # after attention, after MLP
            t_comm = n_collectives * (per_gpu / (a.link_gbs * GB)
                                      + a.link_latency_us * 1e-6)
        else:
            t_comm = 0.0

        model = max(t_mem, t_compute) + t_comm + a.fixed_ms * 1e-3
        return {
            "mem": t_mem * 1e3,
            "compute": t_compute * 1e3,
            "comm": t_comm * 1e3,
            "model": model * 1e3,
            "tpot": model * 1e3 / a.efficiency,
            "kv_read_gb": kv_read / GB,
            "weights_read_gb": weights_read / GB,
        }

    d = step_ms(a.batch)
    node_decode_tps = replicas * a.batch / (d["tpot"] * 1e-3)

    # Batch at which KV traffic overtakes weight traffic: past this point,
    # bigger batches stop buying throughput cheaply.
    kv_per_stream = a.avg_ctx * kv_per_token / a.tp
    knee_batch = (weight_bytes / a.tp) / kv_per_stream

    # ---- Prefill ----------------------------------------------------------
    linear_flops = 2 * params * a.prompt
    attn_flops = 4 * a.layers * a.prompt ** 2 * a.d_model
    prefill_model_s = (linear_flops + attn_flops) / flops_per_replica
    prefill_s = prefill_model_s / a.efficiency
    node_prefill_tps = replicas * a.prompt / prefill_s

    # ---- KV-token budget implied by the TPOT SLO --------------------------
    # Solve for the largest batch whose TPOT meets the SLO. Monotone in batch,
    # so bisect; no closed form once the max() and the comm term are included.
    lo, hi = 0, 1
    while step_ms(hi)["tpot"] <= a.tpot_slo_ms and hi < 1 << 22:
        lo, hi = hi, hi * 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if step_ms(mid)["tpot"] <= a.tpot_slo_ms:
            lo = mid
        else:
            hi = mid
    slo_batch = lo
    slo_tokens = slo_batch * a.avg_ctx

    # ---- Cost -------------------------------------------------------------
    node_usd_per_s = a.gpus * a.gpu_hour_usd / 3600.0
    out_usd_per_m = node_usd_per_s / (node_decode_tps * a.utilization) * 1e6
    in_usd_per_m = node_usd_per_s / (node_prefill_tps * a.utilization) * 1e6

    # ---- Report -----------------------------------------------------------
    line = "-" * 72
    print(line)
    print("LLM SERVING SIZING  --  all figures are estimates derived from the")
    print("assumptions below, not measurements.")
    print(line)
    print("ASSUMPTIONS")
    print(f"  model            {a.params_b:g}B params, {a.layers} layers, "
          f"d_model {a.d_model}, {a.kv_heads} KV heads x {a.head_dim}")
    print(f"  dtypes           weights {a.weight_dtype}, KV {a.kv_dtype}, "
          f"activations {a.act_dtype}")
    print(f"  node             {a.gpus} GPUs, {a.hbm_gb:g} GB HBM each, "
          f"{a.bw_tbs:g} TB/s peak, {a.tflops:g} TFLOP/s peak")
    print(f"  achievable       {a.bw_eff:.0%} of HBM BW, {a.flops_eff:.0%} of FLOP/s, "
          f"{a.efficiency:.0%} end-to-end vs roofline")
    print(f"  fabric           {a.link_gbs:g} GB/s per GPU all-reduce, "
          f"{a.link_latency_us:g} us per collective")
    print(f"  parallelism      TP {a.tp}  ->  {replicas} replica(s) per node")
    print(f"  workload         batch {a.batch}, avg context {a.avg_ctx}, "
          f"prompt {a.prompt}")
    print(f"  cost             ${a.gpu_hour_usd:.2f}/GPU-hour, "
          f"{a.utilization:.0%} average utilization")

    print()
    print("MEMORY")
    print(f"  weights                    {params:,.0f} x {w_bytes:g} B "
          f"= {fmt_gb(weight_bytes)} per replica")
    print(f"  weights on the node        {fmt_gb(node_weight_bytes)} "
          f"({replicas} replica(s))")
    print(f"  runtime reserve            {fmt_gb(reserve_bytes)} "
          f"({a.reserve_frac:.0%} of HBM)")
    print(f"  node HBM                   {fmt_gb(node_hbm)}")
    print(f"  KV pool                    {fmt_gb(kv_pool)}")
    print(f"  KV bytes per token         2 x {a.layers} x {d_kv} x {kv_bytes:g} "
          f"= {kv_per_token:,.0f} B  ({kv_per_token / 1024:,.1f} KiB)")
    print(f"  KV pool capacity           {kv_tokens:,.0f} tokens")
    print(f"  max concurrent sequences   {max_seqs:,.0f} at {a.avg_ctx}-token context")

    print()
    print(f"DECODE  (batch {a.batch}, context {a.avg_ctx})")
    print(f"  weights read per GPU/step  {d['weights_read_gb']:,.2f} GB")
    print(f"  KV read per GPU/step       {d['kv_read_gb']:,.2f} GB")
    for label, value in (
        ("t_memory", d["mem"]),
        ("t_compute", d["compute"]),
        (f"t_allreduce (TP {a.tp})", d["comm"]),
        ("t_fixed", a.fixed_ms),
    ):
        print(f"  {label:<26}{value:8.2f} ms")
    bound = "memory" if d["mem"] >= d["compute"] else "compute"
    print(f"  {'-> roofline step':<26}{d['model']:8.2f} ms   ({bound}-bound)")
    print(f"  {f'-> TPOT at {a.efficiency:.0%} efficiency':<26}{d['tpot']:8.2f} ms")
    print(f"  node decode throughput     {node_decode_tps:,.0f} output tokens/s")
    print(f"  KV-vs-weights knee         batch {knee_batch:,.0f} "
          f"(above this, KV traffic dominates)")

    print()
    print(f"PREFILL  (prompt {a.prompt})")
    print(f"  linear FLOPs               2 x P x N       = {linear_flops / 1e12:,.1f} TFLOP")
    print(f"  attention FLOPs            4 x L x N^2 x d = {attn_flops / 1e12:,.1f} TFLOP "
          f"({attn_flops / (linear_flops + attn_flops):.1%} of total)")
    print(f"  time at {a.efficiency:.0%} efficiency     {prefill_s * 1e3:,.1f} ms")
    print(f"  node prefill throughput    {node_prefill_tps:,.0f} input tokens/s")

    print()
    print(f"SCHEDULER BUDGET  (TPOT SLO {a.tpot_slo_ms:g} ms)")
    print(f"  max batch within SLO       {slo_batch:,} streams at {a.avg_ctx}-token context")
    print(f"  -> admit at most           {slo_tokens:,.0f} KV tokens per decode batch")
    print("  The scheduler needs a two-dimensional cap. KV bytes read scale with total")
    print("  tokens, so one 32k request costs the same bandwidth as sixteen 2k requests;")
    print("  but the all-reduce scales with the stream count, so a wide batch of short")
    print("  requests hits a different ceiling. Cap on both and re-run this for each")
    print("  context length you actually serve.")

    print()
    print("COST")
    print(f"  node                       ${a.gpus * a.gpu_hour_usd:,.2f}/hour")
    print(f"  per 1M output tokens       ${out_usd_per_m:,.2f}")
    print(f"  per 1M input tokens        ${in_usd_per_m:,.2f}")
    print("  Input tokens get cheaper again by the prefix-cache hit rate: a 60% hit")
    print(f"  rate bills 1M tokens while computing 400k, i.e. "
          f"${in_usd_per_m * 0.4:,.2f} per billed 1M.")
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
