#!/usr/bin/env python3
"""Static vs continuous batching on one GPU, simulated.

Every number this program prints is a MODEL OUTPUT, not a hardware measurement.
The point is to make the scheduling lesson measurable without a GPU, and to make
the latency-throughput tradeoff something you can sweep rather than recite.

Cost model (one GPU, one model replica, times in milliseconds):

    prefill step : launch + prefill_ms_per_token * tokens_in_step
    decode step  : launch + decode_base + kv_ms_per_token * resident_tokens
    mixed step   : launch + decode_base + kv_ms_per_token * resident_tokens
                          + prefill_ms_per_token * prefill_tokens

`decode_base` is the weight-streaming floor: one pass over the model weights,
paid once per step no matter how many sequences are in the batch. That single
term is *why* batching raises throughput. `kv_ms_per_token` is the marginal
per-resident-token cost, so a bigger batch or a longer context still makes the
step slower -- that is the latency half of the tradeoff. The mixed step shares
one weight pass between a prefill chunk and the ongoing decodes, which is the
arithmetic-intensity slack that chunked prefill exploits.

Default parameters are order-of-magnitude figures for a ~7B-class model in
bf16 on one data-center GPU. Override them if you want a different regime; the
*shape* of the result is the lesson, not the absolute values.

Stdlib only. Deterministic for a fixed --seed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

POLICIES = ("static", "continuous", "continuous_chunked")


# --------------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineModel:
    decode_base_ms: float = 4.7
    kv_ms_per_token: float = 4.3e-5
    prefill_ms_per_token: float = 0.047
    launch_ms: float = 0.3

    def prefill_step_ms(self, prefill_tokens: int) -> float:
        return self.launch_ms + self.prefill_ms_per_token * prefill_tokens

    def decode_step_ms(self, resident_tokens: int) -> float:
        return self.launch_ms + self.decode_base_ms + self.kv_ms_per_token * resident_tokens

    def mixed_step_ms(self, resident_tokens: int, prefill_tokens: int) -> float:
        return self.decode_step_ms(resident_tokens) + self.prefill_ms_per_token * prefill_tokens


# --------------------------------------------------------------------------- #
# Workload
# --------------------------------------------------------------------------- #


@dataclass
class Request:
    rid: int
    arrival_ms: float
    prompt_len: int
    output_len: int
    prefilled: int = 0
    generated: int = 0
    first_token_ms: Optional[float] = None
    last_token_ms: Optional[float] = None
    finish_ms: Optional[float] = None
    # Every individual inter-token gap. TPOT is their mean; the tail is where a
    # prefill stall shows up, and the mean hides it completely.
    itl_ms: List[float] = field(default_factory=list)

    @property
    def context_len(self) -> int:
        return self.prefilled + max(0, self.generated - 1)

    @property
    def ttft_ms(self) -> float:
        assert self.first_token_ms is not None
        return self.first_token_ms - self.arrival_ms

    @property
    def tpot_ms(self) -> float:
        """Mean inter-token latency after the first token.

        Single-token responses have no inter-token interval; they report 0.0 and
        are therefore never the reason a TPOT SLO fails.
        """
        assert self.first_token_ms is not None and self.finish_ms is not None
        if self.output_len <= 1:
            return 0.0
        return (self.finish_ms - self.first_token_ms) / (self.output_len - 1)


def make_trace(
    n_requests: int,
    rate_rps: float,
    seed: int,
    prompt_mu: float = 6.0,
    prompt_sigma: float = 0.8,
    output_mu: float = 4.6,
    output_sigma: float = 1.0,
) -> List[Request]:
    """Poisson arrivals with lognormal prompt and output lengths.

    The output-length spread is the load-bearing part of the trace: static
    batching only looks bad when sequences in one batch finish at different
    times, which is exactly what real traffic does.
    """
    rng = random.Random(seed)
    trace: List[Request] = []
    now_ms = 0.0
    for rid in range(n_requests):
        now_ms += rng.expovariate(rate_rps) * 1000.0
        prompt = int(min(4096, max(8, rng.lognormvariate(prompt_mu, prompt_sigma))))
        output = int(min(2048, max(1, rng.lognormvariate(output_mu, output_sigma))))
        trace.append(Request(rid=rid, arrival_ms=now_ms, prompt_len=prompt, output_len=output))
    return trace


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #


@dataclass
class Accumulator:
    """Time-weighted bookkeeping for one run."""

    busy_ms: float = 0.0
    idle_ms: float = 0.0
    launch_ms: float = 0.0
    slot_ms_computed: float = 0.0
    slot_ms_useful: float = 0.0

    def step(self, dt_ms: float, slots_computed: int, slots_useful: int, launch_ms: float) -> None:
        self.busy_ms += dt_ms
        self.launch_ms += launch_ms
        self.slot_ms_computed += dt_ms * slots_computed
        self.slot_ms_useful += dt_ms * slots_useful


@dataclass
class Result:
    policy: str
    max_batch: int
    requests: int
    wall_ms: float
    output_tokens: int
    ttft_ms: List[float] = field(default_factory=list)
    tpot_ms: List[float] = field(default_factory=list)
    itl_ms: List[float] = field(default_factory=list)
    acc: Accumulator = field(default_factory=Accumulator)
    goodput_requests: int = 0

    # -- derived ------------------------------------------------------------ #
    @property
    def tokens_per_s(self) -> float:
        return self.output_tokens / (self.wall_ms / 1000.0)

    @property
    def requests_per_s(self) -> float:
        return self.requests / (self.wall_ms / 1000.0)

    @property
    def goodput_per_s(self) -> float:
        return self.goodput_requests / (self.wall_ms / 1000.0)

    @property
    def goodput_pct(self) -> float:
        return 100.0 * self.goodput_requests / self.requests

    @property
    def mean_occupancy(self) -> float:
        """Time-weighted mean number of sequences doing useful work per step."""
        return self.acc.slot_ms_useful / self.acc.busy_ms if self.acc.busy_ms else 0.0

    @property
    def padded_slot_waste(self) -> float:
        """Share of computed slot-time spent on sequences that had already finished."""
        if not self.acc.slot_ms_computed:
            return 0.0
        wasted = self.acc.slot_ms_computed - self.acc.slot_ms_useful
        return wasted / self.acc.slot_ms_computed

    @property
    def gpu_idle_fraction(self) -> float:
        """Fraction of wall time the engine had no sequence to run at all."""
        return self.acc.idle_ms / self.wall_ms if self.wall_ms else 0.0

    @property
    def slot_idle_fraction(self) -> float:
        """Fraction of the batch-slot x time rectangle that did no useful work.

        Counts both empty slots (queue starvation or a scheduler that refuses to
        admit) and slots held by finished sequences. This is the single number
        that separates static from continuous batching.
        """
        capacity_ms = self.wall_ms * self.max_batch
        return 1.0 - (self.acc.slot_ms_useful / capacity_ms) if capacity_ms else 0.0

    @property
    def launch_overhead_pct(self) -> float:
        return 100.0 * self.acc.launch_ms / self.acc.busy_ms if self.acc.busy_ms else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. Deterministic and dependency-free."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[idx]


def _finalize(
    policy: str,
    max_batch: int,
    trace: Sequence[Request],
    wall_ms: float,
    acc: Accumulator,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
) -> Result:
    res = Result(
        policy=policy,
        max_batch=max_batch,
        requests=len(trace),
        wall_ms=wall_ms,
        output_tokens=sum(r.generated for r in trace),
        acc=acc,
    )
    for r in trace:
        res.ttft_ms.append(r.ttft_ms)
        res.tpot_ms.append(r.tpot_ms)
        res.itl_ms.extend(r.itl_ms)
        if r.ttft_ms <= slo_ttft_ms and r.tpot_ms <= slo_tpot_ms:
            res.goodput_requests += 1
    return res


def simulate_static(
    trace: Sequence[Request],
    engine: EngineModel,
    max_batch: int,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
) -> Result:
    """Fixed batch: nobody new is admitted until the whole batch has finished.

    Finished slots are modelled as still costing their final context length --
    padding to the batch maximum would be worse still, so this is the *charitable*
    version of static batching.
    """
    reqs = list(trace)
    acc = Accumulator()
    now = 0.0
    next_idx = 0

    while next_idx < len(reqs):
        if reqs[next_idx].arrival_ms > now:
            acc.idle_ms += reqs[next_idx].arrival_ms - now
            now = reqs[next_idx].arrival_ms

        batch: List[Request] = []
        while next_idx < len(reqs) and len(batch) < max_batch and reqs[next_idx].arrival_ms <= now:
            batch.append(reqs[next_idx])
            next_idx += 1

        prefill_tokens = sum(r.prompt_len for r in batch)
        dt = engine.prefill_step_ms(prefill_tokens)
        now += dt
        acc.step(dt, len(batch), len(batch), engine.launch_ms)
        for r in batch:
            r.prefilled = r.prompt_len
            r.first_token_ms = now
            _emit_token(r, now)

        while any(r.finish_ms is None for r in batch):
            resident = sum(r.context_len for r in batch)
            dt = engine.decode_step_ms(resident)
            now += dt
            alive = sum(1 for r in batch if r.finish_ms is None)
            acc.step(dt, len(batch), alive, engine.launch_ms)
            _advance_decodes(batch, now)

    return _finalize("static", max_batch, reqs, now, acc, slo_ttft_ms, slo_tpot_ms)


def simulate_continuous(
    trace: Sequence[Request],
    engine: EngineModel,
    max_batch: int,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
    chunk_tokens: Optional[int] = None,
) -> Result:
    """Iteration-level scheduling: admit and evict at every step.

    `chunk_tokens=None` is plain continuous batching -- a prefill consumes a
    whole step and every ongoing decode stalls for its duration. Setting
    `chunk_tokens` splits the prefill and piggybacks the decodes onto the same
    step, which is the chunked-prefill fix for that stall.
    """
    reqs = list(trace)
    acc = Accumulator()
    now = 0.0
    next_idx = 0
    active: List[Request] = []
    policy = "continuous_chunked" if chunk_tokens else "continuous"

    while next_idx < len(reqs) or active:
        if not active and next_idx < len(reqs) and reqs[next_idx].arrival_ms > now:
            acc.idle_ms += reqs[next_idx].arrival_ms - now
            now = reqs[next_idx].arrival_ms

        # Admission: one new sequence per step keeps the scheduler simple and
        # bounds how much prefill a single step can drag in.
        prefilling = next((r for r in active if r.prefilled < r.prompt_len), None)
        if (
            prefilling is None
            and len(active) < max_batch
            and next_idx < len(reqs)
            and reqs[next_idx].arrival_ms <= now
        ):
            prefilling = reqs[next_idx]
            next_idx += 1
            active.append(prefilling)

        decoding = [r for r in active if r.prefilled >= r.prompt_len]

        if prefilling is not None:
            remaining = prefilling.prompt_len - prefilling.prefilled
            tokens = min(remaining, chunk_tokens) if chunk_tokens else remaining
            if chunk_tokens:
                resident = sum(r.context_len for r in decoding)
                dt = engine.mixed_step_ms(resident, tokens)
            else:
                dt = engine.prefill_step_ms(tokens)
            now += dt
            acc.step(dt, len(active), len(active), engine.launch_ms)

            prefilling.prefilled += tokens
            if prefilling.prefilled >= prefilling.prompt_len:
                prefilling.first_token_ms = now
                _emit_token(prefilling, now)
            if chunk_tokens:
                _advance_decodes(decoding, now)
        else:
            resident = sum(r.context_len for r in decoding)
            dt = engine.decode_step_ms(resident)
            now += dt
            acc.step(dt, len(active), len(decoding), engine.launch_ms)
            _advance_decodes(decoding, now)

        active = [r for r in active if r.finish_ms is None]

    return _finalize(policy, max_batch, reqs, now, acc, slo_ttft_ms, slo_tpot_ms)


def _emit_token(r: Request, now: float) -> None:
    if r.last_token_ms is not None:
        r.itl_ms.append(now - r.last_token_ms)
    r.last_token_ms = now
    r.generated += 1
    if r.generated >= r.output_len:
        r.finish_ms = now


def _advance_decodes(decoding: Sequence[Request], now: float) -> None:
    for r in decoding:
        if r.finish_ms is None:
            _emit_token(r, now)


def run_policy(policy: str, args: argparse.Namespace, engine: EngineModel, max_batch: int) -> Result:
    trace = make_trace(args.requests, args.rate, args.seed)
    if policy == "static":
        return simulate_static(trace, engine, max_batch, args.slo_ttft_ms, args.slo_tpot_ms)
    chunk = args.chunk_tokens if policy == "continuous_chunked" else None
    return simulate_continuous(
        trace, engine, max_batch, args.slo_ttft_ms, args.slo_tpot_ms, chunk_tokens=chunk
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

BANNER = "SIMULATED (analytic cost model) -- not measured on hardware"


def result_row(res: Result) -> Dict[str, float]:
    return {
        "policy": res.policy,
        "max_batch": res.max_batch,
        "tokens_per_s": round(res.tokens_per_s, 1),
        "requests_per_s": round(res.requests_per_s, 3),
        "ttft_mean_ms": round(sum(res.ttft_ms) / len(res.ttft_ms), 1),
        "ttft_p95_ms": round(percentile(res.ttft_ms, 95), 1),
        "tpot_mean_ms": round(sum(res.tpot_ms) / len(res.tpot_ms), 2),
        "tpot_p95_ms": round(percentile(res.tpot_ms, 95), 2),
        "itl_p99_ms": round(percentile(res.itl_ms, 99), 2),
        "itl_max_ms": round(max(res.itl_ms) if res.itl_ms else 0.0, 2),
        "mean_occupancy": round(res.mean_occupancy, 2),
        "slot_idle_pct": round(100.0 * res.slot_idle_fraction, 1),
        "padded_slot_waste_pct": round(100.0 * res.padded_slot_waste, 1),
        "gpu_idle_pct": round(100.0 * res.gpu_idle_fraction, 2),
        "launch_overhead_pct": round(res.launch_overhead_pct, 1),
        "goodput_req_per_s": round(res.goodput_per_s, 3),
        "goodput_pct": round(res.goodput_pct, 1),
    }


def print_table(title: str, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    widths = [max(len(c), *(len(f"{r[c]}") for r in rows)) for c in columns]
    print(title)
    print("  ".join(c.rjust(w) for c, w in zip(columns, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(f"{r[c]}".rjust(w) for c, w in zip(columns, widths)))
    print()


def report(rows: Sequence[Dict[str, object]]) -> None:
    print_table(
        "Throughput and latency",
        rows,
        ["policy", "max_batch", "tokens_per_s", "requests_per_s",
         "ttft_mean_ms", "ttft_p95_ms", "tpot_mean_ms", "tpot_p95_ms"],
    )
    print_table(
        "Utilization, jitter and goodput",
        rows,
        ["policy", "mean_occupancy", "slot_idle_pct", "padded_slot_waste_pct", "gpu_idle_pct",
         "launch_overhead_pct", "itl_p99_ms", "itl_max_ms", "goodput_req_per_s", "goodput_pct"],
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--rate", type=float, default=12.0, help="Poisson arrival rate, requests/s")
    p.add_argument("--max-batch", type=int, default=32, help="admission cap on concurrent sequences")
    p.add_argument("--chunk-tokens", type=int, default=512, help="prefill token budget per step")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    p.add_argument("--slo-tpot-ms", type=float, default=50.0)
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument(
        "--sweep-batch",
        default="",
        help="comma-separated batch caps; sweeps continuous_chunked to expose the SLO knee",
    )
    p.add_argument("--json", action="store_true", help="emit rows as JSON instead of tables")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    engine = EngineModel()

    requested = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown = [p for p in requested if p not in POLICIES]
    if unknown:
        raise SystemExit(f"unknown policy/policies: {', '.join(unknown)}; choose from {POLICIES}")

    rows = [result_row(run_policy(p, args, engine, args.max_batch)) for p in requested]

    sweep_rows: List[Dict[str, object]] = []
    if args.sweep_batch:
        for token in args.sweep_batch.split(","):
            cap = int(token)
            sweep_rows.append(result_row(run_policy("continuous_chunked", args, engine, cap)))

    if args.json:
        print(json.dumps({"banner": BANNER, "policies": rows, "sweep": sweep_rows}, indent=2))
        return 0

    print(f"=== {BANNER} ===")
    print(
        f"trace: {args.requests} requests, {args.rate} req/s Poisson, seed={args.seed} | "
        f"max_batch={args.max_batch}, chunk={args.chunk_tokens} tok | "
        f"SLO: TTFT <= {args.slo_ttft_ms:.0f} ms, TPOT <= {args.slo_tpot_ms:.0f} ms"
    )
    print(
        f"engine model: decode_base={engine.decode_base_ms} ms/step, "
        f"prefill={engine.prefill_ms_per_token * 1000:.0f} us/token, "
        f"kv={engine.kv_ms_per_token * 1000:.3f} us/resident-token, "
        f"launch={engine.launch_ms} ms/step\n"
    )
    report(rows)
    if sweep_rows:
        print("=== Batch-cap sweep (continuous_chunked): the latency-throughput tradeoff ===")
        report(sweep_rows)
        best = max(sweep_rows, key=lambda r: r["goodput_req_per_s"])
        regressed = any(
            r["max_batch"] > best["max_batch"] and r["goodput_req_per_s"] < best["goodput_req_per_s"]
            for r in sweep_rows
        )
        print(
            f"Peak goodput at max_batch={best['max_batch']}: "
            f"{best['goodput_req_per_s']} req/s ({best['goodput_pct']}% of offered load)."
        )
        print("Below the peak, queueing delay breaks the TTFT budget.")
        print(
            "Above the peak, the fuller batch breaks the TPOT budget while raw token"
            " throughput is flat or still rising -- goodput, not throughput, is the objective."
            if regressed
            else "No larger cap regressed at this offered load: the engine's own capacity, not the"
            " admission cap, is the ceiling. Raise --rate or tighten --slo-tpot-ms to expose the knee."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
