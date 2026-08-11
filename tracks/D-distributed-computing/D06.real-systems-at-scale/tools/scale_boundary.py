#!/usr/bin/env python3
"""Where does a described workload stop scaling, and what binds it when it does?

This is Track D's capstone calculator. It assembles the five constraints the track
derived -- serial fraction, collective latency, collective bandwidth, straggler
variance and component failure -- into ONE prediction for a workload you describe,
and names the regime of the scale-boundary ladder (module README section 6.4) that
each rank count lands on.

Everything here is an ANALYTIC MODEL. No cluster is contacted, no collective is
timed, no failure is observed. That warning matters more in this tool than anywhere
else in the repository, because this is the only tool here that predicts things
nobody on this machine can measure: the behaviour of 65,536 ranks and the failure
rate of a fleet.

The model, per step, for a fixed problem on N ranks:

    T(N) = s*C + (1-s)*C/N          compute: the serial part does not divide
         + 2*(N-1)*alpha            collective latency: ring step count
         + 2*m*(N-1)/N / B          collective bandwidth: ring traffic per rank
         + (E[max of N] - 1) * (everything above)   straggler variance

    efficiency(N) = T(1) / (N * T(N))

  C = --compute-ms, the single-rank time for the step's work
  s = --serial-fraction, m = --model-bytes, alpha = --alpha-us, B = --bandwidth-gbs

The collective is modelled as a flat ring all-reduce: 2(N-1) steps and about 2m
bytes per rank, which Patarasuk and Yuan proved bandwidth-optimal. A real library
switches algorithms and reduces hierarchically, so the latency term here is an
UPPER bound at large N -- see module D01 section 3.3 and D05 section 3.6.

Failure is a separate Poisson estimate, not a term in the step:

    nodes = ceil(N / --gpus-per-node)
    expected interruptions = --job-hours * nodes / --node-mtbf-hours

When a job expects an interruption before it ends, goodput SUPERSEDES the step-time
constraint: no step-time fix delivers a finished run, so the named binding constraint
becomes failure and the step-time term is reported as the second problem. That is the
whole argument of the module README section 3.9.

Stdlib only, and no site packages required:
    python -S -I tools/scale_boundary.py --preset dp-training --ranks 4096
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap

BANNER = "analytic scale-boundary model, NOT a measurement: nothing here was run on a machine"
BANNER_2 = ("the failure arithmetic is a Poisson model over an ASSUMED node MTBF, "
            "not a fleet observation")

# Per-rank step-time jitter, as a fraction of the step. 2% is the illustrative figure
# D01 section 3.4 uses for its seeded straggler simulation; it is unremarkable jitter
# from clock throttling, ECC scrubbing or an OS timer, and it is an assumption, not a
# measurement. There is deliberately no flag for it: this tool already has ten inputs.
JITTER_SIGMA = 0.02

# An exact tie at the efficiency target can land a few ulps on the wrong side of a
# float comparison, so compare with slack. Same reason as D01's scaling_model.py.
TOL = 1e-12

# Bisection ceiling for the knee. 2**40 ranks is far past any machine that will ever
# exist and keeps the search bounded for tiny serial fractions.
SEARCH_CEILING = 1 << 40

# A step whose overheads are under this share is still paying for more ranks, so the
# honest label is "compute", not whichever overhead happens to be largest.
COMPUTE_BOUND_SHARE = 0.10

# Rank counts where the ladder in the module README section 6.4 changes row. The
# boundaries are SOFT -- they depend on message size, fabric and workload, which is
# the entire reason this tool exists rather than a lookup table.
REGIMES = ((64, "8-64", "D03, D04"),
           (1_000, "64-1k", "D05"),
           (10_000, "1k-10k", "D01, D05"),
           (100_000, "10k-100k", "D06, C03"))

CONSTRAINTS = {
    "compute": "compute (still scaling)",
    "serial": "serial fraction",
    "latency": "collective latency",
    "bandwidth": "collective bandwidth",
    "variance": "straggler variance",
    "failure": "goodput: failures",
}

# Every preset is built from PUBLIC, defensible assumptions and states them under
# --explain. None of them reproduces a measured figure from any system, and none of
# them is a datasheet quote. "why" is the sentence --explain prints for that input.
PRESETS = {
    "dp-training": {
        "what": "data-parallel training: one gradient all-reduce on the critical path per step",
        "values": {
            "ranks": 4096,
            "model_bytes": 2.0e9,
            "compute_ms": 60_000.0,
            "serial_fraction": 0.001,
            "alpha_us": 5.0,
            "bandwidth_gbs": 25.0,
            "gpus_per_node": 8,
            "node_mtbf_hours": 40_000.0,
            "job_hours": 168.0,
        },
        "why": {
            "model_bytes": "the gradient buffer of a 1e9-parameter model at 2 bytes per "
                           "parameter, all-reduced once per step",
            "compute_ms": "single-rank time for one global batch, chosen so the ladder "
                          "crosses several regimes; substitute your own measured step",
            "serial_fraction": "0.1% of the step that does not divide (optimizer bookkeeping, "
                               "logging); illustrative",
            "alpha_us": "inter-node RDMA order of magnitude from D01 section 3.1, which "
                        "gives 2-5 us; not a datasheet quote",
            "bandwidth_gbs": "inter-node RDMA order of magnitude from D01 section 3.1, which "
                             "gives 10s of GB/s achievable per rank",
            "gpus_per_node": "8 accelerators per node, the shape of every system in the "
                             "module README section 3.8 and of most AI clusters",
            "node_mtbf_hours": "an ASSUMED per-node MTBF. Operators rarely publish this "
                               "(README section 6.2) -- replace it with your own fleet's data",
            "job_hours": "a one-week run",
        },
    },
    "hpc-halo": {
        "what": "structured-mesh HPC: small per-step exchange plus a global reduction, "
                "Fugaku's workload class",
        "values": {
            "ranks": 65_536,
            "model_bytes": 8.0e6,
            "compute_ms": 1_800_000.0,
            "serial_fraction": 0.00001,
            "alpha_us": 2.5,
            "bandwidth_gbs": 25.0,
            "gpus_per_node": 4,
            "node_mtbf_hours": 100_000.0,
            "job_hours": 24.0,
        },
        "why": {
            "model_bytes": "8 MB per step, the order of a halo face plus a convergence "
                           "reduction. NOTE the model shape: this tool costs ONE global "
                           "collective, and a real halo exchange is nearest-neighbour and "
                           "does not grow with rank count, so the latency term over-states "
                           "a torus-mapped code (README section 3.5)",
            "compute_ms": "single-rank time for one time step of a large mesh; substitute "
                          "your own",
            "serial_fraction": "1e-5, a well-optimized code with almost no undivided work; "
                               "illustrative",
            "alpha_us": "2.5 us, D01 section 3.1's reference inter-node figure",
            "bandwidth_gbs": "25 GB/s, D01 section 3.1's reference inter-node figure",
            "gpus_per_node": "4 ranks per node, a common one-rank-per-NUMA-domain mapping. "
                             "The flag is --gpus-per-node because that is the common case; "
                             "it means ranks per failure domain",
            "node_mtbf_hours": "an ASSUMED per-node MTBF for a well-run national-scale "
                               "system. Not published by any site cited here",
            "job_hours": "a one-day allocation",
        },
    },
    "llm-16k": {
        "what": "LLM pre-training at the 16k-accelerator scale: the regime where failure, "
                "not throughput, is the binding constraint",
        "values": {
            "ranks": 16_384,
            "model_bytes": 8.0e9,
            "compute_ms": 1_200_000.0,
            "serial_fraction": 0.0002,
            "alpha_us": 5.0,
            "bandwidth_gbs": 25.0,
            "gpus_per_node": 8,
            "node_mtbf_hours": 30_000.0,
            "job_hours": 720.0,
        },
        "why": {
            "ranks": "16,384 ranks, the order of the 16k-GPU scale the Llama 3 paper "
                     "documents for pre-training (arXiv:2407.21783)",
            "model_bytes": "8 GB exchanged per step. THIS REPO'S ASSUMPTION, not a figure "
                           "from any paper. The Llama 3 paper describes 4D parallelism, so a "
                           "flat all-reduce of a whole model is NOT what that run did; this "
                           "preset deliberately models the naive shape so the tool can show "
                           "you why nobody uses it at this scale",
            "compute_ms": "single-rank time for one global batch; illustrative",
            "serial_fraction": "0.02% undivided work per step; illustrative",
            "alpha_us": "inter-node RDMA order of magnitude from D01 section 3.1",
            "bandwidth_gbs": "inter-node RDMA order of magnitude from D01 section 3.1. Not "
                             "the fabric rate of any specific cluster",
            "gpus_per_node": "8 accelerators per node, the common public server shape",
            "node_mtbf_hours": "an ASSUMED per-node MTBF. The Llama 3 paper publishes "
                               "interruption counts for its own cluster and its own 54-day "
                               "snapshot; those are NOT used as inputs here, because reusing "
                               "one run's reliability record as a planning constant is the "
                               "error README section 2.4 warns about. Read the paper for its "
                               "figures and supply your own here",
            "job_hours": "a 30-day run; illustrative, and not the snapshot length of any "
                         "published run",
        },
    },
}

MODEL_KEYS = ("ranks", "model_bytes", "compute_ms", "serial_fraction", "alpha_us",
              "bandwidth_gbs", "gpus_per_node", "node_mtbf_hours", "job_hours")


def validate(cfg: dict, efficiency_target: float) -> None:
    """Reject inputs the model has no meaning for, naming the offending value."""
    if cfg["ranks"] < 1:
        raise ValueError(f"--ranks must be at least 1, got {cfg['ranks']}")
    if not 0.0 < efficiency_target <= 1.0:
        raise ValueError(
            f"--efficiency-target must be in (0, 1], got {efficiency_target}")
    if not 0.0 <= cfg["serial_fraction"] <= 1.0:
        raise ValueError(
            f"--serial-fraction must be in [0, 1], got {cfg['serial_fraction']}")
    if cfg["compute_ms"] <= 0.0:
        raise ValueError(f"--compute-ms must be positive, got {cfg['compute_ms']}")
    if cfg["model_bytes"] < 0.0:
        raise ValueError(f"--model-bytes must be non-negative, got {cfg['model_bytes']}")
    if cfg["alpha_us"] < 0.0:
        raise ValueError(f"--alpha-us must be non-negative, got {cfg['alpha_us']}")
    if cfg["bandwidth_gbs"] <= 0.0:
        raise ValueError(f"--bandwidth-gbs must be positive, got {cfg['bandwidth_gbs']}")
    if cfg["gpus_per_node"] < 1:
        raise ValueError(f"--gpus-per-node must be at least 1, got {cfg['gpus_per_node']}")
    if cfg["node_mtbf_hours"] <= 0.0:
        raise ValueError(
            f"--node-mtbf-hours must be positive, got {cfg['node_mtbf_hours']}")
    if cfg["job_hours"] < 0.0:
        raise ValueError(f"--job-hours must be non-negative, got {cfg['job_hours']}")


def expected_max_factor(ranks: int) -> float:
    """Expected slowest of `ranks` step times, relative to the mean.

    Per-rank step time is taken as Normal(1, JITTER_SIGMA). A collective is a barrier,
    so the step advances at the MAXIMUM over ranks, and the expected maximum of N
    samples grows with N even though the distribution never changes -- which is why
    jitter is a scaling term and not an annoyance (D01 section 3.4).

    N = 1 and N = 2 are exact (E[max of two standard normals] = 1/sqrt(pi)). Above
    that this is the standard extreme-value approximation to E[max]. It runs a few
    tenths of a percent below D01's seeded simulation above 1,000 ranks and a little
    low under about 16 ranks, where the term is negligible anyway.
    """
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    if ranks == 1:
        return 1.0
    if ranks == 2:
        return 1.0 + JITTER_SIGMA / math.sqrt(math.pi)
    ln_n = math.log(ranks)
    root = math.sqrt(2.0 * ln_n)
    z = root - (math.log(ln_n) + math.log(4.0 * math.pi)) / (2.0 * root)
    return 1.0 + JITTER_SIGMA * z


def step_terms(ranks: int, cfg: dict) -> dict:
    """The four modelled terms of one step at `ranks`, in seconds."""
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    compute_total = cfg["compute_ms"] / 1e3
    serial = cfg["serial_fraction"] * compute_total
    compute = (1.0 - cfg["serial_fraction"]) * compute_total / ranks
    if ranks == 1:
        latency = 0.0
        bandwidth = 0.0
    else:
        latency = 2.0 * (ranks - 1) * cfg["alpha_us"] * 1e-6
        bandwidth = (2.0 * cfg["model_bytes"] * (ranks - 1) / ranks
                     / (cfg["bandwidth_gbs"] * 1e9))
    exposed = serial + compute + latency + bandwidth
    variance = (expected_max_factor(ranks) - 1.0) * exposed
    # compute_total is what the table shows: the serial part is compute the machine does,
    # it just refuses to divide. Reporting it inside the compute column is what makes the
    # four displayed terms sum exactly to the step.
    return {"compute": compute, "serial": serial, "compute_total": serial + compute,
            "latency": latency, "bandwidth": bandwidth, "variance": variance,
            "step": exposed + variance}


def parallel_efficiency(ranks: int, cfg: dict) -> float:
    """Speedup per rank against a single rank: what you bought over what you paid for.

    Strictly decreasing in `ranks`: N * T(N) = N*serial + (1-s)*C + N*(latency +
    bandwidth + variance), and every one of those terms is non-decreasing in N while
    at least one strictly increases. So bisecting for the knee is well defined.
    """
    single = step_terms(1, cfg)["step"]
    return single / (ranks * step_terms(ranks, cfg)["step"])


def failure_estimate(ranks: int, cfg: dict) -> dict:
    """Poisson interruption estimate over the job, from node MTBF and node count.

    Nodes fail independently with rate 1/MTBF, so a job holding `nodes` of them has
    rate nodes/MTBF and a mean time between interruptions of MTBF/nodes. This is the
    arithmetic behind "failure is not an exception, it is the operating regime": the
    expected count grows with BOTH rank count and job length, so better components
    move the threshold and cannot remove it.
    """
    nodes = math.ceil(ranks / cfg["gpus_per_node"])
    mtbi_hours = cfg["node_mtbf_hours"] / nodes
    expected = cfg["job_hours"] * nodes / cfg["node_mtbf_hours"]
    return {"nodes": nodes, "mtbi_hours": mtbi_hours, "expected_interruptions": expected,
            "p_uninterrupted": math.exp(-expected)}


def regime_of(ranks: int, gpus_per_node: int) -> tuple[str, str]:
    """(regime label, Track D modules that own it) from the section 6.4 ladder."""
    if ranks <= gpus_per_node:
        return "1-node", "D04"
    for ceiling, label, owners in REGIMES:
        if ranks <= ceiling:
            return label, owners
    return "100k+", "D06, C03"


def step_constraint(terms: dict) -> str:
    """Name the step-time term that binds, from the section 6.4 vocabulary.

    The binding constraint is the largest OVERHEAD term. Compute is excluded, because
    compute dividing by N is the thing you are buying rather than a problem. When the
    overheads together are still a small share of the step, the honest label is
    compute: ranks are still paying and no overhead is worth attacking yet.
    """
    overheads = {name: terms[name]
                 for name in ("serial", "latency", "bandwidth", "variance")}
    if sum(overheads.values()) < COMPUTE_BOUND_SHARE * terms["step"]:
        return "compute"
    return max(overheads, key=lambda name: overheads[name])


def rank_ladder(max_ranks: int) -> list[int]:
    """Powers of two up to max_ranks, with max_ranks itself appended if it is not one."""
    ladder = []
    ranks = 1
    while ranks <= max_ranks:
        ladder.append(ranks)
        ranks *= 2
    if ladder[-1] != max_ranks:
        ladder.append(max_ranks)
    return ladder


def knee(cfg: dict, target: float) -> int | None:
    """Largest rank count whose efficiency still meets `target`, by bisection.

    Returns None when the target never breaks inside the search range. Bisection
    rather than algebra: the moment the variance term is in the model there is no
    closed form left, and monotonicity is all bisection needs.
    """
    if parallel_efficiency(SEARCH_CEILING, cfg) >= target - TOL:
        return None
    low, high = 1, SEARCH_CEILING
    while high - low > 1:
        mid = (low + high) // 2
        if parallel_efficiency(mid, cfg) >= target - TOL:
            low = mid
        else:
            high = mid
    return low


def build_rows(cfg: dict, target: float) -> list[dict]:
    """One row per rung of the rank ladder: the term split, the regime, the constraint.

    Failure SUPERSEDES the step-time constraint rather than replacing it. If the job
    expects an interruption before it ends, no amount of step-time engineering delivers
    a finished job, so goodput is what binds -- but the step-time term is still the
    right answer to "and what would I fix second", so both are kept.
    """
    rows = []
    for ranks in rank_ladder(cfg["ranks"]):
        terms = step_terms(ranks, cfg)
        failure = failure_estimate(ranks, cfg)
        label, owners = regime_of(ranks, cfg["gpus_per_node"])
        step_binds = step_constraint(terms)
        superseded = failure["expected_interruptions"] >= 1.0
        efficiency = parallel_efficiency(ranks, cfg)
        rows.append({"ranks": ranks, "regime": label, "owners": owners,
                     "efficiency": efficiency,
                     "meets_target": efficiency >= target - TOL,
                     "step_constraint": step_binds,
                     "constraint": "failure" if superseded else step_binds,
                     "constraint_label": CONSTRAINTS["failure" if superseded
                                                     else step_binds],
                     "nodes": failure["nodes"],
                     "expected_interruptions": failure["expected_interruptions"],
                     **terms})
    return rows


def explain_row(row: dict) -> str:
    """One sentence naming the term that dominates, why, and where it is covered."""
    if row["constraint"] == "failure":
        return (f"{row['expected_interruptions']:.1f} interruptions expected before this "
                f"job ends, so no step-time fix delivers a finished run; goodput binds and "
                f"{CONSTRAINTS[row['step_constraint']]} is only second "
                f"(README section 3.9, C03)")
    overhead = row["latency"] + row["bandwidth"] + row["variance"] + row["serial"]
    share = 100.0 * overhead / row["step"] if row["step"] else 0.0
    if row["step_constraint"] == "compute":
        overheads = {name: row[name]
                     for name in ("serial", "latency", "bandwidth", "variance")}
        nextup = max(overheads, key=lambda name: overheads[name])
        return (f"overheads are {share:.1f}% of the step, so ranks still pay; the largest "
                f"of them, and so the next thing to bind, is "
                f"{CONSTRAINTS[nextup]} ({row['owners']})")
    if row["step_constraint"] == "serial":
        return (f"the serial {fmt_ms(row['serial'])} ms does not divide by N, so it is a "
                f"floor no fabric can lower (D01 section 2, Amdahl)")
    if row["step_constraint"] == "latency":
        return (f"the ring's 2(N-1) steps cost {fmt_ms(row['latency'])} ms of a "
                f"{fmt_ms(row['step'])} ms step; attack the STEP COUNT with hierarchical "
                f"collectives, not the wire (D01 section 3.3, D05 section 3.6)")
    if row["step_constraint"] == "bandwidth":
        return (f"the ring moves about 2 x model-bytes whatever N is, so "
                f"{fmt_ms(row['bandwidth'])} ms of wire time does not shrink with more "
                f"ranks; only fewer bytes or a faster fabric help ({row['owners']})")
    return (f"the barrier waits on the slowest of {row['ranks']} ranks, and that tax "
            f"({fmt_ms(row['variance'])} ms) is the largest overhead; uniformity, not "
            f"bandwidth, is the fix (D01 section 3.4, D05)")


def emit(prefix: str, text: str, width: int = 92) -> None:
    """Print `prefix + text` wrapped to `width`, hanging-indented under the prefix."""
    pad = " " * len(prefix)
    chunks = textwrap.wrap(text, width=max(24, width - len(prefix))) or [""]
    for index, chunk in enumerate(chunks):
        print((prefix if index == 0 else pad) + chunk)


def fmt_pct(fraction: float) -> str:
    """A percentage that stays informative when efficiency collapses to a thousandth."""
    value = fraction * 100.0
    if value >= 1.0:
        return f"{value:.1f}%"
    if value >= 0.01:
        return f"{value:.2f}%"
    if value <= 0.0:
        return "0%"
    return f"{value:.1g}%"


def fmt_ms(seconds: float) -> str:
    """Milliseconds at a readable precision across the decades this tool spans."""
    value = seconds * 1e3
    if value == 0.0:
        return "0"
    if value >= 100.0:
        return f"{value:,.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def print_table(rows: list[dict]) -> None:
    header = (f"{'ranks':>7}{'regime':>9}{'compute':>10}{'latency':>10}"
              f"{'wire':>10}{'variance':>10}{'step':>10}{'eff':>7}{'interr':>8}"
              f"  binding constraint")
    print("step-time split in milliseconds, and the four terms sum to the step. compute")
    print("includes the serial part that does not divide; eff is parallel efficiency against")
    print("one rank; interr is expected interruptions for a job of this length at this scale.")
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['ranks']:>7}{row['regime']:>9}"
              f"{fmt_ms(row['compute_total']):>10}{fmt_ms(row['latency']):>10}"
              f"{fmt_ms(row['bandwidth']):>10}{fmt_ms(row['variance']):>10}"
              f"{fmt_ms(row['step']):>10}{fmt_pct(row['efficiency']):>7}"
              f"{row['expected_interruptions']:>8.2f}"
              f"  {row['constraint_label']}")
    print("-" * len(header))


SHARED_ASSUMPTIONS = (
    ("collective shape", "a flat ring all-reduce: 2(N-1) steps and about 2m bytes per rank, "
                         "which Patarasuk and Yuan (2009) proved bandwidth-optimal. A real "
                         "library reduces hierarchically, so the latency term here is an "
                         "UPPER bound at large N"),
    ("jitter", f"per-rank step time varies by {JITTER_SIGMA * 100:g}%, D01 section 3.4's "
               "illustrative figure, and the barrier pays the expected maximum over ranks"),
    ("failures", "independent and exponential per node, which ignores correlated failure "
                 "(a rack, a cooling event, a bad batch) and every recovery cost"),
    ("not modelled", "I/O, checkpoint write time, detection time, restart time and lost "
                     "work. Those are C03's subject and they decide real goodput"),
)


def assumption_lines(preset: str, overridden: set) -> list[tuple[str, str]]:
    """(prefix, text) pairs stating what the preset assumes and what the caller replaced."""
    spec = PRESETS[preset]
    lines = [("preset: ", f"{preset} -- {spec['what']}"),
             ("", "what it assumes. Every line is this repo's assumption, not a "
                  "measurement and not a datasheet quote:")]
    for key in MODEL_KEYS:
        flag = "  --" + key.replace("_", "-") + ": "
        if key in overridden:
            lines.append((flag, "from your flag, so the preset's assumption does not apply"))
        elif key in spec["why"]:
            lines.append((flag, spec["why"][key]))
    lines.append(("", "shared by every preset:"))
    for name, text in SHARED_ASSUMPTIONS:
        lines.append((f"  {name}: ", text))
    return lines


def print_failure(cfg: dict, target_rows: dict) -> None:
    print("failure and goodput at the requested scale")
    print(f"  ranks {cfg['ranks']} over {cfg['gpus_per_node']} per node "
          f"= {target_rows['nodes']} nodes")
    print(f"  assumed node MTBF                  {cfg['node_mtbf_hours']:>12,.0f} h")
    print(f"  job MTBI = MTBF / nodes            {target_rows['mtbi_hours']:>12,.2f} h")
    print(f"  job length                         {cfg['job_hours']:>12,.0f} h")
    print(f"  expected interruptions             "
          f"{target_rows['expected_interruptions']:>12,.2f}")
    print(f"  chance of finishing uninterrupted  "
          f"{target_rows['p_uninterrupted'] * 100:>11.2f}%")
    if target_rows["expected_interruptions"] >= 1.0:
        print("  this job expects to be interrupted, so its first design problem is")
        print("  checkpoint cadence, detection and restart -- not step time (C03)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict where a described workload stops scaling, name the binding "
                    "constraint per regime, and estimate its failure budget.",
        epilog="Every number printed is an analytic model, never a measurement.")
    parser.add_argument("--preset", default="dp-training",
                        metavar="{dp-training,hpc-halo,llm-16k}",
                        help="a described workload to start from (default dp-training); "
                             "any flag you pass overrides the preset's value")
    parser.add_argument("--ranks", type=int, default=None,
                        help="largest rank count to tabulate; the ladder runs 1, 2, 4 ... N")
    parser.add_argument("--model-bytes", type=float, default=None,
                        help="bytes entering the per-step collective (a gradient buffer, "
                             "a halo plus reduction)")
    parser.add_argument("--compute-ms", type=float, default=None,
                        help="single-rank compute time for one step of the FIXED problem; "
                             "the parallel part of it divides by rank count")
    parser.add_argument("--serial-fraction", type=float, default=None,
                        help="fraction of --compute-ms that does not divide (Amdahl's s)")
    parser.add_argument("--alpha-us", type=float, default=None,
                        help="per-collective-step latency alpha, in microseconds")
    parser.add_argument("--bandwidth-gbs", type=float, default=None,
                        help="achievable per-rank bandwidth in GB/s (1 GB = 1e9 bytes)")
    parser.add_argument("--gpus-per-node", type=int, default=None,
                        help="ranks per failure domain; sets the node count for the "
                             "failure estimate and the one-node regime")
    parser.add_argument("--node-mtbf-hours", type=float, default=None,
                        help="assumed mean time between failures for ONE node, in hours")
    parser.add_argument("--job-hours", type=float, default=None,
                        help="how long the job needs to run, in hours")
    parser.add_argument("--efficiency-target", type=float, default=0.5,
                        help="efficiency floor that defines the knee (default 0.5)")
    parser.add_argument("--explain", action="store_true",
                        help="add a sentence per regime naming the dominant term, plus "
                             "what the preset assumes")
    parser.add_argument("--json", action="store_true",
                        help="emit the same numbers as JSON instead of tables")
    args = parser.parse_args(argv)

    if args.preset not in PRESETS:
        print(f"error: unknown preset {args.preset!r}; choose from "
              f"{', '.join(sorted(PRESETS))}", file=sys.stderr)
        return 2

    cfg = dict(PRESETS[args.preset]["values"])
    overridden = set()
    for key in MODEL_KEYS:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
            overridden.add(key)

    try:
        validate(cfg, args.efficiency_target)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    target = args.efficiency_target
    rows = build_rows(cfg, target)
    final = rows[-1]
    at_target = knee(cfg, target)
    failure = failure_estimate(cfg["ranks"], cfg)
    explanations = {row["ranks"]: explain_row(row) for row in rows}
    assumptions = assumption_lines(args.preset, overridden)

    if args.json:
        payload = {"banner": BANNER, "banner_note": BANNER_2, "preset": args.preset,
                   "overridden": sorted(overridden), "inputs": cfg,
                   "jitter_sigma": JITTER_SIGMA, "efficiency_target": target,
                   "knee_ranks": at_target, "regimes": rows, "failure": failure}
        if args.explain:
            payload["explanations"] = explanations
            payload["assumptions"] = [(prefix + text).strip()
                                      for prefix, text in assumptions]
        print(json.dumps(payload, indent=2))
        return 0

    print(BANNER)
    print(BANNER_2)
    print()
    if args.explain:
        for prefix, text in assumptions:
            emit(prefix, text)
        print()
    print(f"workload: {cfg['model_bytes']:,.0f} collective bytes per step, "
          f"{cfg['compute_ms']:,.0f} ms single-rank compute, "
          f"serial {cfg['serial_fraction']:g}")
    print(f"fabric:   alpha {cfg['alpha_us']:g} us, {cfg['bandwidth_gbs']:g} GB/s per rank, "
          f"{cfg['gpus_per_node']} ranks per node")
    print(f"job:      {cfg['job_hours']:g} h at up to {cfg['ranks']:,} ranks, "
          f"efficiency target {target * 100:g}%")
    print()
    print_table(rows)

    if args.explain:
        print()
        print("why each regime binds where it does")
        for row in rows:
            emit(f"  N={row['ranks']:<8}", explanations[row["ranks"]])

    print()
    if at_target is None:
        emit("predicted knee: ", f"none -- efficiency never falls below {target * 100:g}% "
                                 f"inside the search range")
    elif at_target == 1:
        emit("predicted knee: ",
             f"N = 1. This workload does not scale at all in this model: one step of the "
             f"collective already costs more than the whole single-rank step, so even two "
             f"ranks are below the {target * 100:g}% target. Cut bytes per step or raise "
             f"work per step before buying ranks")
    else:
        emit("predicted knee: ",
             f"efficiency holds at or above {target * 100:g}% up to N = {at_target:,} "
             f"ranks, and falls below it at N = {at_target + 1:,}")
    tail = ""
    if final["constraint"] == "failure":
        tail = (f" -- the step time is {CONSTRAINTS[final['step_constraint']]}-bound, "
                f"which is the second problem and not the first")
    emit(f"at N = {final['ranks']:,}: ",
         f"efficiency {fmt_pct(final['efficiency'])}, binding constraint "
         f"{final['constraint_label']}{tail}")
    if at_target is not None and at_target > 1 and cfg["ranks"] > at_target:
        waste = 1.0 - final["efficiency"]
        emit("  ", f"you are past the knee by {cfg['ranks'] / at_target:.1f}x; "
                   f"{fmt_pct(waste)} of that capacity is rented and idle")
    print()
    print_failure(cfg, failure)
    print()
    print("a model, not a measurement. It is right about orderings, ratios and which")
    print("regime you are in, and silent about congestion, injection limits, topology,")
    print("algorithm switching inside a real collective library, and correlated failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
