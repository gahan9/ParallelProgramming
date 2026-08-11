"""Drill 1 solution - the scale-boundary model, with the knee found by bisection.

This is ../exercises/scale_boundary.py with every TODO filled in and one function
added: knee_by_bisection, which question (b) asks for. The commentary that matters is
in ../solutions/README.md; this file is the code it discusses.

Its table reproduces, exactly,
    python ../tools/scale_boundary.py --preset dp-training --ranks 4096
because the constants here are that preset. If your version disagrees with either,
one of the three has a bug -- find out which before trusting any of them.

Everything here is an ANALYTIC MODEL, never a measurement.

Run: python scale_boundary.py
"""

import math

BANNER = "analytic scale-boundary model, NOT a measurement: nothing here was run"

COMPUTE_MS = 60_000.0
SERIAL_FRACTION = 0.001
MODEL_BYTES = 2.0e9
ALPHA_US = 5.0
BANDWIDTH_GBS = 25.0
GPUS_PER_NODE = 8
NODE_MTBF_HOURS = 40_000.0
JOB_HOURS = 168.0
MAX_RANKS = 4096
EFFICIENCY_TARGET = 0.5

JITTER_SIGMA = 0.02
COMPUTE_BOUND_SHARE = 0.10
CONSTRAINTS = ("compute", "serial", "latency", "bandwidth", "variance", "failure")

# An exact tie at the target can land a few ulps on the wrong side of a float
# comparison, so the bisection compares with slack. Same reason as D01's drill 1.
TOL = 1e-12
SEARCH_CEILING = 1 << 40


def expected_max_factor(ranks):
    """Expected slowest of `ranks` step times, relative to the mean."""
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    if ranks == 1:
        return 1.0
    if ranks == 2:
        return 1.0 + JITTER_SIGMA / math.sqrt(math.pi)
    ln_n = math.log(ranks)
    root = math.sqrt(2.0 * ln_n)
    return 1.0 + JITTER_SIGMA * (root - (math.log(ln_n) + math.log(4.0 * math.pi))
                                 / (2.0 * root))


def step_terms(ranks):
    """The modelled terms of one step at `ranks`, in seconds."""
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    compute_total = COMPUTE_MS / 1e3
    serial = SERIAL_FRACTION * compute_total
    compute = (1.0 - SERIAL_FRACTION) * compute_total / ranks
    if ranks == 1:
        # Not an optimization: a collective over one rank is a no-op, and the formulas
        # below happen to agree only because (ranks - 1) is zero. Say it explicitly so
        # a later change to the algorithm cannot quietly charge a lone rank for a ring.
        latency = 0.0
        bandwidth = 0.0
    else:
        latency = 2.0 * (ranks - 1) * ALPHA_US * 1e-6
        bandwidth = (2.0 * MODEL_BYTES * (ranks - 1) / ranks
                     / (BANDWIDTH_GBS * 1e9))
    exposed = serial + compute + latency + bandwidth
    variance = (expected_max_factor(ranks) - 1.0) * exposed
    return {"serial": serial, "compute": compute, "latency": latency,
            "bandwidth": bandwidth, "variance": variance, "step": exposed + variance}


def parallel_efficiency(ranks):
    """Efficiency at `ranks` against a single rank: T(1) / (ranks * T(ranks))."""
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    if not 0.0 <= SERIAL_FRACTION <= 1.0:
        raise ValueError(f"serial fraction must be in [0, 1], got {SERIAL_FRACTION}")
    return step_terms(1)["step"] / (ranks * step_terms(ranks)["step"])


def binding_constraint(ranks):
    """The one constraint that binds at `ranks`, from the section 6.4 vocabulary."""
    if expected_interruptions(ranks) >= 1.0:
        return "failure"
    terms = step_terms(ranks)
    overheads = {name: terms[name]
                 for name in ("serial", "latency", "bandwidth", "variance")}
    if sum(overheads.values()) < COMPUTE_BOUND_SHARE * terms["step"]:
        return "compute"
    return max(overheads, key=lambda name: overheads[name])


def expected_interruptions(ranks):
    """Expected interruptions for a JOB_HOURS job holding `ranks` ranks."""
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    nodes = math.ceil(ranks / GPUS_PER_NODE)
    return JOB_HOURS * nodes / NODE_MTBF_HOURS


def knee_by_bisection(target, ceiling=SEARCH_CEILING):
    """Largest rank count whose efficiency still meets `target`. Question (b).

    Efficiency is strictly decreasing in rank count, which is the only property
    bisection needs -- and the only one that survives adding a term to the model.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target efficiency must be in (0, 1], got {target}")
    if parallel_efficiency(ceiling) >= target - TOL:
        return None
    low, high = 1, ceiling
    while high - low > 1:
        mid = (low + high) // 2
        if parallel_efficiency(mid) >= target - TOL:
            low = mid
        else:
            high = mid
    return low


def rank_ladder(max_ranks):
    ladder = []
    ranks = 1
    while ranks <= max_ranks:
        ladder.append(ranks)
        ranks *= 2
    if ladder[-1] != max_ranks:
        ladder.append(max_ranks)
    return ladder


def fmt_ms(seconds):
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


def main():
    print(BANNER)
    print(f"C = {COMPUTE_MS:,.0f} ms single-rank, s = {SERIAL_FRACTION:g}, "
          f"m = {MODEL_BYTES:,.0f} bytes")
    print(f"alpha = {ALPHA_US:g} us, B = {BANDWIDTH_GBS:g} GB/s, "
          f"{GPUS_PER_NODE} ranks per node, {JOB_HOURS:g} h job, "
          f"target {EFFICIENCY_TARGET * 100:g}%")
    print()

    header = (f"{'ranks':>7}{'compute':>10}{'latency':>10}{'wire':>10}"
              f"{'variance':>10}{'step':>10}{'eff':>8}{'interr':>8}  binds")
    print(header)
    print("-" * len(header))
    for ranks in rank_ladder(MAX_RANKS):
        terms = step_terms(ranks)
        print(f"{ranks:>7}{fmt_ms(terms['serial'] + terms['compute']):>10}"
              f"{fmt_ms(terms['latency']):>10}{fmt_ms(terms['bandwidth']):>10}"
              f"{fmt_ms(terms['variance']):>10}{fmt_ms(terms['step']):>10}"
              f"{parallel_efficiency(ranks) * 100:>7.1f}%"
              f"{expected_interruptions(ranks):>8.2f}"
              f"  {binding_constraint(ranks)}")
    print("-" * len(header))

    knee = knee_by_bisection(EFFICIENCY_TARGET)
    print(f"\nknee: efficiency holds at or above {EFFICIENCY_TARGET * 100:g}% up to "
          f"N = {knee}, and falls below it at N = {knee + 1}")
    print(f"at N = {MAX_RANKS}: efficiency {parallel_efficiency(MAX_RANKS) * 100:.1f}%, "
          f"binds on {binding_constraint(MAX_RANKS)}, "
          f"{expected_interruptions(MAX_RANKS):.2f} interruptions expected")

    print("\nrejection checks:")
    for ranks in (0, -8):
        try:
            parallel_efficiency(ranks)
        except ValueError as exc:
            print(f"  ranks={ranks:<4} -> ValueError: {exc}")
        else:
            print(f"  ranks={ranks:<4} -> NO ValueError, which is a bug")


if __name__ == "__main__":
    main()

# TODO 5 answers:
#
# a) At 4,096 ranks this workload is on the 10k-100k row's problem one row early: the
#    model reports 2.15 expected interruptions over the 168-hour job, so goodput binds
#    and the mitigation that belongs to it is asynchronous, sharded checkpointing with
#    fast detection and elastic restart -- C03's subject. The step-time constraint
#    underneath is collective bandwidth (160 ms of a 295 ms step), which is the second
#    problem, not the first. What buys nothing here: raising arithmetic intensity or
#    keeping tensor parallelism in-node (the 1-node row) does not touch a gradient
#    all-reduce; topology-aware placement and rail wiring (the 64-1k row) do not shrink
#    the roughly 2m bytes a ring must move; hierarchical collectives (the 1k-10k row)
#    attack a 41 ms latency term and leave the 160 ms wire term alone. Each of those is
#    the right answer to a different row, which is the whole point of the ladder.
#
# b) knee_by_bisection reports 243, and efficiency there is 0.5003 against 0.4993 at
#    244, so the bracket is tight. Bisection is right even though the compute-only
#    model has a closed form -- N <= (1/target - 1 + s)/s -- because the model this
#    drill actually builds has four terms, one of which contains sqrt(2*ln(N)) inside a
#    product, and there is no closed form left. Bisection needs only monotonicity, which
#    the model has by construction, so it survives every term you add later. Derive the
#    closed form anyway and use it to check the compute-only case; that is how you find
#    an inverted comparison.
#
# c) Both are true because they are statements about different quantities. The variance
#    FACTOR grows with rank count -- expected_max_factor goes 1.011 at 2 ranks to 1.072
#    at 16,384 -- exactly as D01 section 3.4 says. The variance COLUMN is that factor
#    applied to a step that is shrinking much faster, because compute divides by N. So
#    the absolute tax falls (340 ms at 2 ranks to 19 ms at 4,096) while its SHARE of the
#    step rises (1.1% to 6.5%). Share is what matters: the tax is a fixed percentage of
#    whatever the step costs, and no fabric purchase touches it. Note also what the
#    model does not have: a fat tail. One sick node is worth more than the whole
#    Gaussian bulk, and this model cannot see it.
#
# d) With BANDWIDTH_GBS = 50 the knee moves from 243 to 373 ranks -- 1.5x, not the 2x
#    you paid for -- and efficiency at 4,096 ranks goes from 5.0% to 7.0%. The binding
#    constraint at 4,096 does not move at all: it is still failure, because doubling the
#    fabric does not change the node count, the MTBF or the job length. That is
#    recurring lesson 4 as arithmetic: past the knee, the fabric is not what you are
#    short of, and the honest conclusion is to run at a few hundred ranks with the
#    fabric you have, or to cut bytes per step (sharding, gradient accumulation, lower
#    precision) rather than buy bandwidth.
#
# e) Two of many. (1) Congestion and injection limits: the model charges one rank's
#    bandwidth for a ring whose logical neighbours may be many hops apart on a shared,
#    oversubscribed fabric, so the real wire term can be several times this one. Bound
#    it by timing the collective in isolation and comparing the implied bus bandwidth
#    against the link rate, then repeating with the allocation deliberately spread
#    across leaves. (2) Fat-tailed stragglers: a single throttling or sick node is worth
#    more than the entire Gaussian jitter term. Bound it by logging per-rank arrival
#    timestamps at one barrier and looking at the histogram rather than the mean -- if
#    one node is 6% slow, that is an eviction, not an engineering project. Others worth
#    naming: algorithm switching inside a real collective library, checkpoint write time
#    and restart cost, correlated failures, and I/O.
