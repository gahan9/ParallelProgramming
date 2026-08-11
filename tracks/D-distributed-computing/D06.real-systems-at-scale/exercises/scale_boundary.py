"""Drill 1 - place your own workload on Track D's scale-boundary ladder.

Every module in Track D derived one constraint. This drill assembles five of them into
one model and makes it name the row of the ladder (module README section 6.4) that a
given rank count lands on:

    serial fraction, collective latency, collective bandwidth, straggler variance,
    and component failure

The model, per step, for a fixed problem on N ranks:

    T(N) = s*C + (1-s)*C/N          compute: the serial part does not divide
         + 2*(N-1)*alpha            collective latency: the ring's step count
         + 2*m*(N-1)/N / B          collective bandwidth: ring traffic per rank
         + (E[max of N] - 1) * (everything above)   straggler variance

    efficiency(N) = T(1) / (N * T(N))

Your job:
  TODO 1  implement step_terms
  TODO 2  implement parallel_efficiency, rejecting inputs the model cannot mean
  TODO 3  implement binding_constraint, naming the row of the 6.4 ladder
  TODO 4  implement expected_interruptions
  TODO 5  read the table and answer the questions at the bottom of this file

The harness runs as it ships and prints "not implemented" until you fill the TODOs
in. Everything it computes is an ANALYTIC MODEL, never a measurement -- and this one
predicts the behaviour of tens of thousands of ranks, which nobody reading this can
check on the hardware in front of them. Say "model" out loud every time you quote it.

Worked solution with commentary: ../solutions/README.md and ../solutions/scale_boundary.py
Reference implementation with presets and --explain: ../tools/scale_boundary.py

Run: python scale_boundary.py
"""

import math

BANNER = "analytic scale-boundary model, NOT a measurement: nothing here was run"

# The workload under test. These are the tool's dp-training preset, so a correct
# implementation reproduces `python ../tools/scale_boundary.py --preset dp-training
# --ranks 4096` exactly, and a disagreement means one of the two has a bug.
COMPUTE_MS = 60_000.0      # single-rank compute time for one step of the FIXED problem
SERIAL_FRACTION = 0.001    # share of it that does not divide by rank count
MODEL_BYTES = 2.0e9        # bytes entering the per-step collective
ALPHA_US = 5.0             # per-collective-step latency
BANDWIDTH_GBS = 25.0       # achievable per-rank bandwidth, 1 GB = 1e9 bytes
GPUS_PER_NODE = 8          # ranks per failure domain
NODE_MTBF_HOURS = 40_000.0  # ASSUMED, not measured; see the module README section 6.2
JOB_HOURS = 168.0          # how long the job must run
MAX_RANKS = 4096
EFFICIENCY_TARGET = 0.5

# Per-rank step-time jitter, as a fraction of the step. D01 section 3.4's illustrative
# figure for unremarkable jitter -- clock throttling, ECC scrubbing, an OS timer.
JITTER_SIGMA = 0.02

# A step whose overheads are under this share is still paying for more ranks, so the
# honest label is compute rather than whichever overhead happens to be largest.
COMPUTE_BOUND_SHARE = 0.10

# The vocabulary of the ladder in module README section 6.4. Your binding_constraint
# must return one of these keys.
CONSTRAINTS = ("compute", "serial", "latency", "bandwidth", "variance", "failure")


def expected_max_factor(ranks):
    """GIVEN, not a TODO: expected slowest of `ranks` step times, relative to the mean.

    A collective is a barrier, so a step advances at the MAXIMUM over ranks, and the
    expected maximum of N samples grows with N even though the distribution never
    changes. That is why jitter is a scaling term (D01 section 3.4). N = 2 is exact;
    above it this is the standard extreme-value approximation to E[max].
    """
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
    """TODO 1: the modelled terms of one step at `ranks`, in SECONDS.

    Return a dict with these keys, or None while unimplemented:
        serial     s * C, and it does NOT divide by ranks
        compute    (1 - s) * C / ranks
        latency    2 * (ranks - 1) * alpha        (exactly 0.0 at ranks == 1)
        bandwidth  2 * m * (ranks - 1) / ranks / B   (exactly 0.0 at ranks == 1)
        variance   (expected_max_factor(ranks) - 1) * (the four terms above)
        step       the sum of all of them

    Watch the units: COMPUTE_MS is milliseconds, ALPHA_US microseconds, and
    BANDWIDTH_GBS is GB/s with 1 GB = 1e9 bytes. Return seconds throughout, so that
    one place in the file owns the conversion and the rest of it cannot get it wrong.

    Why the variance term multiplies everything else rather than only the compute:
    the collective's own duration varies too, from congestion and from adaptive
    routing's non-determinism (D05 section 6.2).
    """
    return None


def parallel_efficiency(ranks):
    """TODO 2: efficiency at `ranks` against a single rank: T(1) / (ranks * T(ranks)).

    Validate first, and raise ValueError with a ONE-LINE message that names the
    offending value:
      * ranks < 1 is not a machine
      * SERIAL_FRACTION outside [0, 1] is not a fraction
    A tool built on this has to turn the exception into a one-line error and a
    non-zero exit code, so the message must be readable without a traceback.

    Convince yourself the result is strictly decreasing in `ranks` before you go on:
    ranks * T(ranks) = ranks*serial + (1-s)*C + ranks*(latency + bandwidth + variance),
    and every term there is non-decreasing in ranks while at least one strictly
    increases. That monotonicity is what makes the knee bisectable in TODO 3's
    companion question.
    """
    return None


def binding_constraint(ranks):
    """TODO 3: name the ONE constraint that binds at `ranks`, from CONSTRAINTS.

    The rules, in order:
      1. If expected_interruptions(ranks) >= 1, return "failure". Goodput supersedes
         everything: if the job expects to be interrupted before it ends, no
         step-time fix delivers a finished run (README section 3.9).
      2. Otherwise consider only the OVERHEAD terms -- serial, latency, bandwidth,
         variance. Compute is excluded on purpose: compute dividing by N is what you
         are buying, not a problem.
      3. If those overheads together are under COMPUTE_BOUND_SHARE of the step,
         return "compute": ranks still pay and no overhead is worth attacking yet.
      4. Otherwise return the name of the largest overhead term.

    Return None while unimplemented.
    """
    return None


def expected_interruptions(ranks):
    """TODO 4: expected interruptions for a JOB_HOURS job holding `ranks` ranks.

    Nodes fail independently, so:
        nodes = ceil(ranks / GPUS_PER_NODE)          -- math.ceil
        expected = JOB_HOURS * nodes / NODE_MTBF_HOURS

    Return a float, or None while unimplemented. Note what the arithmetic says and
    the module's section 3.9 repeats: the count grows with BOTH rank count and job
    length, so better components move the threshold and cannot remove it.
    """
    return None


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
        efficiency = parallel_efficiency(ranks)
        interruptions = expected_interruptions(ranks)
        constraint = binding_constraint(ranks)
        if terms is None or efficiency is None:
            print(f"{ranks:>7}  not implemented (TODO 1, 2)")
            continue
        print(f"{ranks:>7}{fmt_ms(terms['serial'] + terms['compute']):>10}"
              f"{fmt_ms(terms['latency']):>10}{fmt_ms(terms['bandwidth']):>10}"
              f"{fmt_ms(terms['variance']):>10}{fmt_ms(terms['step']):>10}"
              f"{efficiency * 100:>7.1f}%"
              f"{'  n/a' if interruptions is None else f'{interruptions:>8.2f}'}"
              f"  {constraint or 'not implemented (TODO 3)'}")
    print("-" * len(header))

    print("\nrejection checks (each must raise ValueError, not return a number):")
    for ranks in (0, -8):
        try:
            outcome = parallel_efficiency(ranks)
        except ValueError as exc:
            outcome = f"ValueError: {exc}"
        else:
            outcome = "no ValueError raised" if outcome is None else f"returned {outcome}"
        print(f"  ranks={ranks:<4} -> {outcome}")

    print("\nTODO 5 - answer these in a comment at the bottom of this file:")
    print("  a) Which row of the module README section 6.4 ladder is this workload on")
    print("     at 4,096 ranks, and what is the ONE mitigation that belongs to that")
    print("     row? Name the mitigations from the rows above it that would buy")
    print("     nothing here, and say why.")
    print("  b) The knee is the largest rank count that still meets the target. Find")
    print("     it by BISECTION rather than by algebra, and say why bisection is the")
    print("     right tool even though the compute-only version has a closed form.")
    print("  c) The variance column FALLS as ranks grow, while D01 section 3.4 says")
    print("     variance gets worse with scale. Both are true. Explain.")
    print("  d) Double BANDWIDTH_GBS and re-run. Report what happens to the knee and")
    print("     to the binding constraint at 4,096 ranks, and say what that tells you")
    print("     about buying a faster fabric for this workload.")
    print("  e) Name two things this model does not know that would change the answer")
    print("     on a real machine, and the measurement that would bound each one.")


if __name__ == "__main__":
    main()

# TODO 5 answers:
# a)
# b)
# c)
# d)
# e)
