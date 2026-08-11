# D01 Solutions — Scaling Limits & the Cost of Talking

Worked answers for the four drills in [`../exercises/`](../exercises/), with the reasoning
that matters. Read your own attempt first: the code here is short, and the commentary is
the part being assessed.

Every block below is the body a `TODO` asked for, dropped into the starter file unchanged.
Every table is that file's own output — nothing here was typed by hand, and nothing was
measured on a cluster.

| Solution | Drill | The one thing to take away |
|---|---|---|
| [Drill 1](#drill-1--efficiency-and-the-wall) | Efficiency and bisection | A fixed communication cost is algebraically indistinguishable from a bigger serial fraction |
| [Drill 2](#drill-2--the-crossover) | Crossover finder | `m*` is a *product*, so a slow link can have a smaller crossover than a fast one |
| [Drill 3](#drill-3--algorithm-selection) | Ring vs recursive doubling | The tie point moves with rank count, which is why it cannot be a constant |
| [Drill 4](#drill-4--variance-amplification) | Straggler simulation | The barrier waits on a maximum, and maxima grow with sample size |

---

## Drill 1 — efficiency and the wall

```python
def amdahl_speedup(serial_fraction, ranks):
    """Fixed-problem speedup: the serial part does not divide, the parallel part does."""
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / ranks)


def parallel_efficiency(serial_fraction, ranks):
    """Return (speedup, efficiency). Efficiency is the share of the machine doing work."""
    if not 0.0 <= serial_fraction <= 1.0:
        raise ValueError(f"serial fraction must be in [0, 1], got {serial_fraction}")
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    speedup = amdahl_speedup(serial_fraction, ranks)
    return speedup, speedup / ranks


def largest_ranks_at_efficiency(serial_fraction, target, ceiling=1 << 40):
    """Largest rank count whose efficiency is still at or above `target`, by bisection."""
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target efficiency must be in (0, 1], got {target}")
    if parallel_efficiency(serial_fraction, ceiling)[1] >= target - TOL:
        return None
    low, high = 1, ceiling
    while high - low > 1:
        mid = (low + high) // 2
        if parallel_efficiency(serial_fraction, mid)[1] >= target - TOL:
            low = mid
        else:
            high = mid
    return low
```

```
   ranks     speedup   efficiency
---------------------------------
       1       1.00x       100.0%
       8       7.02x        87.7%
      64      28.32x        44.2%
     512      45.63x         8.9%
    1024      47.72x         4.7%
---------------------------------
largest rank count holding 50% efficiency: 51 (closed form agrees)
```

**Why validate in `parallel_efficiency` and not in `amdahl_speedup`?** Because one of them
is a formula and the other is an interface. The tool built on this has to turn a bad input
into a one-line error and an exit code, so the exception has to carry a message a user can
act on without reading the traceback — which is why the message names the offending value.
Validating in both would duplicate the check; validating in neither is how a `nan`
reaches a table and gets quoted in a review.

**Why bisection when a closed form exists?** The closed form
`N <= (1/target - 1 + s)/s` is two lines and exact, and you should derive it as a check.
Bisection earns its place because it survives the next question. Add a fixed per-step
communication cost `c`, and efficiency becomes `1/(N*s + p + N*c)` — still monotonic,
still bisectable, and no longer something you can rearrange in one step. The habit of
solving the monotone problem rather than the algebraic one is what transfers.

**Why the tolerance?** N=51 is an *exact* tie at `s = 0.02` against a 50% target, so `>=`
handles it and `>` does not. But exactness is luck. At `s = 0.001` against a 40% target the
closed form says 1501, and the efficiency there evaluates to `0.39999999999999997`, so a
bare `>=` reports 1500 and disagrees with the algebra by one rank. Comparing a
floating-point quantity against a round decimal target needs slack; a scan over 1,400
(s, target) pairs finds 31 such ties, so this is not a corner case you can hope to miss.

**The questions.**

- **(a)** At `s = 0.02` and 1,024 ranks the efficiency is **4.7%**. The honest answer to a
  request for 1,024 ranks is: the ceiling is `1/s = 50x`, you already have 47.7x, and 64
  ranks would give you 28.3x at 44% efficiency. Ranks 64 through 1,024 buy 1.7x more speed
  for 16x the machine. If 50x is not enough, no rank count fixes it — the serial section is
  the only thing left to attack.
- **(b)** `s = 0.001`, target 0.4, N = 1501: see above. Trust the closed form here, and
  make the bisection agree with it by comparing with a tolerance. If the two ever disagree
  by more than one rank, the bug is in the bisection's invariant, not in floating point.
- **(c)** Add a fixed per-step communication cost `c` and the efficiency becomes
  `1/(N*(s + c) + p)`. The `N*c` term behaves exactly like the `N*s` term, so **`c` acts as
  extra serial fraction**: the wall moves **in**, to a smaller rank count, and it moves in
  proportionally. That is why section 2.4's worked example collapses at 64 ranks even
  though its serial fraction is zero — Amdahl and communication are the same shape of
  problem, and the second one is usually the larger term.

---

## Drill 2 — the crossover

```python
def crossover_bytes(alpha_s, beta_s_per_byte):
    """Message size at which the latency and transfer terms are equal: m* = alpha/beta."""
    if beta_s_per_byte <= 0.0:
        raise ValueError(f"beta must be positive, got {beta_s_per_byte}")
    if alpha_s < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha_s}")
    return alpha_s / beta_s_per_byte


def classify(nbytes, alpha_s, beta_s_per_byte):
    """Name the regime, which decides which optimization can possibly help."""
    if nbytes < 0.0:
        raise ValueError(f"message size must be non-negative, got {nbytes}")
    star = crossover_bytes(alpha_s, beta_s_per_byte)
    if nbytes < star / BALANCED_BAND:
        return "latency-bound"
    if nbytes > star * BALANCED_BAND:
        return "bandwidth-bound"
    return "balanced"
```

```
link                        alpha us    GB/s    m* bytes          m*
--------------------------------------------------------------------
intra-node GPU link              1.5  200.00      300000   293.0 KiB
inter-node RDMA                  2.5   25.00       62500    61.0 KiB
datacenter Ethernet, TCP       100.0   10.00     1000000   976.6 KiB
--------------------------------------------------------------------

        size       intra-node GPU link           inter-node RDMA  datacenter Ethernet, TCP
------------------------------------------------------------------------------------------
     4.0 KiB             latency-bound             latency-bound             latency-bound
    64.0 KiB             latency-bound                  balanced             latency-bound
     1.0 MiB           bandwidth-bound           bandwidth-bound                  balanced
    64.0 MiB           bandwidth-bound           bandwidth-bound           bandwidth-bound
```

**Why a "balanced" band rather than a single threshold?** Because a threshold implies a
confidence the model has not earned. Within a factor of two of `m*` neither term dominates,
so a prediction of which algorithm wins there is inside the model's own error bars —
contention, protocol switches and injection limits are all larger effects than the gap you
are resolving. Naming the band is how the tool declines to over-claim, and it is also
where a real library's tuned thresholds sit, for the same reason.

**Why does `classify` recompute the crossover instead of taking it as an argument?**
Because `m*` is derived from the link and the caller has no business supplying it
separately. Passing both invites the bug where the size and the crossover come from
different links — which is exactly the mistake the drill is about, expressed as an API.

**The questions.**

- **(a)** All three links call 4 KiB latency-bound, which is the general result: 4 KiB sits
  below `m*` on every fabric ever built. The one fix that helps everywhere is **fewer,
  bigger messages** — aggregation, gradient bucketing, batching. Buying bandwidth changes
  nothing below `m*`.
- **(b)** `m* = alpha * bandwidth` is a property of the *link*, and these three differ by
  16x. "Fuse messages under 1 MB" is correct on the Ethernet link, wrong by 3x on the
  intra-node link, and wrong by 16x on the RDMA fabric. It is not a tuning value, it is a
  value that must be *computed* from the fabric you are on, which is what a collective
  library does at initialization.
- **(c)** **`alpha`.** The Ethernet link has 20x less bandwidth than the GPU link but 67x
  more latency, and since `m*` is the product, latency wins the argument. The lesson for
  procurement: a vendor number in GB/s alone cannot tell you where your messages land.
  Ask for both, and ask what message size the bandwidth figure was measured at.
- **(d)** Compression below `m*` saves the *byte* term, and the byte term is what is
  already nearly free. A 4 KiB message on the RDMA link costs 2.664 us: 2.500 us of latency
  and 0.164 us of wire. Perfect, free compression would save **6%** of it — and it is
  neither perfect nor free, since it costs cycles on both ends and usually accuracy. Below
  `m*`, compress nothing and merge instead. Above `m*` the same arithmetic inverts and
  compression becomes one of the strongest levers available.

---

## Drill 3 — algorithm selection

```python
def validate(ranks, nbytes):
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    if nbytes < 0.0:
        raise ValueError(f"message size must be non-negative, got {nbytes}")


def recursive_doubling_cost(ranks, nbytes, alpha_s, beta_s_per_byte):
    """ceil(log2(N)) rounds, each exchanging the whole buffer with one partner."""
    validate(ranks, nbytes)
    if ranks == 1:
        return 0.0
    steps = (ranks - 1).bit_length()
    return steps * alpha_s + steps * nbytes * beta_s_per_byte


def ring_cost(ranks, nbytes, alpha_s, beta_s_per_byte):
    """Reduce-scatter then all-gather: 2(N-1) steps, 2m(N-1)/N bytes per rank."""
    validate(ranks, nbytes)
    if ranks == 1:
        return 0.0
    steps = 2 * (ranks - 1)
    volume = 2.0 * nbytes * (ranks - 1) / ranks
    return steps * alpha_s + volume * beta_s_per_byte


def choose(ranks, nbytes, alpha_s, beta_s_per_byte):
    """Return (name, seconds) for the cheaper algorithm, ties going to fewer steps."""
    doubling = recursive_doubling_cost(ranks, nbytes, alpha_s, beta_s_per_byte)
    ring = ring_cost(ranks, nbytes, alpha_s, beta_s_per_byte)
    if ring < doubling:
        return "ring", ring
    return "recursive doubling", doubling


def tie_bytes(ranks, alpha_s, beta_s_per_byte, high=1 << 40):
    """Message size at which the two algorithms cost the same, by bisection."""
    validate(ranks, 0.0)
    if ranks <= 2:
        return None

    def gap(nbytes):
        return (ring_cost(ranks, nbytes, alpha_s, beta_s_per_byte)
                - recursive_doubling_cost(ranks, nbytes, alpha_s, beta_s_per_byte))

    if gap(0.0) <= 0.0 or gap(float(high)) >= 0.0:
        return None
    low, high = 0.0, float(high)
    while high - low > 1.0:
        mid = (low + high) / 2.0
        if gap(mid) > 0.0:
            low = mid
        else:
            high = mid
    return low
```

```
winner per cell, and how much cheaper it is than the loser:
      size               N=1               N=2               N=8              N=64             N=512
----------------------------------------------------------------------------------------------------
   1.0 KiB  doubling      --  doubling    2.0x  doubling    4.6x  doubling   20.7x  doubling  111.7x
  64.0 KiB  doubling      --  doubling    1.5x  doubling    2.6x  doubling   10.4x  doubling   55.5x
   1.0 MiB  doubling      --  doubling    1.1x      ring    1.2x  doubling    1.5x  doubling    6.6x
  16.0 MiB  doubling      --  doubling    1.0x      ring    1.7x      ring    2.5x      ring    1.6x
  1.00 GiB  doubling      --  doubling    1.0x      ring    1.7x      ring    3.0x      ring    4.4x
----------------------------------------------------------------------------------------------------

tie point, and how it moves with rank count:
  N=8     tie at       549999 bytes (537.1 KiB)
  N=64    tie at      1860465 bytes (1.8 MiB)
  N=512   tie at      9039598 bytes (8.6 MiB)
```

**Why `(ranks - 1).bit_length()` and not `math.ceil(math.log2(ranks))`?** Because the float
version is a landmine at exact powers of two, which is precisely where every real rank
count sits. `log2(2**k)` is not guaranteed to be exactly `k` once the value has been
through a division, and one spurious step is a whole `alpha` of error on the algorithm that
is supposed to be winning on steps. `bit_length` is integer arithmetic and cannot round.

**Why does `ranks == 1` have to return exactly zero?** Because the formulas do not. At
`N = 1`, `2(N-1) = 0` steps is right, but `(ranks - 1).bit_length()` is also 0, and a
different-looking formula might yield a *negative* time — which would then win every
comparison silently, and the tool would confidently recommend an algorithm for a collective
that does not exist. An all-reduce over one rank is a no-op; say so explicitly rather than
hoping the algebra agrees.

**Why bisect the tie instead of solving it?** Solve it too — the closed form is
`m = alpha * (2(N-1) - log2 N) / (beta * (log2 N - 2(N-1)/N))`, which is what
`tools/collective_choice.py` prints, and the two agree to within the bisection's
one-byte bracket (549999 against 550000, 1860465 against 1860465). The bisection is worth
writing because the *reason* it works is the insight: `ring_cost - doubling_cost` is
monotonic in message size, since the ring starts behind on latency and gains on every byte.
Once you have seen that, adding a third algorithm to the comparison is a loop, not new
algebra.

**The questions.**

- **(a)** Down the `N=64` column, recursive doubling wins at 1 KiB (20.7x), 64 KiB (10.4x)
  and 1 MiB (1.5x); the ring wins at 16 MiB (2.5x) and 1 GiB (3.0x). The flip is at the
  tie, 1,860,465 bytes. The **byte term** caused it: the ring moves about `2m` regardless
  of rank count while recursive doubling moves `m * log2(64) = 6m`, so at large `m` the
  ring's 126 steps of latency are cheap next to the 3x extra traffic.
- **(b)** The tie **grows** with rank count: 537 KiB at 8 ranks, 1.8 MiB at 64, 8.6 MiB at
  512. The ring's latency penalty grows linearly in `N` while its byte advantage saturates
  at `2m`, so it needs a bigger message to amortize each extra pair of steps. A threshold
  baked into a build is therefore wrong at every rank count except the one it was tuned on —
  which is the entire argument for a runtime decision tree, documented for MPICH by
  [Thakur, Rabenseifner and Gropp](../../../../docs/REFERENCES.md#collective-algorithms-in-mpich-thakur-rabenseifner-and-gropp-2005).
- **(c)** At `N = 2` both algorithms move exactly `m` bytes — one exchange with one partner
  — but the ring spends two steps doing it. Equal bytes and strictly more latency means
  recursive doubling wins at every size, so there is nothing for a crossover to cross.
- **(d)** Any oversubscribed or heterogeneous fabric: a ring's logical neighbours are not
  necessarily physically adjacent, so its `2(N-1)` hops can cross contended uplinks and see
  far less than the quoted `beta`. On a heterogeneous fabric the *slowest* link sets the
  pace of the whole ring. To catch it, time the collective in isolation and compare the
  implied bus bandwidth against `2m/busbw` and against the link rate; then time a zero-byte
  collective to separate the latency term from the byte term.

---

## Drill 4 — variance amplification

```python
def barrier_step_time(step_times):
    """Wall-clock time of one barrier-synchronized step: the maximum, not the mean."""
    if not step_times:
        raise ValueError("a step needs at least one rank time")
    return max(step_times)


def waiting_fraction(step_times):
    """Share of aggregate rank-time burned waiting for the last arrival."""
    if not step_times:
        raise ValueError("a step needs at least one rank time")
    slowest = max(step_times)
    if slowest <= 0.0:
        return 0.0
    return 1.0 - (sum(step_times) / len(step_times)) / slowest


def sample_step_times(ranks, sigma, rng):
    """One step's per-rank times, drawn from Normal(MEAN_STEP, sigma)."""
    if ranks < 1:
        raise ValueError(f"ranks must be at least 1, got {ranks}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    return [rng.gauss(MEAN_STEP, sigma) for _ in range(ranks)]
```

```
per-rank step time ~ Normal(mean 1, sd 0.02), 100 steps per rank count, seed 20260811

   ranks   slowest rank   throughput lost   rank-time idle
----------------------------------------------------------
       8         1.030x              2.9%             2.9%
      64         1.049x              4.6%             4.6%
    1024         1.066x              6.2%             6.2%
   16384         1.080x              7.4%             7.4%
----------------------------------------------------------
```

**Why is an empty sequence an error rather than zero?** Because a step with no ranks has no
duration, and `max([])` raising while `sum([])/0` divides by zero means the two functions
would fail differently on the same bad input. Returning `0.0` would be worse than either:
it reports a *perfect* step, and a monitoring dashboard fed that value shows a healthy
cluster where there is no cluster at all.

**Why seed the generator?** Because this table is cited by the module README, and the repo
does not publish a number it cannot reproduce. An unseeded simulation prints slightly
different figures every run, so it can illustrate but never support. Seeding converts the
output from an anecdote into an artifact.

**Why does the mean not appear in the step time at all?** That is the whole lesson. A
collective is a barrier, so the step advances at the speed of the last arrival — the
**maximum** of N samples, not the mean. The mean is the number a single-node benchmark
reports, which is why jitter is invisible until you scale.

**The questions.**

- **(a)** Same dice, more rolls. The distribution never changes, but each step waits for the
  worst of N draws, and the more draws you take the deeper into the tail the worst one
  lands. At 8 ranks you are asking "how slow is the slowest of 8?"; at 16,384 you are asking
  the same question of 16,384 — and the answer is necessarily larger.
- **(b)** The loss grows, but with sharply diminishing returns: 2.9% to 6.2% from 8 to 1,024
  ranks (a factor of 128 in scale), then only 6.2% to 7.4% from 1,024 to 16,384 (a factor of
  16). It is **unbounded but slow**, tracking the Gaussian tail, so the expected maximum
  grows roughly as `sqrt(2 * ln N)`. You cannot outrun it and you also cannot be ruined by
  it — what ruins a run is a *fat* tail, one sick node, not the Gaussian bulk.
- **(c)** Two levers, both structural. Reduce **sigma**: pin clock frequencies, isolate
  cores from OS work, keep noisy neighbours off the node, and drain nodes whose step times
  drift. Reduce **N per barrier**: hierarchical collectives reduce within a node first, so
  the global barrier waits on a maximum over nodes instead of over ranks. An operator can
  apply the first today, through health-gated scheduling, which is why large clusters run
  health checks before every job.
- **(d)** At `SIGMA = 0.10` the same simulation reports a slowest rank of **1.329x** at
  1,024 ranks and **24.7%** of throughput lost — from a single distribution-wide change
  that a per-node dashboard would call a 10% variation. Yes, drain the node: a quarter of a
  1,024-GPU cluster is a far larger prize than any kernel optimization in Track B, and the
  fix is an eviction, not an engineering project.

---

## Grading yourself

You have the right answer if you can defend all four of these:

1. **Every rejected input has a one-line message that names the value.** A tool built on
   these functions must be able to print it and exit non-zero. If a user sees a traceback,
   the function was written for the author, not the caller.
2. **Every edge case is deliberate, not accidental.** `ranks == 1`, an empty rank list, a
   serial fraction of exactly 0 or 1, an exact tie at the target: state what each one
   returns and why that is the honest answer.
3. **You named the regime before you named the fix.** "Latency-bound at this size on this
   fabric, therefore merge messages" is an answer. "Use a ring" is a guess that happens to
   be right about a third of the time.
4. **You said what the model does not know.** These four drills know nothing about
   congestion, injection rate, topology, protocol switches, fat-tailed stragglers or the
   several other algorithms in a real collective library. A Principal-Engineer-grade answer
   names at least one and says what it would measure to bound it.

## Three results worth remembering past this module

**1. A fixed communication cost *is* a serial fraction.** Under Amdahl with an added
per-step cost `c`, efficiency is `1/(N*(s + c) + p)`, so `c` enters exactly where `s` does.
This is why a perfectly parallel code still has a scaling wall, and why "our code has no
serial section" is not a defence.

**2. `m*` is a product, so intuition about "fast" and "slow" links fails.** The TCP-over-
Ethernet link in drill 2 has the *largest* crossover of the three, because its 100 us of
latency outweighs its missing bandwidth — so *more* of its messages are latency-bound, not
fewer. "Slow link, therefore bandwidth-bound" is a guess the arithmetic contradicts, and it
cuts the other way too: drop the bandwidth far enough, as the 1 GbE-class link in the
module's Lab 2 does, and `m*` falls to a few kilobytes.

**3. The tie between two collective algorithms moves with rank count, monotonically.**
537 KiB at 8 ranks becomes 8.6 MiB at 512. Any constant threshold is a bet on one cluster
size, and it is the reason production libraries decide per call rather than per build.

## Verifying the arithmetic

```powershell
# from the module directory
make run-exercises                                     # all four harnesses
python tools/scaling_model.py --serial-fraction 0.02 --ranks 1024
python tools/message_cost.py --alpha-us 2.5 --bandwidth-gbs 25 --bytes 4096
python tools/collective_choice.py --ranks 64 --bytes 4096
```

Remember what these are: **closed-form analytic models, not measurements**, each saying so
in its own banner. They are right about ratios, orderings and orders of magnitude, and
silent about congestion, injection limits, topology, and everything else that decides
whether a real job finishes.
