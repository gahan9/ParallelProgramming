# D04 Solutions - The Wire

Worked answers for the four drills in [`../exercises/`](../exercises/), with the reasoning
that matters. Read your own attempt first: the code here is short, and the commentary is
the part being assessed.

Every code block below is the body a `TODO` asked for, dropped into the starter file
unchanged. Every output block is that file's own output, captured by running it - nothing
here was typed by hand.

| Solution | Drill | The one thing to take away |
|---|---|---|
| [Drill 1](#drill-1-the-topology-classifier) | Topology classifier | A class you cannot name is one you cannot certify, so `unknown` must rank worst, not best |
| [Drill 2](#drill-2-staged-against-direct) | Staged vs direct cost | The crossover is a bandwidth ratio, not a message size - the size cancels |
| [Drill 3](#drill-3-the-injection-ceiling) | Injection ceiling | Demand scales with GPUs, supply with adapters; a single-node benchmark cannot see the gap |
| [Drill 4](#drill-4-the-peer-copy-harness) | Peer-copy harness | Consistency has to be relative, because this repository publishes no absolute number to compare against |

> [!IMPORTANT]
> Drills 1 to 3 were run to produce the output below. Drill 4's reference,
> [`04_peer_copy_harness.cpp`](04_peer_copy_harness.cpp), **has never been compiled or
> run**: there is no GPU toolchain on the machine these files were authored on. It is
> hand-reviewed only, and no bandwidth figure appears anywhere in it or in this page.

---

## Drill 1: the topology classifier

Full file: [`01_topology_classifier.py`](01_topology_classifier.py). The three TODOs:

```python
NVIDIA_CODES = {
    "X": SELF,
    "PIX": SWITCH,   # at most one PCIe bridge
    "PXB": SWITCH,   # several bridges, still no host bridge
    "PHB": HOST,     # through the host bridge
    "NODE": HOST,    # between host bridges inside one NUMA node
    "SYS": SOCKET,   # between NUMA nodes
}


def classify_nvidia_code(code):
    token = code.strip().upper()
    if NVLINK_RE.match(token):
        return DIRECT
    return NVIDIA_CODES.get(token, UNKNOWN)


def classify_amd_link(code, same_device, numa_src, numa_dst):
    token = code.strip().upper()
    if same_device and token in ("0", "X", "SELF", "-", "N/A"):
        return SELF
    if token.startswith("XGMI"):
        return DIRECT
    if token in ("PCIE", "PCI-E", "PCI"):
        if numa_src is not None and numa_dst is not None and numa_src != numa_dst:
            return SOCKET
        return HOST
    return UNKNOWN


def worst_class_in_group(pairs, group):
    if len(group) < 2:
        raise ValueError("a group needs at least two GPUs to have a pair in it, got %d"
                         % len(group))
    worst = SELF
    for src in group:
        for dst in group:
            if src == dst:
                continue
            klass = pairs.get((src, dst)) or UNKNOWN
            if SEVERITY[klass] > SEVERITY[worst]:
                worst = klass
    return worst
```

The table lookup is the design decision, not the dictionary contents. Anything absent from
it is `UNKNOWN` by construction, so adding a vendor code is an edit in one visible place
and *forgetting* to add one produces a refusal rather than a wrong answer. A chain of
`if`/`elif` ending in a plausible default has the opposite failure mode.

```text
NVIDIA capture
  parsed from ..\samples\nvidia_topo_synthetic.txt
          GPU0   GPU1   GPU2   GPU3 mlx5_0 mlx5_1
GPU0         .      D      S      S      W      S
GPU1         D      .      S      S      W      S
GPU2         S      S      .      D      S      W
GPU3         S      S      D      .      S      W

ROCm capture (NUMA nodes: GPU0->0, GPU1->0, GPU2->1, GPU3->1)
  parsed from ..\samples\rocm_showtopo_synthetic.txt
      GPU0 GPU1 GPU2 GPU3
GPU0     .    D    S    S
GPU1     D    .    S    S
GPU2     S    S    .    D
GPU3     S    S    D    .

NVIDIA capture carrying a code this curriculum has never seen
  parsed from ..\samples\nvidia_topo_unknown_code_synthetic.txt
      GPU0 GPU1 GPU2
GPU0     .    D    ?
GPU1     D    .    ?
GPU2     ?    ?    .

  codes not classified as a known tier: QQ9

worst class inside a proposed parallelism group:
  GPU0,GPU1                    -> direct-link
  GPU2,GPU3                    -> direct-link
  GPU0,GPU1,GPU2,GPU3          -> cross-socket

rejection checks (each must raise ValueError):
  single-GPU group -> ValueError: a group needs at least two GPUs to have a pair in it, got 1
  empty group      -> ValueError: a group needs at least two GPUs to have a pair in it, got 0

truncated capture, with (GPU2, GPU0) removed:
  GPU0,GPU2 -> unknown
```

The two captures describe the same invented node in two vendors' formats and produce the
same classes. That agreement is the deliverable: a launcher written against this
enumeration does not care which vendor's machine it lands on.

**a) What the collapses throw away.** `PIX` to `PXB` loses how many PCIe bridges the path
crosses; `PHB` to `NODE` loses whether the host bridge was crossed once or between two
bridges inside a NUMA node. Both are real differences in latency and contention. They are
safe to collapse *here* because this classifier answers one question - is this pair in the
scale-up tier, and if not, how far outside it - and both members of each pair answer it
the same way. The collapse would not be safe in a tool that predicts bandwidth. That is
the test: collapse only what the consumer of the answer cannot act on differently.

**b) Ranking `unknown` worst.** For: a wrong "fine" costs a job that runs a tier down for
its entire life with no error anywhere, while a wrong "refuse" costs one message to a
human - asymmetric costs justify an asymmetric default. Against: `unknown` will most often
mean "this tool is older than this machine", and refusing every new machine until someone
edits a table is a self-inflicted outage that also trains people to pass `--force`, which
destroys the check entirely. A batch scheduler should refuse and page a human, because it
places thousands of jobs and a silent tier drop multiplies. An interactive tool should
warn loudly, print the raw code, and continue - the human is right there, and what they
need is the code, not a verdict.

**c) The truncated capture.** Return `unknown`. A real matrix is symmetric and complete,
so a missing entry means the capture was truncated, not that the pair does not exist.
Skipping it would let the function certify a group from the pairs that happened to
survive - and in a hand-trimmed paste, the pair most likely to be missing is the ugly one
somebody was not proud of. Best-of-what-is-present is the failure mode where a tool is
confidently wrong, which is the one to design against.

**d) Distinguishing switch-local from host-bridge on the AMD side.** `lspci -tv` shows the
bridge tree, so you can see whether a GPU and an adapter hang off one switch or meet only
at the root complex; `lspci -vv` gives `LnkCap` against `LnkSta` per device; and
`/sys/class/infiniband/<dev>/device/numa_node` gives the adapter's socket. What to do with
it: pair each rank with the adapter on its own switch, bind its CPU and memory to that
NUMA node, and file any trained-down or narrow link as a hardware fault.
[Section 3.7](../README.md#37-pcie-the-hidden-bottleneck-in-the-middle) lists five
configuration bugs of exactly this kind, and every one is free to fix and invisible to a
correctness test.

---

## Drill 2: staged against direct

Full file: [`02_staging_cost.py`](02_staging_cost.py). The three TODOs:

```python
def staged_cost(nbytes, pcie_bps, net_bps):
    _check(nbytes, pcie_bps, net_bps)
    serial = 4.0 * nbytes / pcie_bps + nbytes / net_bps
    pipelined = max(2.0 * nbytes / pcie_bps, nbytes / net_bps)
    return (serial, pipelined)


def direct_cost(nbytes, pcie_bps, net_bps):
    _check(nbytes, pcie_bps, net_bps)
    serial = 2.0 * nbytes / pcie_bps + nbytes / net_bps
    pipelined = max(nbytes / pcie_bps, nbytes / net_bps)
    return (serial, pipelined)


def crossover_pcie_bps(net_bps):
    if net_bps <= 0:
        raise ValueError("network bandwidth must be positive, got %g B/s" % net_bps)
    return 2.0 * net_bps
```

Four PCIe traversals in the staged path, two in the direct one, one wire traversal in
both. The pipelined bound is a `max()` because once the segments overlap, the cost is
whichever single resource is busiest - and in the staged case one PCIe complex is carrying
`2m` while the wire carries `m`.

```text
check against README section 2.3: 1e+09 bytes, PCIe 20 GB/s, wire 10 GB/s
  staged serial  0.3000 s  (want 0.30 s)  OK
  direct serial  0.2000 s  (want 0.20 s)  OK

case                             staged ser   staged pipe   direct ser   direct pipe   pipe x
---------------------------------------------------------------------------------------------
PCIe 4x the wire                   0.2000 s      0.1000 s     0.1500 s      0.1000 s    1.00x
PCIe 2x the wire                   0.3000 s      0.1000 s     0.2000 s      0.1000 s    1.00x
PCIe equal to the wire             0.5000 s      0.2000 s     0.3000 s      0.1000 s    2.00x
PCIe narrower than the wire        0.9000 s      0.4000 s     0.5000 s      0.2000 s    2.00x
---------------------------------------------------------------------------------------------

crossover PCIe bandwidth, per network bandwidth:
  wire   5.0 GB/s -> 10.0 GB/s
  wire  10.0 GB/s -> 20.0 GB/s
  wire  25.0 GB/s -> 50.0 GB/s

does the message size move the crossover? (it should not)
         65536 bytes -> pipelined ratio 1.000
       1048576 bytes -> pipelined ratio 1.000
    1000000000 bytes -> pipelined ratio 1.000

rejection checks (each must raise ValueError):
  zero bytes       -> ValueError: message size must be positive, got 0 bytes
  negative PCIe    -> ValueError: PCIe bandwidth must be positive, got -1 B/s
  zero wire        -> ValueError: network bandwidth must be positive, got 0 B/s
```

**a) Which bound describes which message.** The serial bound describes the 4 KiB gradient
bucket; the pipelined bound describes the 1 GiB all-reduce. What decides is whether the
transfer is big enough to be chunked into pieces that keep every segment busy at once: a
4 KiB message has nothing to overlap with itself and pays each hop in turn, while a 1 GiB
message is chunked by the library and the hops run concurrently after the first chunk
lands. Neither bound is the truth in between - the real curve rises from the serial line
to the pipelined one as the message crosses a few times the library's chunk size, which is
a property of the library, not of the hardware.

**b) Why the size cancels.** Every term in both bounds is `m` over a bandwidth, so
`max(2m/P, m/N) == max(m/P, m/N)` divides through by `m` to `max(2/P, 1/N) ==
max(1/P, 1/N)`, and the size is gone. The crossover is `P = 2N` at any size. So "GPUDirect
matters for large messages" is true for a reason that is not in this model: the size
dependence lives in the *latency* term - the extra copies, completions and
synchronisations, all fixed per transfer - which is why small messages suffer most. The
bandwidth argument for the direct path is a statement about the PCIe-to-wire ratio, and it
holds at every size or none.

**c) A measured 1.05x where the model says 1.00x.** Two costs the model does not contain,
either of which can dominate the decision. First, latency: the fixed per-transfer cost of
the extra copies and their completions, which this bandwidth model has no term for at all.
Measure it with `ib_write_lat` between two nodes at the smallest supported size, once with
buffers in GPU memory and once in host memory, and compare the half-round-trip - that
isolates the fixed cost from any bandwidth effect. Second, CPU and host-memory cost: the
staged path spends host cores and roughly `4m` of host memory bandwidth that the rest of
the job also wants, which a microbenchmark measuring only the transfer will never show. In
both cases the number that decides is step time on the real job, not the transfer time in
isolation.

**d) Eight ranks staging at once.** Host memory bandwidth breaks first: eight ranks asking
for roughly `4m` each, against one socket's memory controllers that are already serving
everything else, with the pinned staging buffers competing for the same channels. The part
that notices first is the input pipeline - decode and augmentation are host-memory-hungry
and nothing is protecting them - so the symptom arrives as "the data loader is starving
the GPUs", which sends people to tune worker counts and prefetch depth. The collective is
protected by overlap and looks fine until the host is fully saturated.

---

## Drill 3: the injection ceiling

Full file: [`03_injection_ceiling.py`](03_injection_ceiling.py). The three TODOs:

```python
def per_gpu_injection(n_nic, nic_bps, n_gpu):
    if n_nic < 1:
        raise ValueError("a node needs at least one adapter to leave the node, got %d"
                         % n_nic)
    if n_gpu < 1:
        raise ValueError("GPU count must be at least 1, got %d" % n_gpu)
    if nic_bps <= 0:
        raise ValueError("adapter bandwidth must be positive, got %g B/s" % nic_bps)
    return n_nic * nic_bps / n_gpu


def scaleup_ratio(n_nic, nic_bps, n_gpu, scaleup_bps):
    if scaleup_bps <= 0:
        raise ValueError("scale-up bandwidth must be positive, got %g B/s" % scaleup_bps)
    return per_gpu_injection(n_nic, nic_bps, n_gpu) / scaleup_bps


def min_nics_for_ratio(target, nic_bps, n_gpu, scaleup_bps):
    if target <= 0:
        raise ValueError("target ratio must be positive, got %g" % target)
    ...
    return int(math.ceil(target * scaleup_bps * n_gpu / nic_bps))
```

Two decisions carry the drill. Zero adapters is a `ValueError`, not a zero: a node that
cannot leave the node is a configuration to fix, and `0.0` would read like a measured
share of nothing. And `min_nics_for_ratio` rounds **up**, because you cannot buy 3.2
adapters and an answer that rounds down does not meet the target it was asked for.

```text
per-GPU network share, GB/s
      GPUs      1 adapter(s)      2 adapter(s)      4 adapter(s)      8 adapter(s)
----------------------------------------------------------------------------------
         2            12.500            25.000            50.000           100.000
         4             6.250            12.500            25.000            50.000
         8             3.125             6.250            12.500            25.000
        16             1.562             3.125             6.250            12.500
----------------------------------------------------------------------------------

the same table as a ratio against the scale-up tier
      GPUs      1 adapter(s)      2 adapter(s)      4 adapter(s)      8 adapter(s)
----------------------------------------------------------------------------------
         2            0.0312            0.0625            0.1250            0.2500
         4            0.0156            0.0312            0.0625            0.1250
         8            0.0078            0.0156            0.0312            0.0625
        16            0.0039            0.0078            0.0156            0.0312
----------------------------------------------------------------------------------

adapters needed to hit a target ratio on an 8-GPU node:
  target 0.0625 -> 8 adapter(s)
  target 0.1250 -> 16 adapter(s)
  target 0.2500 -> 32 adapter(s)
  target 1.0000 -> 128 adapter(s)

three properties the functions must have (check them, do not assume them):
  linear in adapters:  doubling n_nic gives 2.000x  (want exactly 2)
  inverse in GPUs:     doubling n_gpu gives 0.500x  (want exactly 0.5)
  n_nic == n_gpu:      ratio is 0.0625, 0.0625, 0.0625 at 2, 8 and 64 GPUs
                       - constant, so at parity the GPU count drops out

rejection checks (each must raise ValueError, never divide by zero):
  zero adapters   -> ValueError: a node needs at least one adapter to leave the node, got 0
  zero GPUs       -> ValueError: GPU count must be at least 1, got 0
  zero adapter BW -> ValueError: adapter bandwidth must be positive, got 0 B/s
  zero scale-up   -> ValueError: scale-up bandwidth must be positive, got 0 B/s
  zero target     -> ValueError: target ratio must be positive, got 0
```

The adapter and scale-up figures are illustrative orders of magnitude, not a quote from
any datasheet. Only the ratio between them is load-bearing, and it is the one quantity
that has stayed roughly stable across generations while both absolute numbers moved.

**a) The missing limit.** The PCIe path between the GPU and the adapter, which this model
assumes is free. An adapter reaches line rate only on a link wide enough to carry it and,
for the direct path, on the same switch as the GPU it serves. Add adapters past the point
where each has its own switch-local lane budget and they start sharing bridges,
root-complex bandwidth and eventually the inter-socket link. The curve bends first at the
adapter whose only path to its GPU crosses a host bridge, and again, harder, above the
node once the fabric is oversubscribed - which is [D05](../../D05.topology-and-scale-boundaries/README.md)'s
subject.

**b) Parity.** It is the design goal because it is the only configuration where the
per-GPU network share stops depending on how many GPUs share the node, which makes a
node's off-node performance a property of the node rather than of its population. It costs
adapters, switch ports, lanes and power in proportion to GPU count. Note the ratio at
parity is still 0.0625, not 1: matching the scale-up tier is not the goal and is not
affordable; what parity buys is that the ratio stops degrading as nodes get denser.

**c) Eight GPUs with one adapter, or four with four.** 25 / 8 = **3.125 GB/s** per GPU
against 100 / 4 = **25 GB/s** per GPU - eight times more. The eight-GPU node wins on
anything that fits inside one node and communicates over the scale-up links: a model whose
parallelism is all intra-node, or inference where the batch is the unit of work. The
four-GPU node wins on anything that crosses the node boundary often - pipeline or data
parallelism spanning nodes, or a model too large for one node. The quantity you had to
know is the workload's off-node communication per unit of compute, its arithmetic
intensity measured against the network instead of against memory. Without that number the
two options are not comparable at all, which is why this decision cannot be made before
the sharding decision.

**d) Healthy single-node, quarter-speed two-node.** The model predicts it directly: inside
the node a flow uses the scale-up tier, and at 8 GPUs behind 1 adapter the off-node share
is 0.0078 of that tier, about 1:128. A single-node sweep never touches the adapter, so it
measures the tier that is not the constraint; the moment the job spans nodes, every rank
is funnelled into one adapter. The experiment that separates injection from fabric: run
the same two-node test with one rank pair, then two, then four, then eight, on the same
node pair, and watch the *aggregate*. If it is injection, the aggregate flattens at
`n_nic * B_nic` and per-pair bandwidth falls as `1/k`; if it were the fabric, one pair
alone would already be slow. Moving half the ranks to a third node is the second
confirmation: injection improves, fabric does not.

---

## Drill 4: the peer-copy harness

Full file: [`04_peer_copy_harness.cpp`](04_peer_copy_harness.cpp).

> [!WARNING]
> This file has never been compiled or run. There is no GPU toolchain on the machine it
> was written on, and no node with two GPUs to run it on. Read it as a reference for the
> structure and the decisions, not as tested code, and expect to fix something on the
> first build. No bandwidth figure appears in it, in its output template, or on this page.

The bookkeeping - sweeping ordered pairs, three sizes, `HIP_CHECK` everywhere, a
byte-for-byte check at every pair and size, warm-up inside the measurement function, one
`hipFree` per `hipMalloc` - is the part you can get right by being careful. Four decisions
are the part worth discussing.

**Peer access is enabled and disabled around each measurement, not once for the run.**
Leaving every direction mapped for the whole sweep makes the result depend on the order
pairs were visited: mapping resources are finite, so a late enable can fail on a large
node, and a measurement of `(0,1)` taken while `(0,2)` and `(0,3)` are also mapped is not
the same experiment as one taken alone. The cost is a pair of runtime calls plus path
setup per measurement, which is exactly what the warm-up iterations absorb. It costs run
time, not accuracy, and buys rows that are comparable with each other.

**The class comes from a file, and its absence is admitted rather than papered over.** The
harness takes an optional `src dst class` file produced from
[`../tools/topology_report.py`](../tools/topology_report.py). Without it, it falls back to
the only classification it can make for itself - peer-capable or not - and says so in its
header, because a check that quietly invents its reference is worse than one that admits
it has none.

**Iteration count scales with message size** so every size gets a comparable time budget.
One size cannot separate a latency problem from a bandwidth problem, and a fixed iteration
count makes the small sizes noise and the large ones slow.

**The consistency rule is relative, stated in two sentences, and its blind spot is
named:**

```c
// 1. A pair is flagged if it reaches less than kSlowFraction of the median of the
//    OTHER pairs sharing its class, which catches one bad link among siblings.
// 2. A direct-link pair is flagged if it fails to reach kTierFactor times the median
//    of the fastest non-direct class, which catches a whole tier being lost.
```

It has to be relative because this curriculum publishes no absolute figure to compare
against and your hardware's is not derivable from the class alone. Comparing pairs against
each other, in one run, on one machine, with one binary, removes almost everything that
would otherwise have to be assumed.

Both halves need a population, so both stay silent on a class with fewer than three
members - the harness says which classes it went quiet on rather than passing them in
silence. The blind spot that matters more: on a **uniformly slow** machine every pair
agrees with every other and this check reports nothing at all. Only an absolute
expectation catches that, and that is what
[lab 3](../README.md#4-hands-on-labs) and the vendor's own bandwidth test are for. A check
that cannot say what it does not cover is a check people will over-trust.

**a) Two numbers 8% apart.** Not a difference until you know the spread, and two numbers
have no spread - the honest answer is that the harness cannot tell. Repeat each
measurement at least five times, keep the individual timings, and report the median with
the min and max rather than a single number, so a reader can see whether 8% sits inside
the noise. If the spread is wide, the interesting output is not the bandwidth but why:
another tenant, a clock or power cap, or a warm-up that was too short.

**b) Enabling every pair for the whole run.** Covered above: finite mapping resources make
a late enable fail on a large node, and concurrent mappings make the first row and the
last row different experiments. The cost of enabling per measurement is run time absorbed
by the warm-up.

**c) Whose stream.** The stream should belong to the device performing the copy - for
`hipMemcpyPeerAsync` as written, the destination device, whose engine and ordering the
copy joins. The call is legal with the source device's stream, and what changes is which
engine performs the transfer and which stream's ordering it serialises against, so the
wrong choice can queue behind unrelated work on the wrong device. Test rather than argue:
run the sweep twice, once with the stream on each device, compare the matrices, then run
it again with a long unrelated kernel occupying the source device - the slowdown will
appear in exactly one of the two variants.

**d) One pair at half its siblings' bandwidth, reproducibly.** Three candidates from
[section 3.7](../README.md#37-pcie-the-hidden-bottleneck-in-the-middle): a link trained to
a lower generation than both ends support; fewer lanes than expected, from bifurcation or
an electrically narrower slot; a shared switch uplink this pair crosses and its siblings
do not. Run `lspci -vv` on the two devices first and read `LnkCap` against `LnkSta`: one
output separates the first two immediately, since a trained-down link and a narrow link
each show as a `LnkSta` that does not match `LnkCap`, in the speed field and the width
field respectively. If both match, the endpoint link is not the cause and `lspci -tv` for
the bridge tree is next.

---

Related: [the exercises](../exercises/README.md) | [module README](../README.md) |
[tools](../tools/) | [glossary](../../../../docs/GLOSSARY.md)
