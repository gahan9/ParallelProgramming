# D04 Exercises - The Wire

Four drills, one per thing the module claims you should be able to do: read a topology
capture and turn it into a decision, price the host-staged path against the direct one,
find the injection ceiling a node was built with, and measure the fabric well enough to
catch it lying to you. They are the "Coding & Algorithms" drills from
[section 9](../README.md#9-self-assessment--interview-drills) of the module, turned into
starter files with TODOs.

The three Python drills **run as they ship**. Each prints its tables with
`not implemented` in every cell until you fill the TODOs in, so you always have a working
harness and never a blank page. Standard library only - no numpy, no torch, nothing to
install.

| Drill | File | What it tests |
|---|---|---|
| 1 | [`01_topology_classifier.py`](01_topology_classifier.py) | Normalize an NVIDIA and an AMD topology capture into one enumeration, then refuse a placement whose worst pair is not a direct link |
| 2 | [`02_staging_cost.py`](02_staging_cost.py) | Serial and pipelined bounds for the staged and direct paths, and the PCIe-to-wire ratio where they diverge |
| 3 | [`03_injection_ceiling.py`](03_injection_ceiling.py) | Per-GPU network share, the ratio against the scale-up tier, and the adapter count a target ratio costs |
| 4 | [`04_peer_copy_harness.md`](04_peer_copy_harness.md) | Extend lab 2 into a sweep over every ordered pair and several sizes, and flag any pair whose bandwidth contradicts its class |

Drill 4 is a specification rather than a starter file, because the thing being exercised
is a design decision - what "inconsistent with its class" means on a machine whose
absolute numbers you are not allowed to assume - and not an algorithm. It needs at least
two GPUs to run.

## How to work these

1. **Do the arithmetic by hand first.** Everything drills 2 and 3 need is derived in
   [section 2.3](../README.md#23-what-a-copy-costs-when-it-is-staged-through-the-host) and
   [section 3.6](../README.md#36-injection-bandwidth-why-one-adapter-for-eight-gpus-is-a-ceiling).
   Reaching for the tool before you have a number practises the wrong skill.
2. **Then check against the tools.** [`../tools/`](../tools/) implements the same closed
   forms and the same parser, so a disagreement means one of the two has a bug. Find out
   which before moving on.
3. **Answer the questions at the bottom of each file.** They are the part an interview
   actually asks. The code gets you the number; the question is what the number licenses
   you to say.

```bash
# From the module directory (one level up):
make run-exercises                      # runs all three Python harnesses
python exercises/01_topology_classifier.py
python tools/topology_report.py --help

# Or from here:
python 02_staging_cost.py
```

Drill 1 reads the format samples in [`../samples/`](../samples/).

> [!IMPORTANT]
> Those samples are **synthetic examples of the output format**, hand-written to exercise
> the parser. They are not a capture from any machine, they contain no hostname, user or
> serial number, and the topology they describe is invented. Every number they lead to is
> a property of the file, not of hardware. Run the same code on your own capture -
> [lab 1](../README.md#4-hands-on-labs) shows how to take one.

## Acceptance criteria

A drill is done when its harness prints these and you can defend the answers at the
bottom of the file. Every figure below is what the reference solution prints, so a
mismatch is a bug in one of the two.

**Drill 1 - the classifier.**

- On the NVIDIA sample, `GPU0`-`GPU1` and `GPU2`-`GPU3` come out `direct-link`, every
  pair that crosses the two islands comes out `cross-socket`, and each GPU reaches its
  own adapter through `pcie-switch` and the far one through `cross-socket`.
- The ROCm sample describes the same node and must produce the same classes, which is the
  whole point of normalizing two vendors onto one enumeration. `PCIE` between GPUs in
  different NUMA nodes is `cross-socket`; within one node it is `pcie-host`, because
  rocm-smi cannot tell a switch-local path from a host-bridge one and guessing the better
  of the two would be inventing a fact.
- The third sample carries the code `QQ9`, which is not a real nvidia-smi code. It must
  come out `unknown` and be listed as an unclassified code. A classifier that maps it to
  anything else has failed the only test in this drill that matters.
- `worst_class_in_group` returns `direct-link` for `GPU0,GPU1`, `cross-socket` for all
  four, and raises `ValueError` for a group with fewer than two GPUs.
- A pair missing from the capture is `unknown`, not skipped. Confirm with the truncated
  case the harness prints last.
- Cross-check the whole thing with
  `python ../tools/topology_report.py --input ../samples/nvidia_topo_synthetic.txt --group 0,1,2,3`.

**Drill 2 - staged against direct.**

- The reference case reproduces [section 2.3](../README.md#23-what-a-copy-costs-when-it-is-staged-through-the-host)
  exactly: 1e9 bytes at PCIe 20 GB/s and wire 10 GB/s gives **0.30 s** staged and
  **0.20 s** direct on the serial bound. If you get 0.322 and 0.215 you used a gibibyte;
  that 7% is the smallest version of the unit mistake
  [section 5.2](../README.md#52-how-to-read-a-bandwidth-test-result) warns about.
- The pipelined bound gives a ratio of **1.00x** when PCIe is 2x or 4x the wire, and
  **2.00x** when PCIe is equal to or narrower than the wire. Both bounds are correct and
  they disagree; question (a) is about which one your workload lives on.
- `crossover_pcie_bps` returns exactly **twice** the network bandwidth: 10, 20 and 50 GB/s
  for wires of 5, 10 and 25 GB/s.
- The pipelined ratio is **identical at 64 KiB, 1 MiB and 1 GB**. The crossover does not
  move with message size, and question (b) asks you to show why from the algebra.
- All three rejection cases raise `ValueError` with a one-line message naming the
  offending value.

**Drill 3 - the injection ceiling.**

- With the shipped illustrative figures, eight GPUs behind one adapter get **3.125 GB/s**
  each, a ratio of **0.0078** against the scale-up tier - about 1:128.
- Doubling adapters doubles the share exactly; doubling GPUs halves it exactly. At
  `n_nic == n_gpu` the ratio is **0.0625** at 2, 8 and 64 GPUs alike, which is the
  property that makes parity the design goal.
- Reaching a 0.25 ratio on an eight-GPU node needs **32 adapters** with these figures.
  The answer rounds **up**: 3.2 adapters is not a thing you can buy, and rounding down
  returns a count that misses the target.
- Zero adapters raises `ValueError`, never `ZeroDivisionError` and never `0.0`. A node
  with no adapter is a configuration to fix, not a share of nothing.
- Cross-check with `python ../tools/injection_model.py --gpus-per-node 8 --nics-per-node 1
  --nic-gbs 25 --scaleup-gbs 400`.

**Drill 4 - the harness.**

- Builds clean, checks every return code, verifies byte-for-byte at every pair and every
  size, and frees everything it allocates.
- Sweeps **ordered** pairs, both directions. A harness that measures only `i < j` reports
  the better direction and hides the worse one.
- States its consistency rule in a comment in two sentences, in your own words, with its
  blind spot named. The reference's blind spot is a uniformly slow machine, where no pair
  contradicts any other and the check says nothing at all.
- Exits with a one-line message on a single-GPU machine, not a crash and not a table of
  zeroes.

## Shared assumptions unless a drill says otherwise

| Quantity | Value | Why this one |
|---|---|---|
| Link classes | `self`, `direct-link`, `pcie-switch`, `pcie-host`, `cross-socket`, `unknown` | one enumeration for both vendors, per module [section 5.1](../README.md#51-how-to-read-a-topology-matrix) |
| Unknown handling | reported, never guessed, and ranked worst | a class you cannot name is one you cannot certify |
| Bandwidth unit | GB/s with 1 GB = 1e9 bytes | the convention link vendors quote; see [bandwidth](../../../../docs/GLOSSARY.md#bandwidth) |
| Staged path | 4 PCIe traversals, 4m of host traffic, 1 wire traversal | module [section 2.3](../README.md#23-what-a-copy-costs-when-it-is-staged-through-the-host) |
| Direct path | 2 PCIe traversals, no host traffic | [RDMA and GPUDirect](../../../../docs/GLOSSARY.md#rdma-and-gpudirect) |
| Adapter, scale-up figures | 25 GB/s and 400 GB/s per GPU | illustrative orders of magnitude, not a datasheet quote - only the ratio is load-bearing |

Drills 2 and 3 compute an **analytic model, never a measurement**, and both harnesses say
so in their banner. Drill 1 reports what a file says and names the file. Drill 4 is the
only one that measures anything, and what it measures is your machine.

Reference answers with commentary: [`../solutions/`](../solutions/). Read them *after* you
have written yours down - they contain the reasoning you are being assessed on, not just
the code.

Related: [module README](../README.md) | [glossary](../../../../docs/GLOSSARY.md) |
[references](../../../../docs/REFERENCES.md)
