# Exercise 4 - peer-copy bandwidth harness, cross-checked against the topology

This drill is a specification rather than a starter file, because the thing being
exercised is a design decision and not an algorithm. You extend [Lab 2's
program](../hip/peer_copy_bw.cpp) from one ordered pair at one size into a sweep over
every ordered pair at several sizes, and then you use the result to catch the failure
[section 3.5](../README.md#35-gpudirect-rdma-and-peer-to-peer-dma-taking-the-host-off-the-path)
describes: a copy that succeeds, returns `hipSuccess`, and quietly staged through host
memory.

Write it as `04_peer_copy_harness.cpp` in this directory. A worked reference is in
[`../solutions/04_peer_copy_harness.cpp`](../solutions/04_peer_copy_harness.cpp) - read
it after you have your own version failing, not before.

> [!IMPORTANT]
> Neither your program nor the reference solution has been compiled or run by this
> repository. There is no GPU toolchain on the machine these files were authored on, and
> the reference is hand-reviewed only. Treat the first build as part of the exercise, and
> treat any number it prints as yours, not as one this curriculum has published.

## What it must do

1. **Sweep every ordered pair.** For `n` devices that is `n * (n - 1)` measurements, and
   both directions matter: a pair can be asymmetric in practice, and a harness that
   measures only `i < j` will report the better direction and hide the worse one.
2. **Sweep several message sizes.** At least 1 MiB, 16 MiB and 256 MiB. One size cannot
   separate a latency problem from a bandwidth problem, which is the whole content of
   [D02's alpha-beta model](../../D02.message-passing-and-mpi/README.md).
3. **Report `hipDeviceCanAccessPeer` / `cudaDeviceCanAccessPeer` for the pair**, next to
   the measured bandwidth. This is the flag whose value the measurement is about to
   corroborate or contradict.
4. **Keep every return code checked** - `HIP_CHECK` on every runtime call, no exceptions
   for the ones that "cannot fail".
5. **Keep the byte-for-byte verification** at every pair and every size. A fast wrong
   answer is worse than a slow right one, and a peer copy that silently transferred
   nothing is fast.
6. **Free every allocation**, disable any peer access you enabled, and destroy every
   stream and event. A sweep allocates `n` times more than Lab 2 does and runs for `n`
   times longer, so a leak that Lab 2 could get away with will end this program.
7. **Warm up outside the timed region** at every pair and size, not once at the start.
   The first copy on a newly enabled peer path pays setup that the rest do not.
8. **Cross-check against the topology classes** from
   [exercise 1](01_topology_classifier.py) and flag any pair whose measurement disagrees
   with its class. Read the class from a file the classifier wrote, or re-derive it -
   your choice, but say in a comment which you chose and why.

## The cross-check is the point

Everything above item 8 is bookkeeping. Item 8 is the drill.

A pair reported as `direct-link` whose measured bandwidth sits at the PCIe tier is the
signature of [section
3.5](../README.md#35-gpudirect-rdma-and-peer-to-peer-dma-taking-the-host-off-the-path):
the fast path was available and was not taken. A pair reported as `pcie-host` running at
the direct-link tier means your classifier is wrong, or the capture and the machine are
not the same machine. Both are worth an alert; neither should be a crash.

Deciding what counts as "inconsistent" is the hard part, and there is no clean
threshold hiding in the problem:

- You cannot compare against an absolute number, because this repository does not
  publish one and your hardware's is not knowable from the class alone.
- You *can* compare pairs against each other on the same machine, in the same run,
  with the same code. Group the measurements by class and compare the groups.
- Decide what to do when a class has only one pair in it, or none. A rule that needs a
  population is a rule that is silent on a two-GPU node.
- Decide what your flag means when *every* pair is slow. A uniformly slow machine has
  no inconsistency in it at all, and this check will say nothing - name that blind spot
  in a comment rather than pretending the check covers it.

Write the rule down as a comment before you implement it. If you cannot state it in two
sentences, it is not a rule, it is a threshold you picked.

## Suggested output shape

Something a human can scan and a script can parse. One row per ordered pair per size,
with the class and the flag on the row, and the alerts collected at the end:

```text
src dst      size    GB/s  peer  class          note
  0   1   1048576   xx.xx   yes  direct-link
  0   1  16777216   xx.xx   yes  direct-link
  ...
  0   2   1048576   xx.xx    no  cross-socket
  ...
alerts:
  (0,1) reported direct-link but measured at the same tier as the pcie-host group
```

The bandwidth column is deliberately left as `xx.xx` here. Filling it in with a
plausible-looking number would make this document a source of a figure nobody measured,
which is the one thing this curriculum does not do.

## Acceptance

- [ ] Builds with `hipcc -O2` (and, if you port it, `nvcc -O2`) with no warnings.
- [ ] Every runtime call is inside `HIP_CHECK` / `CUDA_CHECK`.
- [ ] Every pair and size verifies byte-for-byte; a mismatch makes the process exit
      non-zero and says which pair and size.
- [ ] Runs cleanly under a leak check if you have one, and at minimum has one `hipFree`
      per `hipMalloc` on every path including the error paths.
- [ ] On a single-GPU machine it exits with a one-line message, not a crash and not a
      table of zeroes.
- [ ] The consistency rule is stated in a comment in your own words, with its blind spot
      named.

## Questions to answer in a comment at the bottom of your file

a) You measure a pair twice and get two numbers 8% apart. Is that a real difference?
   Say what you would add to the harness to answer that question, and what you would
   report instead of a single number.

b) The sweep enables peer access between many pairs. Name one reason not to leave every
   pair enabled for the whole run, and say what it would cost you to enable and disable
   around each measurement instead.

c) `hipMemcpyPeerAsync` on a stream created on the destination device: whose stream
   should it be, and what changes if you use the source device's stream instead? Say how
   you would test your answer rather than reasoning it out.

d) Your matrix shows one pair at half the bandwidth of its three siblings in the same
   class, reproducibly. Give three candidate causes from
   [section 3.7](../README.md#37-pcie-the-hidden-bottleneck-in-the-middle) and the
   single command you would run first to separate them.

---

Back to [the exercises index](README.md) | [module README](../README.md) |
[glossary](../../../../docs/GLOSSARY.md)
