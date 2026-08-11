# D04 Samples — synthetic inputs for the two parsers

> [!IMPORTANT]
> **Nothing in this directory was captured from a machine.** Every file here is
> hand-written to illustrate an output *format*, so that
> [`tools/topology_report.py`](../tools/topology_report.py) and
> [`tools/parse_ib_bw.py`](../tools/parse_ib_bw.py) can be exercised on a laptop with no
> GPU and no fabric. The bandwidth figures in `ib_write_bw_synthetic.txt` are **round
> invented numbers**, and the topology matrices describe a **made-up node**. None of it
> is a measurement, a specification, or a claim about any product. Do not quote a number
> from this directory, and do not treat a class in these matrices as evidence about any
> real machine.

No file here contains a hostname, a username, an IP or GUID, a serial number, a device
UUID or any other identifying data, because none of it came from a real system in the
first place.

Every file carries a `.txt` suffix, including the one that illustrates a `.log`, because
`build_docs.py` stages only the suffixes in its `PUBLISH_SUFFIXES` set: a `.log` resolves
on GitHub and 404s on the docs site, and `check_links.py` cannot see that because it reads
the repository rather than the staged output. The parsers do not care about the extension.

| File | Format it illustrates | What it exercises |
|---|---|---|
| [`nvidia_topo_synthetic.txt`](nvidia_topo_synthetic.txt) | `nvidia-smi topo -m` | `direct-link`, `pcie-switch` and `cross-socket` classes; adapter rows; a node split into two islands |
| [`rocm_showtopo_synthetic.txt`](rocm_showtopo_synthetic.txt) | `rocm-smi --showtopo` | the same node in ROCm's vocabulary: link type only, plus the NUMA section that lets a `PCIE` pair be called `cross-socket` instead of falling back to the conservative `pcie-host` |
| [`nvidia_topo_unknown_code_synthetic.txt`](nvidia_topo_unknown_code_synthetic.txt) | a future/unfamiliar code | that an unrecognized code is reported as `unknown` and never approximated |
| [`ib_write_bw_synthetic.txt`](ib_write_bw_synthetic.txt) | `ib_write_bw --report_gbits` | unit handling, both-unit reporting, and locating the size at which bandwidth saturates |

## Using them

```bash
# from the module directory
python3 tools/topology_report.py --input samples/nvidia_topo_synthetic.txt --vendor auto
python3 tools/topology_report.py --input samples/rocm_showtopo_synthetic.txt --vendor amd
python3 tools/topology_report.py --input samples/nvidia_topo_unknown_code_synthetic.txt \
        --vendor nvidia
python3 tools/parse_ib_bw.py --input samples/ib_write_bw_synthetic.txt --report gbits
```

The interesting invocation is the group check, because it is the one a launcher would
make before it places ranks:

```bash
python3 tools/topology_report.py --input samples/nvidia_topo_synthetic.txt --group 0,1
python3 tools/topology_report.py --input samples/nvidia_topo_synthetic.txt --group 0,1,2,3
```

The first group is entirely inside one island and comes back `direct-link`. The second
spans both and comes back `cross-socket` — a four-way tensor-parallel group placed on
that node would run its collective one tier down the ladder, for free, forever, with no
error message. That is [§3.5](../README.md#3-deep-dive)'s silent degradation, caught
before the job starts instead of after the invoice.

## Why these are committed when a performance CSV would not be

[D02](../../D02.message-passing-and-mpi/exercises/README.md) deliberately ships **no**
sample measurement, on the rule that this repository never publishes a number it cannot
reproduce. These files do not break that rule, and the distinction is worth being precise
about:

- A topology matrix is a **description of a format**, and the classes here are the
  arithmetic consequence of the codes in the file. The tool's output on this input is
  fully reproducible by anyone, because the input is in the repository.
- The `ib_write_bw` log is the awkward case, since it does contain bandwidth-shaped
  numbers. They are labelled as invented in the file itself, in every table row's
  provenance (the parser prints the filename it read), and here. They exist so the parser
  can be tested, not so anyone learns what a fabric does.

Get real numbers by running [Lab 1](../README.md#4-hands-on-labs) on your own node and
Lab 3 on your own fabric. Sanitized transcripts from the curriculum's own hardware live
under [`outputs/`](../../../../outputs/README.md).

## Reference

- Parent module: [`../README.md`](../README.md) — [§4](../README.md#4-hands-on-labs) for
  the capture commands, [§5.1](../README.md#5-performance-analysis) for how to read a
  matrix, [§5.2](../README.md#5-performance-analysis) for the seven questions to ask a
  bandwidth number.
- Terms: [`GLOSSARY.md`](../../../../docs/GLOSSARY.md) —
  [NVLink, xGMI and Infinity Fabric](../../../../docs/GLOSSARY.md#nvlink-xgmi-and-infinity-fabric),
  [InfiniBand, RoCE and Slingshot](../../../../docs/GLOSSARY.md#infiniband-roce-and-slingshot),
  [RDMA and GPUDirect](../../../../docs/GLOSSARY.md#rdma-and-gpudirect).
