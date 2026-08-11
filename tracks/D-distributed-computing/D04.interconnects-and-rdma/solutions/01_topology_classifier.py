"""Solution 1 - normalize a captured topology matrix, and refuse a bad placement.

The three TODOs from ../exercises/01_topology_classifier.py, filled in, with the
reasoning in ../solutions/README.md. The parsing half is unchanged from the starter.

This drill PARSES OUTPUT, so it prints no model banner. It names the files it read.

Run: python 01_topology_classifier.py
"""

import os
import re

SELF = "self"
DIRECT = "direct-link"
SWITCH = "pcie-switch"
HOST = "pcie-host"
SOCKET = "cross-socket"
UNKNOWN = "unknown"

SEVERITY = {SELF: 0, DIRECT: 1, SWITCH: 2, HOST: 3, SOCKET: 4, UNKNOWN: 5}
SHORT = {SELF: ".", DIRECT: "D", SWITCH: "W", HOST: "H", SOCKET: "S", UNKNOWN: "?"}

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "..", "samples")
NVIDIA_SAMPLE = os.path.join(SAMPLES, "nvidia_topo_synthetic.txt")
AMD_SAMPLE = os.path.join(SAMPLES, "rocm_showtopo_synthetic.txt")
UNKNOWN_SAMPLE = os.path.join(SAMPLES, "nvidia_topo_unknown_code_synthetic.txt")

GPU_RE = re.compile(r"^GPU\d+$")
NUMA_RE = re.compile(r"GPU\[(\d+)\].*?Numa\s+Node\s*:\s*(-?\d+)", re.IGNORECASE)
LINK_SECTION_RE = re.compile(r"link\s+type\s+between\s+two\s+gpus", re.IGNORECASE)
BANNER_RE = re.compile(r"^\s*={4,}")

NVLINK_RE = re.compile(r"^NV(\d+)$")

# The legend, as a table. Anything absent from it is UNKNOWN by construction, which is
# the property that makes this classifier safe to build a launcher on: adding a code is
# an edit here, and forgetting to add one is a refusal rather than a wrong answer.
NVIDIA_CODES = {
    "X": SELF,
    "PIX": SWITCH,   # at most one PCIe bridge
    "PXB": SWITCH,   # several bridges, still no host bridge
    "PHB": HOST,     # through the host bridge
    "NODE": HOST,    # between host bridges inside one NUMA node
    "SYS": SOCKET,   # between NUMA nodes
}


def classify_nvidia_code(code):
    """Map one `nvidia-smi topo -m` cell to the shared enumeration."""
    token = code.strip().upper()
    if NVLINK_RE.match(token):
        return DIRECT
    return NVIDIA_CODES.get(token, UNKNOWN)


def classify_amd_link(code, same_device, numa_src, numa_dst):
    """Map one `rocm-smi --showtopo` link-type cell to the same enumeration.

    PCIE is the judgement call. rocm-smi says the path is PCIe and stops there, so the
    switch-local and host-bridge readings are indistinguishable from this input alone.
    Report the worse of the two, and use the NUMA nodes when they let you say something
    stronger and definitely true.
    """
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
    """Worst class among all ordered pairs inside a proposed parallelism group.

    A pair missing from the capture becomes UNKNOWN rather than being skipped. A real
    matrix is symmetric and complete, so a gap means the capture is truncated -- and
    skipping it would let a launcher certify a group on the strength of the pairs that
    happened to be present.
    """
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


# --------------------------------------------------------------------- given: parsing


def read_nvidia(path):
    """Return {(row, col): raw_code} from an `nvidia-smi topo -m` capture."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()
    columns, rows = _find_matrix(lines, 0, len(lines), _nvidia_label)
    return {(label, col): cell
            for label, cells in rows for col, cell in zip(columns, cells)}


def read_amd(path):
    """Return ({(row, col): raw_code}, {gpu_index: numa_node}) from rocm-smi output."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    lines = text.splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if LINK_SECTION_RE.search(line)),
                 None)
    stop = len(lines)
    if start is not None:
        for i in range(start, len(lines)):
            if BANNER_RE.match(lines[i]):
                stop = i
                break
    columns, rows = _find_matrix(lines, start or 0, stop, GPU_RE.match)
    numa = {int(m.group(1)): int(m.group(2))
            for m in (NUMA_RE.search(line) for line in lines) if m}
    pairs = {(label, col): cell
             for label, cells in rows for col, cell in zip(columns, cells)}
    return pairs, numa


def _nvidia_label(token):
    """A header token in nvidia-smi's matrix: a device name, not a class code."""
    if token.lower() in ("cpu", "numa", "gpu", "affinity", "id"):
        return False
    if token.upper() in ("X", "PIX", "PXB", "PHB", "NODE", "SYS"):
        return False
    if re.match(r"^NV\d+$", token.upper()):
        return False
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9_.:+-]*$", token))


def _find_matrix(lines, start, stop, label_ok):
    """Header plus rows, choosing the candidate header that yields the most rows."""
    best = (None, [])
    for idx in range(start, stop):
        tokens = lines[idx].split()
        if len(tokens) < 2 or not label_ok(tokens[0]) or not label_ok(tokens[1]):
            continue
        columns = []
        for token in tokens:
            if not label_ok(token):
                break
            columns.append(token)
        rows = []
        for row_idx in range(idx + 1, stop):
            row_tokens = lines[row_idx].split()
            if not row_tokens or row_tokens[0] not in columns:
                continue
            cells = row_tokens[1:1 + len(columns)]
            if len(cells) == len(columns):
                rows.append((row_tokens[0], cells))
        if len(rows) > len(best[1]):
            best = (columns, rows)
    return best


def gpu_index(label):
    return int(label[3:]) if GPU_RE.match(label) else None


def attempt(function, *args):
    """Report what `function` did with bad input: raised, returned, or not written yet."""
    try:
        value = function(*args)
    except ValueError as exc:
        return "ValueError: %s" % exc
    return "no ValueError raised" if value is None else "returned %r" % (value,)


def show(title, path, pairs):
    labels = sorted({label for label, _ in pairs} | {col for _, col in pairs},
                    key=lambda s: (not GPU_RE.match(s), s))
    gpus = [label for label in labels if GPU_RE.match(label)]
    print("%s\n  parsed from %s" % (title, os.path.relpath(path, HERE)))
    width = max(len(label) for label in labels) + 1
    print(" " * width + "".join("%*s" % (width, col) for col in labels))
    for row in gpus:
        cells = []
        for col in labels:
            klass = pairs.get((row, col))
            cells.append("%*s" % (width, SHORT.get(klass, "n/i")))
        print("%-*s" % (width, row) + "".join(cells))
    print("")


def main():
    print("PARSED, not modelled: this drill reads captured-format files and reports what")
    print("they say. The files under ../samples/ are SYNTHETIC examples of the format;")
    print("they describe no real machine. Run the same code on your own capture.\n")
    print("legend  " + "   ".join("%s = %s" % (SHORT[k], k)
                                  for k in (SELF, DIRECT, SWITCH, HOST, SOCKET, UNKNOWN))
          + "   n/i = not implemented\n")

    raw_nvidia = read_nvidia(NVIDIA_SAMPLE)
    nvidia = {key: classify_nvidia_code(code) for key, code in raw_nvidia.items()}
    show("NVIDIA capture", NVIDIA_SAMPLE, nvidia)

    raw_amd, numa = read_amd(AMD_SAMPLE)
    amd = {}
    for (row, col), code in raw_amd.items():
        amd[(row, col)] = classify_amd_link(code, row == col,
                                            numa.get(gpu_index(row)),
                                            numa.get(gpu_index(col)))
    show("ROCm capture (NUMA nodes: %s)"
         % (", ".join("GPU%d->%d" % item for item in sorted(numa.items())) or "none"),
         AMD_SAMPLE, amd)

    raw_odd = read_nvidia(UNKNOWN_SAMPLE)
    odd = {key: classify_nvidia_code(code) for key, code in raw_odd.items()}
    show("NVIDIA capture carrying a code this curriculum has never seen", UNKNOWN_SAMPLE,
         odd)
    codes = sorted({code for key, code in raw_odd.items()
                    if odd.get(key) in (UNKNOWN, None)})
    print("  codes not classified as a known tier: %s" % (", ".join(codes) or "none"))
    print("")

    print("worst class inside a proposed parallelism group:")
    for group in (["GPU0", "GPU1"], ["GPU2", "GPU3"], ["GPU0", "GPU1", "GPU2", "GPU3"]):
        verdict = worst_class_in_group(nvidia, group)
        print("  %-28s -> %s" % (",".join(group), verdict or "not implemented"))
    print("  A group whose worst class is anything but direct-link runs its collective")
    print("  at that pair's tier. Placement, not tuning, is the fix.")
    print("")

    print("rejection checks (each must raise ValueError):")
    print("  single-GPU group -> %s" % attempt(worst_class_in_group, nvidia, ["GPU0"]))
    print("  empty group      -> %s" % attempt(worst_class_in_group, nvidia, []))
    print("")

    truncated = dict(nvidia)
    del truncated[("GPU2", "GPU0")]
    print("truncated capture, with (GPU2, GPU0) removed:")
    print("  GPU0,GPU2 -> %s   (the surviving direction says cross-socket; the missing"
          % worst_class_in_group(truncated, ["GPU0", "GPU2"]))
    print("  one is unknown, and unknown is worse, so the group is refused either way)")


if __name__ == "__main__":
    main()

# TODO 4 answers:
#
# a) PIX -> PXB throws away how many PCIe bridges the path crosses; PHB -> NODE throws
#    away whether the host bridge was crossed once or between two bridges inside a NUMA
#    node. Both are real differences in latency and in contention. They are safe to
#    collapse HERE because this classifier answers exactly one question -- is this pair
#    in the scale-up tier, and if not, how far outside it -- and both members of each
#    pair give the same answer to it. The collapse would NOT be safe in a tool that
#    predicts bandwidth, and that is the test to apply before collapsing anything:
#    collapse only what the consumer of the answer cannot act on differently.
#
# b) FOR: a class you cannot name is a class you cannot certify, and the cost of a wrong
#    "fine" is a job that runs a tier down for its whole life with no error, while the
#    cost of a wrong "refuse" is one message to a human. Asymmetric costs justify an
#    asymmetric default. AGAINST: `unknown` will most often mean "this tool is older
#    than the machine", and refusing every new machine until someone edits a table is a
#    self-inflicted outage; it also trains people to pass --force, which destroys the
#    check. So: a batch scheduler should refuse and page a human, because it places
#    thousands of jobs and a silent tier drop multiplies. An interactive tool should
#    warn loudly, print the raw code, and continue -- the human is right there, and the
#    thing they need is the code, not a verdict.
#
# c) UNKNOWN. A real topology matrix is symmetric and complete, so a missing entry means
#    the capture is truncated, not that the pair is absent. Skipping it would let the
#    function certify a group using only the pairs that happened to survive the capture
#    -- and the pair most likely to be missing from a hand-trimmed paste is the ugly one
#    somebody was not proud of. Returning the best-of-what-is-present is the failure mode
#    where a tool is confidently wrong, which is the one to design against.
#
# d) `lspci -tv` shows the actual bridge tree, so you can see whether a GPU and an
#    adapter hang off the same switch or meet only at the root complex; `lspci -vv` on
#    each device gives LnkCap against LnkSta, which is negotiated width and speed, and
#    /sys/class/infiniband/<dev>/device/numa_node gives the adapter's socket. What to do
#    with it: pair each rank with the adapter on its own switch, bind its CPU and memory
#    to that NUMA node, and file the trained-down or narrow links as hardware faults --
#    README 3.7's list is five configuration bugs, and every one of them is free to fix
#    and invisible in a correctness test.
