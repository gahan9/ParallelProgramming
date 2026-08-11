"""Exercise 1 - normalize a captured topology matrix, and refuse a bad placement.

A topology matrix is a table of vendor-specific codes. It becomes useful the moment you
can turn it into ONE enumeration that both vendors map onto, because then a launcher can
answer the only question that matters before it places ranks: does every pair in this
group have a direct scale-up link, or does one of them quietly sit a tier down?

This drill PARSES OUTPUT, so it prints no model banner. It names the files it read.

Your job:
  TODO 1  implement classify_nvidia_code
  TODO 2  implement classify_amd_link
  TODO 3  implement worst_class_in_group
  TODO 4  read the output and answer the questions at the bottom of this file

The file-reading half is given, deliberately: splitting whitespace is not the skill being
assessed, and you should spend the time on the two places where judgement enters -- what
to do with a code you do not recognize, and how a group's worst pair is defined.

The harness runs as-is and prints "not implemented" until you fill the TODOs in.
Reference implementation, with the messy cases handled: ../tools/topology_report.py
Worked solution with commentary: ../solutions/README.md

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

# Worst last. Note where `unknown` sits, and be ready to defend it in question (b):
# this ordering is a policy decision, not a fact about hardware.
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


def classify_nvidia_code(code):
    """TODO 1: map one `nvidia-smi topo -m` cell to a class from the enumeration above.

    The codes, from the legend the tool prints above its matrix:
        X       self
        NV<n>   a bonded set of n NVLinks  -- the scale-up tier
        PIX     at most a single PCIe bridge
        PXB     multiple PCIe bridges, no host bridge
        PHB     through a PCIe host bridge (typically the CPU)
        NODE    between PCIe host bridges inside one NUMA node
        SYS     across the interconnect between NUMA nodes

    Two of those pairs collapse onto one class each. Decide which, and be able to say
    why the collapse is safe.

    The requirement that carries the most weight: a code NOT in that list must come back
    as UNKNOWN. Do not map it to the nearest thing you do know. A wrong class here is
    worse than no class, because a launcher will act on it.

    Return one of the six strings. Do not raise: an unfamiliar code is data, not an
    error.
    """
    return None


def classify_amd_link(code, same_device, numa_src, numa_dst):
    """TODO 2: map one `rocm-smi --showtopo` link-type cell to the same enumeration.

    ROCm gives you far less than NVIDIA does here, and the drill is mostly about being
    honest about that. The cells you will see:
        0       on the diagonal, a device against itself
        XGMI    the scale-up tier (possibly suffixed, e.g. XGMI2)
        PCIE    a PCIe path -- and nothing about how far it goes

    `numa_src` and `numa_dst` are the NUMA nodes of the two GPUs, or None when the
    capture did not include the Numa Nodes section.

    The judgement call is PCIE. rocm-smi cannot tell you whether that path stays inside
    a switch complex or crosses a host bridge, so PIX/PXB and PHB/NODE are
    indistinguishable to it. Pick the reading that cannot mislead someone acting on it,
    and use the NUMA nodes where they let you say something stronger.

    As in TODO 1: anything you do not recognize is UNKNOWN.
    """
    return None


def worst_class_in_group(pairs, group):
    """TODO 3: worst class among all ordered pairs inside a proposed parallelism group.

    `pairs` maps (source_label, destination_label) to a class string. `group` is a list
    of GPU labels such as ["GPU0", "GPU1"]. Skip the diagonal.

    Two cases decide whether this function is usable in a launcher:
      * the group has fewer than two GPUs -- raise ValueError with a one-line message
        naming the problem, because there is no pair to be worst;
      * a pair is missing from `pairs` entirely -- an asymmetric or truncated capture.
        Decide what that means and defend it in question (c).
    """
    return None


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

    print("TODO 4 - answer these in a comment at the bottom of this file:")
    print("  a) PIX and PXB collapse to one class here, and so do PHB and NODE. What")
    print("     does each collapse throw away, and why is it safe for the decision this")
    print("     classifier exists to make?")
    print("  b) SEVERITY puts `unknown` last, so a group containing one is rejected as")
    print("     firmly as one crossing a socket. Argue for that choice, then argue")
    print("     against it. Which way should a batch scheduler go, and which way should")
    print("     an interactive debugging tool go?")
    print("  c) A capture where (GPU0, GPU2) is present and (GPU2, GPU0) is missing is")
    print("     truncated, not asymmetric - a real matrix is symmetric. What should")
    print("     worst_class_in_group return, and what would go wrong with the other")
    print("     choice?")
    print("  d) The ROCm capture cannot distinguish a switch-local PCIe path from one")
    print("     through a host bridge. Name one thing you could read on the machine")
    print("     that WOULD distinguish them, and say what you would do with it.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
# d)
