#!/usr/bin/env python3
"""Normalize a captured topology matrix into one comparable table, either vendor.

This is lab 1's second half. It PARSES OUTPUT YOU CAPTURED on a real machine:

    nvidia-smi topo -m  | tee topo.txt
    rocm-smi --showtopo | tee topo.txt
    python -S -I tools/topology_report.py --input topo.txt --vendor auto

It measures nothing and models nothing, so unlike tools/staging_cost.py and
tools/injection_model.py it prints no model banner. It reports what it read, names the
file it read it from, and says `unknown` for any code it cannot justify rather than
picking the nearest plausible class.

The normalized classes, best first:

    self          a device against itself
    direct-link   a scale-up link -- NVLink (NV#) or xGMI. Addressable; README 3.1
    pcie-switch   PCIe without leaving the switch complex (nvidia-smi PIX, PXB)
    pcie-host     PCIe through a host bridge (nvidia-smi PHB, NODE)
    cross-socket  a traversal between root complexes (nvidia-smi SYS)
    unknown       a code this tool does not recognize. Never silently bucketed

`unknown` sorts WORST on purpose. A launcher that refuses a placement it cannot certify
is behaving correctly, and an unfamiliar code is far more likely to be a new distant
path than a new fast one.

What the two vendors can and cannot tell you differs, and this tool does not paper over
it. `nvidia-smi topo -m` distinguishes five PCIe distances and lists network adapters as
rows. `rocm-smi --showtopo` reports a link TYPE only -- XGMI or PCIE -- with no switch
versus host-bridge distinction and no adapter rows, so a PCIE pair is reported at the
conservative end of that range and the report says so in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

PROGRAM = "topology_report"

SELF = "self"
DIRECT = "direct-link"
SWITCH = "pcie-switch"
HOST = "pcie-host"
SOCKET = "cross-socket"
UNKNOWN = "unknown"

# Ordering used by worst_class_in_group. Unknown is deliberately the worst: see module
# docstring.
SEVERITY = {SELF: 0, DIRECT: 1, SWITCH: 2, HOST: 3, SOCKET: 4, UNKNOWN: 5}
SHORT = {SELF: ".", DIRECT: "D", SWITCH: "W", HOST: "H", SOCKET: "S", UNKNOWN: "?"}

# nvidia-smi's legend, as of the tool versions this module was written against. A code
# absent from this table becomes `unknown`; it is not approximated to a neighbour.
NVIDIA_CODES = {
    "X": SELF,
    "SELF": SELF,
    "PIX": SWITCH,
    "PXB": SWITCH,
    "PHB": HOST,
    "NODE": HOST,
    "SYS": SOCKET,
}
NVLINK_RE = re.compile(r"^NV(\d+)$")

GPU_RE = re.compile(r"^GPU(\d+)$", re.IGNORECASE)
IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:+-]*$")

# Trailing header words in `nvidia-smi topo -m`: "CPU Affinity NUMA Affinity GPU NUMA ID".
# The device columns are the leading run that stops at the first of these.
HEADER_STOP = {"cpu", "numa", "gpu", "affinity", "id", "legend", "mig"}

# rocm-smi section titles, matched case-insensitively on the banner line.
AMD_LINK_SECTION = re.compile(r"link\s+type\s+between\s+two\s+gpus", re.IGNORECASE)
AMD_HOPS_SECTION = re.compile(r"hops?\s+between\s+two\s+gpus", re.IGNORECASE)
AMD_SECTION_BANNER = re.compile(r"^\s*={4,}")
AMD_NUMA_RE = re.compile(
    r"GPU\[(\d+)\].*?Numa\s+Node\s*:\s*(-?\d+)", re.IGNORECASE)


def die(message):
    """One line on stderr and a non-zero exit. Never a traceback."""
    sys.stderr.write("%s: %s\n" % (PROGRAM, message))
    raise SystemExit(2)


# --------------------------------------------------------------------------- shared


def is_column_token(token):
    """True for a device label in a matrix header row.

    Class codes are excluded explicitly, because otherwise a data row such as
    `GPU0  X  NV12 ...` would look exactly like a header row: two identifier-shaped
    tokens in a row.
    """
    if token.lower() in HEADER_STOP:
        return False
    if token.upper() in NVIDIA_CODES or NVLINK_RE.match(token.upper()):
        return False
    return bool(IDENT_RE.match(token))


def is_gpu_token(token):
    """True only for a GPU label. Used where a looser test would misread a data row."""
    return bool(GPU_RE.match(token))


def find_matrix(lines, start=0, stop=None, token_ok=is_column_token):
    """Locate a square-ish matrix in `lines[start:stop]`.

    Returns (columns, rows) where rows is a list of (label, cells, lineno). The header is
    the first line carrying two consecutive device labels; the data rows are the lines
    after it whose first token is one of those labels.

    `token_ok` decides what counts as a device label. The NVIDIA path accepts adapter
    names too, since `nvidia-smi topo -m` puts them in the matrix; the ROCm path accepts
    GPU labels only, because a rocm-smi row such as `GPU2 PCIE PCIE 0 XGMI` would
    otherwise look like a header row under the broader test.
    """
    stop = len(lines) if stop is None else stop
    best = None
    short_row = None

    # Every line that could be a header is tried, and the candidate that yields the most
    # data rows wins. A single-pass "first plausible header" rule is not enough on real
    # captures: a tee'd transcript carries a shell prompt, a banner or a note above the
    # matrix, and any two words in a row would otherwise be mistaken for column labels.
    for idx in range(start, stop):
        tokens = lines[idx].split()
        if len(tokens) < 2 or not token_ok(tokens[0]) or not token_ok(tokens[1]):
            continue
        columns = []
        for token in tokens:
            if not token_ok(token):
                break
            columns.append(token)
        known = set(columns)

        rows = []
        for row_idx in range(idx + 1, stop):
            row_tokens = lines[row_idx].split()
            if not row_tokens or row_tokens[0] not in known:
                continue
            cells = row_tokens[1:1 + len(columns)]
            if len(cells) < len(columns):
                short_row = (row_idx + 1, row_tokens[0], len(cells), len(columns),
                             columns[0], columns[-1])
                continue
            rows.append((row_tokens[0], cells, row_idx + 1))

        if rows and (best is None or len(rows) > len(best[1])):
            best = (columns, rows)

    if best is not None:
        return best
    if short_row is not None:
        die("line %d: row '%s' has %d cells, expected %d (one per column %s..%s)"
            % short_row)
    return None, None


def gpu_index(label):
    match = GPU_RE.match(label)
    return int(match.group(1)) if match else None


def declares_synthetic(text):
    """True when the file says of itself that it is not a capture.

    The samples/ directory labels its files this way. Echoing the label is cheap
    insurance against a generated example being read as a description of real hardware.
    """
    return "SYNTHETIC" in "\n".join(text.splitlines()[:25]).upper()


# --------------------------------------------------------------------------- NVIDIA


def classify_nvidia(code):
    """Map one nvidia-smi cell to (class, detail). Unrecognized codes stay unknown."""
    token = code.strip().upper()
    link = NVLINK_RE.match(token)
    if link:
        return DIRECT, "NVLink x%s" % link.group(1)
    if token in NVIDIA_CODES:
        return NVIDIA_CODES[token], None
    return UNKNOWN, None


def parse_nvidia(text):
    lines = text.splitlines()
    columns, rows = find_matrix(lines)
    if columns is None:
        die("no topology matrix found. Expected the header row of "
            "`nvidia-smi topo -m`, which lists GPU0 GPU1 ... as column labels")

    pairs = {}
    for label, cells, _lineno in rows:
        for column, cell in zip(columns, cells):
            klass, detail = classify_nvidia(cell)
            pairs[(label, column)] = {"class": klass, "code": cell.strip(),
                                      "detail": detail}
    return {
        "vendor": "nvidia",
        "source_command": "nvidia-smi topo -m",
        "devices": columns,
        "pairs": pairs,
        "notes": [],
    }


# ------------------------------------------------------------------------------ AMD


def parse_amd_numa(lines):
    """GPU index -> NUMA node, from the `Numa Nodes` section. Empty if absent."""
    numa = {}
    for line in lines:
        match = AMD_NUMA_RE.search(line)
        if match:
            numa[int(match.group(1))] = int(match.group(2))
    return numa


def section_bounds(lines, title_re):
    """Half-open [start, stop) line range of a rocm-smi section, or (None, None)."""
    start = None
    for idx, line in enumerate(lines):
        if title_re.search(line):
            start = idx + 1
            break
    if start is None:
        return None, None
    for idx in range(start, len(lines)):
        if AMD_SECTION_BANNER.match(lines[idx]):
            return start, idx
    return start, len(lines)


def classify_amd(code, same_device, numa_src, numa_dst):
    """Map one rocm-smi link-type cell to (class, detail, conservative).

    `conservative` marks a class this tool widened to the worst reading the input can
    support: rocm-smi reports PCIE without saying whether the path stays inside a switch
    complex or crosses a host bridge, so a PCIE pair on one NUMA node is reported as
    pcie-host rather than as the pcie-switch it might be.
    """
    token = code.strip().upper()
    if same_device:
        if token in ("0", "X", "SELF", "-", "N/A"):
            return SELF, None, False
    if token.startswith("XGMI"):
        return DIRECT, "xGMI (%s)" % code.strip(), False
    if token in ("PCIE", "PCI-E", "PCI", "PCIE-BRIDGE"):
        if numa_src is not None and numa_dst is not None and numa_src != numa_dst:
            return SOCKET, "PCIE across NUMA nodes %d and %d" % (numa_src, numa_dst), False
        return HOST, "PCIE, NUMA node not distinguishing", True
    return UNKNOWN, None, False


def parse_amd(text):
    lines = text.splitlines()
    if "amd-smi" in text.lower() and not AMD_LINK_SECTION.search(text):
        die("this looks like `amd-smi topology` output, whose layout differs. "
            "Capture with `rocm-smi --showtopo | tee topo.txt` instead")

    start, stop = section_bounds(lines, AMD_LINK_SECTION)
    if start is None:
        die("no 'Link Type between two GPUs' section found. Expected the whole output "
            "of `rocm-smi --showtopo`; a partial capture with only the weight or hop "
            "table is not enough, and neither is an NVIDIA capture read as --vendor amd")

    columns, rows = find_matrix(lines, start, stop, token_ok=is_gpu_token)
    if columns is None:
        die("the 'Link Type between two GPUs' section has no readable matrix "
            "(expected a GPU0 GPU1 ... header row followed by one row per GPU)")

    numa = parse_amd_numa(lines)
    hop_start, hop_stop = section_bounds(lines, AMD_HOPS_SECTION)
    hops = {}
    if hop_start is not None:
        hop_columns, hop_rows = find_matrix(lines, hop_start, hop_stop,
                                            token_ok=is_gpu_token)
        if hop_columns is not None:
            for label, cells, _lineno in hop_rows:
                for column, cell in zip(hop_columns, cells):
                    if cell.strip().lstrip("-").isdigit():
                        hops[(label, column)] = int(cell.strip())

    pairs = {}
    conservative = 0
    for label, cells, _lineno in rows:
        for column, cell in zip(columns, cells):
            klass, detail, widened = classify_amd(
                cell, label == column, numa.get(gpu_index(label)),
                numa.get(gpu_index(column)))
            conservative += 1 if widened else 0
            entry = {"class": klass, "code": cell.strip(), "detail": detail}
            if (label, column) in hops:
                entry["hops"] = hops[(label, column)]
            pairs[(label, column)] = entry

    notes = []
    if conservative:
        notes.append(
            "%d pair(s) came from a link type of PCIE. rocm-smi reports the link type "
            "only -- it does not separate a switch-local path (nvidia-smi PIX/PXB) from "
            "a host-bridge path (PHB/NODE) -- so those pairs are reported as pcie-host, "
            "the worst reading this input supports, not the one measured."
            % conservative)
    if not numa:
        notes.append(
            "No 'Numa Nodes' section in this capture, so a PCIE pair could not be "
            "promoted to cross-socket even where it is one. Capture the full "
            "`rocm-smi --showtopo` output to get it.")
    notes.append(
        "This format carries no network-adapter rows. For the GPU-to-adapter question "
        "that README section 3.5 turns on, read `lspci -tv` and the adapter's NUMA node "
        "under /sys/class/infiniband/<dev>/device/numa_node.")
    return {
        "vendor": "amd",
        "source_command": "rocm-smi --showtopo",
        "devices": columns,
        "pairs": pairs,
        "notes": notes,
    }


# ------------------------------------------------------------------------- reporting


def detect_vendor(text):
    upper = text.upper()
    lowered = text.lower()
    nvidia = (
        "nvidia-smi" in lowered
        or "gpu numa id" in lowered
        or bool(re.search(r"\bNV\d+\b", upper))
        or bool(re.search(r"\b(PIX|PXB|PHB|SYS)\b", upper))
    )
    amd = (
        "rocm system management interface" in lowered
        or bool(AMD_LINK_SECTION.search(text))
        or "rocm-smi" in lowered
        or "amd-smi" in lowered
        or "XGMI" in upper
    )
    if nvidia and amd:
        return None, ("both vendors' markers are present; pass --vendor amd or "
                      "--vendor nvidia to say which this file is")
    if nvidia:
        return "nvidia", None
    if amd:
        return "amd", None
    return None, ("no vendor markers found. Expected `nvidia-smi topo -m` output "
                  "(codes such as PIX/SYS/NV2) or `rocm-smi --showtopo` output "
                  "(a 'Link Type between two GPUs' section)")


def worst_class_in_group(pairs, gpu_labels):
    """Worst normalized class among all ordered pairs inside a proposed rank group.

    This is what a launcher calls before it places a tensor-parallel group: if the answer
    is worse than direct-link, that group's collective runs one tier down the ladder for
    no reason, and the fix is placement rather than tuning.
    """
    worst = SELF
    for src in gpu_labels:
        for dst in gpu_labels:
            if src == dst:
                continue
            entry = pairs.get((src, dst))
            klass = UNKNOWN if entry is None else entry["class"]
            if SEVERITY[klass] > SEVERITY[worst]:
                worst = klass
    return worst


def split_devices(devices):
    gpus = [d for d in devices if GPU_RE.match(d)]
    adapters = [d for d in devices if not GPU_RE.match(d)]
    return gpus, adapters


def print_matrix(title, row_labels, col_labels, pairs):
    print(title)
    width = max([len(c) for c in col_labels] + [len(r) for r in row_labels] + [4])
    print(" " * (width + 1) + "".join("%*s" % (width + 1, c) for c in col_labels))
    for row in row_labels:
        cells = []
        for col in col_labels:
            entry = pairs.get((row, col))
            cells.append("%*s" % (width + 1,
                                  "-" if entry is None else SHORT[entry["class"]]))
        print("%-*s " % (width, row) + "".join(cells))
    print("")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Normalize a captured nvidia-smi or rocm-smi topology matrix into "
                    "one comparable table of link classes.",
        epilog="This tool reports what it PARSED from a real capture. It prints no "
               "model banner because it models nothing, and it reports `unknown` for "
               "any code it does not recognize rather than guessing.")
    parser.add_argument("--input", "-i", required=True,
                        help="file holding `nvidia-smi topo -m` or "
                             "`rocm-smi --showtopo` output")
    parser.add_argument("--vendor", choices=("auto", "amd", "nvidia"), default="auto",
                        help="which format to expect (default auto)")
    parser.add_argument("--group", default="",
                        help="comma-separated GPU indices, e.g. 0,1,2,3: report the "
                             "worst class inside that proposed parallelism group")
    parser.add_argument("--json", action="store_true",
                        help="emit the same result as JSON")
    return parser


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        die("cannot read %s: %s" % (path, exc.strerror or exc))
    if not text.strip():
        die("%s is empty; capture the matrix first, e.g. "
            "`nvidia-smi topo -m | tee %s`" % (path, path))
    return text


def parse_group(spec, gpus):
    if not spec.strip():
        return []
    labels = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if not piece.lstrip("+").isdigit():
            die("bad --group entry %r: expected GPU indices such as 0,1,2,3" % piece)
        label = "GPU%d" % int(piece)
        if label not in gpus:
            die("--group names %s, which is not in this capture (it has %s)"
                % (label, ", ".join(gpus) if gpus else "no GPUs"))
        labels.append(label)
    if len(labels) < 2:
        die("--group needs at least two distinct GPUs to have a pair in it")
    return labels


def main(argv=None):
    args = build_parser().parse_args(argv)
    text = read_text(args.input)

    vendor = args.vendor
    if vendor == "auto":
        vendor, problem = detect_vendor(text)
        if vendor is None:
            die("%s: %s" % (args.input, problem))

    parsed = parse_amd(text) if vendor == "amd" else parse_nvidia(text)
    pairs = parsed["pairs"]
    gpus, adapters = split_devices(parsed["devices"])
    if not gpus:
        die("%s: parsed %d device column(s) but none of them is a GPU"
            % (args.input, len(parsed["devices"])))

    counts = {}
    unknown_codes = {}
    for (src, dst), entry in pairs.items():
        if src == dst:
            continue
        counts[entry["class"]] = counts.get(entry["class"], 0) + 1
        if entry["class"] == UNKNOWN:
            unknown_codes.setdefault(entry["code"], []).append("%s-%s" % (src, dst))

    no_direct = sorted(
        "%s-%s" % (src, dst)
        for (src, dst), entry in pairs.items()
        if src != dst and src in gpus and dst in gpus and entry["class"] != DIRECT)

    group = parse_group(args.group, gpus)
    group_worst = worst_class_in_group(pairs, group) if group else None

    if args.json:
        print(json.dumps({
            "source_file": args.input,
            "vendor": parsed["vendor"],
            "source_command": parsed["source_command"],
            "gpus": gpus,
            "adapters": adapters,
            "pairs": [{"source": src, "destination": dst, "code": entry["code"],
                       "class": entry["class"], "detail": entry.get("detail"),
                       "hops": entry.get("hops")}
                      for (src, dst), entry in sorted(pairs.items())],
            "class_counts": counts,
            "gpu_pairs_without_direct_link": no_direct,
            "unknown_codes": unknown_codes,
            "group": group,
            "group_worst_class": group_worst,
            "notes": parsed["notes"],
        }, indent=2))
        return 0

    print("=" * 78)
    print("PARSED, not modelled: %s" % args.input)
    print("read as %s `%s` output: %d GPU(s), %d adapter row(s), %d pair(s)"
          % (parsed["vendor"], parsed["source_command"], len(gpus), len(adapters),
             len(pairs)))
    if declares_synthetic(text):
        print("")
        print("!! This file DECLARES ITSELF SYNTHETIC in its own header. It is an example")
        print("!! of the format, not a capture. It says nothing about any real machine.")
    print("=" * 78)
    print("")

    print("legend  %s" % "   ".join(
        "%s = %s" % (SHORT[k], k) for k in (SELF, DIRECT, SWITCH, HOST, SOCKET, UNKNOWN)))
    print("")
    print_matrix("GPU to GPU (row = source, column = destination)", gpus, gpus, pairs)

    if adapters:
        print_matrix("GPU to network adapter -- the rows README 3.5 turns on",
                     gpus, adapters, pairs)
    else:
        print("GPU to network adapter: this capture has no adapter rows.")
        print("")

    print("class counts, excluding the diagonal:")
    for klass in (DIRECT, SWITCH, HOST, SOCKET, UNKNOWN):
        if counts.get(klass):
            print("  %-13s %4d" % (klass, counts[klass]))
    print("")

    if no_direct:
        print("GPU pairs WITHOUT a direct scale-up link (%d):" % len(no_direct))
        print("  %s" % ", ".join(no_direct))
        print("  A parallelism group spanning any of these runs its collective one tier")
        print("  down. That is a placement bug and it is free to fix.")
    else:
        print("Every GPU pair in this capture has a direct scale-up link.")
    print("")

    if unknown_codes:
        print("codes this tool does not recognize, reported as `unknown` rather than")
        print("guessed at:")
        for code, where in sorted(unknown_codes.items()):
            print("  %-10s %d pair(s), e.g. %s" % (code, len(where), where[0]))
        print("  Check the legend your tool printed. If the code is real and current,")
        print("  it belongs in this tool's table, not in a reader's head.")
        print("")

    if group:
        print("proposed group %s: worst class inside it is %s"
              % (",".join(group), group_worst))
        if group_worst == DIRECT:
            print("  Every pair has a direct link. Safe placement for tensor parallelism.")
        elif group_worst == UNKNOWN:
            print("  At least one pair could not be classified. Refuse the placement and")
            print("  find out what the code means; do not assume it is fine.")
        else:
            print("  At least one pair is %s. A collective over this group runs at that"
                  % group_worst)
            print("  pair's tier, not at the group's best tier -- move the group.")
        print("")

    for note in parsed["notes"]:
        print("note: %s" % note)
    if parsed["notes"]:
        print("")

    print("What this tool does NOT know: negotiated PCIe width and speed, whether the")
    print("device-memory fast path is actually in use, and what any pair measures. The")
    print("first two are `lspci -vv` and your driver; the third is lab 2.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(0)
