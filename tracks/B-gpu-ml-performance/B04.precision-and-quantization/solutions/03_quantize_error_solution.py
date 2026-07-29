"""Solution 3 - group-wise quantization, and what it actually buys.

Reference answer for ../exercises/03_quantize_error.py. The three TODO implementations
are ~6 lines each; the value is in the discussion under TODO 4 at the bottom, which is
where the exercise is actually won or lost.

Run: python 03_quantize_error_solution.py
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment guard, not logic
    raise SystemExit(
        "03_quantize_error_solution.py needs numpy (its only dependency).\n"
        "  install it with:  python -m pip install numpy"
    )

SYM_MAX = 127          # symmetric int8 uses [-127, 127] so that 0.0 maps exactly to 0
TINY = 1e-12
BAND = slice(100, 108)  # the 8 wide-range columns


def make_structured_weights(rows: int = 64, cols: int = 256, seed: int = 0, band=BAND):
    """Small Gaussian weights, plus a band of large columns shared by every row.

    `band` is a parameter here (it is hard-coded in the exercise) so that answer (a) can
    be demonstrated rather than asserted: the same data with the band moved onto a group
    boundary quantizes measurably better.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.05, size=(rows, cols)).astype(np.float32)
    w[:, band] = rng.normal(0.0, 4.0, size=(rows, band.stop - band.start)).astype(np.float32)
    return w


def _quantize_last_axis(blocks):
    """Symmetric int8 round-trip where the LAST axis shares one scale."""
    amax = np.maximum(np.abs(blocks).max(axis=-1, keepdims=True), TINY)
    scale = amax / SYM_MAX
    q = np.clip(np.rint(blocks / scale), -SYM_MAX, SYM_MAX)
    return q * scale, scale.size


def quantize_per_tensor(w):
    """ONE scale for the whole matrix - the baseline."""
    deq, n = _quantize_last_axis(w.reshape(1, -1))
    return deq.reshape(w.shape).astype(np.float32), n


def quantize_per_channel(w):
    """TODO 1 - one scale per row. The row itself is the block, so no reshape is needed."""
    deq, n = _quantize_last_axis(w)
    return deq.astype(np.float32), n


def quantize_per_group(w, group: int):
    """TODO 2 - one scale per contiguous run of `group` weights inside a row.

    The ValueError is not defensive padding: a kernel that silently truncates the last
    partial group produces weights that are wrong only at the end of every row, which is
    among the least pleasant bugs in this whole area to find.
    """
    rows, cols = w.shape
    if cols % group:
        raise ValueError(f"group size {group} does not divide {cols} columns")
    deq, n = _quantize_last_axis(w.reshape(rows, cols // group, group))
    return deq.reshape(w.shape).astype(np.float32), n


def total_bytes(n_weights: int, n_scales: int) -> int:
    """TODO 3 - 1 byte per int8 code, 2 bytes per fp16 scale."""
    return n_weights + 2 * n_scales


def report(name, result, w, x, band=BAND):
    if result is None:
        print(f"{name:<26} not implemented yet (fill in the TODO)")
        return
    deq, n_scales = result
    rel_w = float(np.linalg.norm(w - deq) / np.linalg.norm(w))
    band_mask = np.zeros(w.shape[1], dtype=bool)
    band_mask[band] = True
    err = np.abs(w - deq)
    rest_max = float(err[:, ~band_mask].max())
    rest_rms = float(np.sqrt((err[:, ~band_mask] ** 2).mean()))
    rel_y = float(np.linalg.norm(w @ x - deq @ x) / np.linalg.norm(w @ x))
    nbytes = total_bytes(w.size, n_scales)
    ratio = (w.nbytes / nbytes) if nbytes else float("nan")
    print(f"{name:<26} {rel_w:10.5f} {rel_y:10.5f} {rest_max:12.6f} {rest_rms:12.6f} "
          f"{nbytes:9d} {ratio:8.2f}x")


def main() -> None:
    w = make_structured_weights()
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(w.shape[1], 32)).astype(np.float32)

    print("=== Solution 3 - group-wise int8 quantization ===\n")
    print(f"W: {w.shape[0]} x {w.shape[1]} fp32, {w.nbytes / 1024:.1f} KiB")
    print(f"  |w|max inside the band  : {np.abs(w[:, BAND]).max():.4f}")
    print(f"  |w|max outside the band : "
          f"{np.abs(np.delete(w, np.s_[BAND], axis=1)).max():.4f}\n")

    header = (f"{'scheme':<26} {'rel err W':>10} {'rel err Wx':>10} {'rest max err':>12} "
              f"{'rest rms err':>12} {'bytes':>9} {'vs fp32':>9}")
    print(header)
    print("-" * len(header))

    report("per-tensor", quantize_per_tensor(w), w, x)
    report("per-channel", quantize_per_channel(w), w, x)
    for g in (128, 64, 32, 16, 8):
        report(f"per-group g={g}", quantize_per_group(w, g), w, x)

    # The alignment experiment behind answer (a): the same 8-wide band, generated the
    # same way, but starting at column 96 so that at g = 8 it IS one whole group.
    # Nothing about the data got easier - only which groups the large values land in.
    aligned_band = slice(96, 104)
    aligned = make_structured_weights(band=aligned_band)
    print()
    report("per-group g=8, band 100..107", quantize_per_group(w, 8), w, x)
    report("per-group g=8, band  96..103", quantize_per_group(aligned, 8), aligned, x,
           band=aligned_band)
    print("  ^ same 8 wide columns, moved onto the group boundary. Now the contaminated")
    print("    group contains ONLY band values, so no ordinary weight pays for them and")
    print("    'rest max err' collapses to the per-column-scale ideal.")

    print("\n" + "\n".join(ANSWERS))


ANSWERS = [
    "TODO 4 answers",
    "",
    "(a) Why the rms improves and the max does not.",
    "    A group's scale is set by the largest magnitude anywhere in that group. With",
    "    g = 32 a row's groups are columns 0-31, 32-63, 64-95, 96-127, ... and the band",
    "    (columns 100-107) lands inside the 96-127 group. The other 24 columns of that",
    "    group - 96-99 and 108-127 - inherit a step sized for a value of ~12.5 instead of",
    "    ~0.2, so their error is essentially unchanged from per-channel (0.0473 vs",
    "    0.0490). They are only 24 of the 248 non-band columns, so they barely move the",
    "    rms - which improves ~3.2x, from 0.0165 to 0.0052 - while completely determining",
    "    the max. Seven of the eight groups per row did get ~60x finer; that is the gain,",
    "    and the max column simply cannot see it.",
    "    Alignment: the run above repeats g = 8 with the same 8 wide columns starting at",
    "    column 96 instead of 100, so the band IS one whole group. 'rest max err' drops",
    "    from 0.042 to 0.00073 - a factor of 58, which is the ratio of the two |w|max",
    "    values, i.e. the theoretical best. Two lessons: group-wise CONTAINS outlier",
    "    damage rather than removing it, and how well it contains it depends on an",
    "    alignment you do not control - which is why methods that rescale channels (AWQ)",
    "    or split them out entirely (LLM.int8()) exist.",
    "",
    "(b) Where to stop shrinking g.",
    "    Not at a favourite number - at the point where the error stops being the binding",
    "    constraint. The rule: shrink g until the quantization error is comfortably below",
    "    the error your task-level eval can already tolerate, then stop, because every",
    "    further halving costs bytes and kernel complexity for accuracy you cannot spend.",
    "    Here rel err W improves ~2.9x from per-channel (0.0234) to g=32 (0.0082), then",
    "    only ~1.4x per further halving while bytes/weight keeps climbing - g = 64 or 128",
    "    with fp16 scales is the shipped answer for int4 kernels for this reason. Note the",
    "    second constraint: g must divide the reduction tile of the GEMM kernel, or the",
    "    dequantization stops being free.",
    "",
    "(c) Bytes per weight.",
    "    g = 128: 1 + 2/128 = 1.0156 bytes/weight -> 3.94x vs fp32.",
    "    g = 8  : 1 + 2/8   = 1.25   bytes/weight -> 3.20x vs fp32.",
    "    On a memory-bound decode the weight bytes ARE the token latency, so g = 8 gives",
    "    up ~19% of the throughput you were quantizing to get, in exchange for a small",
    "    error improvement. Pick g = 128 unless the eval says otherwise; that is the",
    "    whole decision, and it is a throughput decision informed by a quality number.",
    "",
    "(d) A scheme that exploits the column structure better.",
    "    The band is the same columns in every row, i.e. it is a property of the INPUT",
    "    feature (activation) dimension, not of any output channel. Two options:",
    "      * mixed-precision decomposition (LLM.int8(), arXiv:2208.07339): keep those",
    "        columns in fp16 and int8 the rest, then add the two partial products. Cost:",
    "        two GEMMs instead of one, an irregular memory layout, and a runtime",
    "        threshold that decides which columns are outliers.",
    "      * per-channel activation scaling (AWQ, arXiv:2306.00978): fold a per-input",
    "        scale s_j into the weights (W_ij / s_j) and its inverse into the activations,",
    "        which shrinks the band's weight magnitudes before quantization. Cost: a",
    "        calibration pass to choose s from observed activations, and the scaling has",
    "        to be foldable into an adjacent op or it becomes an extra kernel.",
    "    Both beat group-wise here because they act along the axis the structure lives on.",
    "    Group-wise wins when the outliers are scattered rather than aligned to a column.",
]


if __name__ == "__main__":
    main()
