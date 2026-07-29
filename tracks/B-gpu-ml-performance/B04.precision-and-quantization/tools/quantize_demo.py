"""B04 Lab 3 - int8 quantization schemes, and why the fancy ones exist.

No GPU and no framework: the whole lesson is arithmetic on one synthetic weight
matrix, so it runs anywhere numpy does. We quantize the SAME matrix six ways and
measure what each scheme costs:

  1. per-tensor  symmetric   - one scale for the whole matrix
  2. per-tensor  asymmetric  - one scale + one zero-point (handles a shifted range)
  3. per-channel symmetric   - one scale per output channel (row)
  4. per-channel asymmetric  - scale + zero-point per row
  5. per-channel symmetric, outlier row kept in fp16 - the LLM.int8() idea in
     miniature: mixed-precision decomposition instead of a better scale
  6. per-group   symmetric   - one scale per group of G weights inside a row, the
     scheme GPTQ/AWQ-style int4/int8 kernels actually ship (exercise 3 extends this)

The matrix is deliberately built with ONE outlier row whose weights are ~60x larger
than the rest. That single row is what makes per-tensor quantization fail, and
watching the per-tensor error collapse onto the typical rows is the point of the
whole exercise: a scale is only as good as the largest value it has to cover.

We report three different errors on purpose:
  * weight-space error   - easy to compute, and the one people quote
  * output-space error   - ||Wx - W_q x|| / ||Wx||, which is what the next layer sees
  * per-row error        - where the damage actually lives

Run: python tools/quantize_demo.py     (or: make python)
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment guard, not logic
    raise SystemExit(
        "quantize_demo.py needs numpy (it is the only dependency).\n"
        "  install it with:  python -m pip install numpy\n"
        "If you cannot install packages, the arithmetic is small enough to follow on\n"
        "paper - see README section 3 'Post-training quantization' for the formulas."
    )

INT8_MIN, INT8_MAX = -128, 127
# Symmetric quantization deliberately gives up the -128 code so the grid is
# symmetric about zero: with a range of [-127, 127] the value 0.0 maps exactly to
# the integer 0, which keeps zeros (padding, pruned weights, ReLU output) exact.
SYM_MAX = 127
TINY = 1e-12


def make_weights(rows: int = 64, cols: int = 256, outlier_row: int = 7, seed: int = 0):
    """A plausible weight matrix: mostly small, with one wide-dynamic-range channel."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.05, size=(rows, cols)).astype(np.float32)
    # The outlier channel. Real transformers show this in specific hidden dimensions
    # once the model is large enough (LLM.int8(), arXiv:2208.07339, section 4).
    w[outlier_row] = rng.normal(0.0, 3.0, size=cols).astype(np.float32)
    # A shifted channel too, so the asymmetric schemes have something to earn their
    # extra zero-point with.
    w[outlier_row + 1] = rng.uniform(0.8, 1.2, size=cols).astype(np.float32)
    return w


def quant_symmetric(w, axis=None, group=None):
    """Round to the [-127, 127] grid with a scale per tensor / row / group.

    Returns the dequantized matrix and the number of scale values used, because the
    scales are real bytes and a scheme that needs one per 8 weights is not 4x
    smaller no matter what the marketing says.
    """
    blocks, shape = _blocks(w, axis, group)
    amax = np.maximum(np.abs(blocks).max(axis=-1, keepdims=True), TINY)
    scale = amax / SYM_MAX
    q = np.clip(np.rint(blocks / scale), -SYM_MAX, SYM_MAX)
    return (q * scale).reshape(shape).astype(np.float32), scale.size


def quant_asymmetric(w, axis=None, group=None):
    """Affine (scale + zero-point) quantization over the full [-128, 127] range.

    The zero-point buys you the ability to cover a range that is not centred on
    zero - all-positive activations after a ReLU, for instance - at the cost of one
    extra integer per block and a slightly more expensive kernel.
    """
    blocks, shape = _blocks(w, axis, group)
    lo = blocks.min(axis=-1, keepdims=True)
    hi = blocks.max(axis=-1, keepdims=True)
    scale = np.maximum((hi - lo) / (INT8_MAX - INT8_MIN), TINY)
    zero_point = np.rint(-lo / scale) + INT8_MIN          # maps `lo` exactly onto -128
    q = np.clip(np.rint(blocks / scale) + zero_point, INT8_MIN, INT8_MAX)
    deq = (q - zero_point) * scale
    return deq.reshape(shape).astype(np.float32), 2 * scale.size


def _blocks(w, axis, group):
    """Reshape `w` so that the last axis is exactly the set of values sharing a scale."""
    if axis is None and group is None:
        return w.reshape(1, -1), w.shape                  # per-tensor: one block
    if group is not None:
        rows, cols = w.shape
        if cols % group:
            raise ValueError(f"group size {group} does not divide {cols} columns")
        return w.reshape(rows, cols // group, group), w.shape
    if axis != 1:
        raise ValueError("this demo quantizes per output channel, i.e. along axis 1")
    return w, w.shape                                     # per-row: the row IS the block


def errors(w, deq, x):
    """Weight-space and output-space error for one scheme."""
    denom_w = np.linalg.norm(w)
    rel_w = np.linalg.norm(w - deq) / denom_w
    y, yq = w @ x, deq @ x
    rel_y = np.linalg.norm(y - yq) / np.linalg.norm(y)
    per_row = np.abs(w - deq).max(axis=1)
    return rel_w, rel_y, per_row


def main() -> None:
    w = make_weights()
    rows, cols = w.shape
    outlier = 7
    typical = [i for i in range(rows) if i not in (outlier, outlier + 1)]

    rng = np.random.default_rng(1)
    # A batch of activations, so the output-space error is an average over inputs
    # rather than one lucky vector.
    x = rng.normal(0.0, 1.0, size=(cols, 32)).astype(np.float32)

    print("=== B04 Lab 3 - int8 quantization schemes on one matrix ===\n")
    print(f"W: {rows} x {cols} fp32, {w.nbytes / 1024:.1f} KiB")
    print(f"  typical row |w|max : {np.abs(w[typical]).max():.4f}")
    print(f"  outlier row  |w|max : {np.abs(w[outlier]).max():.4f}   <-- row {outlier}")
    print(f"  whole-tensor |w|max : {np.abs(w).max():.4f}   <-- what a per-tensor scale "
          f"must cover")
    print(f"  per-tensor step     : {np.abs(w).max() / SYM_MAX:.6f} "
          f"(vs {np.abs(w[typical]).max() / SYM_MAX:.6f} if the outlier were gone)\n")

    schemes = [
        ("per-tensor  symmetric", lambda: quant_symmetric(w)),
        ("per-tensor  asymmetric", lambda: quant_asymmetric(w)),
        ("per-channel symmetric", lambda: quant_symmetric(w, axis=1)),
        ("per-channel asymmetric", lambda: quant_asymmetric(w, axis=1)),
        ("per-chan sym + fp16 outlier", lambda: _with_fp16_outlier(w, outlier)),
        ("per-group   symmetric g=32", lambda: quant_symmetric(w, group=32)),
    ]

    header = (f"{'scheme':<28} {'rel err W':>10} {'rel err Wx':>11} "
              f"{'max err (typ)':>14} {'max err (out)':>14} {'bytes':>9} {'vs fp32':>8}")
    print(header)
    print("-" * len(header))

    for name, fn in schemes:
        deq, n_params = fn()
        rel_w, rel_y, per_row = errors(w, deq, x)
        # int8 codes + fp32 scale/zero-point parameters. The outlier row in scheme 5
        # stays 2 bytes per weight, which is why its byte count is not w.size.
        payload = w.size if "fp16 outlier" not in name else w.size + cols
        total = payload + 4 * n_params
        print(f"{name:<28} {rel_w:10.4f} {rel_y:11.4f} {per_row[typical].max():14.6f} "
              f"{per_row[outlier]:14.6f} {total:9d} {w.nbytes / total:7.2f}x")

    print("\nHow to read this table")
    print("  1. Per-tensor vs per-channel is not a small effect. One scale has to span the")
    print("     outlier row, so the step size everywhere is ~60x too coarse and the typical")
    print("     rows - which are most of the matrix - are quantized into near-uniformity.")
    print("     Per-channel gives every row its own step and the typical-row error drops by")
    print("     about the ratio of the two |w|max values above.")
    print("  2. Asymmetric only helps a block whose range is not centred on zero (row 8")
    print("     here, drawn from [0.8, 1.2]). For zero-centred weights it buys almost")
    print("     nothing and costs a zero-point plus a slightly slower kernel - which is why")
    print("     weights are usually quantized symmetrically and activations asymmetrically.")
    print("  3. Keeping ONE row in fp16 fixes most of the damage for ~1/64 of the memory.")
    print("     That is LLM.int8()'s mixed-precision decomposition in miniature: outliers")
    print("     are rare, so paying full precision for them is cheap.")
    print("  4. Per-group beats per-channel here only slightly, and that is worth")
    print("     understanding rather than explaining away: these rows are i.i.d. Gaussian, so")
    print("     every group inside a row sees almost the same maximum and inherits almost the")
    print("     same scale. Group-wise pays off when the range VARIES INSIDE a row, which is")
    print("     what exercise 3 constructs. Meanwhile the cost is immediate - watch the 'vs")
    print("     fp32' column: g=32 with fp32 scales already spends 12.5% of the payload on")
    print("     scales, which is why shipped int4 kernels use fp16 scales and g = 64 or 128.")
    print("  5. Output error tracks weight error closely in this demo, because x is isotropic")
    print("     Gaussian and spreads the error evenly over all directions. Real activations")
    print("     are not isotropic - they concentrate in a few channels - so a weight-space")
    print("     norm can badly UNDER-predict what the next layer feels. That is the gap AWQ")
    print("     exploits (scale by observed activation magnitude, not weight magnitude) and")
    print("     the reason the decision rule in README section 6 is a task-level eval rather")
    print("     than either of these numbers.")


def _with_fp16_outlier(w, outlier_row: int):
    """Per-channel int8 everywhere, except one row kept in fp16 (LLM.int8()-style)."""
    deq, n_scales = quant_symmetric(w, axis=1)
    # float16 is a real 16-bit round-trip, not a copy: the point is that the outlier
    # row is cheap to keep at HIGHER precision, not that it is kept exactly.
    deq[outlier_row] = w[outlier_row].astype(np.float16).astype(np.float32)
    return deq, n_scales


if __name__ == "__main__":
    main()
