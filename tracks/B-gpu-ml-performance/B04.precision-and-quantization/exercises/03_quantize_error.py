"""Exercise 3 - group-wise quantization, and what it actually buys.

`tools/quantize_demo.py` showed that per-channel scales beat one per-tensor scale when a
whole CHANNEL is an outlier. This drill asks the next question: what if the outlier lives
*inside* a channel?

The weight matrix below has a narrow band of large-magnitude columns (a "wide" feature
region) shared by every row. A per-channel scale is set by that band and is therefore too
coarse for the other 95% of the row. A group-wise scale confines the damage to the groups
that actually contain the band - which is why every shipped int4 kernel is group-wise.

Your job:
  TODO 1  implement per-channel symmetric int8 quantization
  TODO 2  implement group-wise symmetric int8 quantization
  TODO 3  account for the scale bytes honestly (fp16 scales, as real kernels use)
  TODO 4  read the sweep and answer the questions at the bottom of this file

The harness runs as-is and prints "not implemented" rows until you fill the TODOs in.
That gap is the exercise. Reference answers in
../solutions/03_quantize_error_solution.py

Run: python 03_quantize_error.py
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - environment guard, not logic
    raise SystemExit(
        "03_quantize_error.py needs numpy (its only dependency).\n"
        "  install it with:  python -m pip install numpy"
    )

SYM_MAX = 127          # symmetric int8 uses [-127, 127] so that 0.0 maps exactly to 0
TINY = 1e-12
BAND = slice(100, 108)  # the 8 wide-range columns


def make_structured_weights(rows: int = 64, cols: int = 256, seed: int = 0):
    """Small Gaussian weights, plus a band of large columns shared by every row."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.05, size=(rows, cols)).astype(np.float32)
    w[:, BAND] = rng.normal(0.0, 4.0, size=(rows, BAND.stop - BAND.start)).astype(np.float32)
    return w


def quantize_per_tensor(w):
    """Reference implementation: ONE scale for the whole matrix. Provided as the baseline.

    Returns (dequantized_matrix, number_of_scale_values).
    """
    scale = max(float(np.abs(w).max()), TINY) / SYM_MAX
    q = np.clip(np.rint(w / scale), -SYM_MAX, SYM_MAX)
    return (q * scale).astype(np.float32), 1


def quantize_per_channel(w):
    """TODO 1: one scale per row.

    Steps:
      * take the max absolute value along axis 1 with keepdims=True so it broadcasts
      * divide by SYM_MAX to get the step size, and floor it with TINY so an all-zero
        row cannot produce a division by zero
      * round-to-nearest, clip to [-SYM_MAX, SYM_MAX], multiply back by the scale
      * return (dequantized, number_of_scales)  -- the count matters for TODO 3
    """
    return None


def quantize_per_group(w, group: int):
    """TODO 2: one scale per contiguous run of `group` weights within a row.

    Steps:
      * reject a `group` that does not divide w.shape[1] (raise ValueError - a silently
        truncated last group is a real bug in real kernels)
      * reshape to (rows, cols // group, group) so the LAST axis is the set of values
        that share a scale, then reuse the logic from TODO 1 on that axis
      * reshape back to w.shape before returning
      * return (dequantized, number_of_scales)
    """
    return None


def total_bytes(n_weights: int, n_scales: int) -> int:
    """TODO 3: bytes for int8 codes plus fp16 scales.

    Shipped int4/int8 kernels store the scales in fp16, not fp32 - the scale only needs
    to be accurate to a fraction of a quantization step, so 11 mantissa bits is plenty
    and the memory saving is real. Return: 1 byte per weight + 2 bytes per scale.
    """
    return 0


def report(name, result, w, x):
    if result is None:
        print(f"{name:<26} not implemented yet (fill in the TODO)")
        return
    deq, n_scales = result
    rel_w = float(np.linalg.norm(w - deq) / np.linalg.norm(w))
    band_mask = np.zeros(w.shape[1], dtype=bool)
    band_mask[BAND] = True
    err = np.abs(w - deq)
    # Both the worst case and the typical case outside the band, because for this
    # matrix they tell very different stories - see question (a).
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

    print("=== Exercise 3 - group-wise int8 quantization ===\n")
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
        result = quantize_per_group(w, g)
        report(f"per-group g={g}", result, w, x)

    print("\nTODO 4 - answer these, in a comment at the bottom of this file:")
    print("  a) Going from per-channel to g=32, 'rest rms err' improves substantially")
    print("     but 'rest max err' hardly moves at all. Which specific columns are still")
    print(f"     badly quantized, and why? (The band is columns {BAND.start}..{BAND.stop - 1}.)")
    print("     Then: what would change if the band were 32-column aligned?")
    print("  b) 'rel err W' keeps improving as g shrinks while 'vs fp32' keeps falling.")
    print("     Pick the g you would ship, and state the rule you used - not the value.")
    print("  c) Compute bytes per weight at g=128 and at g=8. Which one would you pick for")
    print("     a memory-bound decode, and does the error column change your mind?")
    print("  d) The band sits in the same columns of every row. Name a scheme that exploits")
    print("     that structure better than group-wise does, and say what it costs you.")


if __name__ == "__main__":
    main()

# TODO 4 answers:
# a)
# b)
# c)
# d)
