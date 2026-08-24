#!/usr/bin/env bash
# Render an animated SVG from assets/ into a GIF or MP4 for a surface that cannot
# embed SVG - PowerPoint, and some social video pipelines.
#
#   bash scripts/export_visual.sh coalescing            # 16:9 GIF, the default
#   bash scripts/export_visual.sh divergence mp4        # 16:9 MP4 for an editor
#   bash scripts/export_visual.sh reduction-tree gif 9x16
#
# Output goes to exports/, which is gitignored. THAT IS DELIBERATE AND IS NOT A BUG:
# committing the render would put an unreviewable binary in the tree and create a
# second artifact to regenerate on every content edit. The SVG stays the single
# source of truth; this produces a throwaway copy for one specific destination.
# See the "If you need a GIF" section of assets/README.md.
#
# This is NOT part of any build and nothing in the repo depends on it. It needs two
# tools the repo does not otherwise require:
#   - Chrome or Chromium, for --headless --screenshot
#   - ffmpeg, for assembling frames
# Both are checked for below rather than failing halfway through.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ASSET="${1:-}"
FORMAT="${2:-gif}"
ASPECT="${3:-16x9}"
FPS="${FPS:-25}"
SECONDS_TO_CAPTURE="${SECONDS_TO_CAPTURE:-6}"

if [[ -z "$ASSET" ]]; then
  echo "usage: bash scripts/export_visual.sh <asset-name> [gif|mp4] [16x9|9x16]" >&2
  echo "" >&2
  echo "available assets:" >&2
  for f in assets/*.svg; do echo "  $(basename "$f" .svg)" >&2; done
  exit 2
fi

SRC="assets/${ASSET}.svg"
[[ -f "$SRC" ]] || { echo "no such asset: $SRC" >&2; exit 1; }

CHROME="${CHROME:-}"
if [[ -z "$CHROME" ]]; then
  for c in google-chrome chromium chromium-browser msedge; do
    if command -v "$c" >/dev/null 2>&1; then CHROME="$c"; break; fi
  done
fi
[[ -n "$CHROME" ]] || {
  echo "error: no Chrome/Chromium on PATH. Set CHROME=/path/to/chrome and retry." >&2
  echo "       (this script is an optional one-off export, not a repo dependency)" >&2
  exit 1
}
command -v ffmpeg >/dev/null 2>&1 || {
  echo "error: ffmpeg is not on PATH. Install it, or screen-capture the SVG in a browser tab" >&2
  echo "       instead - for a talk or a Short that is usually good enough." >&2
  exit 1
}

case "$ASPECT" in
  16x9) W=1200; H=675 ;;
  9x16) W=1080; H=1920 ;;
  *) echo "unknown aspect: $ASPECT (use 16x9 or 9x16)" >&2; exit 2 ;;
esac

WORK="$(mktemp -d)"
OUT_DIR="$ROOT/exports"
mkdir -p "$OUT_DIR"
trap 'rm -rf "$WORK"' EXIT

# CSS animations in an SVG do not expose a seek API to --screenshot, so the frames
# are captured by loading the same file with a negative animation-delay that fast
# forwards it to a fixed offset. Deterministic, and no video-capture dependency.
TOTAL=$(( FPS * SECONDS_TO_CAPTURE ))
echo "capturing $TOTAL frames of $SRC at ${W}x${H}"

for (( i = 0; i < TOTAL; i++ )); do
  OFFSET=$(awk -v i="$i" -v fps="$FPS" 'BEGIN { printf "%.4f", i / fps }')
  FRAME_HTML="$WORK/frame.html"
  cat > "$FRAME_HTML" <<HTML
<!DOCTYPE html><meta charset="utf-8">
<style>
  html,body { margin:0; background:#0F172A; }
  object { display:block; width:${W}px; height:${H}px; }
  /* fast-forward every animation in the document to a fixed offset */
  * { animation-delay: -${OFFSET}s !important; animation-play-state: paused !important; }
</style>
<object type="image/svg+xml" data="file://$ROOT/$SRC"></object>
HTML
  "$CHROME" --headless --disable-gpu --hide-scrollbars \
            --window-size="${W},${H}" \
            --screenshot="$WORK/$(printf 'frame_%04d.png' "$i")" \
            "file://$FRAME_HTML" >/dev/null 2>&1
done

DEST="$OUT_DIR/${ASSET}-${ASPECT}.${FORMAT}"
if [[ "$FORMAT" == "gif" ]]; then
  ffmpeg -y -loglevel error -framerate "$FPS" -i "$WORK/frame_%04d.png" \
    -vf "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer" \
    "$DEST"
else
  ffmpeg -y -loglevel error -framerate "$FPS" -i "$WORK/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$DEST"
fi

echo "wrote $DEST"
echo "reminder: exports/ is gitignored. Re-run this rather than committing the file."
