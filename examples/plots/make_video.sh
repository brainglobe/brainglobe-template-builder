#!/usr/bin/env bash
# Convert frames -> mp4 -> looping gif.
# Meant to be run after `atlas_annotations.py` (which generates the frames).
# Run from the `video/` directory (containing `frames/slice_%03d.png`).
set -euo pipefail

MP4=output.mp4
GIF=output.gif
PALETTE=palette.png
FPS_MP4=60
FPS_GIF=30
GIF_WIDTH=480

# 1. Frames -> mp4
ffmpeg -y -framerate "$FPS_MP4" -i frames/slice_%03d.png \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 \
    -crf 18 -pix_fmt yuv420p "$MP4"

# 2. mp4 -> palette
ffmpeg -y -i "$MP4" \
    -vf "fps=${FPS_GIF},scale=${GIF_WIDTH}:-1:flags=lanczos,palettegen" \
    "$PALETTE"

# 3. mp4 + palette -> looping gif
ffmpeg -y -i "$MP4" -i "$PALETTE" \
    -lavfi "fps=${FPS_GIF},scale=${GIF_WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse" \
    -loop 0 "$GIF"

rm -f "$PALETTE"
echo "Done: $MP4, $GIF"
