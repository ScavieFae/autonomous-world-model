#!/usr/bin/env python3
"""Convert a parsed Slippi replay into visualizer JSON.

Usage:
    python viz/replay_to_json.py <parsed-game-file> [-o output.json] [--start 0] [--frames 300]

The parsed game file is a zlib-compressed parquet file from nojohns-training/data/parsed-v2/games/.
Output is a JSON array of frames matching the visualizer's expected format.

Example:
    # Pick a game from the parsed dataset
    python viz/replay_to_json.py ../nojohns-training/data/parsed-v2/games/00058755679eb4170b0102180884b670

    # Extract 5 seconds starting at frame 1000
    python viz/replay_to_json.py ../nojohns-training/data/parsed-v2/games/00058755679eb4170b0102180884b670 \
        --start 1000 --frames 300 -o viz/fixtures/real-sequence.json
"""

import argparse
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


BUTTON_NAMES = ["A", "B", "X", "Y", "Z", "L", "R", "D_UP"]


def safe_field(struct, name, num_frames, dtype=np.float32):
    try:
        return np.array(struct.field(name).to_pylist(), dtype=dtype)
    except (KeyError, Exception):
        return np.zeros(num_frames, dtype=dtype)


def load_and_convert(path, compression="zlib", start=0, num_frames=None):
    """Load a parsed game and convert to visualizer JSON format."""
    with open(path, "rb") as f:
        data = f.read()

    if compression == "zlib":
        data = zlib.decompress(data)

    table = pq.read_table(io.BytesIO(data))
    root = table.column("root").combine_chunks()
    total_frames = len(root)

    stage = root.field("stage")[0].as_py()

    # Clamp range
    end = min(start + (num_frames or total_frames), total_frames)
    start = max(0, min(start, total_frames - 1))

    frames = []
    for i in range(start, end):
        frame = {"players": [], "stage": int(stage)}

        for port in ["p0", "p1"]:
            p = root.field(port)

            # Position
            x = float(p.field("x")[i].as_py())
            y = float(p.field("y")[i].as_py())

            # Core state
            percent = float(p.field("percent")[i].as_py())
            shield = float(p.field("shield_strength")[i].as_py())
            facing_raw = p.field("facing")[i].as_py()
            facing = 1 if facing_raw else 0
            on_ground_raw = p.field("on_ground")[i].as_py()
            on_ground = 1 if on_ground_raw else 0

            # Categoricals
            action = int(p.field("action")[i].as_py())
            jumps = int(p.field("jumps_left")[i].as_py())
            character = int(safe_field(p, "character", total_frames, np.int64)[i])

            # Velocity (v2 fields — zeros if old schema)
            speed_air_x = float(safe_field(p, "speed_air_x", total_frames)[i])
            speed_y = float(safe_field(p, "speed_y", total_frames)[i])
            speed_ground_x = float(safe_field(p, "speed_ground_x", total_frames)[i])
            speed_attack_x = float(safe_field(p, "speed_attack_x", total_frames)[i])
            speed_attack_y = float(safe_field(p, "speed_attack_y", total_frames)[i])

            # Dynamics (v2 fields)
            state_age = float(safe_field(p, "state_age", total_frames)[i])
            hitlag = float(safe_field(p, "hitlag", total_frames)[i])
            stocks = float(safe_field(p, "stocks", total_frames)[i])

            frame["players"].append({
                "x": round(x, 3),
                "y": round(y, 3),
                "percent": round(percent, 1),
                "shield_strength": round(shield, 2),
                "speed_air_x": round(speed_air_x, 4),
                "speed_y": round(speed_y, 4),
                "speed_ground_x": round(speed_ground_x, 4),
                "speed_attack_x": round(speed_attack_x, 4),
                "speed_attack_y": round(speed_attack_y, 4),
                "state_age": round(state_age, 1),
                "hitlag": round(hitlag, 1),
                "stocks": int(stocks),
                "facing": facing,
                "on_ground": on_ground,
                "action_state": action,
                "jumps_left": jumps,
                "character": character,
            })

        frames.append(frame)

    return frames, total_frames


def main():
    parser = argparse.ArgumentParser(description="Convert parsed Slippi replay to visualizer JSON")
    parser.add_argument("input", type=Path, help="Path to parsed game file (zlib-compressed parquet)")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output JSON path (default: stdout)")
    parser.add_argument("--start", type=int, default=0, help="Start frame (default: 0)")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames to extract (default: all)")
    parser.add_argument("--no-compression", action="store_true", help="Input is not zlib-compressed")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    compression = "none" if args.no_compression else "zlib"
    frames, total = load_and_convert(args.input, compression, args.start, args.frames)

    print(f"Extracted {len(frames)} frames (of {total} total)", file=sys.stderr)

    output = json.dumps(frames, separators=(",", ":"))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
