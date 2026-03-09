#!/usr/bin/env python3
"""Pre-encode large datasets in chunks, writing directly to disk.

Avoids OOM by never holding more than one chunk in RAM at a time.
Each chunk is saved separately, then a final pass concatenates via
numpy memory mapping (no full-dataset-in-RAM required).

Usage:
    .venv/bin/python worldmodel/scripts/pre_encode_chunked.py \
        --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
        --config worldmodel/experiments/mamba2-medium-gpu.yaml \
        --max-games 22000 \
        --output /tmp/encoded-22k.pt \
        --chunk-size 2000
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.dataset import MeleeDataset
from data.parse import load_game
from models.encoding import EncodingConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-games", type=int, default=22000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--stage", type=int, default=None, help="Filter by stage ID (e.g. 32 for FD)")
    parser.add_argument("--characters", type=int, nargs="+", default=None,
                        help="Filter: both players must be in this character ID set")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    enc_cfg_dict = cfg.get("encoding", {})
    enc_cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if v is not None})

    # Also read filters from config data section (CLI overrides)
    data_cfg = cfg.get("data", {})
    stage_filter = args.stage if args.stage is not None else data_cfg.get("stage_filter")
    char_filter = args.characters if args.characters is not None else data_cfg.get("character_filter")
    char_set = None
    if char_filter is not None:
        char_set = {char_filter} if isinstance(char_filter, int) else set(char_filter)

    dataset_path = Path(args.dataset)
    with open(dataset_path / "meta.json") as f:
        meta_list = json.load(f)

    training = []
    for e in meta_list:
        if not e.get("is_training", False):
            continue
        if stage_filter is not None and e.get("stage") != stage_filter:
            continue
        if char_set is not None:
            players = e.get("players", [])
            if not all(p.get("character") in char_set for p in players):
                continue
        training.append(e)
        if args.max_games and len(training) >= args.max_games:
            break

    hitbox_table = None
    if enc_cfg.hitbox_data:
        hb_path = Path(__file__).parent.parent / "research" / "hitbox-data" / "hitbox_table.json"
        with open(hb_path) as f:
            hitbox_table = json.load(f)["table"]
        print(f"Loaded hitbox table: {len(hitbox_table)} entries")

    filter_desc = []
    if stage_filter is not None:
        filter_desc.append(f"stage={stage_filter}")
    if char_set is not None:
        filter_desc.append(f"chars={sorted(char_set)}")
    filter_str = f" (filters: {', '.join(filter_desc)})" if filter_desc else ""
    print(f"Will encode {len(training)} games in chunks of {args.chunk_size}{filter_str}")

    # Phase 1: Encode chunks and save to temp files
    tmp_dir = Path(args.output).parent / f".encode_chunks_{os.getpid()}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_lengths = []
    total_frames = 0
    total_games = 0
    float_width = None
    int_width = None

    t_start = time.time()

    for chunk_idx, start in enumerate(range(0, len(training), args.chunk_size)):
        chunk_entries = training[start:start + args.chunk_size]
        t0 = time.time()

        games = []
        for entry in chunk_entries:
            game_path = dataset_path / "games" / entry["slp_md5"]
            if not game_path.exists():
                continue
            try:
                g = load_game(game_path, compression=entry.get("compression", "zlib"))
                games.append(g)
            except Exception:
                continue

        if not games:
            continue

        ds = MeleeDataset(games, enc_cfg, hitbox_table=hitbox_table)

        # Save chunk tensors to disk immediately
        chunk_path = tmp_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save({"floats": ds.floats, "ints": ds.ints, "lengths": ds.game_lengths}, chunk_path)

        if float_width is None:
            float_width = ds.floats.shape[1]
            int_width = ds.ints.shape[1]

        all_lengths.extend(ds.game_lengths)
        total_frames += ds.total_frames
        total_games += ds.num_games

        elapsed = time.time() - t0
        print(
            f"  Chunk {chunk_idx}: {ds.num_games} games, {ds.total_frames:,} frames "
            f"({elapsed:.1f}s) | total: {total_games} games, {total_frames:,} frames"
        )

        del games, ds
        gc.collect()

    print(f"\nPhase 1 done: {total_games} games, {total_frames:,} frames in {time.time() - t_start:.0f}s")

    # Phase 2: Concatenate chunks using numpy memory mapping
    print(f"Phase 2: Concatenating to {args.output} ({total_frames:,} × {float_width} floats, {int_width} ints)")

    # Create memory-mapped arrays for the final output
    float_mmap = np.memmap(
        str(tmp_dir / "floats.mmap"), dtype=np.float32, mode="w+",
        shape=(total_frames, float_width),
    )
    int_mmap = np.memmap(
        str(tmp_dir / "ints.mmap"), dtype=np.int64, mode="w+",
        shape=(total_frames, int_width),
    )

    offset = 0
    chunk_files = sorted(tmp_dir.glob("chunk_*.pt"))
    for i, chunk_path in enumerate(chunk_files):
        chunk = torch.load(chunk_path, weights_only=False)
        n = chunk["floats"].shape[0]
        float_mmap[offset:offset + n] = chunk["floats"].numpy()
        int_mmap[offset:offset + n] = chunk["ints"].numpy()
        offset += n
        os.unlink(chunk_path)  # Free disk space as we go
        if (i + 1) % 3 == 0 or i == len(chunk_files) - 1:
            print(f"  Merged {i + 1}/{len(chunk_files)} chunks ({offset:,}/{total_frames:,} frames)")

    # Phase 3: Save final .pt file
    # Use torch.from_numpy on the memmap directly (zero-copy view).
    # torch.save streams from the mmap backing file — no full copy in RAM.
    print("Phase 3: Saving final .pt file...")
    payload = {
        "floats": torch.from_numpy(float_mmap),
        "ints": torch.from_numpy(int_mmap),
        "game_offsets": torch.tensor(np.cumsum([0] + all_lengths)),
        "game_lengths": all_lengths,
        "num_games": total_games,
        "encoding_config": enc_cfg_dict,
    }

    torch.save(payload, args.output)

    # Cleanup mmaps before reporting (must del tensor refs first)
    del payload
    gc.collect()
    del float_mmap, int_mmap

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output} ({size_mb:.1f} MB)")

    for f in tmp_dir.glob("*.mmap"):
        os.unlink(f)
    tmp_dir.rmdir()
    print(f"Done in {time.time() - t_start:.0f}s total")


if __name__ == "__main__":
    main()
