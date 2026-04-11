#!/usr/bin/env python3
"""Run the JEPA identity diagnostic suite on a trained checkpoint.

Callable post-hoc against a saved JEPA checkpoint, independent of the training
loop. Intended for two uses:

    1. Validate a checkpoint after training without re-running the whole loop.
    2. Smoke-test the diagnostic suite itself during development (tiny
       synthetic data + freshly-initialized model).

Usage:
    # Against a real checkpoint + pre-encoded data
    python -m scripts.run_jepa_diagnostics \\
        --checkpoint checkpoints/e028a-jepa-baseline/best.pt \\
        --encoded-file /data/encoded-e012-fd-top5.pt

    # Smoke test with random weights + random inputs
    python -m scripts.run_jepa_diagnostics --smoke

See docs/jepa-data-flow.md for what the diagnostics measure and
training/jepa_diagnostics.py for the implementations.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoding import EncodingConfig
from models.jepa.model import JEPAWorldModel
from training.jepa_diagnostics import run_diagnostic_suite


logger = logging.getLogger("jepa_diagnostics")


def _load_checkpoint(path: str, device: torch.device) -> tuple[JEPAWorldModel, EncodingConfig]:
    """Load a JEPAWorldModel checkpoint.

    The checkpoint schema written by training/jepa_trainer.py:
        {"model_state_dict", "encoding_config" (dict), "history_size",
         "embed_dim", "arch"="jepa", ...}
    """
    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if ckpt.get("arch") != "jepa":
        raise ValueError(
            f"Checkpoint arch is {ckpt.get('arch')!r}, expected 'jepa'. "
            f"This script only handles JEPA checkpoints."
        )

    enc_cfg_dict = ckpt["encoding_config"]
    enc_cfg = EncodingConfig(
        **{k: v for k, v in enc_cfg_dict.items() if hasattr(EncodingConfig, k)}
    )

    model = JEPAWorldModel(
        cfg=enc_cfg,
        embed_dim=ckpt.get("embed_dim", 192),
        history_size=ckpt.get("history_size", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, enc_cfg


def _build_smoke_model(device: torch.device) -> tuple[JEPAWorldModel, EncodingConfig]:
    """Build a tiny JEPAWorldModel with default EncodingConfig for smoke tests.

    No checkpoint required. The model has random weights, so diagnostics
    results are noise — this mode exists to verify the plumbing runs
    without errors, not to measure anything meaningful.
    """
    enc_cfg = EncodingConfig(
        state_flags=True,
        hitstun=True,
        ctrl_threshold_features=True,
        multi_position=True,
        state_age_as_embed=True,
    )
    model = JEPAWorldModel(cfg=enc_cfg, embed_dim=64, history_size=3).to(device).eval()
    return model, enc_cfg


def _build_smoke_batch(
    cfg: EncodingConfig,
    B: int = 32,
    T: int = 4,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic batch matching the shapes the encoder expects."""
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    float_frames = torch.randn(B, T, 2 * fp, device=device) * 0.3

    # Int frames: indices must be within vocab bounds. Mix a few ditto rows
    # in so the swap test has some dittos to bucket.
    int_frames = torch.zeros(B, T, 2 * ipp + 1, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            p0_char = torch.randint(0, cfg.character_vocab, (1,)).item()
            p1_char = p0_char if b % 4 == 0 else torch.randint(0, cfg.character_vocab, (1,)).item()
            int_frames[b, t, 0] = torch.randint(0, cfg.action_vocab, (1,))
            int_frames[b, t, 1] = torch.randint(0, cfg.jumps_vocab, (1,))
            int_frames[b, t, 2] = p0_char
            int_frames[b, t, 3] = torch.randint(0, cfg.l_cancel_vocab, (1,))
            int_frames[b, t, 4] = torch.randint(0, cfg.hurtbox_vocab, (1,))
            int_frames[b, t, 5] = torch.randint(0, cfg.ground_vocab, (1,))
            int_frames[b, t, 6] = torch.randint(0, cfg.last_attack_vocab, (1,))
            if cfg.state_age_as_embed:
                int_frames[b, t, 7] = torch.randint(0, cfg.state_age_embed_vocab, (1,))
            int_frames[b, t, ipp + 0] = torch.randint(0, cfg.action_vocab, (1,))
            int_frames[b, t, ipp + 1] = torch.randint(0, cfg.jumps_vocab, (1,))
            int_frames[b, t, ipp + 2] = p1_char
            int_frames[b, t, ipp + 3] = torch.randint(0, cfg.l_cancel_vocab, (1,))
            int_frames[b, t, ipp + 4] = torch.randint(0, cfg.hurtbox_vocab, (1,))
            int_frames[b, t, ipp + 5] = torch.randint(0, cfg.ground_vocab, (1,))
            int_frames[b, t, ipp + 6] = torch.randint(0, cfg.last_attack_vocab, (1,))
            if cfg.state_age_as_embed:
                int_frames[b, t, ipp + 7] = torch.randint(0, cfg.state_age_embed_vocab, (1,))
            int_frames[b, t, -1] = 32  # stage = FD

    ctrl_inputs = torch.randn(B, T, cfg.ctrl_conditioning_dim, device=device) * 0.3
    return float_frames, int_frames, ctrl_inputs


def _build_batch_from_encoded(
    encoded_file: str,
    cfg: EncodingConfig,
    history_size: int,
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull a diagnostic batch from a pre-encoded .pt dataset.

    Matches the loading pattern in scripts/modal_train.py (reconstruct
    MeleeDataset directly from the saved tensors). Uses JEPAFrameDataset
    to produce (history_size + 1)-length subsequences.
    """
    from data.dataset import MeleeDataset
    from data.jepa_dataset import JEPAFrameDataset

    logger.info("Loading encoded data from %s", encoded_file)
    payload = torch.load(encoded_file, map_location="cpu", weights_only=False)

    dataset = MeleeDataset.__new__(MeleeDataset)
    dataset.cfg = cfg
    dataset.floats = payload["floats"]
    dataset.ints = payload["ints"]
    game_offsets = payload["game_offsets"]
    if isinstance(game_offsets, torch.Tensor):
        game_offsets = game_offsets.clone().numpy()
    dataset.game_offsets = game_offsets
    dataset.num_games = len(dataset.game_offsets) - 1
    dataset.total_frames = int(dataset.game_offsets[-1])
    dataset.game_lengths = [
        int(dataset.game_offsets[i + 1] - dataset.game_offsets[i])
        for i in range(dataset.num_games)
    ]
    logger.info("Dataset: %d games, %d frames", dataset.num_games, dataset.total_frames)

    # Use the last 10% of games as the diagnostic pool (held out from training)
    split_idx = max(1, int(dataset.num_games * 0.9))
    jepa_ds = JEPAFrameDataset(
        dataset,
        range(split_idx, dataset.num_games),
        history_size=history_size,
        num_preds=1,
    )

    # Pull a fixed contiguous slice of the valid_indices for reproducibility
    indices = np.arange(min(n_samples, len(jepa_ds)))
    float_frames, int_frames, ctrl_inputs = jepa_ds.get_batch(indices)
    return (
        float_frames.to(device),
        int_frames.to(device),
        ctrl_inputs.to(device),
    )


def _format_metrics(metrics: dict[str, float]) -> str:
    """Pretty-print diagnostic metrics grouped by section."""
    groups: dict[str, list[tuple[str, float]]] = {}
    for k, v in metrics.items():
        section = k.split("/", 1)[0] if "/" in k else "misc"
        groups.setdefault(section, []).append((k, v))

    lines = []
    for section in sorted(groups):
        lines.append(f"\n[{section}]")
        for k, v in sorted(groups[section]):
            short = k.split("/", 1)[1] if "/" in k else k
            if np_isnan(v):
                lines.append(f"  {short:<28s}  nan")
            else:
                lines.append(f"  {short:<28s}  {v:.4f}")
    return "\n".join(lines)


def np_isnan(x: float) -> bool:
    return x != x


def _interpret(metrics: dict[str, float]) -> str:
    """Short human-readable interpretation of the diagnostic results."""
    lines = []

    mean_swap = metrics.get("swap/mean_cosine_sim", float("nan"))
    ditto_swap = metrics.get("swap/ditto_cosine_sim", float("nan"))

    lines.append("# Interpretation")
    lines.append("")

    if np_isnan(mean_swap):
        lines.append("- Swap test: no data")
    elif mean_swap < 0.5:
        lines.append(f"- Swap test: HEALTHY (mean cos sim {mean_swap:.3f} < 0.5)")
    elif mean_swap < 0.9:
        lines.append(f"- Swap test: WEAK (mean cos sim {mean_swap:.3f}, watch ditto)")
    else:
        lines.append(
            f"- Swap test: COLLAPSED (mean cos sim {mean_swap:.3f} > 0.9) — "
            f"encoder has lost P0/P1 identity"
        )

    if not np_isnan(ditto_swap):
        if ditto_swap > 0.9:
            lines.append(
                f"- Ditto swap: COLLAPSED ({ditto_swap:.3f}) — in same-char matchups "
                f"the encoder cannot distinguish players at all"
            )
        else:
            lines.append(f"- Ditto swap: {ditto_swap:.3f}")

    # Probes
    probe_keys = [k for k in metrics if k.startswith("probe/") and k.endswith("_r2")]
    if probe_keys:
        p0_x = metrics.get("probe/p0_x_r2", float("nan"))
        p1_x = metrics.get("probe/p1_x_r2", float("nan"))
        rel_x = metrics.get("probe/rel_x_r2", float("nan"))
        if not (np_isnan(p0_x) or np_isnan(p1_x)):
            if p0_x > 0.8 and p1_x > 0.8:
                lines.append(f"- Per-player x probes: HEALTHY (P0 R²={p0_x:.3f}, P1 R²={p1_x:.3f})")
            elif abs(p0_x - p1_x) > 0.2:
                lines.append(
                    f"- Per-player x probes: ASYMMETRIC (P0 R²={p0_x:.3f}, P1 R²={p1_x:.3f}) — "
                    f"one player is decodable, the other is not"
                )
            else:
                lines.append(f"- Per-player x probes: WEAK (P0 R²={p0_x:.3f}, P1 R²={p1_x:.3f})")
        if not np_isnan(rel_x):
            if rel_x > 0.8:
                lines.append(f"- Relational rel_x probe: HEALTHY (R²={rel_x:.3f}) — cross-player binding present")
            else:
                lines.append(f"- Relational rel_x probe: WEAK (R²={rel_x:.3f}) — cross-player binding missing")

    straight = metrics.get("emergent/straightness", float("nan"))
    if not np_isnan(straight):
        lines.append(f"- Temporal straightness: {straight:.3f} (should INCREASE over training epochs)")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run JEPA identity diagnostics")
    parser.add_argument("--checkpoint", default=None, help="Path to JEPA checkpoint")
    parser.add_argument("--encoded-file", default=None, help="Path to pre-encoded .pt dataset")
    parser.add_argument("--smoke", action="store_true", help="Smoke test with random model/data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", default=None, help="Write metrics to JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    if args.smoke:
        logger.info("Smoke test mode — random model, random data")
        model, cfg = _build_smoke_model(device)
        float_frames, int_frames, ctrl_inputs = _build_smoke_batch(
            cfg, B=args.batch_size, T=4, device=device
        )
    else:
        if not args.checkpoint:
            parser.error("--checkpoint required when not in --smoke mode")
        model, cfg = _load_checkpoint(args.checkpoint, device)
        if args.encoded_file:
            float_frames, int_frames, ctrl_inputs = _build_batch_from_encoded(
                args.encoded_file, cfg, model.history_size, args.batch_size, device
            )
        else:
            logger.warning(
                "No --encoded-file provided; falling back to synthetic batch. "
                "Diagnostic numbers will be noise."
            )
            float_frames, int_frames, ctrl_inputs = _build_smoke_batch(
                cfg, B=args.batch_size, T=model.history_size + 1, device=device
            )

    logger.info(
        "Input shapes: floats=%s ints=%s ctrls=%s",
        tuple(float_frames.shape),
        tuple(int_frames.shape),
        tuple(ctrl_inputs.shape),
    )

    metrics = run_diagnostic_suite(model, float_frames, int_frames, ctrl_inputs)

    print(_format_metrics(metrics))
    print()
    print(_interpret(metrics))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote metrics to %s", args.json_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
