#!/usr/bin/env python3
"""Nonlinear probe diagnostic for JEPA encoders.

Loads a JEPA checkpoint, encodes a batch of held-out frames through the
frozen encoder, and compares two decoding strategies on the same embeddings:

  1. **Linear probe** — closed-form ridge regression from embedding → target
  2. **Nonlinear probe** — 2-layer MLP trained with SGD for 500 steps

If linear R² is low but nonlinear R² is high, the encoder still carries the
information but in a form that's not linearly decodable — the failure was a
probe methodology issue, not representation collapse. If both are low, the
information is genuinely gone.

This exists to answer a specific question raised after e030b:

  "Per-player probe R² for x/y dropped from 0.904 at epoch 1 to 0.305 at
  epoch 10. Is that prediction-shortcut collapse (encoder lost the info),
  or is it that our linear probe under-measures what a nonlinear decoder
  could still pull out of the embedding?"

The linear probe cares about how well `frame_embs @ W + b` can regress to
`x, y, percent, action_state`. The nonlinear probe cares about how well a
nonlinear function of the embedding can recover the same targets. If the
encoder is representing game state via linearly-separable features, both
should agree. If the encoder is representing game state through nonlinear
combinations (e.g., distances between subspaces), only the nonlinear probe
will recover it.

Usage:

    python -m scripts.nonlinear_probe_diagnostic \\
        --checkpoint ./checkpoints/e030b-jepa-rescale/final.pt \\
        --input-replay /path/to/canonical/fd-game \\
        --num-frames 2000

Prints a side-by-side table of linear vs nonlinear R² for each probe target.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.parse import load_game
from models.encoding import EncodingConfig, encode_player_frames
from models.jepa.model import JEPAWorldModel
from training.jepa_diagnostics import (
    linear_probe_regression_holdout,
    _probe_scale,
    _r2_and_mae,
)


logger = logging.getLogger("nonlinear_probe")


def _load_checkpoint(path: Path, device: torch.device) -> tuple[JEPAWorldModel, EncodingConfig]:
    logger.info("Loading checkpoint: %s", path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if ckpt.get("arch") != "jepa":
        raise ValueError(f"Expected arch='jepa', got {ckpt.get('arch')!r}")

    enc_cfg_dict = ckpt["encoding_config"]
    cfg = EncodingConfig(**{k: v for k, v in enc_cfg_dict.items() if hasattr(EncodingConfig, k)})

    model = JEPAWorldModel(
        cfg=cfg,
        embed_dim=ckpt.get("embed_dim", 192),
        history_size=ckpt.get("history_size", 3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def _load_and_encode_frames(
    replay_path: Path, cfg: EncodingConfig, num_frames: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a parsed game and produce (float_frames, int_frames) for the model.

    Note: the local _encode_game path has the state_flags/hitstun dimensional
    mismatch documented in issue #25. We zero-pad missing dims to match the
    model's expected float_per_player. This is the same workaround the
    visualizer uses — OOD for those dims but the trunk's first linear layer
    treats zero-padded slots as "no signal" rather than noise.
    """
    logger.info("Loading replay: %s", replay_path)
    game = load_game(str(replay_path))
    logger.info("Replay has %d frames", game.num_frames)

    p0 = encode_player_frames(game.p0, cfg)
    p1 = encode_player_frames(game.p1, cfg)

    cont0 = p0["continuous"]
    cont1 = p1["continuous"]
    bin0 = p0["binary"]
    bin1 = p1["binary"]
    ctrl0 = p0["controller"]
    ctrl1 = p1["controller"]

    fp_expected = cfg.float_per_player
    fp_actual = cont0.shape[1] + bin0.shape[1] + ctrl0.shape[1]
    pad_dims = fp_expected - fp_actual
    if pad_dims > 0:
        logger.warning(
            "Zero-padding %d missing float dims per player (cfg expects %d, got %d). "
            "See issue #25 — state_flags/hitstun are declared in cfg but not produced "
            "by encode_player_frames.",
            pad_dims, fp_expected, fp_actual,
        )

    T = cont0.shape[0]
    if num_frames is not None and num_frames < T:
        T = num_frames

    pad0 = torch.zeros(T, max(0, pad_dims))
    pad1 = torch.zeros(T, max(0, pad_dims))

    p0_float = torch.cat([cont0[:T], bin0[:T], ctrl0[:T], pad0], dim=1)
    p1_float = torch.cat([cont1[:T], bin1[:T], ctrl1[:T], pad1], dim=1)
    float_frames = torch.cat([p0_float, p1_float], dim=1)  # (T, 2*fp)

    # Int frames: [p0_cats, p1_cats, stage]
    int_parts = [
        p0["action"][:T, None], p0["jumps_left"][:T, None], p0["character"][:T, None],
        p0["l_cancel"][:T, None], p0["hurtbox_state"][:T, None],
        p0["ground"][:T, None], p0["last_attack_landed"][:T, None],
    ]
    if cfg.state_age_as_embed:
        int_parts.append(p0["state_age_int"][:T, None])
    int_parts.extend([
        p1["action"][:T, None], p1["jumps_left"][:T, None], p1["character"][:T, None],
        p1["l_cancel"][:T, None], p1["hurtbox_state"][:T, None],
        p1["ground"][:T, None], p1["last_attack_landed"][:T, None],
    ])
    if cfg.state_age_as_embed:
        int_parts.append(p1["state_age_int"][:T, None])
    stage_col = torch.full((T, 1), int(game.stage), dtype=torch.long)
    stage_col = stage_col.clamp(0, cfg.stage_vocab - 1)
    int_parts.append(stage_col)
    int_frames = torch.cat(int_parts, dim=1)

    # Shape for encoder: (B=1, T, F/I)
    float_frames = float_frames.unsqueeze(0).to(device)
    int_frames = int_frames.unsqueeze(0).to(device)

    logger.info("Encoded shapes: float=%s int=%s", tuple(float_frames.shape), tuple(int_frames.shape))
    return float_frames, int_frames, cont0[:T], cont1[:T]


@torch.no_grad()
def _get_embeddings_and_targets(
    model: JEPAWorldModel, float_frames: torch.Tensor, int_frames: torch.Tensor,
    cfg: EncodingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run encoder, build probe targets. Returns (embeddings, target_dict)."""
    embs = model.encoder(float_frames, int_frames).squeeze(0)  # (T, D)

    T = embs.shape[0]
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    p0f = float_frames.squeeze(0)[:, :fp]  # (T, fp)
    p1f = float_frames.squeeze(0)[:, fp:2*fp]
    p0i = int_frames.squeeze(0)[:, :ipp]
    p1i = int_frames.squeeze(0)[:, ipp:2*ipp]

    targets = {
        "p0_percent": p0f[:, 0],
        "p0_x": p0f[:, 1],
        "p0_y": p0f[:, 2],
        "p0_shield": p0f[:, 3],
        "p1_percent": p1f[:, 0],
        "p1_x": p1f[:, 1],
        "p1_y": p1f[:, 2],
        "p1_shield": p1f[:, 3],
        "p0_action": p0i[:, 0].long(),
        "p1_action": p1i[:, 0].long(),
    }
    return embs, targets


class MLPProbe(nn.Module):
    """Small 2-layer MLP probe for nonlinear decoding of game state from embeddings.

    Keep it small: 192 → 64 → 1 with ReLU. Small enough that if it can fit
    the data, the encoder genuinely had the information. If we needed a
    massive probe to fit, we'd be testing probe capacity, not encoder fidelity.
    """

    def __init__(self, emb_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def nonlinear_probe_regression(
    fit_embs: torch.Tensor, fit_y: torch.Tensor,
    eval_embs: torch.Tensor, eval_y: torch.Tensor,
    scale: float = 1.0, steps: int = 500, lr: float = 1e-3,
) -> dict[str, float]:
    """Fit a 2-layer MLP probe on fit_embs → fit_y, evaluate on eval_embs → eval_y.

    Returns R² and MAE in game units, same format as linear_probe_regression_holdout.
    """
    device = fit_embs.device
    probe = MLPProbe(fit_embs.shape[1]).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    fit_y = fit_y.float()
    eval_y = eval_y.float()

    probe.train()
    with torch.enable_grad():
        for _ in range(steps):
            pred = probe(fit_embs)
            loss = F.mse_loss(pred, fit_y)
            optim.zero_grad()
            loss.backward()
            optim.step()

    probe.eval()
    with torch.no_grad():
        eval_pred = probe(eval_embs)
        return _r2_and_mae(eval_y, eval_pred, scale)


class MLPClassProbe(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nonlinear_probe_classification(
    fit_embs: torch.Tensor, fit_y: torch.Tensor,
    eval_embs: torch.Tensor, eval_y: torch.Tensor,
    num_classes: int, steps: int = 500, lr: float = 1e-3,
) -> float:
    device = fit_embs.device
    probe = MLPClassProbe(fit_embs.shape[1], num_classes).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    fit_y = fit_y.long()
    eval_y = eval_y.long()

    probe.train()
    with torch.enable_grad():
        for _ in range(steps):
            logits = probe(fit_embs)
            loss = F.cross_entropy(logits, fit_y)
            optim.zero_grad()
            loss.backward()
            optim.step()

    probe.eval()
    with torch.no_grad():
        eval_logits = probe(eval_embs)
        pred = eval_logits.argmax(dim=-1)
        return (pred == eval_y).float().mean().item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-replay", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=2000)
    parser.add_argument("--start-frame", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # Load model + frames
    model, cfg = _load_checkpoint(args.checkpoint, device)
    float_frames, int_frames, _, _ = _load_and_encode_frames(
        args.input_replay, cfg, args.start_frame + args.num_frames, device,
    )
    # Slice the window we want
    float_frames = float_frames[:, args.start_frame:args.start_frame + args.num_frames]
    int_frames = int_frames[:, args.start_frame:args.start_frame + args.num_frames]
    logger.info("Window shape: float=%s int=%s", tuple(float_frames.shape), tuple(int_frames.shape))

    # Encode + get targets
    embs, targets = _get_embeddings_and_targets(model, float_frames, int_frames, cfg)
    T, D = embs.shape
    logger.info("Embeddings: (%d, %d)", T, D)

    # Split 50/50 for fit/eval (use temporally disjoint halves to reduce
    # sample correlation — same reasoning as the stride-sampled holdout in
    # training/jepa_diagnostics.py, scaled down for this simpler test)
    split = T // 2
    fit_embs = embs[:split]
    eval_embs = embs[split:]
    logger.info("Fit: %d samples | Eval: %d samples", split, T - split)

    print()
    print("=" * 72)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Replay: {args.input_replay.name}")
    print(f"  Window: frames [{args.start_frame}, {args.start_frame + args.num_frames})")
    print(f"  Fit/Eval split: {split} / {T - split}")
    print("=" * 72)
    print()
    print(f"{'target':<15} {'linear R²':>12} {'nonlin R²':>12} {'lin MAE':>10} {'nonlin MAE':>12}  unit")
    print("-" * 72)

    # Regression probes — per-player continuous
    reg_keys = ["p0_percent", "p0_x", "p0_y", "p0_shield",
                "p1_percent", "p1_x", "p1_y", "p1_shield"]
    for key in reg_keys:
        fit_y = targets[key][:split]
        eval_y = targets[key][split:]

        scale = _probe_scale(key, cfg)
        unit = (
            "pct" if "percent" in key
            else "px" if key.endswith("_x") or key.endswith("_y")
            else "pct" if "shield" in key
            else "u"
        )

        lin = linear_probe_regression_holdout(fit_embs, fit_y, eval_embs, eval_y, scale=scale)
        nonlin = nonlinear_probe_regression(fit_embs, fit_y, eval_embs, eval_y, scale=scale)

        # Guard against zero-variance eval split (R² undefined — report as —)
        var_y = float(eval_y.float().var().item())
        if var_y < 1e-8:
            lin_r2 = nonlin_r2 = "—"
        else:
            lin_r2 = f"{lin['r2']:>12.4f}"
            nonlin_r2 = f"{nonlin['r2']:>12.4f}"

        print(f"{key:<15} {lin_r2}  {nonlin_r2}  "
              f"{lin['mae_units']:>10.3f}  {nonlin['mae_units']:>12.3f}  {unit}")

    # Classification probes — action state
    print()
    print(f"{'target':<15} {'linear acc':>12} {'nonlin acc':>12}  note")
    print("-" * 72)
    for key in ["p0_action", "p1_action"]:
        from training.jepa_diagnostics import linear_probe_classification_holdout
        fit_y = targets[key][:split]
        eval_y = targets[key][split:]
        lin_acc = linear_probe_classification_holdout(
            fit_embs, fit_y, eval_embs, eval_y, num_classes=cfg.action_vocab,
        )
        nonlin_acc = nonlinear_probe_classification(
            fit_embs, fit_y, eval_embs, eval_y, num_classes=cfg.action_vocab,
        )
        print(f"{key:<15} {lin_acc:>12.4f}  {nonlin_acc:>12.4f}  "
              f"(random = {1.0/cfg.action_vocab:.4f})")

    print()
    print("Interpretation guide:")
    print("  linear R² ≈ nonlin R² → encoder is linearly decodable, numbers are real")
    print("  linear R² « nonlin R² → information is there but not linear; linear probe underreports")
    print("  both low, both ≈ 0   → information genuinely absent; encoder collapsed fine detail")
    print("  both high, both ≈ 1  → encoder is faithful, all's well")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
