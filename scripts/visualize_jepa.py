#!/usr/bin/env python3
"""Visualize a JEPA world model checkpoint by decoding latents via linear probes.

JEPA predicts in latent space with no decoder. This script fits a ridge-regression
linear probe from the frozen encoder's 192-dim latents back to game-state fields
(x, y, percent, shield, action_state, ...), then uses those probe weights to
"decode" either encoder embeddings (mode=encode) or autoregressive rollout
latents (mode=rollout) into frames the existing `viz/visualizer.html` can render.

Emits a comparison JSON ({comparison, tracks, meta}) with up to three tracks:
    Ground Truth, Encoder Recon, Rollout Recon

Usage:
    python scripts/visualize_jepa.py \
        --checkpoint ./checkpoints/e030a-jepa-baseline/latest.pt \
        --input-replay ./data/local_test_games/<hash> \
        --output-json ./viz/e030a-reconstruction.json \
        --mode both --horizon 30

The input replay must be a zlib-compressed parquet game file parseable by
`data.parse.load_game`. Copy one from nojohns-training/data/parsed/games/ to
`./data/local_test_games/` if none are present.

This is a diagnostic tool for a discarded checkpoint (e030a), not production code.
The probe-as-decoder pattern is cheap and mirrors how LeWM's paper generates
reconstructions — the probes used in training/jepa_diagnostics.py double as
our decoder here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import _encode_game
from data.parse import load_game
from models.encoding import EncodingConfig
from models.jepa.model import JEPAWorldModel


# ----------------------------------------------------------------------------
# Feature-layout adapter
# ----------------------------------------------------------------------------
#
# _encode_game() builds frames *without* the extra features the e030a checkpoint
# was trained with (state_flags=True adds 40 binary dims, hitstun=True adds 1
# continuous dim). The training data used a pre-encoded .pt file that was built
# with a different pipeline. Zero-fill those missing features so shapes match
# the trained encoder. The probe decoding is downstream and still learns a
# valid mapping from the resulting (slightly off-distribution) latents.


def _pad_float_frames_for_flags(
    fl: torch.Tensor, cfg: EncodingConfig
) -> torch.Tensor:
    """Zero-pad a (T, F_local) tensor from `_encode_game` up to (T, F_trained).

    Inserts a zero for `hitstun_remaining` after the 5 velocity dims (if cfg has
    hitstun=True) and appends 40 zeros for state_flags at the end of the binary
    block (if cfg has state_flags=True). Applies both per-player.
    """
    T = fl.shape[0]

    # Continuous layout per player (state_age_as_embed=True, which is true for e030a):
    #   [percent, x, y, shield, 5 velocity, hitlag, stocks, combo_count]
    # With hitstun=True we need one more continuous: dynamics block is
    # (hitlag, stocks, [combo], hitstun). The exact ordering doesn't matter
    # for a zero pad because the encoder's trunk is a dense linear layer.
    core_cont_without_hitstun = 4 + 5 + 2 + 1  # percent,x,y,shield + vel5 + hitlag,stocks + combo = 12
    cont_dim_expected = cfg.continuous_dim
    assert cfg.state_age_as_embed, "pad assumes state_age_as_embed=True"
    hitstun_pad = cont_dim_expected - core_cont_without_hitstun  # 1 if hitstun else 0

    base_binary = 3
    state_flags_pad = cfg.binary_dim - base_binary  # 40 if state_flags else 0

    fp_trained = cfg.float_per_player
    fp_local = core_cont_without_hitstun + base_binary + cfg.controller_dim  # 12+3+13 = 28
    assert fl.shape[1] == 2 * fp_local, (
        f"Unexpected local float layout: got {fl.shape[1]}, expected {2 * fp_local}"
    )

    def split_player(player_block: torch.Tensor) -> torch.Tensor:
        """Split one player's 28-dim block and pad to fp_trained=69."""
        cont = player_block[:, :core_cont_without_hitstun]     # (T, 12)
        binary = player_block[:, core_cont_without_hitstun : core_cont_without_hitstun + base_binary]  # (T, 3)
        ctrl = player_block[:, core_cont_without_hitstun + base_binary :]  # (T, 13)

        cont_padded = torch.cat(
            [cont, torch.zeros(T, hitstun_pad, dtype=cont.dtype)], dim=1
        )
        binary_padded = torch.cat(
            [binary, torch.zeros(T, state_flags_pad, dtype=binary.dtype)], dim=1
        )
        return torch.cat([cont_padded, binary_padded, ctrl], dim=1)

    p0_block = split_player(fl[:, :fp_local])
    p1_block = split_player(fl[:, fp_local:])
    padded = torch.cat([p0_block, p1_block], dim=1)

    assert padded.shape == (T, 2 * fp_trained), (
        f"padded float shape {padded.shape} != expected {(T, 2 * fp_trained)}"
    )
    return padded


def _extract_ctrl(float_frames: torch.Tensor, cfg: EncodingConfig) -> torch.Tensor:
    """Mirror JEPAFrameDataset._extract_ctrl on a batch. Returns (T, ctrl_conditioning_dim).

    Ctrl layout: [p0_ctrl(13), p1_ctrl(13), p0_analog_threshold(5), p1_analog_threshold(5)]
    (threshold features only added when cfg.ctrl_threshold_features=True).
    """
    fp = cfg.float_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim

    p0_ctrl = float_frames[:, ctrl_start:ctrl_end]
    p1_ctrl = float_frames[:, fp + ctrl_start : fp + ctrl_end]
    parts = [p0_ctrl, p1_ctrl]
    if cfg.ctrl_threshold_features:
        p0_analog = float_frames[:, ctrl_start : ctrl_start + 5]
        p1_analog = float_frames[:, fp + ctrl_start : fp + ctrl_start + 5]
        parts.append((p0_analog.abs() > 0.3).float())
        parts.append((p1_analog.abs() > 0.3).float())
    return torch.cat(parts, dim=-1)


# ----------------------------------------------------------------------------
# Checkpoint and model loading
# ----------------------------------------------------------------------------


def load_jepa_model(ckpt_path: Path, device: torch.device) -> tuple[JEPAWorldModel, EncodingConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") != "jepa":
        raise ValueError(f"arch mismatch: {ckpt.get('arch')!r} (expected 'jepa')")

    cfg_dict = ckpt["encoding_config"]
    cfg = EncodingConfig(
        **{k: v for k, v in cfg_dict.items() if hasattr(EncodingConfig, k)}
    )
    model = JEPAWorldModel(
        cfg=cfg,
        embed_dim=ckpt["embed_dim"],
        history_size=ckpt["history_size"],
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"])
    if missing or unexpected:
        raise RuntimeError(
            f"state dict mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.to(device).eval()
    return model, cfg


# ----------------------------------------------------------------------------
# Linear-probe decoder — ridge-regression from 192-dim latent → game-state field.
# ----------------------------------------------------------------------------


def _ridge_fit(X: torch.Tensor, y: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """Closed-form ridge regression; X includes a bias column. Returns w: (D+1,)."""
    D = X.shape[-1]
    A = X.T @ X + ridge * torch.eye(D, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ y)


def _ridge_fit_multi(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """Multi-output ridge. Y: (N, K). Returns W: (D+1, K)."""
    D = X.shape[-1]
    A = X.T @ X + ridge * torch.eye(D, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).item()


# Continuous probe targets: (name, float-frame index, scale-attr, game-units label)
CONT_TARGETS = [
    # (name, float-idx offset-style, scale attr, unit)
    ("p0_percent", 0, "percent_scale", "pct"),
    ("p0_x",       1, "xy_scale",      "px"),
    ("p0_y",       2, "xy_scale",      "px"),
    ("p0_shield",  3, "shield_scale",  "pct"),
    ("p1_percent", 0, "percent_scale", "pct"),
    ("p1_x",       1, "xy_scale",      "px"),
    ("p1_y",       2, "xy_scale",      "px"),
    ("p1_shield",  3, "shield_scale",  "pct"),
    # velocities — encoder embeds via controller-conditioned dynamics, worth probing
    ("p0_speed_x", 4, "velocity_scale", "u/f"),
    ("p0_speed_y", 5, "velocity_scale", "u/f"),
    ("p1_speed_x", 4, "velocity_scale", "u/f"),
    ("p1_speed_y", 5, "velocity_scale", "u/f"),
]


def fit_probes(
    fit_embs: torch.Tensor, fit_floats: torch.Tensor, fit_ints: torch.Tensor,
    eval_embs: torch.Tensor, eval_floats: torch.Tensor, eval_ints: torch.Tensor,
    cfg: EncodingConfig,
) -> tuple[dict, dict]:
    """Fit ridge probes on `fit_*`, report R²/MAE on `eval_*`.

    Returns (weights_dict, metrics_dict). weights_dict contains:
        "cont_W":    (D+1, K_cont) ridge weights for continuous targets
        "cont_targets":   list of (name, fp_index, scale) tuples in column order
        "p0_action_W": (D+1, action_vocab) for p0 action logits
        "p1_action_W": similar
    """
    fp = cfg.float_per_player
    D = fit_embs.shape[-1]

    # Bias column
    X_fit = torch.cat([fit_embs, torch.ones(fit_embs.shape[0], 1)], dim=-1)
    X_eval = torch.cat([eval_embs, torch.ones(eval_embs.shape[0], 1)], dim=-1)

    # --- Continuous multi-output ridge ---
    cont_targets_resolved = []
    fit_y_cols = []
    eval_y_cols = []
    for (name, idx, scale_attr, unit) in CONT_TARGETS:
        if name.startswith("p0_"):
            base = 0
        else:
            base = fp
        col = base + idx
        cont_targets_resolved.append((name, col, getattr(cfg, scale_attr), unit))
        fit_y_cols.append(fit_floats[:, col])
        eval_y_cols.append(eval_floats[:, col])
    fit_Y = torch.stack(fit_y_cols, dim=1)   # (N_fit, K)
    eval_Y = torch.stack(eval_y_cols, dim=1)
    cont_W = _ridge_fit_multi(X_fit, fit_Y, ridge=1.0)   # (D+1, K)

    eval_Y_pred = X_eval @ cont_W

    metrics = {}
    for i, (name, _, scale, unit) in enumerate(cont_targets_resolved):
        target_var = eval_Y[:, i].var().item()
        mae_normalized = (eval_Y[:, i] - eval_Y_pred[:, i]).abs().mean().item()
        mae_units = mae_normalized / scale if scale > 0 else mae_normalized
        # R² is only meaningful if target has nonzero variance on the eval split;
        # otherwise ss_tot → 0 and the result is numerically unstable garbage.
        # Flag degenerate targets explicitly instead of reporting -1e10 / 1.000.
        if target_var < 1e-6:
            metrics[name] = {
                "r2": float("nan"), "mae_units": mae_units, "unit": unit,
                "degenerate": True,
            }
        else:
            r2 = _r2(eval_Y[:, i], eval_Y_pred[:, i])
            metrics[name] = {
                "r2": r2, "mae_units": mae_units, "unit": unit,
                "degenerate": False,
            }

    # --- Action classification (multinomial linear, SGD) ---
    def _fit_action(slot_idx: int) -> tuple[torch.Tensor, float]:
        y_fit = fit_ints[:, slot_idx].long()
        y_eval = eval_ints[:, slot_idx].long()
        K = cfg.action_vocab
        W = torch.zeros(D, K, requires_grad=True)
        b = torch.zeros(K, requires_grad=True)
        opt = torch.optim.SGD([W, b], lr=0.1, momentum=0.9)
        with torch.enable_grad():
            for _ in range(300):
                logits = fit_embs @ W + b
                loss = F.cross_entropy(logits, y_fit)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            pred = (eval_embs @ W + b).argmax(dim=-1)
            acc = (pred == y_eval).float().mean().item()
        return torch.cat([W.detach(), b.detach().unsqueeze(0)], dim=0), acc

    p0_action_W, p0_action_acc = _fit_action(0)
    p1_action_W, p1_action_acc = _fit_action(cfg.int_per_player)
    metrics["p0_action"] = {"acc": p0_action_acc}
    metrics["p1_action"] = {"acc": p1_action_acc}

    weights = {
        "cont_W": cont_W,
        "cont_targets": cont_targets_resolved,
        "p0_action_W": p0_action_W,
        "p1_action_W": p1_action_W,
    }
    return weights, metrics


# ----------------------------------------------------------------------------
# Probe decoding → viz-schema dict
# ----------------------------------------------------------------------------


def decode_embeddings_to_frames(
    embs: torch.Tensor,
    weights: dict,
    cfg: EncodingConfig,
    gt_ints: torch.Tensor,
    stage: int,
) -> list[dict]:
    """Decode (T, D) latents into a viz-schema list of frames.

    For un-probed fields (jumps_left, hitlag, etc.) we use the ground-truth
    values from `gt_ints` — the visualizer tolerates their absence via
    `playerDefaults`, but using GT for those keeps the picture honest where
    we know the probe isn't contributing.

    `character` is a pass-through from gt_ints (character is effectively
    constant per-game in the encoding config's treatment).
    """
    T, D = embs.shape
    X = torch.cat([embs, torch.ones(T, 1)], dim=-1)
    cont_preds = X @ weights["cont_W"]   # (T, K)
    cont_targets = weights["cont_targets"]

    # Continuous → game units
    # cont_targets order: p0_{percent,x,y,shield,speed_x,speed_y}, p1_{same}
    decoded = {name: (cont_preds[:, i] / scale).tolist()
               for i, (name, _, scale, _unit) in enumerate(cont_targets)}

    # Action classification: argmax
    p0_action_logits = torch.cat([embs, torch.ones(T, 1)], dim=-1) @ weights["p0_action_W"]
    p1_action_logits = torch.cat([embs, torch.ones(T, 1)], dim=-1) @ weights["p1_action_W"]
    p0_action_pred = p0_action_logits.argmax(dim=-1).tolist()
    p1_action_pred = p1_action_logits.argmax(dim=-1).tolist()

    ipp = cfg.int_per_player
    p0_char_gt = gt_ints[:, 2].tolist()
    p1_char_gt = gt_ints[:, ipp + 2].tolist()
    p0_jumps_gt = gt_ints[:, 1].tolist()
    p1_jumps_gt = gt_ints[:, ipp + 1].tolist()

    frames = []
    for t in range(T):
        p0 = {
            "x": decoded["p0_x"][t],
            "y": decoded["p0_y"][t],
            "percent": max(0.0, decoded["p0_percent"][t]),
            "shield_strength": max(0.0, decoded["p0_shield"][t]),
            "speed_air_x": decoded["p0_speed_x"][t],
            "speed_y": decoded["p0_speed_y"][t],
            "speed_ground_x": 0.0,
            "speed_attack_x": 0.0,
            "speed_attack_y": 0.0,
            "state_age": 0.0,
            "hitlag": 0.0,
            "stocks": 4.0,
            "facing": 1,
            "on_ground": 1,
            "action_state": int(p0_action_pred[t]),
            "jumps_left": int(p0_jumps_gt[t]),
            "character": int(p0_char_gt[t]),
        }
        p1 = {
            "x": decoded["p1_x"][t],
            "y": decoded["p1_y"][t],
            "percent": max(0.0, decoded["p1_percent"][t]),
            "shield_strength": max(0.0, decoded["p1_shield"][t]),
            "speed_air_x": decoded["p1_speed_x"][t],
            "speed_y": decoded["p1_speed_y"][t],
            "speed_ground_x": 0.0,
            "speed_attack_x": 0.0,
            "speed_attack_y": 0.0,
            "state_age": 0.0,
            "hitlag": 0.0,
            "stocks": 4.0,
            "facing": 1,
            "on_ground": 1,
            "action_state": int(p1_action_pred[t]),
            "jumps_left": int(p1_jumps_gt[t]),
            "character": int(p1_char_gt[t]),
        }
        frames.append({"players": [p0, p1], "stage": stage})
    return frames


def ground_truth_frames(
    fl_padded: torch.Tensor,
    ints: torch.Tensor,
    cfg: EncodingConfig,
    stage: int,
) -> list[dict]:
    """Convert normalized float/int frames back to viz-schema game units.

    Uses ONLY the features we have (from dataset.py's _encode_game path);
    leaves state_flags/hitstun as zeros since we padded those.
    """
    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    T = fl_padded.shape[0]

    def get(col):
        return fl_padded[:, col].tolist()

    # Continuous offsets (matching _encode_game layout):
    #   0: percent, 1: x, 2: y, 3: shield,
    #   4: speed_air_x, 5: speed_y, 6: speed_ground_x, 7: speed_attack_x, 8: speed_attack_y,
    #   9: hitstun (zero pad), 10: hitlag, 11: stocks, 12: combo
    #   Note: if cfg.state_age_as_embed, state_age lives in ints, not floats.

    frames = []
    for t in range(T):
        p0 = {
            "x": fl_padded[t, 1].item() / cfg.xy_scale,
            "y": fl_padded[t, 2].item() / cfg.xy_scale,
            "percent": fl_padded[t, 0].item() / cfg.percent_scale,
            "shield_strength": fl_padded[t, 3].item() / cfg.shield_scale,
            "speed_air_x": fl_padded[t, 4].item() / cfg.velocity_scale,
            "speed_y": fl_padded[t, 5].item() / cfg.velocity_scale,
            "speed_ground_x": fl_padded[t, 6].item() / cfg.velocity_scale,
            "speed_attack_x": fl_padded[t, 7].item() / cfg.velocity_scale,
            "speed_attack_y": fl_padded[t, 8].item() / cfg.velocity_scale,
            "state_age": float(ints[t, 7].item()) if cfg.state_age_as_embed else 0.0,
            "hitlag": fl_padded[t, 10].item() / cfg.hitlag_scale,
            "stocks": fl_padded[t, 11].item() / cfg.stocks_scale,
            "facing": int(fl_padded[t, 13].item() > 0.5),  # binary[0] at idx 13 (after 13 cont)
            "on_ground": int(fl_padded[t, 15].item() > 0.5),  # binary[2]
            "action_state": int(ints[t, 0].item()),
            "jumps_left": int(ints[t, 1].item()),
            "character": int(ints[t, 2].item()),
        }
        p1 = {
            "x": fl_padded[t, fp + 1].item() / cfg.xy_scale,
            "y": fl_padded[t, fp + 2].item() / cfg.xy_scale,
            "percent": fl_padded[t, fp + 0].item() / cfg.percent_scale,
            "shield_strength": fl_padded[t, fp + 3].item() / cfg.shield_scale,
            "speed_air_x": fl_padded[t, fp + 4].item() / cfg.velocity_scale,
            "speed_y": fl_padded[t, fp + 5].item() / cfg.velocity_scale,
            "speed_ground_x": fl_padded[t, fp + 6].item() / cfg.velocity_scale,
            "speed_attack_x": fl_padded[t, fp + 7].item() / cfg.velocity_scale,
            "speed_attack_y": fl_padded[t, fp + 8].item() / cfg.velocity_scale,
            "state_age": float(ints[t, ipp + 7].item()) if cfg.state_age_as_embed else 0.0,
            "hitlag": fl_padded[t, fp + 10].item() / cfg.hitlag_scale,
            "stocks": fl_padded[t, fp + 11].item() / cfg.stocks_scale,
            "facing": int(fl_padded[t, fp + 13].item() > 0.5),
            "on_ground": int(fl_padded[t, fp + 15].item() > 0.5),
            "action_state": int(ints[t, ipp].item()),
            "jumps_left": int(ints[t, ipp + 1].item()),
            "character": int(ints[t, ipp + 2].item()),
        }
        frames.append({"players": [p0, p1], "stage": stage})
    return frames


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--input-replay", type=Path, required=True,
                    help="Path to a zlib parquet parsed game (load_game-compatible)")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--mode", choices=["encode", "rollout", "both"], default="both")
    ap.add_argument("--horizon", type=int, default=30,
                    help="Number of autoregressive rollout steps (mode=rollout/both)")
    ap.add_argument("--num-frames", type=int, default=300,
                    help="Number of ground-truth frames to use from the replay")
    ap.add_argument("--start-frame", type=int, default=500,
                    help="Index into the replay where the sequence starts")
    ap.add_argument("--fit-frac", type=float, default=0.8,
                    help="Fraction of frames used to fit the linear probes")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    # 1. Load model
    print(f"[1/6] Loading checkpoint: {args.checkpoint}")
    model, cfg = load_jepa_model(args.checkpoint, device)
    print(f"      arch=jepa embed_dim={model.embed_dim} history={model.history_size}")

    # 2. Load and encode game
    print(f"[2/6] Loading replay: {args.input_replay}")
    game = load_game(str(args.input_replay))
    print(f"      stage={game.stage} num_frames={game.num_frames} "
          f"p0_char={int(game.p0.character[0])} p1_char={int(game.p1.character[0])}")
    fl_local, ints = _encode_game(game, cfg)
    fl = _pad_float_frames_for_flags(fl_local, cfg)  # (T, 138)
    print(f"      floats (padded): {tuple(fl.shape)}  ints: {tuple(ints.shape)}")

    start = args.start_frame
    end = min(start + args.num_frames, game.num_frames)
    if end - start < model.history_size + max(args.horizon, 10):
        raise ValueError(
            f"Not enough frames: want {model.history_size + max(args.horizon, 10)} but slice is {end - start}"
        )
    fl_seq = fl[start:end].to(device)
    int_seq = ints[start:end].to(device)
    T = fl_seq.shape[0]
    print(f"      using frames [{start}, {end}) = {T} frames")

    # 3. Fit probes
    print(f"[3/6] Fitting linear probes (fit_frac={args.fit_frac})...")
    with torch.no_grad():
        all_embs = model.encoder(
            fl_seq.unsqueeze(0), int_seq.unsqueeze(0)
        ).squeeze(0)  # (T, D)
    split = int(T * args.fit_frac)
    weights, metrics = fit_probes(
        fit_embs=all_embs[:split], fit_floats=fl_seq[:split], fit_ints=int_seq[:split],
        eval_embs=all_embs[split:], eval_floats=fl_seq[split:], eval_ints=int_seq[split:],
        cfg=cfg,
    )

    # Report table
    print(f"\n{'target':<16}{'R²':>10}{'MAE':>12}{'unit':>8}  {'note':<10}")
    print("-" * 58)
    for (name, _col, _s, unit) in weights["cont_targets"]:
        m = metrics[name]
        note = "ZEROVAR" if m.get("degenerate") else ""
        r2_str = f"{m['r2']:>10.4f}" if not m.get("degenerate") else f"{'—':>10}"
        print(f"{name:<16}{r2_str}{m['mae_units']:>12.3f}{unit:>8}  {note:<10}")
    print("-" * 58)
    print(f"{'p0_action acc':<16}{metrics['p0_action']['acc']:>10.4f}")
    print(f"{'p1_action acc':<16}{metrics['p1_action']['acc']:>10.4f}")
    print()

    # 4. Ground-truth track (always emitted)
    print(f"[4/6] Building ground-truth frames...")
    gt_frames = ground_truth_frames(fl_seq.cpu(), int_seq.cpu(), cfg, game.stage)

    tracks = {"Ground Truth": gt_frames}
    default_track = "Ground Truth"

    # 5. Encoder reconstruction
    if args.mode in ("encode", "both"):
        print(f"[5/6] Encoder reconstruction (per-frame encode → probe decode)...")
        with torch.no_grad():
            enc_embs = all_embs.cpu()
        enc_frames = decode_embeddings_to_frames(
            enc_embs, weights, cfg, int_seq.cpu(), game.stage
        )
        tracks["Encoder Recon"] = enc_frames

    # 6. Rollout reconstruction
    if args.mode in ("rollout", "both"):
        print(f"[6/6] Autoregressive rollout (horizon={args.horizon})...")
        H = model.history_size
        if args.horizon + H > T:
            raise ValueError(f"horizon {args.horizon} + H {H} > available {T}")
        # Seed: first H frames
        float_ctx = fl_seq[:H].unsqueeze(0)   # (1, H, F)
        int_ctx = int_seq[:H].unsqueeze(0)    # (1, H, I)
        # Controllers: extract from raw float frames (pre-encoder representation)
        ctrls_all = _extract_ctrl(fl_seq, cfg)  # (T, C)
        initial_ctrls = ctrls_all[:H].unsqueeze(0)           # (1, H, C)
        ctrl_sequence = ctrls_all[H : H + args.horizon].unsqueeze(0)  # (1, N, C)

        with torch.no_grad():
            pred_embs = model.rollout(
                float_ctx=float_ctx,
                int_ctx=int_ctx,
                initial_ctrls=initial_ctrls,
                ctrl_sequence=ctrl_sequence,
            ).squeeze(0).cpu()  # (N, D)

        # Decode predicted latents; for the seed frames we use the GT encoder
        # reconstruction to keep the rollout track aligned.
        with torch.no_grad():
            seed_embs = model.encoder(float_ctx, int_ctx).squeeze(0).cpu()
        full_embs = torch.cat([seed_embs, pred_embs], dim=0)  # (H+N, D)
        rollout_len = full_embs.shape[0]
        rollout_frames = decode_embeddings_to_frames(
            full_embs, weights, cfg, int_seq[:rollout_len].cpu(), game.stage
        )
        tracks["Rollout Recon"] = rollout_frames
        default_track = "Rollout Recon"

    # Keep each track at its natural length. The visualizer displays one
    # track at a time and scrubs by index within whichever is active, so
    # different lengths are fine. Rollout is short (H + horizon); encoder
    # and ground truth cover the full requested window.
    track_lens = {k: len(v) for k, v in tracks.items()}

    # 7. Emit comparison JSON
    out = {
        "comparison": True,
        "default_track": default_track,
        "tracks": tracks,
        "meta": {
            "checkpoint": str(args.checkpoint.name),
            "arch": "jepa",
            "epoch": 3,  # e030a stopped at epoch 3
            "seed_game": args.input_replay.name,
            "start_frame": start,
            "track_lens": track_lens,
            "num_frames": track_lens[default_track],
            "history_size": model.history_size,
            "horizon": args.horizon if args.mode in ("rollout", "both") else 0,
            "mode": args.mode,
            "device": str(device),
            "probe_metrics": {
                k: v for k, v in metrics.items() if isinstance(v, dict)
            },
        },
    }
    # Python's json.dump writes NaN / Infinity by default but those are not
    # valid JSON per the spec — JavaScript's JSON.parse rejects them, so the
    # viz HTML fails to load with "Unexpected token 'N'". Walk the output
    # once and replace any non-finite floats with None (JSON null). Affects
    # probe metrics for ZEROVAR targets where R² is undefined.
    def _sanitize(obj):
        import math
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj

    out = _sanitize(out)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    lens_str = ", ".join(f"{k}={v}" for k, v in track_lens.items())
    print(f"\nWrote {args.output_json} ({lens_str})")


if __name__ == "__main__":
    main()
