"""Shared autoregressive utilities for rollout, evaluation, and training.

Contains the frame reconstruction logic used by:
- rollout.py (demo generation)
- eval_rollout.py (rollout coherence evaluation)
- Self-Forcing training (e018a, future)

The reconstruction must be identical everywhere to ensure the eval
measures the same behavior as the demos.
"""

import torch

from models.encoding import EncodingConfig


def reconstruct_frame(
    preds: dict[str, torch.Tensor],
    prev_float: torch.Tensor,
    prev_int: torch.Tensor,
    ctrl_float: torch.Tensor,
    cfg: EncodingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build next frame tensors from model predictions.

    Works with arbitrary leading batch dimensions (single frame or batched).

    Args:
        preds: Model output dict with keys:
            continuous_delta: (..., 8) — [p0_Δpct, p0_Δx, p0_Δy, p0_Δshield, p1_...]
            binary_logits: (..., 6) — [p0_facing, p0_invuln, p0_on_ground, p1_...]
            p0_action_logits: (..., 400)
            p0_jumps_logits: (..., 8)
            p1_action_logits: (..., 400)
            p1_jumps_logits: (..., 8)
        prev_float: (..., F) — previous frame's float tensor.
        prev_int: (..., I) — previous frame's int tensor.
        ctrl_float: (..., ctrl_dim*2) — controller input for this frame
            [p0_ctrl(13), p1_ctrl(13)].
        cfg: Encoding config for dimension offsets.

    Returns:
        (next_float, next_int) with same leading dimensions as input.
    """
    fp = cfg.float_per_player
    cd = cfg.continuous_dim
    bd = cfg.binary_dim
    ccd = cfg.core_continuous_dim  # 4: percent, x, y, shield
    ipp = cfg.int_per_player

    next_float = prev_float.clone()
    next_int = prev_int.clone()

    # --- Continuous deltas (first 4 per player: percent, x, y, shield) ---
    delta = preds["continuous_delta"]  # (..., 8)
    next_float[..., :ccd] += delta[..., :ccd]
    next_float[..., fp:fp + ccd] += delta[..., ccd:]

    # --- Velocity deltas (indices 4:9 per player) ---
    if "velocity_delta" in preds:
        vel_start = ccd  # 4
        vel_end = vel_start + cfg.velocity_dim  # 9
        vel_d = preds["velocity_delta"]  # (..., 10)
        next_float[..., vel_start:vel_end] += vel_d[..., :cfg.velocity_dim]
        next_float[..., fp + vel_start:fp + vel_end] += vel_d[..., cfg.velocity_dim:]

    # --- Dynamics predictions (absolute: hitlag, stocks, combo [, hitstun]) ---
    if "dynamics_pred" in preds:
        dyn_start = ccd + cfg.velocity_dim + (0 if cfg.state_age_as_embed else 1)
        dyn_per_player = cfg.predicted_dynamics_dim // 2  # 3 or 4
        dyn_d = preds["dynamics_pred"]  # (..., 6 or 8)
        next_float[..., dyn_start:dyn_start + dyn_per_player] = dyn_d[..., :dyn_per_player]
        next_float[..., fp + dyn_start:fp + dyn_start + dyn_per_player] = dyn_d[..., dyn_per_player:]

    # --- Binary predictions (threshold logits at 0) ---
    binary = (preds["binary_logits"] > 0).float()  # (..., 6)
    p0_bin_start = cd
    p1_bin_start = fp + cd
    next_float[..., p0_bin_start:p0_bin_start + bd] = binary[..., :bd]
    next_float[..., p1_bin_start:p1_bin_start + bd] = binary[..., bd:]

    # --- Controller input (copy raw controller into frame, not threshold features) ---
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim
    # ctrl_float may include threshold features beyond the raw 2*controller_dim;
    # only copy the raw controller values into the frame tensor.
    raw_cd = cfg.controller_dim  # 13
    next_float[..., ctrl_start:ctrl_end] = ctrl_float[..., :raw_cd]
    next_float[..., fp + ctrl_start:fp + ctrl_end] = ctrl_float[..., raw_cd:raw_cd * 2]

    # --- Categorical predictions (argmax) ---
    next_int[..., 0] = preds["p0_action_logits"].argmax(dim=-1)
    next_int[..., 1] = preds["p0_jumps_logits"].argmax(dim=-1)
    next_int[..., ipp] = preds["p1_action_logits"].argmax(dim=-1)
    next_int[..., ipp + 1] = preds["p1_jumps_logits"].argmax(dim=-1)
    # character, l_cancel, hurtbox, ground, last_attack, stage: carry forward

    return next_float, next_int


def _append_threshold_features(ctrl: torch.Tensor, cfg: EncodingConfig) -> torch.Tensor:
    """Append ctrl_threshold binary features if enabled.

    Thresholds the 5 analog axes (main_x, main_y, c_x, c_y, shoulder)
    at 0.3 for each player. Produces 10 binary features (5 per player).

    Args:
        ctrl: (..., ctrl_dim * 2) — raw controller input [p0(13), p1(13)].
        cfg: Encoding config.

    Returns:
        (..., ctrl_conditioning_dim) — with threshold features appended if enabled.
    """
    if not cfg.ctrl_threshold_features:
        return ctrl
    cd = cfg.controller_dim  # 13
    p0_analog = ctrl[..., :5]       # main_x, main_y, c_x, c_y, shoulder
    p1_analog = ctrl[..., cd:cd+5]
    threshold = 0.3
    p0_thresh = (p0_analog.abs() > threshold).float()
    p1_thresh = (p1_analog.abs() > threshold).float()
    return torch.cat([ctrl, p0_thresh, p1_thresh], dim=-1)


def build_ctrl(
    float_data: torch.Tensor,
    t: int,
    cfg: EncodingConfig,
) -> torch.Tensor:
    """Extract controller input for frame t from float data.

    Args:
        float_data: (T, F) — full game float tensors.
        t: Frame index.
        cfg: Encoding config.

    Returns:
        (ctrl_conditioning_dim,) — [p0_ctrl(13), p1_ctrl(13), threshold(10 if enabled)]
    """
    fp = cfg.float_per_player
    ctrl_start = cfg.continuous_dim + cfg.binary_dim
    ctrl_end = ctrl_start + cfg.controller_dim

    if t < float_data.shape[0]:
        raw = torch.cat([
            float_data[t, ctrl_start:ctrl_end],
            float_data[t, fp + ctrl_start:fp + ctrl_end],
        ])
    else:
        raw = torch.zeros(cfg.controller_dim * 2)

    return _append_threshold_features(raw, cfg)


def build_ctrl_batch(
    float_data: torch.Tensor,
    indices: torch.Tensor,
    cfg: EncodingConfig,
) -> torch.Tensor:
    """Extract controller inputs for multiple frames (batched).

    Args:
        float_data: (T, F) — full game float tensors.
        indices: (N,) — frame indices (int64).
        cfg: Encoding config.

    Returns:
        (N, ctrl_conditioning_dim) — controller inputs with threshold features if enabled.
    """
    fp = cfg.float_per_player
    ctrl_start = cfg.continuous_dim + cfg.binary_dim
    ctrl_end = ctrl_start + cfg.controller_dim

    frames = float_data[indices]  # (N, F)
    raw = torch.cat([
        frames[:, ctrl_start:ctrl_end],
        frames[:, fp + ctrl_start:fp + ctrl_end],
    ], dim=1)

    return _append_threshold_features(raw, cfg)
