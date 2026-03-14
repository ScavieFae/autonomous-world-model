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

    # --- Binary predictions (threshold logits at 0) ---
    binary = (preds["binary_logits"] > 0).float()  # (..., 6)
    p0_bin_start = cd
    p1_bin_start = fp + cd
    next_float[..., p0_bin_start:p0_bin_start + bd] = binary[..., :bd]
    next_float[..., p1_bin_start:p1_bin_start + bd] = binary[..., bd:]

    # --- Controller input (copy from ground truth) ---
    ctrl_start = cd + bd
    ctrl_end = ctrl_start + cfg.controller_dim
    next_float[..., ctrl_start:ctrl_end] = ctrl_float[..., :cfg.controller_dim]
    next_float[..., fp + ctrl_start:fp + ctrl_end] = ctrl_float[..., cfg.controller_dim:]

    # --- Categorical predictions (argmax) ---
    next_int[..., 0] = preds["p0_action_logits"].argmax(dim=-1)
    next_int[..., 1] = preds["p0_jumps_logits"].argmax(dim=-1)
    next_int[..., ipp] = preds["p1_action_logits"].argmax(dim=-1)
    next_int[..., ipp + 1] = preds["p1_jumps_logits"].argmax(dim=-1)
    # character, l_cancel, hurtbox, ground, last_attack, stage: carry forward

    return next_float, next_int


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
        (ctrl_dim * 2,) — [p0_ctrl(13), p1_ctrl(13)]
    """
    fp = cfg.float_per_player
    ctrl_start = cfg.continuous_dim + cfg.binary_dim
    ctrl_end = ctrl_start + cfg.controller_dim

    if t < float_data.shape[0]:
        return torch.cat([
            float_data[t, ctrl_start:ctrl_end],
            float_data[t, fp + ctrl_start:fp + ctrl_end],
        ])
    else:
        return torch.zeros(cfg.controller_dim * 2)


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
        (N, ctrl_dim * 2) — controller inputs.
    """
    fp = cfg.float_per_player
    ctrl_start = cfg.continuous_dim + cfg.binary_dim
    ctrl_end = ctrl_start + cfg.controller_dim

    frames = float_data[indices]  # (N, F)
    return torch.cat([
        frames[:, ctrl_start:ctrl_end],
        frames[:, fp + ctrl_start:fp + ctrl_end],
    ], dim=1)
