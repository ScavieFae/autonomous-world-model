"""State encoding: game frames → flat tensors for the world model.

Converts per-player per-frame data into a fixed-size tensor representation.
Continuous values are normalized, categoricals get learned embeddings.

Normalization scales match slippi-ai's embed.py:
  - xy: ×0.05    (positions range roughly -200 to 200)
  - percent: ×0.01  (damage 0-999)
  - shield: ×0.01   (0-60)
  - velocity: ×0.05  (same scale as position — units/frame)
  - state_age: ×0.01 (action frames, typically 0-100+)
  - hitlag: ×0.1     (typically 0-10 frames)
  - stocks: ×0.25    (0-4 stocks → 0-1 range)
  - combo_count: ×0.1 (ordinal, typically 0-10)
  - controller: already [0, 1], no scaling needed

Originally copied from nojohns/worldmodel/model/encoding.py.
Now canonical — includes both config and tensor encoding for training + inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from data.parse import PlayerFrame


@dataclass
class EncodingConfig:
    """Configuration for state encoding dimensions."""

    # Normalization scales
    xy_scale: float = 0.05
    percent_scale: float = 0.01
    shield_scale: float = 0.01
    velocity_scale: float = 0.05
    state_age_scale: float = 0.01
    hitlag_scale: float = 0.1
    stocks_scale: float = 0.25
    combo_count_scale: float = 0.1

    # Embedding sizes for categoricals
    action_vocab: int = 400  # action states 0..399
    action_embed_dim: int = 32
    jumps_vocab: int = 8  # jumps_left 0..7
    jumps_embed_dim: int = 4
    character_vocab: int = 33  # internal character IDs 0..32
    character_embed_dim: int = 8
    stage_vocab: int = 33  # stage IDs 0..32 (legal stages are a subset)
    stage_embed_dim: int = 4
    # Combat context categoricals (v2.1)
    l_cancel_vocab: int = 3  # 0=N/A, 1=success, 2=missed
    l_cancel_embed_dim: int = 2
    hurtbox_vocab: int = 3  # 0=vulnerable, 1=invulnerable, 2=intangible
    hurtbox_embed_dim: int = 2
    ground_vocab: int = 32  # remapped surface IDs (0=airborne, 1+=surfaces)
    ground_embed_dim: int = 4
    last_attack_vocab: int = 64  # ~60 distinct attack IDs
    last_attack_embed_dim: int = 8

    # Experiment flags
    state_age_as_embed: bool = False  # Exp 1a: learned embedding instead of scaled float
    state_age_embed_vocab: int = 150  # max animation frames
    state_age_embed_dim: int = 8
    press_events: bool = False  # Exp 2a: 16 binary features for newly-pressed buttons
    lookahead: int = 0  # Exp 3a: predict frame t+d given ctrl(t) through ctrl(t+d)
    projectiles: bool = False  # Exp 4: item/projectile encoding (per-player nearest)
    state_flags: bool = False  # all 40 bits from 5 state_flags bytes as binary features
    hitstun: bool = False  # hitstun_remaining as continuous feature
    hitstun_scale: float = 0.02  # normalization: range 0-50 → 0-1
    focal_offset: int = 0  # frame offset for focal loss weighting
    multi_position: bool = False  # multi-position encoding (edge/center)
    ctrl_residual_to_action: bool = False  # controller residual connection to action head
    ctrl_threshold_features: bool = False  # threshold-based controller binary features
    character_conditioning: bool = False  # character-conditioned heads

    @property
    def core_continuous_dim(self) -> int:
        return 4  # percent, x, y, shield

    @property
    def velocity_dim(self) -> int:
        return 5  # speed_air_x, speed_y, speed_ground_x, speed_attack_x, speed_attack_y

    @property
    def dynamics_dim(self) -> int:
        base = 2 if self.state_age_as_embed else 3  # hitlag, stocks [, state_age]
        if self.hitstun:
            base += 1  # hitstun_remaining
        return base

    @property
    def combat_continuous_dim(self) -> int:
        return 1  # combo_count

    @property
    def projectile_continuous_dim(self) -> int:
        return 3 if self.projectiles else 0

    @property
    def continuous_dim(self) -> int:
        return (self.core_continuous_dim + self.velocity_dim
                + self.dynamics_dim + self.combat_continuous_dim
                + self.projectile_continuous_dim)

    @property
    def binary_dim(self) -> int:
        base = 3  # facing, invulnerable, on_ground
        if self.state_flags:
            base += 40  # all 40 bits from 5 state_flags bytes
        return base

    @property
    def controller_dim(self) -> int:
        return 13  # 2 sticks (4) + shoulder (1) + 8 buttons

    @property
    def float_per_player(self) -> int:
        return self.continuous_dim + self.binary_dim + self.controller_dim

    @property
    def embed_dim(self) -> int:
        base = (self.action_embed_dim + self.jumps_embed_dim + self.character_embed_dim
                + self.l_cancel_embed_dim + self.hurtbox_embed_dim
                + self.ground_embed_dim + self.last_attack_embed_dim)
        if self.state_age_as_embed:
            base += self.state_age_embed_dim
        return base

    @property
    def int_per_player(self) -> int:
        return 8 if self.state_age_as_embed else 7

    @property
    def int_per_frame(self) -> int:
        return self.int_per_player * 2 + 1  # +1 for stage

    @property
    def ctrl_extra_dim(self) -> int:
        return 16 if self.press_events else 0

    @property
    def ctrl_threshold_dim(self) -> int:
        return 5 * 2 if self.ctrl_threshold_features else 0  # 5 analog axes × 2 players

    @property
    def ctrl_conditioning_dim(self) -> int:
        base = self.controller_dim * 2 + self.ctrl_extra_dim + self.ctrl_threshold_dim
        return base * (1 + self.lookahead)

    @property
    def player_dim(self) -> int:
        return self.float_per_player + self.embed_dim

    @property
    def predicted_binary_dim(self) -> int:
        return self.binary_dim * 2  # both players

    @property
    def predicted_velocity_dim(self) -> int:
        return self.velocity_dim * 2  # 10 — both players

    @property
    def predicted_dynamics_dim(self) -> int:
        base = 6  # hitlag + stocks + combo, both players
        if self.hitstun:
            base += 2  # hitstun × 2 players
        return base

    @property
    def target_int_dim(self) -> int:
        return 12

    @property
    def frame_dim(self) -> int:
        return self.player_dim * 2 + self.stage_embed_dim


# --- Tensor encoding (numpy → torch, no learned params) ---


def encode_player_frames(pf: PlayerFrame, cfg: EncodingConfig) -> dict[str, torch.Tensor]:
    """Convert a PlayerFrame's numpy arrays into normalized torch tensors.

    Returns dict with keys:
        continuous: (T, continuous_dim) — core + velocity + dynamics + combat_continuous
        binary: (T, 3) — [facing, invulnerable, on_ground]
        controller: (T, 13) — [main_x, main_y, c_x, c_y, shoulder, A..D_UP]
        action: (T,) — int64 action indices
        jumps_left: (T,) — int64 jumps indices
        character: (T,) — int64 character indices
        l_cancel: (T,) — int64 l-cancel status
        hurtbox_state: (T,) — int64 hurtbox state
        ground: (T,) — int64 ground surface ID
        last_attack_landed: (T,) — int64 last attack ID
        state_age_int: (T,) — int64 (only when cfg.state_age_as_embed)
    """
    T = len(pf.percent)

    # Build continuous columns — dynamics section varies with state_age_as_embed
    cont_cols = [
        # Core continuous (predict deltas for these)
        pf.percent * cfg.percent_scale,
        pf.x * cfg.xy_scale,
        pf.y * cfg.xy_scale,
        pf.shield_strength * cfg.shield_scale,
        # Velocity (input features)
        pf.speed_air_x * cfg.velocity_scale,
        pf.speed_y * cfg.velocity_scale,
        pf.speed_ground_x * cfg.velocity_scale,
        pf.speed_attack_x * cfg.velocity_scale,
        pf.speed_attack_y * cfg.velocity_scale,
    ]
    # Dynamics — state_age is float only when not embedded
    if not cfg.state_age_as_embed:
        cont_cols.append(pf.state_age * cfg.state_age_scale)
    cont_cols.extend([
        pf.hitlag * cfg.hitlag_scale,
        pf.stocks * cfg.stocks_scale,
        # Combat continuous
        pf.combo_count * cfg.combo_count_scale,
    ])
    continuous = np.stack(cont_cols, axis=1)  # (T, 13) or (T, 12)

    binary = np.stack(
        [pf.facing, pf.invulnerable, pf.on_ground],
        axis=1,
    )  # (T, 3)

    controller = np.concatenate(
        [
            pf.main_stick_x[:, None],
            pf.main_stick_y[:, None],
            pf.c_stick_x[:, None],
            pf.c_stick_y[:, None],
            pf.shoulder[:, None],
            pf.buttons,  # (T, 8)
        ],
        axis=1,
    )  # (T, 13)

    # Clamp categoricals to valid range
    action = np.clip(pf.action, 0, cfg.action_vocab - 1)
    jumps = np.clip(pf.jumps_left, 0, cfg.jumps_vocab - 1)
    character = np.clip(pf.character, 0, cfg.character_vocab - 1)
    l_cancel = np.clip(pf.l_cancel, 0, cfg.l_cancel_vocab - 1)
    hurtbox = np.clip(pf.hurtbox_state, 0, cfg.hurtbox_vocab - 1)
    ground = np.clip(pf.ground, 0, cfg.ground_vocab - 1)
    last_attack = np.clip(pf.last_attack_landed, 0, cfg.last_attack_vocab - 1)

    result = {
        "continuous": torch.from_numpy(continuous).float(),
        "binary": torch.from_numpy(binary).float(),
        "controller": torch.from_numpy(controller).float(),
        "action": torch.from_numpy(action).long(),
        "jumps_left": torch.from_numpy(jumps).long(),
        "character": torch.from_numpy(character).long(),
        "l_cancel": torch.from_numpy(l_cancel).long(),
        "hurtbox_state": torch.from_numpy(hurtbox).long(),
        "ground": torch.from_numpy(ground).long(),
        "last_attack_landed": torch.from_numpy(last_attack).long(),
    }

    if cfg.state_age_as_embed:
        # Clamp to vocab range as integer for embedding lookup
        state_age_int = np.clip(pf.state_age.astype(np.int64), 0, cfg.state_age_embed_vocab - 1)
        result["state_age_int"] = torch.from_numpy(state_age_int).long()

    return result


# --- Embedding module (learned params, part of the model) ---


class StateEncoder(nn.Module):
    """Encodes per-frame game state into a flat vector via learned embeddings.

    Takes pre-encoded tensors (from encode_player_frames) and produces
    a single flat vector per frame by embedding categoricals and concatenating.
    """

    def __init__(self, cfg: EncodingConfig):
        super().__init__()
        self.cfg = cfg
        self.action_embed = nn.Embedding(cfg.action_vocab, cfg.action_embed_dim)
        self.jumps_embed = nn.Embedding(cfg.jumps_vocab, cfg.jumps_embed_dim)
        self.character_embed = nn.Embedding(cfg.character_vocab, cfg.character_embed_dim)
        self.stage_embed = nn.Embedding(cfg.stage_vocab, cfg.stage_embed_dim)
        # Combat context embeddings (v2.1)
        self.l_cancel_embed = nn.Embedding(cfg.l_cancel_vocab, cfg.l_cancel_embed_dim)
        self.hurtbox_embed = nn.Embedding(cfg.hurtbox_vocab, cfg.hurtbox_embed_dim)
        self.ground_embed = nn.Embedding(cfg.ground_vocab, cfg.ground_embed_dim)
        self.last_attack_embed = nn.Embedding(cfg.last_attack_vocab, cfg.last_attack_embed_dim)
        # Exp 1a: state_age as learned embedding
        if cfg.state_age_as_embed:
            self.state_age_embed = nn.Embedding(cfg.state_age_embed_vocab, cfg.state_age_embed_dim)

    def forward_player(
        self,
        continuous: torch.Tensor,  # (B, 13) or (B, 12) when state_age_as_embed
        binary: torch.Tensor,  # (B, 3)
        controller: torch.Tensor,  # (B, 13)
        action: torch.Tensor,  # (B,)
        jumps_left: torch.Tensor,  # (B,)
        character: torch.Tensor,  # (B,)
        l_cancel: torch.Tensor,  # (B,)
        hurtbox_state: torch.Tensor,  # (B,)
        ground: torch.Tensor,  # (B,)
        last_attack_landed: torch.Tensor,  # (B,)
        state_age: Optional[torch.Tensor] = None,  # (B,) int — only when state_age_as_embed
    ) -> torch.Tensor:
        """Encode one player's state. Returns (B, player_dim)."""
        action_emb = self.action_embed(action)  # (B, 32)
        jumps_emb = self.jumps_embed(jumps_left)  # (B, 4)
        char_emb = self.character_embed(character)  # (B, 8)
        lc_emb = self.l_cancel_embed(l_cancel)  # (B, 2)
        hb_emb = self.hurtbox_embed(hurtbox_state)  # (B, 2)
        gnd_emb = self.ground_embed(ground)  # (B, 4)
        la_emb = self.last_attack_embed(last_attack_landed)  # (B, 8)
        parts = [
            continuous, binary, controller,
            action_emb, jumps_emb, char_emb,
            lc_emb, hb_emb, gnd_emb, la_emb,
        ]
        if self.cfg.state_age_as_embed and state_age is not None:
            parts.append(self.state_age_embed(state_age))  # (B, 8)
        return torch.cat(parts, dim=-1)
