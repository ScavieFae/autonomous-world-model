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

Copied from nojohns/worldmodel/model/encoding.py — encoding config only.
The encode_player_frames() and StateEncoder class are not needed for inference.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    def ctrl_conditioning_dim(self) -> int:
        base = self.controller_dim * 2 + self.ctrl_extra_dim
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
