"""Constraint violation detection for world model outputs.

Checks reconstructed frames for physically impossible states.
Measurement only — no penalties applied to the loss.

Two tiers:
  Tier 1 (hard): physically impossible, stateless checks
    - percent_negative: percent < 0
    - stocks_increased: stocks(t) > stocks(t-1)
    - hitlag_negative: hitlag < 0
    - blast_zone_alive: outside FD blast zones without losing a stock
    - phantom_damage: percent increased but hitlag == 0

  Tier 3 (coherence): internal consistency checks
    - ground_air_mismatch: on_ground=1 but y > stage_height + threshold
"""

import torch

from models.encoding import EncodingConfig


# Final Destination blast zone approximations (game units)
FD_BLAST_X_MIN = -224.0
FD_BLAST_X_MAX = 224.0
FD_BLAST_Y_MIN = -109.0
FD_BLAST_Y_MAX = 200.0

# FD main platform height (game units)
FD_STAGE_Y = 0.0
# Threshold above stage height where on_ground=1 is suspicious
GROUND_AIR_Y_THRESHOLD = 5.0

# Minimum percent delta to count as damage (game units, post-denorm)
PHANTOM_DAMAGE_THRESHOLD = 0.5


class ConstraintChecker:
    """Checks batched reconstructed frames for constraint violations.

    All methods expect float tensors of shape (B, F) where F is the
    full frame float dimension (float_per_player * 2).
    """

    def __init__(self, cfg: EncodingConfig):
        self.cfg = cfg
        fp = cfg.float_per_player
        cd = cfg.continuous_dim
        ccd = cfg.core_continuous_dim  # 4

        # Per-player field indices within the float tensor
        # Player 0 starts at 0, Player 1 starts at fp
        self._p0_percent = 0
        self._p1_percent = fp
        self._p0_x = 1
        self._p0_y = 2
        self._p1_x = fp + 1
        self._p1_y = fp + 2

        # Dynamics indices depend on state_age_as_embed
        dyn_start = ccd + cfg.velocity_dim  # 4 + 5 = 9
        if not cfg.state_age_as_embed:
            # state_age at [9], hitlag at [10], stocks at [11]
            self._p0_hitlag = dyn_start + 1
            self._p0_stocks = dyn_start + 2
        else:
            # no state_age float, hitlag at [9], stocks at [10]
            self._p0_hitlag = dyn_start
            self._p0_stocks = dyn_start + 1

        self._p1_hitlag = fp + self._p0_hitlag
        self._p1_stocks = fp + self._p0_stocks

        # Binary field: on_ground is the 3rd binary (index 2 within binary block)
        self._p0_on_ground = cd + 2  # continuous_dim + 2
        self._p1_on_ground = fp + cd + 2

        # Store scales for denormalization
        self._percent_scale = cfg.percent_scale
        self._xy_scale = cfg.xy_scale
        self._hitlag_scale = cfg.hitlag_scale
        self._stocks_scale = cfg.stocks_scale

    def _denorm_percent(self, val: torch.Tensor) -> torch.Tensor:
        """Denormalize percent: stored as percent * 0.01."""
        return val / self._percent_scale

    def _denorm_xy(self, val: torch.Tensor) -> torch.Tensor:
        """Denormalize position: stored as pos * 0.05."""
        return val / self._xy_scale

    def _denorm_hitlag(self, val: torch.Tensor) -> torch.Tensor:
        """Denormalize hitlag: stored as hitlag * 0.1."""
        return val / self._hitlag_scale

    def _denorm_stocks(self, val: torch.Tensor) -> torch.Tensor:
        """Denormalize stocks: stored as stocks * 0.25."""
        return val / self._stocks_scale

    def check_frame(
        self,
        current_float: torch.Tensor,
        prev_float: torch.Tensor | None,
        cfg: EncodingConfig,
    ) -> dict[str, int]:
        """Check one batch of frames for constraint violations.

        Args:
            current_float: (B, F) current frame float tensor (normalized).
            prev_float: (B, F) previous frame float tensor (normalized), or None.
                Required for stocks_increased and phantom_damage checks.
            cfg: EncodingConfig (used for consistency; indices were cached in __init__).

        Returns:
            Dict mapping violation name to count of violations in the batch.
        """
        B = current_float.shape[0]
        violations: dict[str, int] = {}

        # --- Tier 1: Hard violations ---

        # percent_negative: percent < 0 (either player)
        p0_pct = self._denorm_percent(current_float[:, self._p0_percent])
        p1_pct = self._denorm_percent(current_float[:, self._p1_percent])
        pct_neg = (p0_pct < 0).sum() + (p1_pct < 0).sum()
        violations["percent_neg"] = int(pct_neg.item())

        # hitlag_negative: hitlag < 0 (either player)
        p0_hl = self._denorm_hitlag(current_float[:, self._p0_hitlag])
        p1_hl = self._denorm_hitlag(current_float[:, self._p1_hitlag])
        hl_neg = (p0_hl < 0).sum() + (p1_hl < 0).sum()
        violations["hitlag_neg"] = int(hl_neg.item())

        # blast_zone_alive: outside blast zones AND stocks didn't decrease
        p0_x = self._denorm_xy(current_float[:, self._p0_x])
        p0_y = self._denorm_xy(current_float[:, self._p0_y])
        p1_x = self._denorm_xy(current_float[:, self._p1_x])
        p1_y = self._denorm_xy(current_float[:, self._p1_y])

        p0_outside = (
            (p0_x < FD_BLAST_X_MIN) | (p0_x > FD_BLAST_X_MAX)
            | (p0_y < FD_BLAST_Y_MIN) | (p0_y > FD_BLAST_Y_MAX)
        )
        p1_outside = (
            (p1_x < FD_BLAST_X_MIN) | (p1_x > FD_BLAST_X_MAX)
            | (p1_y < FD_BLAST_Y_MIN) | (p1_y > FD_BLAST_Y_MAX)
        )

        if prev_float is not None:
            # Stock check: stocks didn't decrease means no KO happened
            p0_stocks_cur = self._denorm_stocks(current_float[:, self._p0_stocks])
            p0_stocks_prev = self._denorm_stocks(prev_float[:, self._p0_stocks])
            p1_stocks_cur = self._denorm_stocks(current_float[:, self._p1_stocks])
            p1_stocks_prev = self._denorm_stocks(prev_float[:, self._p1_stocks])

            p0_no_ko = p0_stocks_cur >= p0_stocks_prev
            p1_no_ko = p1_stocks_cur >= p1_stocks_prev

            blast_zone = (p0_outside & p0_no_ko).sum() + (p1_outside & p1_no_ko).sum()

            # stocks_increased: stocks(t) > stocks(t-1)
            stocks_inc = (
                (p0_stocks_cur > p0_stocks_prev + 0.01).sum()
                + (p1_stocks_cur > p1_stocks_prev + 0.01).sum()
            )
            violations["stocks_inc"] = int(stocks_inc.item())

            # phantom_damage: percent increased but hitlag == 0 on the damaged player
            p0_pct_prev = self._denorm_percent(prev_float[:, self._p0_percent])
            p1_pct_prev = self._denorm_percent(prev_float[:, self._p1_percent])
            p0_pct_delta = p0_pct - p0_pct_prev
            p1_pct_delta = p1_pct - p1_pct_prev

            p0_phantom = (p0_pct_delta > PHANTOM_DAMAGE_THRESHOLD) & (p0_hl.abs() < 0.01)
            p1_phantom = (p1_pct_delta > PHANTOM_DAMAGE_THRESHOLD) & (p1_hl.abs() < 0.01)
            phantom = p0_phantom.sum() + p1_phantom.sum()
            violations["phantom_dmg"] = int(phantom.item())
        else:
            blast_zone = p0_outside.sum() + p1_outside.sum()
            violations["stocks_inc"] = 0
            violations["phantom_dmg"] = 0

        violations["blast_zone"] = int(blast_zone.item())

        # --- Tier 3: Coherence ---

        # ground_air_mismatch: on_ground=1 but y > stage_height + threshold
        p0_on_ground = current_float[:, self._p0_on_ground] > 0.5
        p1_on_ground = current_float[:, self._p1_on_ground] > 0.5
        p0_high = p0_y > (FD_STAGE_Y + GROUND_AIR_Y_THRESHOLD)
        p1_high = p1_y > (FD_STAGE_Y + GROUND_AIR_Y_THRESHOLD)
        ground_air = (p0_on_ground & p0_high).sum() + (p1_on_ground & p1_high).sum()
        violations["ground_air"] = int(ground_air.item())

        return violations

    def check_batch_and_log(
        self,
        current_float: torch.Tensor,
        prev_float: torch.Tensor | None,
        cfg: EncodingConfig,
    ) -> dict[str, float]:
        """Check violations and return rates (count / batch_size / 2).

        Division by 2 because each check covers 2 players, so the
        denominator is B*2 player-frames.

        Args:
            current_float: (B, F) current frame float tensor.
            prev_float: (B, F) previous frame float tensor, or None.
            cfg: EncodingConfig.

        Returns:
            Dict mapping violation name to violation rate [0, 1].
        """
        B = current_float.shape[0]
        counts = self.check_frame(current_float, prev_float, cfg)
        # Denominator: B samples * 2 players
        denom = B * 2
        rates = {name: count / denom for name, count in counts.items()}
        # Total violation rate: any violation in a sample
        total_count = sum(counts.values())
        rates["total"] = total_count / denom
        return rates
