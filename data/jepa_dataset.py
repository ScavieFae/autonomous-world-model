"""Dataset for JEPA world model training.

Returns frame subsequences instead of single-frame predictions.
JEPA's training pattern: encode all frames in a subsequence, predict
the next embedding from the context window.

Returns (float_frames, int_frames, ctrl_inputs) where each is a
contiguous subsequence of T = history_size + num_preds frames.
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from data.dataset import MeleeDataset
from models.encoding import EncodingConfig

logger = logging.getLogger(__name__)


class JEPAFrameDataset(Dataset):
    """Dataset for JEPA training — returns frame subsequences.

    Unlike MeleeFrameDataset (single-frame prediction with targets),
    this returns raw subsequences for JEPA's encode-all-then-predict
    training pattern.

    Returns:
        float_frames: (T, F) — T frames of float features
        int_frames: (T, I) — T frames of categorical indices
        ctrl_inputs: (T, C) — T frames of controller inputs
    """

    def __init__(
        self,
        data: MeleeDataset,
        game_range: range,
        history_size: int = 3,
        num_preds: int = 1,
    ):
        self.data = data
        self.history_size = history_size
        self.num_preds = num_preds
        self.seq_len = history_size + num_preds
        cfg = data.cfg

        # Shape guards: _extract_ctrl and get_batch hardcode the single-frame
        # non-press layout. ctrl_conditioning_dim multiplies by (1 + lookahead)
        # and adds ctrl_extra_dim for press_events — toggling either flag
        # would silently shape-mismatch the predictor. Kill the ambiguity now.
        assert cfg.lookahead == 0, (
            f"JEPAFrameDataset requires cfg.lookahead == 0 (got {cfg.lookahead}). "
            "Multi-frame controller stacking is a divergence from LeWM — flag it."
        )
        assert not cfg.press_events, (
            "JEPAFrameDataset requires cfg.press_events == False. "
            "Press-event controller features are a divergence from LeWM — flag them."
        )

        # Controller slice indices (same layout as MeleeFrameDataset)
        fp = cfg.float_per_player
        cd = cfg.continuous_dim
        bd = cfg.binary_dim
        ctrl_start = cd + bd
        ctrl_end = ctrl_start + cfg.controller_dim
        self._p0_ctrl = slice(ctrl_start, ctrl_end)
        self._p1_ctrl = slice(fp + ctrl_start, fp + ctrl_end)
        self._ctrl_threshold = cfg.ctrl_threshold_features
        self._p0_analog = slice(ctrl_start, ctrl_start + 5)
        self._p1_analog = slice(fp + ctrl_start, fp + ctrl_start + 5)

        # Valid starting indices: need seq_len consecutive frames within a game
        indices = []
        for gi in game_range:
            start = data.game_offsets[gi]
            end = data.game_offsets[gi + 1]
            for t in range(start, end - self.seq_len + 1):
                indices.append(t)

        self.valid_indices = np.array(indices, dtype=np.int64)
        logger.info(
            "JEPAFrameDataset: %d examples from %d games (history=%d, preds=%d)",
            len(self.valid_indices), len(game_range), history_size, num_preds,
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.valid_indices[idx])
        T = self.seq_len

        float_frames = self.data.floats[t : t + T]  # (T, F)
        int_frames = self.data.ints[t : t + T]       # (T, I)

        # Extract controller inputs per frame
        ctrl_inputs = torch.stack([
            self._extract_ctrl(float_frames[i]) for i in range(T)
        ])  # (T, C)

        return float_frames, int_frames, ctrl_inputs

    def _extract_ctrl(self, float_frame: torch.Tensor) -> torch.Tensor:
        """Extract controller inputs from a float frame."""
        parts = [float_frame[self._p0_ctrl], float_frame[self._p1_ctrl]]
        if self._ctrl_threshold:
            p0_analog = float_frame[self._p0_analog]
            p1_analog = float_frame[self._p1_analog]
            parts.append((p0_analog.abs() > 0.3).float())
            parts.append((p1_analog.abs() > 0.3).float())
        return torch.cat(parts)

    def get_batch(
        self, indices: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized batch loading (matches MeleeFrameDataset.get_batch pattern)."""
        T = self.seq_len
        ts = self.valid_indices[indices]

        offsets = np.arange(T)
        frame_indices = ts[:, None] + offsets[None, :]  # (B, T)

        float_frames = self.data.floats[frame_indices]  # (B, T, F)
        int_frames = self.data.ints[frame_indices]       # (B, T, I)

        # Vectorized controller extraction
        p0_ctrl = float_frames[:, :, self._p0_ctrl]      # (B, T, 13)
        p1_ctrl = float_frames[:, :, self._p1_ctrl]      # (B, T, 13)
        ctrl_parts = [p0_ctrl, p1_ctrl]
        if self._ctrl_threshold:
            p0_analog = float_frames[:, :, self._p0_analog]
            p1_analog = float_frames[:, :, self._p1_analog]
            ctrl_parts.append((p0_analog.abs() > 0.3).float())
            ctrl_parts.append((p1_analog.abs() > 0.3).float())
        ctrl_inputs = torch.cat(ctrl_parts, dim=-1)       # (B, T, C)

        return float_frames, int_frames, ctrl_inputs
