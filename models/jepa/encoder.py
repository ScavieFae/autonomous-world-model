"""Game state encoder for the JEPA world model.

Replaces LeWM's ViT encoder (accepted divergence — structured game state,
not pixels). The projector (MLP + BatchNorm → embed_dim) matches LeWM.

Reference: le-wm/module.py — Encoder class (ViT + projector).
"""

import torch
import torch.nn as nn

from models.encoding import EncodingConfig


class GameStateEncoder(nn.Module):
    """Encodes per-frame game state into a latent embedding.

    Categorical features get learned embeddings (same vocab sizes as Mamba2).
    Continuous features pass through directly (already normalized).
    All concatenated → MLP trunk → projector with BatchNorm → embed_dim.

    The projector with BatchNorm matches LeWM's pattern — BatchNorm is needed
    because LayerNorm-style normalization interferes with SIGReg.
    """

    def __init__(self, cfg: EncodingConfig, embed_dim: int = 192, hidden_dim: int = 512):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim

        # Categorical embeddings (same vocabs as Mamba2)
        self.action_embed = nn.Embedding(cfg.action_vocab, cfg.action_embed_dim)
        self.jumps_embed = nn.Embedding(cfg.jumps_vocab, cfg.jumps_embed_dim)
        self.character_embed = nn.Embedding(cfg.character_vocab, cfg.character_embed_dim)
        self.stage_embed = nn.Embedding(cfg.stage_vocab, cfg.stage_embed_dim)
        self.l_cancel_embed = nn.Embedding(cfg.l_cancel_vocab, cfg.l_cancel_embed_dim)
        self.hurtbox_embed = nn.Embedding(cfg.hurtbox_vocab, cfg.hurtbox_embed_dim)
        self.ground_embed = nn.Embedding(cfg.ground_vocab, cfg.ground_embed_dim)
        self.last_attack_embed = nn.Embedding(cfg.last_attack_vocab, cfg.last_attack_embed_dim)
        if cfg.state_age_as_embed:
            self.state_age_embed = nn.Embedding(
                cfg.state_age_embed_vocab, cfg.state_age_embed_dim
            )

        # Compute total input dimension
        float_dim = cfg.float_per_player * 2
        cat_embed_per_player = (
            cfg.action_embed_dim + cfg.jumps_embed_dim + cfg.character_embed_dim
            + cfg.l_cancel_embed_dim + cfg.hurtbox_embed_dim
            + cfg.ground_embed_dim + cfg.last_attack_embed_dim
        )
        if cfg.state_age_as_embed:
            cat_embed_per_player += cfg.state_age_embed_dim
        self.total_input_dim = float_dim + 2 * cat_embed_per_player + cfg.stage_embed_dim

        # MLP trunk (2-layer with SiLU, matching LeWM's activation)
        self.trunk = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Projector: MLP + BatchNorm (matches LeWM: hidden → 2048 → embed_dim + BN)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

        # Int column layout (same as Mamba2's _int_cols)
        self._ipp = cfg.int_per_player

    def forward(
        self, float_frames: torch.Tensor, int_frames: torch.Tensor
    ) -> torch.Tensor:
        """Encode frames into latent embeddings.

        Args:
            float_frames: (B, T, F) — continuous + binary + controller, both players
            int_frames: (B, T, I) — categorical indices

        Returns:
            embeddings: (B, T, embed_dim)
        """
        B, T, F = float_frames.shape
        I = int_frames.shape[-1]

        flat_float = float_frames.reshape(B * T, F)
        flat_int = int_frames.reshape(B * T, I)

        ipp = self._ipp

        # Embed categoricals for both players + stage
        parts = [flat_float]

        for offset in [0, ipp]:
            parts.append(self.action_embed(flat_int[:, offset + 0]))
            parts.append(self.jumps_embed(flat_int[:, offset + 1]))
            parts.append(self.character_embed(flat_int[:, offset + 2]))
            parts.append(self.l_cancel_embed(flat_int[:, offset + 3]))
            parts.append(self.hurtbox_embed(flat_int[:, offset + 4]))
            parts.append(self.ground_embed(flat_int[:, offset + 5]))
            parts.append(self.last_attack_embed(flat_int[:, offset + 6]))
            if self.cfg.state_age_as_embed:
                parts.append(self.state_age_embed(flat_int[:, offset + 7]))

        parts.append(self.stage_embed(flat_int[:, -1]))

        x = torch.cat(parts, dim=-1)   # (B*T, total_input_dim)
        x = self.trunk(x)              # (B*T, hidden_dim)
        x = self.projector(x)          # (B*T, embed_dim)

        return x.reshape(B, T, self.embed_dim)
