"""JEPA World Model for Melee.

Wraps encoder + predictor + SIGReg into a single module.
Training: encode all frames → predict next embedding → MSE + SIGReg.
Rollout: autoregressive prediction in latent space.

Reference: le-wm/jepa.py — JEPA class.
"""

import torch
import torch.nn as nn

from models.encoding import EncodingConfig
from models.jepa.encoder import GameStateEncoder
from models.jepa.predictor import ARPredictor
from models.jepa.sigreg import SIGReg


class JEPAWorldModel(nn.Module):
    """JEPA World Model for Melee.

    Training forward returns prediction loss + SIGReg loss.
    Rollout produces autoregressive latent predictions.
    """

    def __init__(
        self,
        cfg: EncodingConfig,
        embed_dim: int = 192,
        history_size: int = 3,
        encoder_hidden_dim: int = 512,
        predictor_layers: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        predictor_dropout: float = 0.1,
        sigreg_lambda: float = 0.1,
        sigreg_knots: int = 17,
        sigreg_projections: int = 1024,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.context_len = history_size  # alias for compatibility
        self.sigreg_lambda = sigreg_lambda

        self.encoder = GameStateEncoder(
            cfg, embed_dim=embed_dim, hidden_dim=encoder_hidden_dim
        )
        self.predictor = ARPredictor(
            embed_dim=embed_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            action_input_dim=cfg.ctrl_conditioning_dim,
            action_embed_dim=embed_dim,
            dropout=predictor_dropout,
        )
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_projections)

    def forward(
        self,
        float_frames: torch.Tensor,
        int_frames: torch.Tensor,
        ctrl_inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            float_frames: (B, T, F) — all frames (context + target), T = history_size + 1
            int_frames: (B, T, I) — categorical indices
            ctrl_inputs: (B, T, C) — controller inputs

        Returns:
            dict with pred_loss, sigreg_loss, total_loss, embeddings
        """
        B, T, F = float_frames.shape
        H = self.history_size
        num_preds = T - H
        assert num_preds == 1, (
            f"Only num_preds=1 supported (got T={T}, H={H}, num_preds={num_preds}). "
            f"Multi-step prediction requires AR unrolling — flag as divergence."
        )

        # 1. Encode ALL frames
        all_embs = self.encoder(float_frames, int_frames)  # (B, T, D)

        # 2. Context + target split
        context_embs = all_embs[:, :H]      # (B, H, D)
        target_embs = all_embs[:, H:]       # (B, 1, D)
        context_ctrl = ctrl_inputs[:, :H]   # (B, H, C)

        # 3. Predict next embedding from context (last position's output)
        pred_embs = self.predictor(context_embs, context_ctrl)  # (B, H, D)
        pred_next = pred_embs[:, -1:]       # (B, 1, D)

        # 4. Prediction loss: MSE in latent space
        pred_loss = (pred_next - target_embs).pow(2).mean()

        # 5. SIGReg: applied to all embeddings
        # SIGReg expects (T, B, D) — transpose from (B, T, D)
        sigreg_loss = self.sigreg(all_embs.transpose(0, 1))

        total_loss = pred_loss + self.sigreg_lambda * sigreg_loss

        return {
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "total_loss": total_loss,
            "embeddings": all_embs.detach(),
        }

    @torch.no_grad()
    def rollout(
        self,
        float_ctx: torch.Tensor,
        int_ctx: torch.Tensor,
        initial_ctrls: torch.Tensor,
        ctrl_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive rollout in latent space.

        Args:
            float_ctx: (B, H, F) — initial context frames
            int_ctx: (B, H, I) — initial context categoricals
            initial_ctrls: (B, H, C) — controller inputs for initial context
            ctrl_sequence: (B, N, C) — controller inputs for N future steps

        Returns:
            pred_embs: (B, N, D) — predicted latent embeddings
        """
        # BN footgun guard: both projectors end in nn.BatchNorm1d.
        # In train mode BN uses batch stats → predictions silently drift.
        # @torch.no_grad() does not handle this — only .eval() does.
        assert not self.training, (
            "JEPAWorldModel.rollout() requires eval mode. "
            "Call model.eval() before rollout() — BatchNorm projectors "
            "will use batch stats in train mode and silently drift predictions."
        )
        N = ctrl_sequence.shape[1]

        # Encode initial context
        embs = self.encoder(float_ctx, int_ctx)  # (B, H, D)
        ctrl_buffer = initial_ctrls  # (B, H, C)

        predictions = []
        for step in range(N):
            ctx = embs[:, -self.history_size:]
            ctx_ctrl = ctrl_buffer[:, -self.history_size:]

            pred = self.predictor(ctx, ctx_ctrl)
            next_emb = pred[:, -1:]
            predictions.append(next_emb)

            embs = torch.cat([embs, next_emb], dim=1)
            ctrl_buffer = torch.cat(
                [ctrl_buffer, ctrl_sequence[:, step : step + 1]], dim=1
            )

        return torch.cat(predictions, dim=1)  # (B, N, D)
