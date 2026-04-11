"""Autoregressive predictor with AdaLN-zero action conditioning.

Matches LeWM's ARPredictor + ConditionalBlock: causal Transformer with
Adaptive Layer Normalization modulated by action (controller) inputs.

Reference: le-wm/module.py — ARPredictor, ConditionalBlock classes.
"""

import torch
import torch.nn as nn


class AdaLNBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization conditioning.

    Matches LeWM's ConditionalBlock. LayerNorm params modulated by action input.
    6 modulation parameters per block: (shift, scale, gate) x 2 sublayers.
    Gate is zero-initialized so conditioning ramps up gradually (DiT pattern).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        action_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        # Multi-head self-attention
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

        # AdaLN modulation: action → 6 * dim parameters
        # (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(action_dim, 6 * dim),
        )
        # Zero-initialize so gates start at 0 — conditioning ramps up gradually
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        action_emb: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim) — sequence of frame embeddings
            action_emb: (B, T, action_dim) — per-step action embeddings
            causal_mask: (T, T) boolean mask for causal attention
        """
        B, T, D = x.shape

        # Compute modulation parameters from action
        mod = self.adaLN_modulation(action_emb)  # (B, T, 6*D)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Attention sublayer with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1

        # Multi-head self-attention
        qkv = self.to_qkv(h).reshape(B, T, 3, self.num_heads, self.dim_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, heads, T, dim_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        attn = attn.masked_fill(~causal_mask, float("-inf"))
        attn = self.attn_dropout(torch.softmax(attn, dim=-1))

        h = torch.matmul(attn, v)
        h = h.transpose(1, 2).reshape(B, T, -1)
        h = self.to_out(h)

        x = x + gate1 * h  # gated residual

        # MLP sublayer with AdaLN
        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        h = self.mlp(h)
        x = x + gate2 * h

        return x


class ARPredictor(nn.Module):
    """Autoregressive predictor with AdaLN-zero action conditioning.

    Matches LeWM's ARPredictor: causal Transformer, learned positional
    embeddings, AdaLN modulation from controller inputs at each layer.
    Predictor projector (MLP + BatchNorm) maps output into encoder space.
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        action_input_dim: int = 26,
        action_embed_dim: int = 192,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Action encoder: controller input → action embedding
        # LeWM uses Conv1d(kernel=1) + MLP for frameskip stacking.
        # We have no frameskip, so plain MLP suffices (minor accepted divergence).
        self.action_encoder = nn.Sequential(
            nn.Linear(action_input_dim, action_embed_dim),
            nn.SiLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.SiLU(),
        )

        # Learned positional embeddings (matching LeWM)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        # Transformer blocks with AdaLN
        self.blocks = nn.ModuleList([
            AdaLNBlock(embed_dim, num_heads, dim_head, mlp_dim, action_embed_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Predictor projector: MLP + BatchNorm (matches LeWM's pred_proj)
        self.pred_proj = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(
        self, frame_embs: torch.Tensor, ctrl_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Predict next frame embedding(s) from context.

        Args:
            frame_embs: (B, T, embed_dim) — encoder outputs for context frames
            ctrl_inputs: (B, T, ctrl_dim) — controller inputs per context frame

        Returns:
            pred_embs: (B, T, embed_dim) — predicted next embeddings per position
        """
        B, T, D = frame_embs.shape

        action_embs = self.action_encoder(ctrl_inputs)  # (B, T, action_embed_dim)
        x = frame_embs + self.pos_embed[:, :T, :]

        causal_mask = torch.tril(
            torch.ones(T, T, device=x.device, dtype=torch.bool)
        )

        for block in self.blocks:
            x = block(x, action_embs, causal_mask)

        x = self.final_norm(x)  # (B, T, embed_dim)

        # pred_proj has BatchNorm1d — needs (N, D) input
        x_flat = x.reshape(B * T, D)
        x_flat = self.pred_proj(x_flat)
        return x_flat.reshape(B, T, D)
