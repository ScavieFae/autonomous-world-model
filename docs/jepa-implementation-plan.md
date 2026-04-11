# JEPA World Model — Implementation Plan

## Overview

Adapt LeWorldModel (arXiv 2603.19312) to structured Melee game state. 8 files, matching LeWM's architecture as closely as possible. The encoder is the one accepted divergence (MLP on game state instead of ViT on pixels).

## Reference Implementations

- **Primary:** LeWM (`lucas-maes/le-wm`) — `jepa.py`, `module.py`, `train.py`
- **Secondary:** EB-JEPA (`facebookresearch/eb_jepa`) — `jepa.py`, `planning.py`
- **Saved summaries:** `research/sources/2603.19312-summary.md`, `research/sources/lewm-repo-analysis.md`

## Architecture

```
Game State (float + int per frame)
    → Encoder (categorical embeddings + MLP + BatchNorm → 192-dim)
    → Predictor (6L Transformer with AdaLN-zero, conditioned on controller inputs)
    → Next latent embedding (192-dim)

Loss = MSE(predicted_emb, target_emb) + 0.1 * SIGReg(all_emb)
```

## File 1: `models/jepa/__init__.py`

```python
from models.jepa.model import JEPAWorldModel
from models.jepa.encoder import GameStateEncoder
from models.jepa.predictor import ARPredictor
from models.jepa.sigreg import sigreg

__all__ = ["JEPAWorldModel", "GameStateEncoder", "ARPredictor", "sigreg"]
```

## File 2: `models/jepa/encoder.py`

GameStateEncoder: maps one frame of game state → 192-dim latent embedding.

**Design:**
- Categorical embeddings matching EncodingConfig vocab sizes (same vocabs as Mamba2: action=400, jumps=8, character=33, stage=33, l_cancel=3, hurtbox=3, ground=32, last_attack=64, optionally state_age=150)
- Continuous features passed through directly (already normalized by EncodingConfig scales)
- All concatenated → MLP trunk → projector with BatchNorm

**Matching LeWM:** LeWM's encoder is ViT → CLS token → `MLP(hidden_dim, 2048, embed_dim)` with BatchNorm. We replace ViT with categorical embeddings + continuous concat, keep the MLP+BN projector.

```python
class GameStateEncoder(nn.Module):
    """Encodes per-frame game state into a latent embedding.
    
    Replaces LeWM's ViT encoder. Accepted divergence: structured game state
    instead of pixels. The projector (MLP + BatchNorm → embed_dim) matches LeWM.
    """
    
    def __init__(self, cfg: EncodingConfig, embed_dim: int = 192):
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
            self.state_age_embed = nn.Embedding(cfg.state_age_embed_vocab, cfg.state_age_embed_dim)
        
        # Input dim: float features (both players) + embedded categoricals (both players) + stage
        float_dim = cfg.float_per_player * 2  # continuous + binary + controller, both players
        cat_embed_per_player = (
            cfg.action_embed_dim + cfg.jumps_embed_dim + cfg.character_embed_dim
            + cfg.l_cancel_embed_dim + cfg.hurtbox_embed_dim
            + cfg.ground_embed_dim + cfg.last_attack_embed_dim
        )
        if cfg.state_age_as_embed:
            cat_embed_per_player += cfg.state_age_embed_dim
        total_input_dim = float_dim + 2 * cat_embed_per_player + cfg.stage_embed_dim
        
        # MLP trunk + projector with BatchNorm (matching LeWM's projector pattern)
        # LeWM: hidden_dim → 2048 → embed_dim with BatchNorm
        hidden_dim = 512  # intermediate representation size
        self.trunk = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Projector: matches LeWM's MLP(hidden, 2048, embed_dim) + BatchNorm
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
    
    def forward(self, float_frames, int_frames):
        """Encode frames into latent embeddings.
        
        Args:
            float_frames: (B, T, F) or (B*T, F) — continuous + binary + controller
            int_frames: (B, T, I) or (B*T, I) — categorical indices
            
        Returns:
            embeddings: same leading dims as input, last dim = embed_dim
        """
        # Handle both (B, T, ...) and (B*T, ...) inputs
        leading_shape = float_frames.shape[:-1]
        F = float_frames.shape[-1]
        I = int_frames.shape[-1]
        
        flat_float = float_frames.reshape(-1, F)
        flat_int = int_frames.reshape(-1, I)
        
        # Int column indices (same layout as Mamba2's _int_cols)
        ipp = self.cfg.int_per_player
        
        # Embed categoricals for both players
        parts = [flat_float]  # continuous + binary + controller already here
        
        for player_offset in [0, ipp]:
            parts.append(self.action_embed(flat_int[:, player_offset + 0]))
            parts.append(self.jumps_embed(flat_int[:, player_offset + 1]))
            parts.append(self.character_embed(flat_int[:, player_offset + 2]))
            parts.append(self.l_cancel_embed(flat_int[:, player_offset + 3]))
            parts.append(self.hurtbox_embed(flat_int[:, player_offset + 4]))
            parts.append(self.ground_embed(flat_int[:, player_offset + 5]))
            parts.append(self.last_attack_embed(flat_int[:, player_offset + 6]))
            if self.cfg.state_age_as_embed:
                parts.append(self.state_age_embed(flat_int[:, player_offset + 7]))
        
        # Stage embedding (last column in int_frames)
        parts.append(self.stage_embed(flat_int[:, -1]))
        
        x = torch.cat(parts, dim=-1)  # (B*T, total_input_dim)
        x = self.trunk(x)             # (B*T, hidden_dim)
        x = self.projector(x)         # (B*T, embed_dim)
        
        return x.reshape(*leading_shape, self.embed_dim)
```

**Notes:**
- BatchNorm1d in the projector: LeWM uses this explicitly because ViT's final LayerNorm interferes with SIGReg. We include it for the same reason — to ensure the embedding distribution can be properly regularized.
- The trunk is 2-layer MLP with SiLU (matching LeWM's activation choice).
- Controller inputs are part of the float_frames (they're in the context). This means the encoder sees the controller state as part of the observation, same as LeWM seeing the full pixel frame.

## File 3: `models/jepa/predictor.py`

ARPredictor: Transformer with AdaLN-zero conditioning, matching LeWM's `ARPredictor` + `ConditionalBlock`.

**Design:**
- Causal Transformer (standard multi-head self-attention + MLP)
- AdaLN-zero: at each layer, controller input generates 6 modulation parameters (shift1, scale1, gate1, shift2, scale2, gate2) for the attention and MLP sub-layers
- Zero-initialized modulation so action conditioning ramps up gradually during training
- Learned positional embeddings

```python
class AdaLNBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization conditioning.
    
    Matches LeWM's ConditionalBlock: LayerNorm params modulated by action input.
    6 modulation parameters per block: (shift, scale, gate) × 2 sublayers.
    Gate is zero-initialized so conditioning ramps up gradually.
    """
    
    def __init__(self, dim: int, num_heads: int, dim_head: int, mlp_dim: int,
                 action_dim: int, dropout: float = 0.1):
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
        # Zero-initialize the linear layer so gates start at 0
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
    
    def forward(self, x, action_emb, causal_mask=None):
        """
        Args:
            x: (B, T, dim) — sequence of frame embeddings
            action_emb: (B, T, action_dim) — per-step action embeddings
            causal_mask: (T, T) boolean mask for causal attention
            
        Returns:
            (B, T, dim)
        """
        B, T, D = x.shape
        
        # Compute modulation parameters from action
        mod = self.adaLN_modulation(action_emb)  # (B, T, 6*D)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)
        
        # Attention sublayer with AdaLN
        h = self.norm1(x)
        h = h * (1 + scale1) + shift1  # modulated normalization
        
        # Multi-head self-attention
        qkv = self.to_qkv(h).reshape(B, T, 3, self.num_heads, self.dim_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, heads, dim_head)
        q = q.transpose(1, 2)  # (B, heads, T, dim_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_head ** 0.5)
        if causal_mask is not None:
            attn = attn.masked_fill(~causal_mask, float('-inf'))
        attn = self.attn_dropout(torch.softmax(attn, dim=-1))
        
        h = torch.matmul(attn, v)  # (B, heads, T, dim_head)
        h = h.transpose(1, 2).reshape(B, T, -1)  # (B, T, inner_dim)
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
    
    Matches LeWM's ARPredictor: causal Transformer, learned positional embeddings,
    AdaLN modulation from action inputs at each layer.
    """
    
    def __init__(
        self,
        embed_dim: int = 192,
        num_layers: int = 6,
        num_heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        action_input_dim: int = 26,  # controller dim (cfg.ctrl_conditioning_dim)
        action_embed_dim: int = 192,  # action embedding size (matches embed_dim)
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Action encoder: controller input → action embedding
        # LeWM uses Conv1d(kernel=1) + MLP with SiLU
        # We skip Conv1d (no frameskip stacking) and just use MLP
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
        
        # Predictor projector: matches LeWM's pred_proj (MLP + BatchNorm)
        self.pred_proj = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.SiLU(),
            nn.Linear(2048, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
    
    def forward(self, frame_embs, ctrl_inputs):
        """Predict next frame embedding(s) from context.
        
        Args:
            frame_embs: (B, T, embed_dim) — encoder outputs for context frames
            ctrl_inputs: (B, T, ctrl_dim) — controller inputs for each context frame
            
        Returns:
            pred_embs: (B, T, embed_dim) — predicted next embeddings for each position
        """
        B, T, D = frame_embs.shape
        
        # Encode actions
        action_embs = self.action_encoder(ctrl_inputs)  # (B, T, action_embed_dim)
        
        # Add positional embeddings
        x = frame_embs + self.pos_embed[:, :T, :]
        
        # Causal mask: each position can attend to itself and earlier positions
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, action_embs, causal_mask)
        
        x = self.final_norm(x)  # (B, T, embed_dim)
        
        # Project through pred_proj (BatchNorm needs (B*T, D) shape)
        x_flat = x.reshape(B * T, D)
        x_flat = self.pred_proj(x_flat)
        pred_embs = x_flat.reshape(B, T, D)
        
        return pred_embs
```

**Notes:**
- Action encoder: LeWM uses `Conv1d(kernel=1) + MLP` because it stacks actions across frameskip. We don't frameskip, so a plain MLP suffices. This is a minor accepted divergence.
- The `pred_proj` with BatchNorm matches LeWM's predictor projector — this is important because it maps predictions into the same space as encoder outputs.
- GELU activation in the MLP sublayer (standard Transformer choice, matching LeWM's usage).

## File 4: `models/jepa/sigreg.py`

SIGReg: Sketch Isotropic Gaussian Regularizer from LeJEPA (arXiv 2511.08544).

**IMPORTANT: Port LeWM's exact implementation.** The LeWM repo (`le-wm/module.py`) is MIT-licensed. Port the SIGReg function directly rather than reimplementing from scratch. The Epps-Pulley test has specific weighting and normalization that a hand-rolled version will get wrong (arbitrary frequency grids, missing kernel weights, etc.).

```python
# Port directly from lucas-maes/le-wm/module.py — MIT license
# Do NOT hand-roll the characteristic function test.
# The reference implementation handles:
#   - Proper Epps-Pulley weighting kernel
#   - Correct standardization
#   - Random projection sampling
#   - Gradient-friendly formulation
#
# Fetch the exact implementation at build time and adapt only
# what's necessary (import paths, device handling).
```

**Notes:**
- The Cramer-Wold theorem guarantees that if all 1D projections are Gaussian, the full joint distribution is Gaussian. SIGReg tests this via random projections.
- Directions are re-sampled each call (not persistent), matching LeWM.
- Per implementation rule 5: use the existing tool, don't hand-roll.

## File 5: `models/jepa/model.py`

JEPAWorldModel: wraps encoder + predictor, provides training forward and rollout.

```python
class JEPAWorldModel(nn.Module):
    """JEPA World Model for Melee.
    
    Training: encode all frames → predict next embeddings → MSE + SIGReg loss.
    Rollout: autoregressive prediction in latent space.
    """
    
    def __init__(
        self,
        cfg: EncodingConfig,
        embed_dim: int = 192,
        history_size: int = 3,
        predictor_layers: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        predictor_dropout: float = 0.1,
        sigreg_lambda: float = 0.1,
        sigreg_projections: int = 1024,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = embed_dim
        self.history_size = history_size
        self.context_len = history_size  # alias for compatibility
        self.sigreg_lambda = sigreg_lambda
        self.sigreg_projections = sigreg_projections
        
        self.encoder = GameStateEncoder(cfg, embed_dim=embed_dim)
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
    
    def forward(self, float_frames, int_frames, ctrl_inputs):
        """Training forward pass.
        
        Args:
            float_frames: (B, T, F) — all frames in subsequence (context + target)
            int_frames: (B, T, I) — categorical indices for all frames
            ctrl_inputs: (B, T, C) — controller inputs for all frames
            
        Returns:
            dict with:
                pred_loss: MSE between predicted and target embeddings
                sigreg_loss: SIGReg regularization loss
                total_loss: pred_loss + lambda * sigreg_loss
                embeddings: (B, T, D) — all frame embeddings (for logging)
        """
        B, T, F = float_frames.shape
        H = self.history_size
        num_preds = T - H
        assert num_preds == 1, (
            f"Currently only num_preds=1 is supported (got T={T}, H={H}, "
            f"num_preds={num_preds}). Multi-step prediction requires "
            f"autoregressive unrolling during training — flag as divergence."
        )
        
        # 1. Encode ALL frames
        all_embs = self.encoder(float_frames, int_frames)  # (B, T, D)
        
        # 2. Context: first H frames; target: frame H onward
        context_embs = all_embs[:, :H]          # (B, H, D)
        target_embs = all_embs[:, H:]           # (B, 1, D)
        context_ctrl = ctrl_inputs[:, :H]       # (B, H, C)
        
        # 3. Predict next embedding from context
        pred_embs = self.predictor(context_embs, context_ctrl)  # (B, H, D)
        # We want the prediction from the last context position
        pred_next = pred_embs[:, -1:]           # (B, 1, D)
        
        # 4. Prediction loss: MSE in latent space (both tensors are (B, 1, D))
        pred_loss = (pred_next - target_embs).pow(2).mean()
        
        # 5. SIGReg: applied to all embeddings in the batch
        # Reshape to (B*T, D) for SIGReg
        all_embs_flat = all_embs.reshape(-1, self.embed_dim)
        sigreg_loss = sigreg(all_embs_flat, self.sigreg_projections)
        
        total_loss = pred_loss + self.sigreg_lambda * sigreg_loss
        
        return {
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "total_loss": total_loss,
            "embeddings": all_embs.detach(),
        }
    
    def rollout(self, float_ctx, int_ctx, initial_ctrls, ctrl_sequence):
        """Autoregressive rollout in latent space.
        
        Args:
            float_ctx: (B, H, F) — initial context frames
            int_ctx: (B, H, I) — initial context categoricals
            initial_ctrls: (B, H, C) — controller inputs for the initial context frames
            ctrl_sequence: (B, N, C) — controller inputs for N future steps
            
        Returns:
            pred_embs: (B, N, D) — predicted embeddings for each future step
        """
        B, H, F = float_ctx.shape
        N = ctrl_sequence.shape[1]
        
        # Encode initial context
        embs = self.encoder(float_ctx, int_ctx)  # (B, H, D)
        
        # Maintain a parallel controller buffer — each context position needs
        # its own historical controller (the action that produced the transition
        # FROM that frame TO the next). This matches LeWM's training where each
        # context frame has its own action.
        ctrl_buffer = initial_ctrls  # (B, H, C)
        
        predictions = []
        for step in range(N):
            # Truncate both buffers to history_size
            ctx = embs[:, -self.history_size:]          # (B, H, D)
            ctx_ctrl = ctrl_buffer[:, -self.history_size:]  # (B, H, C)
            
            pred = self.predictor(ctx, ctx_ctrl)  # (B, H, D)
            next_emb = pred[:, -1:]  # (B, 1, D)
            predictions.append(next_emb)
            
            # Append prediction and its controller to buffers
            embs = torch.cat([embs, next_emb], dim=1)
            ctrl_buffer = torch.cat([ctrl_buffer, ctrl_sequence[:, step:step+1]], dim=1)
        
        return torch.cat(predictions, dim=1)  # (B, N, D)
```

## File 6: `data/jepa_dataset.py`

JEPAFrameDataset: returns subsequences of (frames, actions) for JEPA training.

```python
class JEPAFrameDataset(Dataset):
    """Dataset for JEPA training — returns frame subsequences.
    
    Unlike MeleeFrameDataset (which returns single-frame predictions),
    this returns full subsequences of (float, int, ctrl) for JEPA's
    encode-all-then-predict training pattern.
    
    Returns:
        float_frames: (T, F) — T frames of float features
        int_frames: (T, I) — T frames of categorical indices
        ctrl_inputs: (T, C) — T frames of controller inputs
        
    where T = history_size + num_preds (default 3 + 1 = 4)
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
    
    def __len__(self):
        return len(self.valid_indices)
    
    def _extract_ctrl(self, float_frame):
        """Extract controller inputs from a float frame."""
        parts = [float_frame[self._p0_ctrl], float_frame[self._p1_ctrl]]
        if self._ctrl_threshold:
            p0_analog = float_frame[self._p0_analog]
            p1_analog = float_frame[self._p1_analog]
            parts.append((p0_analog.abs() > 0.3).float())
            parts.append((p1_analog.abs() > 0.3).float())
        return torch.cat(parts)
    
    def __getitem__(self, idx):
        t = int(self.valid_indices[idx])
        T = self.seq_len
        
        float_frames = self.data.floats[t:t + T]  # (T, F)
        int_frames = self.data.ints[t:t + T]       # (T, I)
        
        # Extract controller inputs for each frame
        ctrl_inputs = torch.stack([
            self._extract_ctrl(float_frames[i]) for i in range(T)
        ])  # (T, C)
        
        return float_frames, int_frames, ctrl_inputs
    
    def get_batch(self, indices):
        """Vectorized batch loading (matching MeleeFrameDataset pattern)."""
        T = self.seq_len
        ts = self.valid_indices[indices]
        
        offsets = np.arange(T)
        frame_indices = ts[:, None] + offsets[None, :]  # (B, T)
        
        float_frames = self.data.floats[frame_indices]  # (B, T, F)
        int_frames = self.data.ints[frame_indices]       # (B, T, I)
        
        # Extract controller inputs
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
```

## File 7: `training/jepa_trainer.py`

JEPATrainer: training loop matching LeWM's training regime.

```python
class JEPATrainer:
    """Training loop for the JEPA world model.
    
    Simpler than the main Trainer — no per-field loss heads, no self-forcing.
    Loss is MSE in latent space + SIGReg.
    
    Optimizer: AdamW, lr=5e-5, wd=1e-3, gradient clip 1.0 (matching LeWM)
    """
    
    def __init__(
        self,
        model: JEPAWorldModel,
        train_dataset: JEPAFrameDataset,
        val_dataset: Optional[JEPAFrameDataset],
        *,
        lr: float = 5e-5,
        weight_decay: float = 1e-3,
        batch_size: int = 128,
        num_epochs: int = 100,
        save_dir: Optional[Path] = None,
        device: Optional[str] = None,
        use_amp: bool = False,
        warmup_pct: float = 0.0,
        gradient_clip: float = 1.0,
    ):
        # Device setup (same as main Trainer)
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        
        self._use_amp = use_amp and (device == "cuda")
        self._amp_dtype = torch.float16
        self._scaler = GradScaler(enabled=self._use_amp)
        self.gradient_clip = gradient_clip
        
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Optimizer: AdamW matching LeWM defaults
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # LR schedule: optional warmup + cosine (matching LeWM)
        # ... (same warmup + cosine pattern as main Trainer)
        
        # Fast batch loader (reuse pattern from main Trainer)
        self._use_fast_loader = hasattr(train_dataset, 'get_batch')
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4 if device == "cuda" else 0, drop_last=True,
        )
        self.val_loader = None
        if val_dataset and len(val_dataset) > 0:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            )
        
        self.history = []
    
    def _train_epoch(self):
        self.model.train()
        total_pred_loss = 0.0
        total_sigreg_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            float_frames, int_frames, ctrl_inputs = [x.to(self.device) for x in batch]
            
            self.optimizer.zero_grad()
            
            with autocast("cuda", enabled=self._use_amp, dtype=self._amp_dtype):
                output = self.model(float_frames, int_frames, ctrl_inputs)
                loss = output["total_loss"]
            
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            
            total_pred_loss += output["pred_loss"].item()
            total_sigreg_loss += output["sigreg_loss"].item()
            total_loss += output["total_loss"].item()
            num_batches += 1
        
        return {
            "pred_loss": total_pred_loss / num_batches,
            "sigreg_loss": total_sigreg_loss / num_batches,
            "total_loss": total_loss / num_batches,
        }
    
    def train(self):
        best_val_loss = float("inf")
        
        for epoch in range(self.num_epochs):
            train_metrics = self._train_epoch()
            # ... val loop, checkpointing, logging (similar to main Trainer)
            
            logger.info(
                "Epoch %d/%d: pred=%.4f sigreg=%.4f total=%.4f",
                epoch + 1, self.num_epochs,
                train_metrics["pred_loss"],
                train_metrics["sigreg_loss"],
                train_metrics["total_loss"],
            )
```

## File 8: `scripts/train_jepa.py`

Entry point. Loads data, creates model, trains.

```python
"""Train the JEPA world model.

Usage:
    python -m scripts.train_jepa --config experiments/e030a-jepa-baseline.yaml
"""

def main():
    # Parse config (YAML)
    # Load data (reuse existing data loading pipeline)
    # Create MeleeDataset
    # Create JEPAFrameDataset (train + val splits)
    # Create JEPAWorldModel
    # Create JEPATrainer
    # Train
    # Log results

    # ... (standard script structure matching scripts/train.py)
```

## Experiment Config: `experiments/e030a-jepa-baseline.yaml`

```yaml
# e030a: JEPA baseline — LeWM architecture on Melee game state
# First test of JEPA paradigm on structured (non-pixel) data

encoding:
  # Match b002 data contract exactly
  xy_scale: 0.05
  action_vocab: 400
  action_embed_dim: 32
  state_age_as_embed: true
  state_flags: true
  hitstun: true
  ctrl_threshold_features: true
  multi_position: true

model:
  arch: jepa
  embed_dim: 192          # match LeWM
  history_size: 3          # match LeWM
  predictor_layers: 6      # match LeWM
  predictor_heads: 16      # match LeWM
  predictor_dim_head: 64   # match LeWM
  predictor_mlp_dim: 2048  # match LeWM
  predictor_dropout: 0.1   # match LeWM
  sigreg_lambda: 0.1       # match LeWM
  sigreg_projections: 1024 # match LeWM
  encoder_hidden_dim: 512  # our encoder trunk

training:
  batch_size: 128          # match LeWM
  num_epochs: 100          # match LeWM
  lr: 0.00005              # match LeWM (5e-5)
  weight_decay: 0.001      # match LeWM
  gradient_clip: 1.0       # match LeWM
  use_amp: true
  warmup_pct: 0.05         # from b002
```

## Param Count Estimate

With b002 encoding flags (state_age_as_embed, state_flags, hitstun, ctrl_threshold):
- float_per_player ≈ 74 (continuous+binary+controller with state_flags=40 bits)
- float_dim = 74 × 2 = 148
- cat_embed_per_player = 32+4+8+2+2+4+8+8 = 68 (with state_age_as_embed)
- total_input_dim ≈ 148 + 2×68 + 4 = 288

| Component | Params | Notes |
|-----------|--------|-------|
| Encoder embeddings | ~25K | same vocabs as Mamba2 |
| Encoder trunk MLP | ~410K | 288→512→512 (2 layers) |
| Encoder projector | ~1.1M | 512→2048→192 + BN |
| Predictor action encoder | ~82K | 36→192→192 (with ctrl_threshold) |
| Predictor pos embed | ~12K | 64×192 |
| Predictor blocks (×6) | ~10.8M | attention(192→1024→192) + MLP(192→2048→192) + AdaLN(192→1152) × 6 |
| Predictor final norm | ~384 | LayerNorm |
| Predictor projector | ~0.8M | 192→2048→192 + BN |
| **Total** | **~13.2M** | vs LeWM's ~15M, vs Mamba2's ~15.8M |

Smaller than LeWM due to MLP encoder (~1.5M) vs ViT-tiny (~5M). Same ballpark — not a concern. Could increase encoder depth in a follow-up if the bottleneck is there.

## Open Design Questions

1. ~~**Rollout controller handling:**~~ **RESOLVED.** Maintain a parallel controller buffer alongside the embedding buffer during rollout. Initial context controllers come from seed data; subsequent controllers come from the ctrl_sequence input. Each context position sees its actual historical controller. See updated `rollout()` method above.

2. **Decoder for eval:** Not in v1. Needed for rollout coherence comparison with Mamba2. Plan: train lightweight MLP decoder on frozen encoder, mapping latent → game state fields. **Follow-up after training works.**

3. **SIGReg on what?** LeWM applies SIGReg to all (B*T, D) embeddings (context + target). We match this. Regularizing target embeddings is important — if excluded, the encoder could collapse target representations without penalty. **Resolved: follow LeWM.**

4. **BatchNorm1d with small batch dimension:** B*T = 128*4 = 512 is adequate for BatchNorm (standard vision models run BN with 32-256). **Resolved: not a concern.**

5. **Controller inputs in encoder:** The encoder sees controller state as part of the frame observation (it's in float_frames). The predictor also conditions on controllers via AdaLN. This double-exposure is fine — the encoder needs controller to represent the current state; the predictor needs controller to predict the dynamics. No information leakage since the encoder sees current-frame controllers, not future ones. **Resolved: not a concern.**

## Future Levers (experiment directions after e030a)

These are design decisions locked to LeWM defaults for the baseline. Each is a potential experiment axis:

| Lever | e030a Value | Range to Explore | Why |
|-------|------------|------------------|-----|
| `history_size` | 3 (50ms) | 3–30 (50–500ms) | LeWM uses 3 with frameskip=5. We have no frameskip. Melee may need more context. |
| `embed_dim` | 192 | 128–512 | Higher = more capacity, more SIGReg dimensions to regularize |
| `encoder_depth` | 2 layers | 2–6 layers | MLP encoder is much shallower than LeWM's 12-layer ViT. May bottleneck. |
| `predictor_layers` | 6 | 2–8 | Standard depth sweep |
| `sigreg_lambda` | 0.1 | 0.01–1.0 | LeWM says bisection search works in O(log n) |
| `batch_size` | 128 | 128–1024 | Short sequences = less memory. Can we go bigger? |
| `num_preds` | 1 | 1–3 | Multi-step prediction during training (requires AR unrolling in forward) |
| Frameskip | None | 2–5 | Skip frames, stack actions (LeWM pattern). Longer effective context. |
| Action tokenization | Continuous (raw ctrl) | Discrete vocab | Julian's question — define Melee action primitives |
| Decoder training | Separate (frozen encoder) | Joint (auxiliary loss) | Trade purity for guaranteed decodability |
| Self-forcing | None | From Mamba2 side | Compose JEPA + SF? Major divergence, flag first. |
