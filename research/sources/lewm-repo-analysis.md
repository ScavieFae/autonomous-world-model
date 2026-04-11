# LeWM Repo Analysis — Code-Level Reference

**Repo:** https://github.com/lucas-maes/le-wm
**License:** MIT

## Repo Structure (extremely lean — 5 Python files)

```
le-wm/
├── jepa.py          # Core JEPA model (encoder + predictor + rollout + cost)
├── module.py        # All neural net building blocks
├── train.py         # Training entry point (Hydra + PyTorch Lightning)
├── eval.py          # Evaluation/planning entry point (MPC with CEM or Adam)
├── utils.py         # Image preprocessing, column normalization, checkpoint callback
├── config/
│   ├── train/
│   │   ├── lewm.yaml          # Main training config
│   │   ├── data/              # Per-environment data configs
│   │   └── launcher/local.yaml
│   └── eval/
│       ├── pusht.yaml, tworoom.yaml, cube.yaml, reacher.yaml
│       ├── solver/cem.yaml, adam.yaml
│       └── launcher/local.yaml
```

Heavy lifting delegated to upstream libraries:
- **`stable-worldmodel`** (galilai-group) — environment wrappers, HDF5 data, planning/MPC, eval
- **`stable-pretraining`** (galilai-group) — ViT backbones, transforms, training wrappers

## Model Components (from module.py)

### Encoder
- HuggingFace ViT-tiny, patch size 14, image size 224
- Not pretrained — trained end-to-end from scratch
- CLS token extracted as observation embedding

### Projector
- MLP: `hidden_dim → 2048 → embed_dim` with BatchNorm
- Projects CLS token down to embedding space (default 192-dim)
- BatchNorm is critical: ViT's final LayerNorm interferes with SIGReg

### Action Encoder (Embedder)
- Conv1d(kernel=1) → MLP with SiLU activation
- Input: `action_dim * frameskip` (stacked across skipped frames)
- Output: `embed_dim` (192-dim)

### Predictor (ARPredictor)
- Causal Transformer with AdaLN-zero conditioning
- `ConditionalBlock` class implements the AdaLN modulation
- 6 modulation parameters per block: shift/scale/gate for attention and MLP
- 6 layers, 16 heads, dim_head=64, mlp_dim=2048
- Learned positional embeddings
- Context window of state embeddings → next-state predictions

### Predictor Projector (pred_proj)
- Same MLP structure as encoder projector
- Maps predictor output back to embedding space

### Rollout (Autoregressive)
Given initial pixels + action sequence:
1. Encode first frame
2. Truncate to `history_size` (default 3)
3. Predict next embedding
4. Append to sequence
5. Repeat

## Training Config (from lewm.yaml)

```yaml
# Key training hyperparameters
optimizer: AdamW
lr: 5e-5
weight_decay: 1e-3
gradient_clip: 1.0
precision: bf16
epochs: 100
batch_size: 128
train_val_split: 90/10

# SIGReg
lambda: 0.09  # (paper says 0.1, config says 0.09)
M: 1024       # random projections (insensitive per ablations)

# Data
frameskip: 5
history_size: 3
num_preds: 1  # predict 1 step ahead

# Architecture
embed_dim: 192
predictor_layers: 6
predictor_heads: 16
predictor_dim_head: 64
predictor_mlp_dim: 2048
predictor_dropout: 0.1
```

## Loss Implementation (from train.py::lejepa_forward)

```python
# Pseudocode reconstruction from analysis
def lejepa_forward(batch, model, lambda_sigreg=0.09):
    pixels, actions = batch  # (B, T, C, H, W), (B, T, A)
    
    # Encode all frames
    embeddings = model.encode(pixels)  # (B, T, D)
    action_embs = model.encode_actions(actions)
    
    # Context = first history_size frames
    context_embs = embeddings[:, :history_size]
    context_acts = action_embs[:, :history_size]
    
    # Predict next embedding
    pred_embs = model.predict(context_embs, context_acts)  # (B, 1, D)
    
    # Targets = actual embeddings of predicted frames
    target_embs = embeddings[:, history_size:]  # (B, 1, D)
    
    # Loss
    pred_loss = (pred_embs - target_embs).pow(2).mean()
    sigreg_loss = SIGReg(embeddings)  # applied stepwise across batch
    
    return pred_loss + lambda_sigreg * sigreg_loss
```

## Dependencies

```bash
uv pip install stable-worldmodel[train,env]
```

- `stable-worldmodel` — galilai-group: env wrappers, HDF5 data, planning/MPC
- `stable-pretraining` — galilai-group: ViT backbones, data transforms
- PyTorch + Lightning
- Hydra — config management
- einops — tensor reshaping
- WandB — experiment tracking
- MuJoCo (via MUJOCO_GL=egl) — for 3D environments
- scikit-learn — StandardScaler for eval

## Environments Tested

| Environment | Type | Action Space |
|---|---|---|
| PushT | 2D pushing | Continuous |
| TwoRoom | 2D navigation | Continuous |
| OGBench Cube | 3D manipulation (MuJoCo) | Continuous |
| Reacher (DMControl) | 3D arm (MuJoCo) | Continuous |

All use continuous action spaces. Planning via CEM (300 samples, 30 steps) or gradient-based Adam solver.
