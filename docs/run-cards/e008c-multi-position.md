# Run Card: E008c — Multi-Position Prediction

**Created**: 2026-02-28
**Config**: `worldmodel/experiments/e008c-multi-position.yaml`
**Branch**: scav/E008c-multi-position
**wandb**: `hcujei2k` — https://wandb.ai/shinewave/melee-worldmodel/runs/hcujei2k
**Status**: COMPLETE — BEST CLEAN RESULT

## Goal

GPT-style predict-at-every-position. At each position i in the context window, predict frame i+1 from the causal hidden state. Gives K× more training signal per sample. No future context needed — architecturally clean.

## Results

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| change_acc | **71.5%** | 67.3% | **+4.2pp** |
| action_acc | 97.1% | — | — |
| pos_mae | **0.61** | 0.65 | **-0.04** |
| val_loss | **0.289** | 0.291 | **-0.002** |

**Verdict**: Best clean result of the E008 series. Best pos_mae and val_loss. No train/eval mismatch — inference works the same way (predict from last position). Graduated to E009 with v3 encoding.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-2k.pt` (pre-v3) |
| Games | 2,000 |
| Train examples | 17,725,279 (1,800 games) |
| Val examples | 1,961,679 (200 games) |
| Encoding | state_flags=false, hitstun=false, projectiles=false |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Parameters | 4,282,386 |
| d_model / n_layers | 384 / 4 |
| context_len (K) | 10 |
| chunk_size | 10 |
| multi_position | true |
| focal_offset | 0 |

## Training

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Epochs | 2 |
| Batch size | 4096 |
| LR | 5e-4 |
| Wall time | ~35 min |
| Epoch 1 | loss=0.460, action_acc=0.955, change_acc=56.1%, pos_mae=0.72, val_loss=0.301 |
| Epoch 2 | loss=0.231, action_acc=0.971, change_acc=71.5%, pos_mae=0.61, val_loss=0.289 |

## Key Findings

- 10× training signal per sample acts as regularization — model can't overfit to predicting only the final frame.
- Per-position ctrl conditioning: positions 0..K-2 use ctrl extracted from the next context frame, position K-1 uses next_ctrl.
- Bug on first launch: `_val_epoch` used original (B,K,...) shaped predictions after `compute_loss` had reshaped internally via `.pop()`. Fixed by using `.get()` and adding dim-check reshape in `_val_epoch`.
- Combined with E008a (focal context) in E009 to test whether multi-position regularization makes focal context benefits survive at inference.
