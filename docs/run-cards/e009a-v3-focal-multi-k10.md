# Run Card: E009a — v3 Encoding + Focal + Multi-Position (K=10)

**Created**: 2026-02-28
**Config**: `worldmodel/experiments/e009a-v3-focal-multi-k10.yaml`
**Branch**: scav/E008c-multi-position
**wandb**: `wqkltuzj` — https://wandb.ai/shinewave/melee-worldmodel/runs/wqkltuzj
**Status**: COMPLETE

## Goal

Graduate E008's best ideas (multi-position + focal context) to v3 encoding. Key new signals:
- **byte0.bit2** (84.9% freq): "has control / actionable" — OFF during dash startup, jump startup, etc.
- **byte3.bit5** (33.9% freq): "can cancel / IASA window" — ON when action is interruptible

These directly tell the model when controller inputs should take effect — the core problem identified in E008.

## Target Metrics

| Metric | E008c (no v3) | Target |
|--------|--------------|--------|
| change_acc | 71.5% | >73% |
| pos_mae | 0.61 | <0.60 |
| val_loss | 0.289 | <0.285 |

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-v3-ranked-50k-part-0.pt` (capped to 2K) |
| Games | 2,000 (ranked, GAME outcomes only) |
| Train examples | 21,369,255 (1,800 games) |
| Val examples | 2,320,272 (200 games) |
| Encoding | state_flags=true, hitstun=true, projectiles=false |
| float_per_player | 69 (vs 28 in E008) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Parameters | 4,345,444 |
| d_model / n_layers | 384 / 4 |
| context_len (K) | 10 |
| chunk_size | 10 |
| multi_position | true |
| focal_offset | 3 |

## Training

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Epochs | 2 |
| Batch size | 4096 |
| LR | 5e-4 |
| binary loss weight | **1.0** (up from 0.5) |

## Results

| Metric | Epoch 1 | Epoch 2 | E008c (baseline) | vs baseline |
|--------|---------|---------|-------------------|-------------|
| change_acc (val) | — | **71.5%** | 71.5% | **+0.0pp** |
| pos_mae (val) | — | **0.491** | 0.61 | **-20%** |
| val_loss | — | 0.278 | 0.289 | -0.011 |
| action_acc (val) | — | 96.7% | 97.1% | -0.4pp |

**Verdict**: Matches E008c change_acc (71.5%) but with 20% better position prediction (0.491 vs 0.61). Val_loss improved (0.278 vs 0.289). The v3 encoding helps position/loss but doesn't lift change_acc at K=10 — the actionability bits alone can't compensate for limited context. E009a plateaued after epoch 1 (no improvement in epoch 2), suggesting v3 features are fast learners that saturate quickly at K=10.

## Comparison to E009b

E009a is the K=10 control. E009b (K=30) beat E009a by 3.3pp on change_acc (74.8% vs 71.5%). Despite compressed history features, the longer context window provides additional signal. E009b also continued improving in epoch 2 while E009a plateaued — the 3× multi-position training signal from K=30 gives more room to learn.
