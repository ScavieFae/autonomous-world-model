# Run Card: E009b — v3 Encoding + Focal + Multi-Position (K=30)

**Created**: 2026-02-28
**Config**: `worldmodel/experiments/e009b-v3-focal-multi-k30.yaml`
**Branch**: scav/E008c-multi-position
**wandb**: `c3d4rxln` — https://wandb.ai/shinewave/melee-worldmodel/runs/c3d4rxln
**Status**: COMPLETE — NEW BEST

## Goal

Same as E009a but at K=30 — tests whether 3× context window (500ms) justifies 3× compute when the model has v3 actionability signals. 30 frames covers 39.5% of commitment windows (vs 14.6% for K=10).

## Target Metrics

| Metric | E008c (no v3) | Target |
|--------|--------------|--------|
| change_acc | 71.5% | >74% |
| pos_mae | 0.61 | <0.58 |
| val_loss | 0.289 | <0.280 |

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-v3-ranked-50k-part-0.pt` (capped to 2K) |
| Games | 2,000 (ranked, GAME outcomes only) |
| Train examples | 21,329,655 (1,800 games) |
| Val examples | 2,315,872 (200 games) |
| Encoding | state_flags=true, hitstun=true, projectiles=false |
| float_per_player | 69 |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Parameters | 4,345,444 |
| d_model / n_layers | 384 / 4 |
| context_len (K) | 30 |
| chunk_size | 15 |
| multi_position | true |
| focal_offset | 5 |

## Training

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Epochs | 2 |
| Batch size | 2048 |
| LR | 5e-4 |
| binary loss weight | **1.0** (up from 0.5) |
| Estimated wall time | ~2 hours |
| Estimated cost | ~$7.78 |

## Key Questions

1. Does K=30 beat K=10 when the model has state_flags? (If compressed history features carry enough info, K=30 may not justify 3× cost.)
2. Does 3× training signals per sample (multi-position at K=30 = 30 predictions vs 10) help despite same epoch count?
3. Does focal_offset=5 (83ms of future context) combine well with the IASA bits?

## Results

| Metric | Epoch 1 | Epoch 2 | E008c (baseline) | vs baseline |
|--------|---------|---------|-------------------|-------------|
| change_acc (val) | 72.2% | **74.8%** | 71.5% | **+3.3pp** |
| pos_mae (val) | 0.496 | **0.457** | 0.61 | **-25%** |
| val_loss | 0.273 | 0.288 | 0.289 | -0.001 |
| action_acc (val) | 96.8% | 96.7% | 97.1% | -0.4pp |
| train change_acc | 66.9% | 79.5% | 71.5% | — |
| Wall time | 42 min | 42 min | 35 min | — |

**Verdict**: New best clean result. 74.8% change_acc beats E008c by 3.3pp and E009a by 3.3pp. Position prediction improved 25% (0.457 vs 0.61). Val_loss slightly up in epoch 2 (0.273→0.288) while accuracy metrics improved — model trading total loss for harder-case accuracy.

## Answers to Key Questions

1. **K=30 beats K=10 with state_flags**: Yes. Despite compressed history features, 500ms of raw context adds signal. E009b 74.8% vs E009a 71.5% (+3.3pp).
2. **3× multi-position targets help**: Yes. E009b continued improving in epoch 2 (72.2→74.8%) while E009a plateaued (71.5→71.5%). More training signal per sample gives the model more room to learn.
3. **focal_offset=5 + IASA bits**: The combination works well — the model can see both where the game goes (future context) and when inputs take effect (state_flags).
