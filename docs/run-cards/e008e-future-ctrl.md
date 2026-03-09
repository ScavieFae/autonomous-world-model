# Run Card: E008e — Future Controller Conditioning

**Created**: 2026-02-28
**Config**: `worldmodel/experiments/e008e-future-ctrl-2.yaml`
**Branch**: scav/E008e-future-ctrl
**wandb**: `ok5bww53` — https://wandb.ai/shinewave/melee-worldmodel/runs/ok5bww53
**Status**: COMPLETE

## Goal

Condition prediction on future controller inputs (ctrl at T+1, T+2) without shifting the prediction target. The rollback-inspired idea: future ctrl is available via speculative lookahead or opponent modeling, even when future state isn't. Tests whether knowing what buttons will be pressed helps predict the current frame's state.

## Results

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| change_acc | 64.7% | 67.3% | **-2.6pp** |
| action_acc | 96.4% | — | — |
| pos_mae | 0.65 | 0.65 | 0 |
| val_loss | 0.307 | 0.291 | +0.016 |

**Verdict**: Below baseline. Future ctrl didn't help and may have hurt. Hypothesis: at 60fps, ~80% of consecutive controller frames are identical (human motor autocorrelation). The ~5% with discrete transitions are the only frames where future ctrl differs from current ctrl — and without knowing future actionability state, the model can't tell if those inputs will take effect. The wider conditioning dimension (78 vs 26) likely added noise.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-2k.pt` (pre-v3) |
| Games | 2,000 |
| Train examples | 17,721,679 (1,800 games) |
| Val examples | 1,961,279 (200 games) |
| Encoding | state_flags=false, hitstun=false, projectiles=false |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Parameters | 4,302,354 |
| d_model / n_layers | 384 / 4 |
| context_len (K) | 10 |
| chunk_size | 10 |
| future_ctrl_frames | 2 |

## Training

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Epochs | 2 |
| Batch size | 4096 |
| LR | 5e-4 |
| Wall time | ~31 min |
| Epoch 1 | loss=0.560, action_acc=0.946, change_acc=47.0%, pos_mae=0.79, val_loss=0.359 |
| Epoch 2 | loss=0.307, action_acc=0.964, change_acc=64.7%, pos_mae=0.65, val_loss=0.307 |

## Connection to GGPO/Rollback

See `research/notes/rollback-delay-frames.md`. Future ctrl alone is insufficient because controller inputs are meaningless without actionability context. This informed the E009 direction: add state_flags (which encode actionability) rather than future ctrl.
