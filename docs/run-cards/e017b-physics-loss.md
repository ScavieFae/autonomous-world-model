# Run Card: e017b-physics-loss

**Created**: 2026-03-02
**Config**: `worldmodel/experiments/e017b-physics-loss.yaml`
**Status**: LAUNCHING

## Goal

Add physics consistency loss: penalize |predicted_y| when ground truth says on_ground=true. Teaches the continuous head to couple with the binary on_ground flag, reducing y-drift for grounded characters. Orthogonal to E017a (which changes the target representation) — this one adds a penalty term while keeping deltas.

Works for both delta mode (Δy should be ~0 when on_ground) and absolute mode (y should be ~0 when on_ground).

## Target Metrics

| Metric | E012 (1.9K, baseline) | E016 (7.7K, cascade+SS) | Target | Kill threshold |
|--------|----------------------|-------------------------|--------|---------------|
| val_change_acc | 91.1% | 94.3% | >88% | <80% after epoch 1 |
| val_pos_mae | 0.706 | 0.466 | <0.70 | >0.85 |
| val_loss | 0.527 | 0.327 | <0.55 | not decreasing after 10% |

Physics loss is an auxiliary loss — it should help position accuracy without hurting action prediction. Watch for: physics penalty dominating and suppressing all y predictions (not just grounded ones).

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-e012-fd-top5.pt` |
| File size | 12.0 GB |
| Encoding flags | state_flags=true, hitstun=true, projectiles=false |
| Filters | stage=32 (FD), characters=[1,2,7,18,22] (Fox/Falcon/Sheik/Marth/Falco) |
| Games | 1,988 |
| Total frames | 17,465,842 |
| float_per_player | 69 (float columns = 138) |
| Train examples | ~15,656,435 (90%) |
| Val examples | ~1,789,527 (10%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `worldmodel/experiments/e017b-physics-loss.yaml` |
| Parameters | ~4,357K (same as E016 — cascade embeddings, no new params) |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |

## Training

| Field | Value |
|-------|-------|
| Epochs | 2 |
| Batch size | 4096 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=1.0, action=2.0, action_change_weight=5.0, **physics=0.5** |

### Key flags

| Flag | Value | Source |
|------|-------|--------|
| physics (loss weight) | **0.5** | E017b — ground-consistency penalty |
| cascaded_heads | true | E014/E016 |
| cascade_embed_dim | 16 | E014/E016 |
| scheduled_sampling | 0.3 | E015/E016 |
| ss_true | true | E015/E016 |
| ss_anneal_epochs | 2 | E015/E016 |
| multi_position | true | E008c |
| ctrl_threshold_features | true | E010c |
| action_change_weight | 5.0 | E010b |
| projectiles | false | Matches encoded data (138 cols) |

### Physics loss details

The physics loss computes: `mean(|pred_y_p0| * on_ground_p0) + mean(|pred_y_p1| * on_ground_p1)) / 2`

Only penalizes y predictions when ground truth `on_ground=true`. Weight 0.5 makes it comparable to velocity/dynamics losses. Logged as `loss/physics` in wandb.

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 (Modal) |
| num_workers | 4 |
| wandb | `shinewave/melee-worldmodel` / `e017b-physics-loss` |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. epoch time | ~16 min (H100, 1.9K games) |
| Est. total training | ~32 min |
| Est. data load | ~5 min |
| Est. total wall time | ~40 min |
| **Est. cost** | **~$2.60** |

## Escape Hatches

- **Kill if**: loss not decreasing after epoch 1, physics loss dominating total (>50% of total loss), change_acc < 80%
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /checkpoints/e017b-physics-loss/latest.pt --config worldmodel/experiments/e017b-physics-loss.yaml --encoded-file /encoded-e012-fd-top5.pt --run-name e017b-physics-loss`
- **Fallback plan**: If physics=0.5 is too aggressive, try 0.1. If it helps, combine with E017a absolute-y.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — no cascade, no SS |
| E016 | 7,700 FD top-5 | 3 | 94.3% | 0.466 | 0.327 | Cascade + SS + 4x data |

## What's New in This Run

- **Physics consistency loss**: New auxiliary loss term that couples the continuous y-prediction with the binary on_ground flag. When the model predicts on_ground=true but also predicts large |Δy|, the physics loss penalizes it. Should teach the model that "grounded means y≈0."
- **No architecture changes**: Same model, same tensor dimensions. Just one extra loss term added to the total.
- **Fast iteration**: Uses E012's 1.9K encoded file (~16 min/epoch). If physics loss helps, combine with E017a and scale to larger data.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e017b-physics-loss.yaml \
  --encoded-file /encoded-e012-fd-top5.pt \
  --epochs 2 --run-name e017b-physics-loss
```

## Run Log

### Run: e017b-physics-loss — 2026-03-02 19:22 UTC

**wandb**: [77vmem5u](https://wandb.ai/shinewave/melee-worldmodel/runs/77vmem5u)
**Modal**: [ap-H5zhiI6HaUSBOmlvvJNO70](https://modal.com/apps/scaviefae/main/ap-H5zhiI6HaUSBOmlvvJNO70)
**GPU**: NVIDIA H100 80GB HBM3

| What | Card expected | Actual | Match? |
|------|--------------|--------|--------|
| Encoded file | /encoded-e012-fd-top5.pt | /encoded-e012-fd-top5.pt | ✓ |
| Projectiles (saved) | false | False | ✓ |
| State flags (saved) | true | True | ✓ |
| Hitstun (saved) | true | True | ✓ |
| Float columns | 138 | 138 | ✓ |
| Train examples | ~15,656,435 | 15,656,435 | ✓ |
| Val examples | ~1,789,527 | 1,789,527 | ✓ |
| Model params | ~4,357K | 4,369,188 | ✓ (~close) |
| physics (loss weight) | 0.5 | 0.5 | ✓ |

**No mismatches. Run is clean.**

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
