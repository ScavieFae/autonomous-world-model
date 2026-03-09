# Run Card: e017a-absolute-y

**Created**: 2026-03-02
**Config**: `worldmodel/experiments/e017a-absolute-y.yaml`
**Status**: LAUNCHING

## Goal

Replace delta-y prediction with absolute-y prediction to eliminate compounding y-drift in autoregressive mode. E016 proved teacher-forced metrics don't predict AR quality — val_loss dropped 38% but characters fall through the stage faster. Root cause: continuous Δy errors compound as a random walk. With absolute-y, the model directly predicts y-position (grounded characters → y=0, no accumulation possible).

Percent, x, shield stay as deltas. Only y changes.

## Target Metrics

| Metric | E012 (1.9K, baseline) | E016 (7.7K, cascade+SS) | Target | Kill threshold |
|--------|----------------------|-------------------------|--------|---------------|
| val_change_acc | 91.1% | 94.3% | >88% | <80% after epoch 1 |
| val_pos_mae | 0.706 | 0.466 | <0.70 | >0.85 |
| val_loss | 0.527 | 0.327 | <0.55 | not decreasing after 10% |

The real test is autoregressive demo quality. E017a should keep characters on the ground plane instead of drifting through it.

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

Note: absolute_y only changes the target computation in the dataset (index 2 of cont_delta becomes absolute y instead of Δy). Encoded data is unchanged — same file as E012.

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `worldmodel/experiments/e017a-absolute-y.yaml` |
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
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=1.0, action=2.0, action_change_weight=5.0 |

### Key flags

| Flag | Value | Source |
|------|-------|--------|
| absolute_y | **true** | E017a — predict absolute y instead of Δy |
| cascaded_heads | true | E014/E016 |
| cascade_embed_dim | 16 | E014/E016 |
| scheduled_sampling | 0.3 | E015/E016 |
| ss_true | true | E015/E016 |
| ss_anneal_epochs | 2 | E015/E016 |
| multi_position | true | E008c |
| ctrl_threshold_features | true | E010c |
| action_change_weight | 5.0 | E010b |
| projectiles | false | Matches encoded data (138 cols) |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 (Modal) |
| num_workers | 4 |
| wandb | `shinewave/melee-worldmodel` / `e017a-absolute-y` |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. epoch time | ~16 min (H100, 1.9K games) |
| Est. total training | ~32 min |
| Est. data load | ~5 min |
| Est. total wall time | ~40 min |
| **Est. cost** | **~$2.60** |

## Escape Hatches

- **Kill if**: loss not decreasing after epoch 1, OOM, change_acc < 80%
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /checkpoints/e017a-absolute-y/latest.pt --config worldmodel/experiments/e017a-absolute-y.yaml --encoded-file /encoded-e012-fd-top5.pt --run-name e017a-absolute-y`
- **Fallback plan**: If absolute-y hurts action accuracy (different target distribution confuses the model), try combining with E017b physics loss instead.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — no cascade, no SS |
| E016 | 7,700 FD top-5 | 3 | 94.3% | 0.466 | 0.327 | Cascade + SS + 4x data |

## What's New in This Run

- **Absolute y-position**: The continuous head's index 2 (y) predicts the absolute scaled y-position instead of Δy. For grounded characters the target is just `y * xy_scale` (≈0 on FD's ground plane). No compounding possible — each frame independently predicts y.
- **Surgical change**: Only affects target computation in dataset, frame reconstruction in trainer, and AR loop in generate_demo. Model architecture unchanged (still outputs 8 continuous values). Loss computation unchanged (MSE on pred vs target works regardless).
- **Fast iteration**: Uses E012's 1.9K encoded file (~16 min/epoch) instead of E016's 7.7K data. If this works, scale to E017a-full on larger data.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e017a-absolute-y.yaml \
  --encoded-file /encoded-e012-fd-top5.pt \
  --epochs 2 --run-name e017a-absolute-y
```

## Run Log

### Run: e017a-absolute-y — 2026-03-02 19:22 UTC

**wandb**: [xfarwruv](https://wandb.ai/shinewave/melee-worldmodel/runs/xfarwruv)
**Modal**: [ap-3StVIXiOSeDjNEz9GAM86b](https://modal.com/apps/scaviefae/main/ap-3StVIXiOSeDjNEz9GAM86b)
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
| absolute_y | true | True | ✓ |

**No mismatches. Run is clean.**

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
