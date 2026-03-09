# Run Card: e017d-absolute-xy-vel

**Created**: 2026-03-02
**Config**: `worldmodel/experiments/e017d-absolute-xy-vel.yaml`
**Status**: LAUNCHING

## Goal

Most aggressive absolute-target experiment: predict absolute x, y, AND velocities. Eliminates all compounding in both position and velocity channels. The model predicts the full physics state directly each frame instead of accumulating deltas.

If this works, it means the model can learn to output correct instantaneous physics states. If velocity prediction regresses while positions improve, it tells us velocities specifically need delta structure (Δvel = acceleration is a natural physical quantity).

## Target Metrics

| Metric | E012 (1.9K, baseline) | E017a (absolute y) | Target | Kill threshold |
|--------|----------------------|---------------------|--------|---------------|
| val_change_acc | 91.1% | 88.2% | >84% | <78% after epoch 1 |
| val_pos_mae | 0.706 | 0.979 | any | >2.0 |
| val_loss | 0.527 | 0.507 | <0.60 | not decreasing after 10% |

More aggressive targets — switching 14 channels (4 position + 10 velocity per pair) from delta to absolute is a bigger change. Allow more regression before killing.

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
| Config | `worldmodel/experiments/e017d-absolute-xy-vel.yaml` |
| Parameters | ~4,357K (same as E016/E017a) |
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
| absolute_positions | **true** | E017c — predict absolute x+y |
| absolute_velocities | **true** | E017d — predict absolute velocities |
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
| wandb | `shinewave/melee-worldmodel` / `e017d-absolute-xy-vel` |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. epoch time | ~16 min (H100, 1.9K games) |
| Est. total training | ~32 min |
| Est. data load | ~5 min |
| Est. total wall time | ~40 min |
| **Est. cost** | **~$2.60** |

## Escape Hatches

- **Kill if**: loss not decreasing after epoch 1, change_acc < 78%, velocity loss exploding
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /checkpoints/e017d-absolute-xy-vel/latest.pt --config worldmodel/experiments/e017d-absolute-xy-vel.yaml --encoded-file /encoded-e012-fd-top5.pt --run-name e017d-absolute-xy-vel`
- **Fallback plan**: If velocities regress while positions improve, use E017c (positions-only absolute). Velocities may need delta structure because Δvel = acceleration is physically meaningful.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — no cascade, no SS |
| E017a | 1,988 FD top-5 | 2 | 88.2% | 0.979 | 0.507 | Absolute y only |
| E016 | 7,700 FD top-5 | 3 | 94.3% | 0.466 | 0.327 | Cascade + SS + 4x data |

## What's New in This Run

- **Absolute positions + absolute velocities**: All 14 channels that were previously deltas (4 position + 10 velocity) now predict absolute values. Percent, shield, dynamics stay as-is.
- **Most aggressive delta elimination**: If this works, it means the model can learn instantaneous physics states without accumulation. If velocity regresses, we learn that Δvel (acceleration) is a more natural target.
- **No architecture changes**: Same model, same tensor dimensions. Only target computation and frame reconstruction change.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e017d-absolute-xy-vel.yaml \
  --encoded-file /encoded-e012-fd-top5.pt \
  --epochs 2 --run-name e017d-absolute-xy-vel
```

## Run Log

### Run: e017d-absolute-xy-vel — 2026-03-02 20:49 UTC

**wandb**: [owwxoxpd](https://wandb.ai/shinewave/melee-worldmodel/runs/owwxoxpd)
**Modal**: [ap-oGc7SwIhJU8wQ29vQvJhcu](https://modal.com/apps/scaviefae/main/ap-oGc7SwIhJU8wQ29vQvJhcu)
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
| absolute_positions | true | True | ✓ |
| absolute_velocities | true | True | ✓ |

**No mismatches. Run is clean.**

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
