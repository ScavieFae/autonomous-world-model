# Run Card: e017c-absolute-xy

**Created**: 2026-03-02
**Config**: `worldmodel/experiments/e017c-absolute-xy.yaml`
**Status**: LAUNCHING

## Goal

Extend E017a's absolute-y to both spatial axes. Replace Δx+Δy with absolute x+y predictions to prevent compounding drift on both axes. E017a showed absolute-y trades drift for jitter — a structural improvement. Adding x should prevent horizontal drift too, which is less visible but equally compounding.

On FD, x is bounded ~(-85, 85) and y is well-defined. Absolute positions mean the model directly predicts "character is at (12.3, 0.0)" instead of accumulating per-frame deltas.

## Target Metrics

| Metric | E012 (1.9K, baseline) | E017a (absolute y) | Target | Kill threshold |
|--------|----------------------|---------------------|--------|---------------|
| val_change_acc | 91.1% | 88.2% | >86% | <80% after epoch 1 |
| val_pos_mae | 0.706 | 0.979 | any | >1.5 |
| val_loss | 0.527 | 0.507 | <0.55 | not decreasing after 10% |

pos_mae will inflate further (both x and y have wider absolute distributions than deltas). The real test is AR demo quality — characters should stay on-stage and at ground level.

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
| Config | `worldmodel/experiments/e017c-absolute-xy.yaml` |
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
| wandb | `shinewave/melee-worldmodel` / `e017c-absolute-xy` |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. epoch time | ~16 min (H100, 1.9K games) |
| Est. total training | ~32 min |
| Est. data load | ~5 min |
| Est. total wall time | ~40 min |
| **Est. cost** | **~$2.60** |

## Escape Hatches

- **Kill if**: loss not decreasing after epoch 1, change_acc < 80%
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /checkpoints/e017c-absolute-xy/latest.pt --config worldmodel/experiments/e017c-absolute-xy.yaml --encoded-file /encoded-e012-fd-top5.pt --run-name e017c-absolute-xy`
- **Fallback plan**: If absolute x hurts (x is more variable than y), drop back to E017a's y-only absolute.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — no cascade, no SS |
| E017a | 1,988 FD top-5 | 2 | 88.2% | 0.979 | 0.507 | Absolute y only — jittery but ground-level |
| E016 | 7,700 FD top-5 | 3 | 94.3% | 0.466 | 0.327 | Cascade + SS + 4x data |

## What's New in This Run

- **Absolute x+y positions**: `absolute_positions: true` replaces both Δx and Δy with absolute values. Percent and shield stay as deltas (they work well that way).
- **Subsumes E017a**: `absolute_positions` handles both axes. The old `absolute_y` flag is preserved for backward compat but not used here.
- **No architecture changes**: Same model, same tensor dimensions. Just target computation changes.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e017c-absolute-xy.yaml \
  --encoded-file /encoded-e012-fd-top5.pt \
  --epochs 2 --run-name e017c-absolute-xy
```

## Run Log

### Run: e017c-absolute-xy — 2026-03-02 20:49 UTC

**wandb**: [cc9yk53x](https://wandb.ai/shinewave/melee-worldmodel/runs/cc9yk53x)
**Modal**: [ap-zK5krdPZn7pZHX9aHzBHaH](https://modal.com/apps/scaviefae/main/ap-zK5krdPZn7pZHX9aHzBHaH)
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

**No mismatches. Run is clean.**

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
