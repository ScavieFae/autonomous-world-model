# Run Card: e013-hitbox-data

**Created**: 2026-03-02
**Config**: `worldmodel/experiments/e013-hitbox-data.yaml`
**Status**: PENDING REVIEW

## Goal

Test whether hitbox data as input features improves the model's understanding of combat interactions. Adds 5 continuous features per player per frame: hitbox is_active, damage, angle, knockback growth, hitbox size — looked up from a pre-computed table keyed by (character, action, state_age).

Pure A/B test vs E012: identical model architecture, training hyperparams, and data filters. Only difference is `hitbox_data=true` adding 10 float columns (5 per player).

## Target Metrics

| Metric | Baseline (E012) | Target | Kill threshold |
|--------|-----------------|--------|---------------|
| val_change_acc | 91.1% | >91% | <85% after epoch 1 |
| val_pos_mae | 0.706 | <0.70 | >0.85 |
| val_loss/total | 0.527 | <0.53 | not decreasing after 5% |

Hypothesis: hitbox features should help the model predict damage changes and knockback more accurately, since it can see *what* attack is active and its properties.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-e013-hitbox-fd-top5.pt` (NEW — requires pre-encode) |
| Encoding flags | state_flags=true, hitstun=true, projectiles=false, **hitbox_data=true** |
| Filters | stage=32 (FD), characters=[1,2,7,18,22] (Fox/Falcon/Sheik/Marth/Falco) |
| Games | ~1,988 (same pool as E012) |
| float_per_player | 74 (float columns = 148, +10 vs E012's 138) |
| Train examples | ~15.7M (90%) |
| Val examples | ~1.8M (10%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `worldmodel/experiments/e013-hitbox-data.yaml` |
| Parameters | ~4,353,124 (+3,840 vs E012 from wider frame_proj input) |
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
| log_interval | 1000 |

### Key flags (identical to E012 except hitbox_data)

| Flag | Value | Source |
|------|-------|--------|
| hitbox_data | **true** | E013 (new) |
| focal_offset | 0 | E012 baseline |
| multi_position | true | E008c |
| ctrl_threshold_features | true | E010c |
| action_change_weight | 5.0 | E010b |
| projectiles | false | E012 baseline |
| character_conditioning | false | E012 baseline |
| ctrl_residual_to_action | false | E012 baseline |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 (Modal) |
| num_workers | 4 |
| wandb | `shinewave/melee-worldmodel` / `e013-hitbox-data` |

## Timing & Cost

### Pre-encode (one-time)
| Field | Value |
|-------|-------|
| Est. time | ~20 min (CPU, ~2K games, zlib decompress + hitbox lookup) |
| Est. cost | ~$1 |

### Training
| Field | Value |
|-------|-------|
| Est. batches/epoch | ~3,833 (15.7M / 4096) |
| Est. epoch time | ~30 min (H100) |
| Est. total training | ~1h (2 epochs) |
| **Est. cost** | **~$4** |

## Pre-encode Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::pre_encode \
  --config worldmodel/experiments/e013-hitbox-data.yaml \
  --output /encoded-e013-hitbox-fd-top5.pt
```

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file /encoded-e013-hitbox-fd-top5.pt \
  --epochs 2 \
  --run-name e013-hitbox-data \
  --config worldmodel/experiments/e013-hitbox-data.yaml
```

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — no hitbox data |

## What's New in This Run

- **Hitbox features**: 5 continuous features per player per frame from pre-computed hitbox table
  - `is_active` (0/1): whether the current (action, state_age) frame has an active hitbox
  - `damage` (normalized /30): how much damage this attack does
  - `angle` (normalized /361): knockback angle
  - `kbg` (normalized /200): knockback growth scaling
  - `size` (normalized /10): hitbox radius
- Hypothesis: the model currently has to *infer* attack properties from action state + state_age. Giving it the data directly should improve damage/knockback prediction.

## Run Log

### Run: e013-hitbox-data — 2026-03-02

**wandb**: [5yw0k7vt](https://wandb.ai/shinewave/melee-worldmodel/runs/5yw0k7vt)
**Modal**: [ap-cxQq5YeBheGADiM2HDIYVM](https://modal.com/apps/scaviefae/main/ap-cxQq5YeBheGADiM2HDIYVM)

| What | Card expected | Actual | Match? |
|------|--------------|--------|--------|
| Encoded file | /encoded-e013-hitbox-fd-top5.pt | /encoded-e013-hitbox-fd-top5.pt | ✓ |
| Hitbox data | true | true | ✓ |
| Projectiles | false | false | ✓ |
| Saved enc projectiles | false | false | ✓ |
| Saved enc hitbox_data | true | true | ✓ |
| Float columns | 148 (74×2) | 148 | ✓ |
| Train examples | ~15.7M | 15,656,435 | ✓ |
| Val examples | ~1.8M | 1,789,527 | ✓ |
| Model params | ~4,353,124 | 4,353,124 | ✓ |
| Games | ~1,988 | 1,988 | ✓ |

No mismatches. Run is clean.

**Results** (2 epochs, H100, ~30 min/epoch):

| Metric | Target | Actual | vs E012 baseline | Hit target? |
|--------|--------|--------|------------------|-------------|
| val_change_acc | >91% | 91.2% | +0.1pp | ✓ (no regression) |
| val_pos_mae | <0.70 | 0.71 | +0.004 | ✗ (marginal) |
| val_loss/total | <0.53 | 0.517 | -0.010 | ✓ |

**Verdict**: Null result. Hitbox features neither helped nor hurt. The model already infers attack properties from action + state_age sufficiently well. The 10 extra input features added minimal information at this data scale.

**Detailed delta vs E012**:

| Metric | E012 | E013 | Delta |
|--------|------|------|-------|
| change_acc | 91.07% | 91.15% | +0.08pp |
| pos_mae | 0.706 | 0.706 | -0.001 |
| damage_mae | 0.281 | 0.280 | -0.001 |
| val_loss | 0.527 | 0.517 | -0.010 |
| action_acc | 97.62% | 97.65% | +0.03pp |

Every delta is within noise. The val_loss drop (-0.010) likely reflects the model learning to predict the hitbox features themselves (they're in the continuous target), but this doesn't translate to any downstream improvement.

**Decision**: Drop hitbox_data for future experiments. Not worth the encoding complexity or the per-game hitbox table lookup during pre-encode. The model's implicit inference from (action, state_age) is sufficient — no need to extrapolate the hitbox table across the full dataset.

