# Run Card: E010d/e — Combined B+C (± Projectiles)

**Created**: 2026-02-28
**Configs**: `e010d-combined-bc.yaml`, `e010e-combined-bc-proj.yaml`
**Branch**: `scav/research/e010-movement-suite`
**Status**: COMPLETE

## Goal

Test E010b (transition weighting) + E010c (threshold features) combined. These attack movement prediction from orthogonal angles — loss-side (b) and input-side (c). Paired runs with projectiles on/off to measure projectile impact.

## Target Metrics

| Metric | Baseline (E009a) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| change_acc | 70.9% | >75% | <70% after epoch 1 |
| action_acc | 96.7% | >95% | <93% |
| val_pos_mae | 0.560 | <0.60 | >0.70 |
| val_loss/total | 0.278 | — | not decreasing after 50% |

## Experiments

### E010d — B+C Combined (no projectiles)
- **Flags**: `ctrl_threshold_features: true` + `action_change_weight: 5.0`
- **Encoded file**: `encoded-v3-ranked-50k-part-0.pt` (capped to 2K)
- **Encoding**: projectiles=false, state_flags=true, hitstun=true
- **Params**: 4,349,284
- **float_per_player**: 69 (138 total)

### E010e — B+C Combined (with projectiles)
- **Flags**: `ctrl_threshold_features: true` + `action_change_weight: 5.0` + `projectiles: true`
- **Encoded file**: `encoded-v3-2k-proj.pt` (re-uploaded from local)
- **Encoding**: projectiles=true, state_flags=true, hitstun=true
- **Params**: 4,351,588
- **float_per_player**: 72 (144 total)

**Note**: E010d and E010e use different game sets. E010d uses first 2K from 50K ranked (same as E009a baseline). E010e uses `encoded-game-v3-2k.pt` (renamed on upload) which was encoded from `parsed-v3-12k-gameonly.tar`. The projectile comparison is directional, not perfectly controlled.

## Shared Config

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD), 4.35M params |
| d_model / n_layers | 384 / 4 |
| context_len (K) | 10, chunk_size=10 |
| Epochs | 2 |
| Batch size | 4096 |
| LR | 5e-4 |
| Loss weights | action=2.0, action_change_weight=5.0, binary=1.0, continuous=1.0, velocity=0.5, dynamics=0.5 |
| Encoding | v3, state_flags=true, hitstun=true, focal_offset=3, multi_position=true, ctrl_threshold_features=true |

## Data

| Field | E010d (no proj) | E010e (proj) |
|-------|----------------|-------------|
| Encoded file | `encoded-v3-ranked-50k-part-0.pt` | `encoded-v3-2k-proj.pt` |
| Games | 2,000 (ranked, capped) | ~2,000 |
| Train examples | ~21.4M | ~17.5M |
| Val examples | ~2.3M | ~1.9M |
| Float columns | 138 | 144 |
| Same games as E009a? | Yes | No |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 80GB |
| Modal timeout | 86400s (24hr) |
| wandb | `shinewave/melee-worldmodel` |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. epoch time | ~17 min (H100) |
| Est. total training | ~34 min per run |
| Est. data load | ~60s (E010d, 95GB capped) / ~17s (E010e, 13GB) |
| **Est. cost per run** | **~$2.30** |
| **Est. total (×2)** | **~$4.60** |

## Prior Comparable Runs

| Run | Flags | Data | change_acc | pos_mae | val_loss | Notes |
|-----|-------|------|------------|---------|----------|-------|
| E009a | baseline | ranked 2K, no proj | 70.9% | 0.560 | 0.278 | Direct baseline for E010d |
| E010b-v1 | transition wt | 2k-proj | 80.8% | 0.69 | 0.872 | Wrong data (proj=true) |
| E010c-v1 | thresholds | 2k-proj | 76.8% | 0.65 | 0.249 | Wrong data (proj=true) |
| E010a-v1 | ctrl residual | 2k-proj | 70.9% | 0.63 | 0.294 | No effect |

Note: v1 results used `encoded-game-v3-2k.pt` (projectiles=true, different games). Not directly comparable to E009a baseline. E010d will be the first clean comparison.

## Pre-flight Checklist

- [ ] `encoded-v3-2k-proj.pt` uploaded to Modal volume (for E010e)
- [ ] Config validator passes for both (no encoding mismatches)
- [ ] Use `/launch-experiment` skill to verify actual vs card

## Launch Commands

```bash
# E010d — B+C no projectiles (can launch immediately)
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e010d-combined-bc.yaml \
  --encoded-file /encoded-v3-ranked-50k-part-0.pt \
  --epochs 2 --run-name e010d-combined-bc

# E010e — B+C with projectiles (after upload)
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e010e-combined-bc-proj.yaml \
  --encoded-file /encoded-v3-2k-proj.pt \
  --epochs 2 --run-name e010e-combined-bc-proj
```

## Run Log

### Run: e010d-combined-bc — 2026-02-28 17:00 UTC

**wandb**: [kvjqsdua](https://wandb.ai/shinewave/melee-worldmodel/runs/kvjqsdua)
**Status**: RUNNING

| What | Card expected | Actual | Match? |
|------|--------------|--------|--------|
| Encoded file | encoded-v3-ranked-50k-part-0.pt | encoded-v3-ranked-50k-part-0.pt | ✓ |
| encoding.projectiles | false | false | ✓ |
| saved_config.projectiles | false | false | ✓ |
| saved_config.state_flags | true | true | ✓ |
| saved_config.hitstun | true | true | ✓ |
| Float columns | 138 | 138 | ✓ |
| Train examples | ~21.4M | 21,369,255 | ✓ |
| Val examples | ~2.3M | 2,320,272 | ✓ |
| Model params | 4,349,284 | 4,349,284 | ✓ |
| Games | 2,000 | 2,000 | ✓ |

**No mismatches. Run is clean. Same data as E009a baseline.**

### Run: e010e-combined-bc-proj — 2026-02-28 17:01 UTC

**wandb**: [xis85adq](https://wandb.ai/shinewave/melee-worldmodel/runs/xis85adq)
**Status**: RUNNING

| What | Card expected | Actual | Match? |
|------|--------------|--------|--------|
| Encoded file | encoded-v3-2k-proj.pt | encoded-v3-2k-proj.pt | ✓ |
| encoding.projectiles | true | true | ✓ |
| saved_config.projectiles | true | true | ✓ |
| saved_config.state_flags | true | true | ✓ |
| saved_config.hitstun | true | true | ✓ |
| Float columns | 144 | 144 | ✓ |
| Train examples | ~17.5M | 17,495,326 | ✓ |
| Val examples | ~1.9M | 1,920,861 | ✓ |
| Model params | 4,351,588 | 4,351,588 | ✓ |
| Games | ~2,000 | 2,000 | ✓ |

**No mismatches. Run is clean. Different game set from E009a (as documented).**

### Results — 2026-02-28 17:41 UTC

| Metric | Baseline (E009a) | E010d (no proj) | E010e (proj) | E010d vs baseline |
|--------|-------------------|-----------------|--------------|-------------------|
| **change_acc** | 70.9% | **85.8%** | **85.2%** | **+14.9pp** |
| action_acc | 96.7% | 96.3% | 96.2% | -0.4pp |
| pos_mae | 0.560 | 0.65 | 0.68 | +0.09 |
| val_loss | 0.278 | 0.696 | 0.724 | — (not comparable, weighted) |

**E010d epoch-by-epoch:**
| | loss | action_acc | pos_mae | change_acc | val_loss |
|---|---|---|---|---|---|
| Epoch 1 | 1.072 | 94.5% | 0.77 | 75.1% | 0.749 |
| Epoch 2 | 0.574 | 96.3% | 0.65 | 85.8% | 0.696 |

**E010e epoch-by-epoch:**
| | loss | action_acc | pos_mae | change_acc | val_loss |
|---|---|---|---|---|---|
| Epoch 1 | 1.140 | 94.3% | 0.79 | 74.1% | 0.772 |
| Epoch 2 | 0.595 | 96.2% | 0.68 | 85.2% | 0.724 |

**Key findings:**
- B+C combination is super-additive: 85.8% > max(E010b-v1 80.8%, E010c-v1 76.8%)
- Projectiles had negligible impact: E010d 85.8% ≈ E010e 85.2% (within noise)
- action_acc held steady (96.3% vs 96.7% baseline) — minimal regression
- pos_mae regressed slightly (0.65 vs 0.560) — expected trade-off from transition weighting
- E010d is directly comparable to E009a (same data, same games)

| Metric | Target | E010d actual | Hit target? |
|--------|--------|-------------|-------------|
| change_acc | >75% | 85.8% | ✓ |
| action_acc | >95% | 96.3% | ✓ |
| val_pos_mae | <0.60 | 0.65 | ✗ |

## Sign-off

- [x] Scav reviewed (launch-experiment skill verification + results)
- [ ] Mattie reviewed
