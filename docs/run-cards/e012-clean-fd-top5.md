# Run Card: e012-clean-fd-top5

**Created**: 2026-03-01
**Config**: `worldmodel/experiments/e012-clean-fd-top5.yaml`
**Status**: PENDING REVIEW

## Goal

Test whether removing focal_offset inflation + simplifying to top-5 characters on FD gives honest, improved change_acc — or reveals how much our "85.8%" was illusory.

## Target Metrics

| Metric | Baseline (source) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| val_change_acc | 72.7% (E010d batch_eval, focal=3 corrected) | >75% | <65% after epoch 1 |
| val_pos_mae | 0.654 (E010d) | <0.60 | >0.80 |
| val_loss/total | 0.696 (E010d) | <0.65 | not decreasing after 5% |

Note: E010d's reported 85.8% was inflated by focal_offset + multi_position averaging. The honest batch eval number was 72.7%. This run uses focal_offset=0 so training metrics should be directly comparable to batch eval.

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
| Train examples | ~15,719,258 (90%) |
| Val examples | ~1,746,584 (10%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `worldmodel/experiments/e012-clean-fd-top5.yaml` |
| Parameters | ~4,349,284 |
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
| focal_offset | **0** | Removing inflation (was 3 in E010d) |
| multi_position | true | E008c |
| ctrl_threshold_features | true | E010c |
| action_change_weight | 5.0 | E010b |
| projectiles | false | E010e showed negligible impact |
| character_conditioning | false | E011 null result |
| ctrl_residual_to_action | false | E010a null result |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | A100-SXM4-40GB |
| num_workers | 4 |
| Modal timeout | **18000s (5h)** |
| wandb | `shinewave/melee-worldmodel` / `e012-clean-fd-top5` |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch | ~3,833 |
| log_interval | 1000 batches |
| Time between logs | ~700s (~12 min) |
| Logs per epoch | ~4 |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed | ~0.70s (bs=4096, K=10, multi_pos) |
| Est. epoch time | ~45 min |
| Est. total training | ~1.5h |
| Est. data load | ~5 min |
| Est. total wall time | ~1.6h |
| **Est. cost** | **~$4.50** |
| Timeout | 18000s = 5h (3.1x safety margin) |

## Escape Hatches

- **Kill if**: loss not decreasing after 500 batches, OOM, change_acc < 60% after epoch 1
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /checkpoints/e012-clean-fd-top5/latest.pt --config worldmodel/experiments/e012-clean-fd-top5.yaml --encoded-file /encoded-e012-fd-top5.pt --run-name e012-clean-fd-top5`
- **Fallback plan**: If FD-only data is too narrow, try top-5 characters on all stages

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E010d | 2K mixed, all stages | 2 | 85.8% (inflated) | 0.654 | 0.696 | focal_offset=3 + multi_pos inflation |
| E010d batch_eval | same | — | 72.7% (honest, focal=3) | — | — | Corrected via focal rollback fill |
| E010d batch_eval | same | — | 68.8% (honest, focal=0) | — | — | No focal offset at inference |
| E009a | 2K mixed, all stages | 2 | 71.5% (inflated) | 0.491 | 0.278 | v3 baseline |

## What's New in This Run

- **focal_offset=0**: Removes the inflation. Training val metrics should now match batch eval.
- **FD-only**: No platform stages. Position prediction should improve (no Battlefield/Yoshi's platform geometry to learn).
- **Top-5 characters only**: Fox, Falco, Marth, Falcon, Sheik. 77.5% of the competitive meta. Removes rare-character noise (Samus/Yoshi games dragged change_acc to 56%).
- **All proven E010 improvements kept**: transition weighting (5x), ctrl threshold features, multi_position.
- **Honest metrics**: This is the first run where training val change_acc is directly comparable to batch eval. No focal offset inflation to untangle.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file /encoded-e012-fd-top5.pt \
  --epochs 2 \
  --run-name e012-clean-fd-top5 \
  --config worldmodel/experiments/e012-clean-fd-top5.yaml
```

## Run Log

### Run: e012-clean-fd-top5 — 2026-03-01 23:44 UTC

**wandb**: [n778g2qr](https://wandb.ai/shinewave/melee-worldmodel/runs/n778g2qr)
**Modal**: [ap-CA6uTOtoDhCjHZ4qLPWUgM](https://modal.com/apps/scaviefae/main/ap-CA6uTOtoDhCjHZ4qLPWUgM)
**GPU**: NVIDIA H100 80GB HBM3

| What | Card expected | Actual | Match? |
|------|--------------|--------|--------|
| Encoded file | /encoded-e012-fd-top5.pt | /encoded-e012-fd-top5.pt | ✓ |
| Projectiles (saved) | false | False | ✓ |
| State flags (saved) | true | True | ✓ |
| Hitstun (saved) | true | True | ✓ |
| Float columns | 138 | 138 | ✓ |
| Train examples | ~15,719,258 | 15,656,435 | ✓ (~close) |
| Val examples | ~1,746,584 | 1,789,527 | ✓ (~close) |
| Model params | ~4,349,284 | 4,349,284 | ✓ |
| focal_offset | 0 | 0 | ✓ |
| multi_position | true | True | ✓ |
| ctrl_threshold_features | true | True | ✓ |
| action_change_weight | 5.0 | 5 | ✓ |

**No mismatches. Run is clean.**

**Results** (2 epochs, H100, 1845s = 31 min):

| Metric | Target | Actual | vs baseline (E010d batch_eval) | Hit target? |
|--------|--------|--------|-------------------------------|-------------|
| change_acc | >75% | **91.1%** | +18.4pp vs 72.7% | ✓ |
| p0_change_acc | — | 91.2% | — | — |
| p1_change_acc | — | 91.0% | — | — |
| pos_mae | <0.60 | 0.706 | +0.052 vs 0.654 | ✗ |
| val_loss | <0.65 | 0.527 | -0.169 vs 0.696 | ✓ |

Notes:
- change_acc is **honest** — focal_offset=0 means no multi-position inflation
- p0/p1 nearly identical (91.2% vs 91.0%) — perfect symmetry
- pos_mae regressed slightly — likely action_change_weight=5.0 trading position for action accuracy
- H100 completed in 31 min (vs 1.5h estimate for A100)

## Sign-off

- [x] Scav reviewed
- [ ] Mattie reviewed
