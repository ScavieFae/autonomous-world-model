# Run Card: e014-cascaded-heads

**Created**: 2026-03-01
**Config**: `worldmodel/experiments/e014-cascaded-heads.yaml`
**Status**: PENDING REVIEW

## Goal

Test whether cascaded heads (action → physics) improves autoregressive consistency. Action heads predict first (tier 1), their embeddings feed into all downstream heads (tier 2). This mirrors the game engine's causal chain: action determines what physics happens.

**Problem being solved**: E012 demos showed the independent heads problem — damage drift without attacks, y-position drift while STANDING, on_ground=true during jumps. All heads read from the same `h` independently.

## Target Metrics

| Metric | Baseline (E012) | Target | Kill threshold |
|--------|-----------------|--------|---------------|
| val_change_acc | 91.1% | ≥91% (no regression) | <85% after epoch 1 |
| val_pos_mae | 0.706 | ≤0.70 | >0.85 |
| val_loss/total | 0.527 | <0.55 | not decreasing after 5% |

Note: Primary evaluation is autoregressive demo consistency, not just metrics. The cascade should produce physically coherent predictions during rollout even if teacher-forced metrics are similar.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-e012-fd-top5.pt` |
| File size | 12.0 GB |
| Encoding flags | state_flags=true, hitstun=true, projectiles=false |
| Filters | stage=32 (FD), characters=[1,2,7,18,22] (Fox/Falcon/Sheik/Marth/Falco) |
| Games | 1,988 |
| Total frames | ~17,465,842 |
| float_per_player | 69 (float columns = 138) |
| Train examples | ~15,656,435 (90%) |
| Val examples | ~1,789,527 (10%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) + cascaded heads |
| Config | `worldmodel/experiments/e014-cascaded-heads.yaml` |
| Parameters | ~4,394,980 (+47,232 vs E012 baseline) |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |
| cascade_embed_dim | 16 |

## Training

| Field | Value |
|-------|-------|
| Epochs | 3 |
| Batch size | 1024 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=1.0, action=2.0, action_change_weight=5.0 |

### Key flags

| Flag | Value | Source |
|------|-------|--------|
| cascaded_heads | **true** | E014 (new) |
| cascade_embed_dim | 16 | E014 (new) |
| character_conditioning | true | E011b (included in E012 card as false, testing here) |
| ctrl_residual_to_action | true | E010a (included in E012 card as false, testing here) |
| focal_offset | 0 | E012 baseline |
| multi_position | true | E008c |
| ctrl_threshold_features | true | E010c |
| action_change_weight | 5.0 | E010b |
| projectiles | false | E012 baseline |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 (Modal default) |
| num_workers | 4 |
| Modal timeout | 86400s (24h default) |
| wandb | `shinewave/melee-worldmodel` / `e014-cascaded-heads` |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed | ~0.35s (bs=1024, K=10, multi_pos, H100) |
| Est. batches/epoch | ~15,290 (15.66M / 1024) |
| Est. epoch time | ~90 min |
| Est. total training | ~4.5h (3 epochs) |
| Est. data load | ~5 min |
| Est. total wall time | ~4.6h |
| **Est. cost** | **~$18** (H100 @ $3.95/hr × 4.6h) |

## Escape Hatches

- **Kill if**: loss not decreasing after 1000 batches, OOM, change_acc < 80% after epoch 1
- **Fallback plan**: If cascaded heads regress action accuracy, try with character_conditioning=false and ctrl_residual_to_action=false (isolate cascade effect)

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — independent heads |

## What's New in This Run

- **Cascaded heads**: Action heads predict first, embed result, feed to physics/binary/categorical heads
- **Teacher forcing**: Training uses GT actions for cascade conditioning; inference uses argmax
- **character_conditioning**: Additive char embed projection into hidden state (was false in E012)
- **ctrl_residual_to_action**: Direct ctrl→action head skip connection (was false in E012)
- **3 epochs** (vs E012's 2): extra epoch to let cascade embedding converge
- **LR 0.001** (vs E012's 0.0005): faster convergence for the new parameters
- **Batch size 1024** (vs E012's 4096): more gradient updates per epoch

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file /encoded-e012-fd-top5.pt \
  --epochs 3 \
  --run-name e014-cascaded-heads \
  --config worldmodel/experiments/e014-cascaded-heads.yaml
```

## Run Log

### Run: e014-cascaded-heads — 2026-03-02 01:45 UTC

**wandb**: [yfg3kdsm](https://wandb.ai/shinewave/melee-worldmodel/runs/yfg3kdsm)
**Modal**: [ap-1pjKp3f6lZzoyD9QlJn4Kg](https://modal.com/apps/scaviefae/main/ap-1pjKp3f6lZzoyD9QlJn4Kg)
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
| Model params | ~4,394,980 | 4,404,516 | ~close (+9,536 from ctrl_threshold × ctrl_residual interaction) |
| cascaded_heads | true | True | ✓ |
| cascade_embed_dim | 16 | 16 | ✓ |
| multi_position | true | True | ✓ |
| focal_offset | 0 | 0 | ✓ |
| ctrl_threshold_features | true | True | ✓ |
| character_conditioning | true | True | ✓ |
| ctrl_residual_to_action | true | True | ✓ |

**No critical mismatches. Run is clean.**

