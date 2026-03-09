# Run Card: e015-true-ss

**Created**: 2026-03-02
**Config**: `worldmodel/experiments/e015-true-ss.yaml`
**Status**: PENDING REVIEW

## Goal

Test whether true scheduled sampling (feeding the model's own predictions back as context) reduces autoregressive drift, compared to noise-based SS (E012b, null result) and no SS (E012 baseline). The key metric is autoregressive demo quality — but val change_acc should not regress.

## Target Metrics

| Metric | Baseline (E012) | Target | Kill threshold |
|--------|-----------------|--------|---------------|
| val_change_acc | 91.1% | >90% | <85% after epoch 1 |
| val_pos_mae | 0.706 | <0.75 | >0.85 |
| val_loss/total | 0.527 | <0.55 | not decreasing after 5% |

Note: val runs on clean (uncorrupted) context. SS primarily improves autoregressive rollout quality, not teacher-forced metrics. The real test is demo quality post-training.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-e012-fd-top5.pt` (reuses E012's data — no pre-encode) |
| Encoding flags | state_flags=true, hitstun=true, projectiles=false |
| Filters | stage=32 (FD), characters=[1,2,7,18,22] (Fox/Falcon/Sheik/Marth/Falco) |
| Games | 1,988 |
| float_per_player | 69 (float columns = 138) |
| Train examples | ~15.7M (90%) |
| Val examples | ~1.8M (10%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `worldmodel/experiments/e015-true-ss.yaml` |
| Parameters | ~4,349,284 (identical to E012) |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |

## Training

| Field | Value |
|-------|-------|
| Epochs | 4 (2x E012 — SS needs ramp + recovery) |
| Batch size | 4096 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=1.0, action=2.0, action_change_weight=5.0 |

### Key flags

| Flag | Value | Source |
|------|-------|--------|
| scheduled_sampling | **0.3** | E015 (new) |
| ss_true | **true** | E015 (new — feeds own predictions, not noise) |
| ss_anneal_epochs | 2 | Ramp from 0 to full rate over 2 epochs |
| focal_offset | 0 | E012 baseline |
| multi_position | true | E008c (required for true SS) |
| ctrl_threshold_features | true | E010c |
| action_change_weight | 5.0 | E010b |

### How true SS works

1. No-grad forward pass on ~30% of samples → multi-position predictions
2. Extract position K-2's prediction for frame K-1
3. Reconstruct frame K-1 from predictions (deltas, absolutes, argmax)
4. Replace last context frame with reconstruction
5. Second forward pass (with gradient) trains on corrupted context

Epoch 0 skips SS. Rate anneals: epoch 1 → 15%, epoch 2+ → 30%.

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 (Modal) |
| num_workers | 4 |
| wandb | `shinewave/melee-worldmodel` / `e015-true-ss` |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch | ~3,822 |
| log_interval | 1000 batches |
| Time between logs | ~260s (~4.3 min) |
| Logs per epoch | ~4 |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed | ~0.26s (H100, bs=4096, K=10, multi_pos) |
| Est. epoch time | ~16-19 min (15 min base + up to 20% SS overhead) |
| Est. total training | ~70 min (~1.2h) |
| Est. data load | ~3 min |
| Est. total wall time | ~1.3h |
| **Est. cost** | **~$5** |

## Escape Hatches

- **Kill if**: loss not decreasing after epoch 1, OOM from double forward pass, change_acc < 85% after epoch 1
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /checkpoints/e015-true-ss/latest.pt --config worldmodel/experiments/e015-true-ss.yaml --encoded-file /encoded-e012-fd-top5.pt --run-name e015-true-ss`
- **Fallback plan**: If true SS hurts, revert to no-SS baseline (E012). If it helps teacher-forced but not AR, combine with post-hoc clamping.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — no SS |
| E012b | 1,988 FD top-5 | 2 | 91.0% | 0.708 | 0.528 | Noise SS (rate=0.3) — null result |
| E013 | 1,988 FD top-5 | 2 | 91.2% | 0.710 | 0.517 | Hitbox features — null result |

## What's New in This Run

- **True scheduled sampling**: Feeds the model's own predictions back as context instead of random Gaussian noise. The corruption is structured — it has the model's real biases (y-drift, damage creep, action stickiness). This directly teaches the model to handle its own mistakes.
- **4 epochs** (vs 2): SS needs ramp time (anneals over 2 epochs) and recovery epochs after corruption increases.
- **Same data as E012/E012b**: Pure A/B test of the SS method. No confounding variables.
- **Why noise SS failed (E012b)**: Gaussian noise on position/velocity is random — doesn't match the model's actual error patterns at inference. True SS creates *realistic* errors for the model to learn from.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file /encoded-e012-fd-top5.pt \
  --epochs 4 \
  --run-name e015-true-ss \
  --config worldmodel/experiments/e015-true-ss.yaml
```

## Run Log

### Run: e015-true-ss — 2026-03-02

**wandb**: [1m4fr40y](https://wandb.ai/shinewave/melee-worldmodel/runs/1m4fr40y)
**Modal**: [ap-nyEDNZhjK9LDpuNiDznlB6](https://modal.com/apps/scaviefae/main/ap-nyEDNZhjK9LDpuNiDznlB6)

| What | Card expected | Actual | Match? |
|------|--------------|--------|--------|
| Encoded file | /encoded-e012-fd-top5.pt | /encoded-e012-fd-top5.pt | ✓ |
| Projectiles (saved) | false | False | ✓ |
| State flags (saved) | true | True | ✓ |
| Hitstun (saved) | true | True | ✓ |
| Float columns | 138 | 138 | ✓ |
| Train examples | ~15.7M | 15,656,435 | ✓ |
| Val examples | ~1.8M | 1,789,527 | ✓ |
| Model params | ~4,349,284 | 4,349,284 | ✓ |
| Games | 1,988 | 1,988 | ✓ |
| scheduled_sampling | 0.3 | 0.3 | ✓ |
| ss_true | true | True | ✓ |
| multi_position | true | True | ✓ |

No mismatches. Run is clean.

## Sign-off

- [x] Scav reviewed
- [ ] Mattie reviewed
