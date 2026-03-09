# Run Card: e016-omnibus

**Created**: 2026-03-01
**Config**: `worldmodel/experiments/e016-omnibus.yaml`
**Status**: PENDING REVIEW

## Goal

Consolidate the two best independent improvements (E014 cascaded heads + E015 true scheduled sampling) onto 4x larger data (7.7K games). E014 proved cascade fixes damage drift structurally but overfitted at 1.9K games. E015 tests whether true SS reduces autoregressive drift. Together on 7.7K games: cascade's structural improvement without overfitting, plus SS's drift resistance.

## Target Metrics

| Metric | E012 (1.9K, baseline) | E014-bs4k (1.9K, cascade) | Target | Kill threshold |
|--------|----------------------|---------------------------|--------|---------------|
| val_change_acc | 91.1% | 88.0% (overfit) | >91% | <85% after epoch 1 |
| val_pos_mae | 0.706 | 0.59 | <0.70 | >0.85 |
| val_loss | 0.527 | 0.56 | <0.55 | not decreasing after 10% |

The real test is autoregressive demo quality. Cascade should fix damage drift (proven in E014). True SS should reduce y-drift and on_ground inconsistency.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-v3-ranked-fd-top5.pt` (56.9 GB on Modal volume) |
| Encoding flags | state_flags=true, hitstun=true, projectiles=false |
| Filters | stage=32 (FD), characters=[1,2,7,18,22] (Fox/Falcon/Sheik/Marth/Falco) |
| Games | ~7,700 |
| Est. frames | ~67.4M |
| float_per_player | 69 (float columns = 138, same as E012) |
| Train examples | ~60.7M (90%) |
| Val examples | ~6.7M (10%) |

Note: encoded data has projectiles=false (138 float columns, same as E012). Plan originally assumed projectiles=true but the saved encoding config confirmed otherwise.

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `worldmodel/experiments/e016-omnibus.yaml` |
| Parameters | ~4,357K (est. — slightly more than E012's 4,349K due to cascade embeddings) |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |

## Training

| Field | Value |
|-------|-------|
| Epochs | 3 |
| Batch size | 4096 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=1.0, action=2.0, action_change_weight=5.0 |

### Key flags

| Flag | Value | Source |
|------|-------|--------|
| cascaded_heads | true | E014 — action conditions physics heads |
| cascade_embed_dim | 16 | E014 |
| scheduled_sampling | 0.3 | E015 — true SS rate |
| ss_true | true | E015 — feeds own predictions back |
| ss_anneal_epochs | 2 | E015 — ramp: ep0=0%, ep1=15%, ep2=30% |
| multi_position | true | E008c (required by cascade and SS) |
| ctrl_threshold_features | true | E010c (+5.9pp) |
| action_change_weight | 5.0 | E010b (+9.9pp) |
| projectiles | false | Matches encoded data (138 cols) |
| state_flags | true | v3 encoding |
| hitstun | true | v3 encoding |

### Excluded flags

| Flag | Why excluded |
|------|-------------|
| character_conditioning | E011 null result |
| ctrl_residual_to_action | E010a null result, not the interesting part of cascade |
| hitbox_data | E013 null result, not in encoded data |

### Why 3 epochs

- E014 overfitted at 1,988 games / 2 epochs (95.3% train vs 88.5% val). With 7.7K games (4x data), overfitting risk is much lower.
- True SS anneals over 2 epochs (ep0: no SS, ep1: 15%, ep2: 30%). 3 epochs gives one fully-annealed epoch.
- Total data exposure: ~60.7M × 3 = 182M frames. E014 saw ~31M. This is 5.8x more.
- CosineAnnealingLR over 3 epochs reaches near-zero LR at the end — clean convergence.

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 (Modal) |
| num_workers | 4 |
| wandb | `shinewave/melee-worldmodel` / `e016-omnibus` |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch | ~14,820 (60.7M / 4096) |
| log_interval | 1000 batches |
| Time between logs | ~310s (~5.2 min) |
| Logs per epoch | ~15 |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed | ~0.31s (H100, bs=4096, K=10, with ~20% SS overhead) |
| Est. epoch time | ~77 min |
| Est. total training | ~3.8h |
| Est. data load | ~10 min (56.9 GB) |
| Est. total wall time | ~4h |
| **Est. cost** | **~$16** |

## Escape Hatches

- **Kill if**: loss not decreasing after epoch 1, OOM from double forward pass, change_acc < 85% after epoch 1
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /checkpoints/e016-omnibus/latest.pt --config worldmodel/experiments/e016-omnibus.yaml --encoded-file /encoded-v3-ranked-fd-top5.pt --run-name e016-omnibus`
- **Fallback plan**: If cascade + SS interaction hurts, try cascade-only on 7.7K data (drop SS flags). If 7.7K data with projectiles=true causes issues, use the 1.9K E012 encoded file as a sanity check.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| E012 | 1,988 FD top-5 | 2 | 91.1% | 0.706 | 0.527 | Baseline — no cascade, no SS |
| E014 | 1,988 FD top-5 | 3 | 88.0% | 0.59 | 0.56 | Cascade only — overfit (95.3% train) |
| E015 | 1,988 FD top-5 | 4 | TBD | TBD | TBD | True SS only — currently running |

## What's New in This Run

- **Cascaded heads (E014)**: Action prediction conditions physics/binary/categorical heads via learned embeddings. Mirrors the game engine's causal chain — action determines what physics happens. Fixed damage drift in AR demos.
- **True scheduled sampling (E015)**: Feeds model's own predictions back as context during training. Teaches the model to handle its own errors, reducing y-drift and on_ground inconsistency.
- **4x data scale (7.7K games)**: E014 overfit badly at 1.9K games. With 4x more data, the cascade structure should shine without memorizing.
- **Consolidation**: First run to combine all proven improvements in one config. If this works, it becomes the new baseline for everything after.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e016-omnibus.yaml \
  --encoded-file /encoded-v3-ranked-fd-top5.pt \
  --epochs 3 \
  --run-name e016-omnibus
```

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
