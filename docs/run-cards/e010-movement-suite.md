# Run Card: E010 Movement Suite

**Created**: 2026-02-28
**Configs**: `e010a-ctrl-residual.yaml`, `e010b-transition-weight.yaml`, `e010c-ctrl-thresholds.yaml`
**Branch**: `scav/research/e010-movement-suite`
**Status**: COMPLETE (v1 on wrong data, v2 in progress)

## Goal

Improve movement category change_acc (currently 55.6%, worst action category) by attacking the ctrl→action pathway from three angles: direct ctrl residual to action heads (a), transition-weighted loss (b), and pre-computed stick threshold features (c).

## Target Metrics

| Metric | Baseline (E009a) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| movement change_acc | 55.6% | >60% | <55% after epoch 1 |
| TURNING change_acc | 28.2% | >35% | — |
| overall change_acc | 70.9% | >71% | <68% |
| val_pos_mae | 0.560 | <0.560 | >0.60 |
| val_loss/total | 0.278 | <0.278 | not decreasing after 50% |

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-game-v3-2k.pt` |
| File size | 13.5 GB |
| Games | ~2,000 |
| Total frames | ~18.8M (est. 9,400/game) |
| Train examples | ~16.9M (90%) |
| Val examples | ~1.9M (10%) |

## Experiments (Three Parallel Runs)

### E010a — Ctrl Residual to Action Head
- **Flag**: `ctrl_residual_to_action: true`
- **What it does**: Action heads receive `cat(h, ctrl)` instead of just `h`. Raw ctrl vector bypasses the SSM prior, giving action heads direct access to stick values for threshold decisions.
- **Params**: 4,366,244 (+20,800 from wider action heads)

### E010b — Transition-Weighted Action Loss
- **Flag**: `action_change_weight: 5.0`
- **What it does**: Frames where the action changes get 5× weight in action CE loss. ~10% of frames are transitions, so 5× gives them ~35% of action gradient.
- **Params**: 4,345,444 (same as baseline)

### E010c — Derived Threshold Features
- **Flag**: `ctrl_threshold_features: true`
- **What it does**: Computes 5 binary features per player on-the-fly from stick values: in_deadzone, walk_zone, dash_zone, x_sign, stick_y_up. Matches Melee's engine thresholds (~0.2875 walk, ~0.79 dash).
- **Params**: 4,349,284 (+3,840 from wider ctrl projection)

## Shared Model Config

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Parameters | ~4.35M (varies by experiment) |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |

## Shared Training Config

| Field | Value |
|-------|-------|
| Epochs | 2 |
| Batch size | 4096 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=1.0, action=2.0, jumps=0.5, l_cancel=0.3, hurtbox=0.3, ground=0.3, last_attack=0.3 |
| Encoding | v3, state_flags, hitstun, focal_offset=3, multi_position=true |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | H100 |
| Modal timeout | 86400s (24hr — default) |
| wandb | `shinewave/melee-worldmodel` |
| Runs | 3 parallel (separate Modal invocations) |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch | ~4,126 |
| log_interval | 1000 batches |
| Logs per epoch | ~4 |
| Time between logs | ~5.5 min (H100 est.) |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed (H100) | ~0.33s |
| Est. epoch time | ~23 min |
| Est. training time | ~46 min |
| Est. data load | ~3-5 min |
| Est. total wall time | ~50 min per run |
| **Est. cost per run** | **~$3.30** |
| **Est. total (×3)** | **~$10** |

Note: batch speed estimated from E009a observed 46 min/epoch on A100, halved for H100.

## Escape Hatches

- **Kill if**: movement change_acc below baseline at epoch 1, loss diverging, OOM
- **Resume**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume /data/checkpoints/{slug}/best.pt ...`
- **Fallback**: If all three fail, revisit with larger K (E009b K=30 showed 79.5% change_acc overall — movement gains may come from longer context rather than ctrl pathway)

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| e009a-v3-focal-multi-k10 | 2K v3 | 2 | 70.9% | 0.560 | 0.278 | Direct baseline |
| e009b-v3-focal-multi-k30 | 2K v3 | 2 | 79.5% | 0.499 | 0.288 | K=30 variant, higher change_acc |
| mamba2-first-complete | 2K v2 | 2 | 71.5% | 0.608 | 0.289 | Pre-v3, older encoding |

## What's New

- **E010a**: New ctrl residual pathway — action heads see raw ctrl, bypassing SSM dilution
- **E010b**: Per-sample transition weighting — 5× loss on action-change frames
- **E010c**: 10 pre-computed threshold binary features from Melee's stick deadzones

All three are config-gated with `False`/`1.0` defaults. Baseline behavior unchanged when flags are off.

## Launch Commands

```bash
# E010a — Ctrl Residual
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e010a-ctrl-residual.yaml \
  --encoded-file /encoded-game-v3-2k.pt \
  --epochs 2 --run-name e010a-ctrl-residual

# E010b — Transition Weight
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e010b-transition-weight.yaml \
  --encoded-file /encoded-game-v3-2k.pt \
  --epochs 2 --run-name e010b-transition-weight

# E010c — Ctrl Thresholds
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e010c-ctrl-thresholds.yaml \
  --encoded-file /encoded-game-v3-2k.pt \
  --epochs 2 --run-name e010c-ctrl-thresholds
```

## Run Log

### v1 — Wrong data (2026-02-28 09:20 UTC)

**DATA MISMATCH**: Launched with `encoded-game-v3-2k.pt` instead of `encoded-v3-ranked-50k-part-0.pt`. This file had `projectiles=True` baked in. Config validator caught it but Scav flipped configs to match data instead of switching data files. Results below are **not comparable to E009a baseline**.

| What | Card said | v1 actual | Mismatch? |
|------|-----------|-----------|-----------|
| Encoded file | encoded-v3-ranked-50k-part-0.pt | **encoded-game-v3-2k.pt** | YES |
| projectiles | false | **true** | YES |
| float columns | 138 (69/player) | **144 (72/player)** | YES |
| Train examples | ~21.4M | **17,495,326** | YES |
| Val examples | ~2.3M | **1,920,861** | YES |
| Games | 2,000 | 2,000 | ok |
| GPU | H100 | H100 80GB HBM3 | ok |
| Params (E010a) | 4,366,244 | 4,368,548 | minor (projectile dims) |
| Params (E010b) | 4,345,444 | 4,347,748 | minor |
| Params (E010c) | 4,349,284 | 4,351,588 | minor |

**v1 results** (not comparable to E009a — different data + encoding):

| Metric | E010a ctrl-residual | E010b transition-weight | E010c thresholds |
|--------|--------------------|-----------------------|-----------------|
| change_acc | 70.9% | 80.8% | 76.8% |
| action_acc | 96.8% | 95.3% | 97.4% |
| pos_mae | 0.63 | 0.69 | 0.65 |
| val_loss | 0.294 | 0.872 | 0.249 |
| wall time | 34 min | 34 min | 34 min |
| wandb | [q2pm2p5o](https://wandb.ai/shinewave/melee-worldmodel/runs/q2pm2p5o) | [jexx16wv](https://wandb.ai/shinewave/melee-worldmodel/runs/jexx16wv) | [u1ipsk4p](https://wandb.ai/shinewave/melee-worldmodel/runs/u1ipsk4p) |

**Root cause**: `encoded-game-v3-2k.pt` filename sounds like the right file but was encoded with projectiles=true. Deleted from Modal volume to prevent reuse. Local copy retained at `~/claude-projects/nojohns-training/data/`.

### v2 — Correct data (2026-02-28 16:42 UTC)

Relaunched E010b on `encoded-v3-ranked-50k-part-0.pt` (capped to 2K). Confirmed: projectiles=False, 138 float cols, 21.4M train examples — matches E009a exactly.

- **E010b-v2**: [obj9ab0p](https://wandb.ai/shinewave/melee-worldmodel/runs/obj9ab0p) — IN PROGRESS
- **E010c-v2**: Not yet launched

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
- [ ] ScavieFae reviewed
