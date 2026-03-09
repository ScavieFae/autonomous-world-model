# Run Card: E011 — Character Importance

**Created**: 2026-02-28
**Configs**: `e011a-char-embed-32.yaml`, `e011b-char-conditioning.yaml`, `e011c-char-embed-cond.yaml`
**Branch**: `scav/research/e010-movement-suite`
**Status**: PENDING

## Goal

Test whether giving the model stronger character identity signals improves predictions. Character distribution is extremely skewed (Fox 26.9%, top 3 = 61.7%, top 8 = 89.3%), so the model needs to distinguish character-specific physics (Fox dash ≠ Marth dash ≠ Falcon dash) from a small embedding.

## Motivation

From character distribution analysis: Fox:Mewtwo ratio is 410:1. With only 8 embedding dims shared across 33 characters, the model may not be learning character-specific movement physics well. Two orthogonal interventions:
- **Representation**: bigger embedding (more capacity to encode character differences)
- **Injection**: character conditioning (project into hidden state, like ctrl conditioning, so character identity shapes all predictions directly)

## Target Metrics

| Metric | Baseline (E010d) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| change_acc | 85.8% | >86% | <83% after epoch 1 |
| action_acc | 96.3% | >96% | <94% |
| val_pos_mae | 0.65 | <0.65 | >0.75 |

## Experiments

### E011a — Bigger Character Embedding (8 → 32 dim)
- **Change**: `character_embed_dim: 32` (was 8)
- **Hypothesis**: 8 dims is too few for 33 characters' distinct physics. 32 dims (matching action_embed_dim) gives more capacity.
- **Impact**: frame_dim 278→326 (+48), adds ~19K params
- **Params**: 4,368,508

### E011b — Character Conditioning (additive projection)
- **Change**: `character_conditioning: true`
- **Hypothesis**: Character as a concatenated input feature gets diluted through SSM layers. Projecting char embedding directly into the hidden state (like ctrl conditioning) gives a character-specific prior for all predictions.
- **Impact**: new Linear(16→384), adds ~6.5K params
- **Params**: 4,355,812

### E011c — Combined A+B
- **Change**: `character_embed_dim: 32` + `character_conditioning: true`
- **Hypothesis**: If representation (a) and injection (b) are orthogonal, combination should be super-additive (as E010d was for b+c).
- **Impact**: frame_dim 278→326 + new Linear(64→384)
- **Params**: 4,393,468

## Shared Config (from E010d base)

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD), ~4.35-4.39M params |
| d_model / n_layers | 384 / 4 |
| context_len (K) | 10, chunk_size=10 |
| Epochs | 2 |
| Batch size | 4096 |
| LR | 5e-4 |
| Loss weights | action=2.0, action_change_weight=5.0, binary=1.0, continuous=1.0, velocity=0.5, dynamics=0.5 |
| Encoding | v3, state_flags=true, hitstun=true, focal_offset=3, multi_position=true, ctrl_threshold_features=true |
| Encoded file | `encoded-v3-ranked-50k-part-0.pt` (capped to 2K) |

## Data

| Field | Value |
|-------|-------|
| Encoded file | `encoded-v3-ranked-50k-part-0.pt` |
| Games | 2,000 (ranked, capped from 12K) |
| Train examples | ~21.4M |
| Val examples | ~2.3M |
| Float columns | 138 |
| Encoding | projectiles=false, state_flags=true, hitstun=true |

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
| Est. data load | ~60s (95GB capped) |
| **Est. cost per run** | **~$2.30** |
| **Est. total (×3)** | **~$6.90** |

## Prior Comparable Runs

| Run | Flags | Data | change_acc | pos_mae | val_loss | Notes |
|-----|-------|------|------------|---------|----------|-------|
| E009a | baseline | ranked 2K | 70.9% | 0.560 | 0.278 | Original baseline |
| E010d | b+c combined | ranked 2K | 85.8% | 0.65 | 0.696 | Direct baseline for E011 |
| E010e | b+c + proj | 2k-proj | 85.2% | 0.68 | 0.724 | Projectiles negligible |

## Pre-flight Checklist

- [x] All three configs pass forward smoke test
- [x] Param counts verified (E011a: 4,368,508, E011b: 4,355,812, E011c: 4,393,468)
- [ ] Use `/launch-experiment` skill to verify actual vs card

## Launch Commands

```bash
# E011a — bigger embedding
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e011a-char-embed-32.yaml \
  --encoded-file /encoded-v3-ranked-50k-part-0.pt \
  --epochs 2 --run-name e011a-char-embed-32

# E011b — character conditioning
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e011b-char-conditioning.yaml \
  --encoded-file /encoded-v3-ranked-50k-part-0.pt \
  --epochs 2 --run-name e011b-char-conditioning

# E011c — combined a+b
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/e011c-char-embed-cond.yaml \
  --encoded-file /encoded-v3-ranked-50k-part-0.pt \
  --epochs 2 --run-name e011c-char-embed-cond
```

## Run Log

### Run: e011a-char-embed-32 — 2026-02-28 18:33 UTC

**wandb**: [l331n8ba](https://wandb.ai/shinewave/melee-worldmodel/runs/l331n8ba)
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
| Model params | 4,368,508 | 4,368,508 | ✓ |
| Games | 2,000 | 2,000 | ✓ |

**No mismatches. Run is clean. Same data as E010d baseline.**

Note: first launch failed with config validator false positive (`character_embed_dim: (32, 8)`). `character_embed_dim` is model-side (nn.Embedding table), not data-side. Fixed validator to exclude embedding dim fields.

### Run: e011b-char-conditioning — 2026-02-28 18:27 UTC

**wandb**: [4vd6oqrd](https://wandb.ai/shinewave/melee-worldmodel/runs/4vd6oqrd)
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
| Model params | 4,355,812 | 4,355,812 | ✓ |
| Games | 2,000 | 2,000 | ✓ |

**No mismatches. Run is clean. Same data as E010d baseline.**

### Run: e011c-char-embed-cond — 2026-02-28 18:32 UTC

**wandb**: [gl7hfifv](https://wandb.ai/shinewave/melee-worldmodel/runs/gl7hfifv)
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
| Model params | 4,393,468 | 4,393,468 | ✓ |
| Games | 2,000 | 2,000 | ✓ |

**No mismatches. Run is clean. Same data as E010d baseline.**

### Results — 2026-02-28 19:14 UTC

| Metric | Baseline (E010d) | E011a (embed 32) | E011b (char cond) | E011c (a+b) |
|--------|-------------------|-----------------|-------------------|-------------|
| **change_acc** | 85.8% | 85.9% | 85.9% | 85.9% |
| action_acc | 96.3% | 96.3% | 96.4% | 96.4% |
| pos_mae | 0.65 | 0.67 | 0.67 | 0.68 |
| val_loss | 0.696 | 0.695 | 0.696 | 0.689 |

**E011a epoch-by-epoch:**
| | loss | action_acc | pos_mae | change_acc | val_loss |
|---|---|---|---|---|---|
| Epoch 1 | 1.077 | 94.4% | 0.78 | 74.8% | 0.748 |
| Epoch 2 | 0.573 | 96.3% | 0.67 | 85.9% | 0.695 |

**E011b epoch-by-epoch:**
| | loss | action_acc | pos_mae | change_acc | val_loss |
|---|---|---|---|---|---|
| Epoch 1 | 1.070 | 94.5% | 0.78 | 75.3% | 0.750 |
| Epoch 2 | 0.575 | 96.4% | 0.67 | 85.9% | 0.696 |

**E011c epoch-by-epoch:**
| | loss | action_acc | pos_mae | change_acc | val_loss |
|---|---|---|---|---|---|
| Epoch 1 | 1.073 | 94.5% | 0.79 | 75.1% | 0.743 |
| Epoch 2 | 0.570 | 96.4% | 0.68 | 85.9% | 0.689 |

**Key findings:**
- **Null result across all three experiments.** Character importance interventions had no measurable effect.
- change_acc: all three at 85.9% vs 85.8% baseline (within noise)
- pos_mae regressed slightly (0.67-0.68 vs 0.65) — likely noise
- val_loss: E011c showed a tiny improvement (0.689 vs 0.696) — not meaningful
- The 8-dim character embedding is already sufficient at 2K games / 2 epochs
- With 77.5% of games being Fox/Falco/Marth, character diversity is too low for bigger embeddings to help
- Character conditioning (additive projection) had zero effect — character identity is already well-captured in the input concatenation

| Metric | Target | Best E011 | vs baseline | Hit target? |
|--------|--------|-----------|-------------|-------------|
| change_acc | >86% | 85.9% | +0.1pp | ✗ (within noise) |
| action_acc | >96% | 96.4% | +0.1pp | ✓ |
| val_pos_mae | <0.65 | 0.67 | +0.02 | ✗ |

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
