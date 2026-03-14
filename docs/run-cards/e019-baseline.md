---
id: e019
created: 2026-03-14
status: proposed
type: training-regime
base_build: b001
built_on: []
source_paper: null
rollout_coherence: null
prior_best_rc: null
---

# Run Card: e019-baseline

## Goal

Establish the first rollout coherence baseline using the stable build — all proven improvements combined. This is not a new experiment; it's the reference point every future experiment compares against.

## What This Is

The "stable build" config: every kept improvement from E008–E017, on the largest clean dataset (7.7K FD top-5). The first checkpoint to get a rollout coherence score.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-v3-ranked-fd-top5.pt` |
| Games | ~7,700 FD top-5 |
| Stage | Final Destination (32) |
| Characters | Fox, Falcon, Sheik, Marth, Falco |
| Train/val split | 90/10 by game |

## Model

Mamba-2 (locked):
- d_model=384, d_state=64, n_layers=4, headdim=64
- context_len=10, chunk_size=10
- ~4.35M params

## Training

| Param | Value | Source |
|-------|-------|--------|
| lr | 5e-4 | E012+ |
| weight_decay | 1e-5 | E012+ |
| batch_size | 4096 | E012+ |
| epochs | 2 | baseline |
| scheduled_sampling | 0.3 | E015/E016 |
| ss_true | true | E015/E016 |
| action_change_weight | 5.0 | E010b |

## Encoding Flags

All proven flags enabled:

| Flag | Value | Source |
|------|-------|--------|
| state_age_as_embed | true | E012+ |
| state_flags | true | v3 |
| hitstun | true | v3 |
| ctrl_threshold_features | true | E010c |
| multi_position | true | E008c |
| cascaded_heads | true | E014 |
| cascade_embed_dim | 16 | E014 |
| focal_offset | 0 | E012 |

## Target Metrics

This establishes baselines, not targets:

| Metric | E012 (1.9K) | E016 (7.7K) | E019 target |
|--------|-------------|-------------|-------------|
| change_acc | 91.1% | 94.3% | ~94% (match E016) |
| pos_mae | 0.706 | 0.466 | ~0.47 |
| val_loss | 0.527 | 0.327 | ~0.33 |
| **rollout_coherence** | **unknown** | **unknown** | **establish baseline** |

## Launch Command

```bash
modal run scripts/modal_train.py \
    --config experiments/e019-baseline.yaml \
    --encoded-file /encoded-v3-ranked-fd-top5.pt
```

## Cost Estimate

A100 40GB, ~60-90 min training + ~1 min eval = ~$3-4.50 total.
