---
id: e019
created: 2026-03-14
status: running
type: training-regime
base_build: b001
built_on: []
source_paper: null
rollout_coherence: null
prior_best_rc: 6.8448
---

# Run Card: e019-baseline

## Goal

Establish the first rollout coherence baseline using proven-in-isolation improvements on 7.7K data. This is the reference point every future experiment compares against.

## What This Is

The b001 stable build on the largest clean dataset. No cascaded heads (unproven in isolation at scale), no scheduled sampling (not ported). Pure proven encoding + training practices from E008c–E012.

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
| action_change_weight | 5.0 | E010b |

## Encoding Flags

Proven-in-isolation flags only:

| Flag | Value | Source |
|------|-------|--------|
| state_age_as_embed | true | E012+ |
| state_flags | true | v3 |
| hitstun | true | v3 |
| ctrl_threshold_features | true | E010c |
| multi_position | true | E008c |
| focal_offset | 0 | E012 |

**Excluded:** cascaded_heads (E014 — AR damage fix at 1.9K, TF regression, never isolated at 7.7K). Will be tested as e020.

## Target Metrics

This establishes baselines, not targets:

| Metric | E012 (1.9K, no cascade) | E019 target |
|--------|-------------------------|-------------|
| change_acc | 91.1% | TBD (expect higher with 4x data) |
| pos_mae | 0.706 | TBD |
| val_loss | 0.527 | TBD |
| **rollout_coherence** | **6.8448** | **establish baseline on 7.7K** |

## Launch Command

```bash
modal run --detach scripts/modal_train.py \
    --config experiments/e019-baseline.yaml \
    --encoded-file /encoded-v3-ranked-fd-top5.pt
```

## Cost Estimate

A100 40GB, ~60-90 min training + ~1 min eval = ~$3-4.50 total.
