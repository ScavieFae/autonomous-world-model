---
id: e023b
created: 2026-03-20
status: running
type: architectural
base_build: b001
built_on: [e018c]
source_paper: null
rollout_coherence: null
prior_best_rc: 6.03
---

# Run Card: e023b-dmodel768

## Goal

Phase 1 Tier 1 architecture grid — test whether d_model=384 is capacity-constrained by doubling to 768. E023a (d_model=192) showed clear underfitting (RC 6.065, change_acc -7.6pp, sf_loss +45%), confirming the model is not over-parameterized at 384. This tests the opposite direction.

## Context

| Config | d_model | Params | RC | Notes |
|--------|---------|--------|----|-------|
| E023a | 192 | ~1.3M | 6.065 | Underfitting — change_acc -7.6pp, sf_loss +45% |
| E018c | 384 | 4,275,810 | 6.03 | Current best |
| **E023b** | **768** | **15,648,882** | **?** | **This experiment** |

15.6M params at ~1.9K training games = ~8,200 params/game. Overfitting risk is real but managed by dropout=0.1 and weight_decay=1e-5.

## What Changes

One config change from E018c: `d_model: 768` (was 384).

- d_inner = 2 * 768 = 1536
- num_heads = 1536 / 64 = 24 (headdim=64 unchanged)
- All other hyperparameters identical

## Target Metrics

- **Keep:** RC < 6.03 (improvement over E018c)
- **Kill:** RC > 6.10 or val loss divergence (overfitting)

## Model

| Param | E018c | E023b |
|-------|-------|-------|
| d_model | 384 | **768** |
| d_state | 64 | 64 |
| n_layers | 4 | 4 |
| headdim | 64 | 64 |
| d_inner | 768 | 1536 |
| num_heads | 12 | 24 |
| Total params | 4,275,810 | **15,648,882** |

## Training

Identical to E018c:
- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Cost

~$7.50 Scout tier (A100 40GB, ~3.5hr estimate). 3.7x more params will increase step time but A100 40GB has headroom.

## Confounds

- 8,200 params/game ratio is high. Watch for val loss divergence vs train loss as primary overfitting signal.
- Dropout 0.1 and WD 1e-5 are the only regularization. If overfitting is severe, a follow-up with higher dropout or WD would isolate capacity vs regularization.
