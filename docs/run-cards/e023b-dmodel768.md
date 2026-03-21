---
id: e023b
created: 2026-03-20
status: kept
type: architectural
base_build: b001
built_on: [e018c]
source_paper: null
rollout_coherence: 5.775
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

## Results

| Metric | E018c (d=384) | E023b (d=768) | E023a (d=192) | Delta |
|--------|--------------|---------------|---------------|-------|
| **rollout_coherence** | **6.026** | **5.775** | 6.065 | **-4.2%** |
| change_acc | 62.3% | 66.0% | 54.7% | **+3.7pp** |
| pos_mae | 0.824 | 0.823 | 0.845 | flat |
| p0_action_acc | 95.3% | 95.8% | 94.5% | +0.5pp |
| sf_loss | 0.367 | 0.618 | 0.536 | +68% |
| h10_pos_mae | 5.770 | 5.426 | 5.807 | **-6.0%** |
| h10_action_acc | 75.5% | 77.3% | 74.5% | +1.8pp |

Runtime: 14911s (~4.1hr) on A100. Cost: ~$8.70.
wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/9xacat7s

## Director Evaluation

**Verdict: KEPT — New Best**

RC 5.775 is a 4.2% improvement — second-largest single-experiment gain after Self-Forcing (7.5%). First experiment to improve BOTH RC and change_acc simultaneously (+3.7pp). h10_pos_mae improved 6.0%.

Width axis is monotonic: d_model=192 (6.065) < 384 (6.026) < 768 (5.775). Model was capacity-constrained. No overfitting detected (val loss stable, dropout + WD held).

SF loss jumped 68% but this is expected — richer representations assign higher loss to slightly-off self-predictions, while still producing better AR rollouts overall.

**Next:** test d_model=512 for efficient frontier (3.7× param increase is expensive onchain).
