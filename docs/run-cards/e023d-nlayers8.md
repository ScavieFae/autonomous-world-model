---
id: e023d
created: 2026-03-20
status: discarded
type: architectural
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: 7.108
prior_best_rc: 5.775
---

# Run Card: e023d-nlayers8

## Goal

Phase 1 Tier 1 architecture grid — test whether doubling depth (n_layers 4 → 8) at the wider d_model=768 compounds the capacity gain from E023b. Width improved RC by 4.2%; does depth add further?

## Context

| Config | n_layers | d_model | Params | RC | Notes |
|--------|----------|---------|--------|----|-------|
| E018c | 4 | 384 | 4,275,810 | 6.026 | Former best |
| E023b | 4 | 768 | 15,648,882 | 5.775 | Current best |
| **E023d** | **8** | **768** | **~30.5M** | **?** | **This experiment** |

~30.5M params at ~1.9K training games = ~16,000 params/game. High overfitting risk — 2x the ratio of E023b (8,200).

## What Changes

One config change from E023b: `n_layers: 8` (was 4).

- d_model = 768, d_inner = 1536, num_heads = 24 (all unchanged)
- All other hyperparameters identical

## Target Metrics

- **Keep:** RC < 5.775 (improvement over E023b)
- **Kill:** RC > 6.10 or val loss divergence

## Model

| Param | E023b | E023d |
|-------|-------|-------|
| d_model | 768 | 768 |
| d_state | 64 | 64 |
| n_layers | 4 | **8** |
| headdim | 64 | 64 |
| d_inner | 1536 | 1536 |
| num_heads | 24 | 24 |
| Total params | 15,648,882 | **~30.5M** |

## Training

Identical to E023b:
- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Cost

~$13 (H100, ~4.7hr). 30.5M params doubled step time vs E023b. H100 used instead of A100 due to memory/throughput.

## Confounds

- 16,000 params/game ratio is very high. Overfitting is the primary risk.
- H100 vs A100 — different GPU than E023b. Hardware difference is unlikely to affect final metrics (deterministic eval) but runtime/cost is not directly comparable.
- Doubling depth at already-wide 768 tests a combined scale-up. If it fails, unclear whether depth itself is bad or the combo is over-parameterized.

## Results

| Metric | E023b (4 layers) | E023d (8 layers) | Delta |
|--------|-----------------|-----------------|-------|
| **RC** | **5.775** | **7.108** | **+23.1%** |
| change_acc | 66.0% | 66.0% | flat |
| pos_mae | 0.823 | 0.824 | flat |
| p0_action_acc | 95.8% | 95.8% | flat |
| sf_loss | 0.618 | 0.413 | -33% |
| h10_pos_mae | 5.426 | 6.749 | **+24.4%** |
| h10_action_acc | 77.3% | 77.7% | flat |

Runtime: ~16960s (~4.7hr) on H100. Cost: ~$13.00.
wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/jg3cnzxs

## Director Evaluation

**Verdict: DISCARDED**

RC 7.108 is a 23.1% regression — the largest single-experiment regression in the architecture grid. TF metrics are flat (change_acc, pos_mae, action_acc all unchanged), but AR rollout quality degrades severely (h10_pos_mae +24.4%). The lower sf_loss (-33%) is misleading — the deeper model memorizes self-forcing trajectories better during training but generalizes worse at inference.

Depth hurts at d_model=768 with 1.9K data. The 16,000 params/game ratio is likely over-parameterized for this data scale. The model fits training data well (flat TF metrics, lower SF loss) but doesn't transfer to AR rollouts.

**Architecture grid status after E023d:**
- Width: monotonic improvement (192 < 384 < 768). High-value axis. KEPT.
- Depth: 8 layers regressed 23.1% at d_model=768. 0/1 experiments improved. Depth is low-priority.
- d_state, headdim: untested but lower priority given depth result.

Phase 1 width axis is the clear winner. Depth scaling requires more data (7.7K+) or better regularization to be revisitable.
