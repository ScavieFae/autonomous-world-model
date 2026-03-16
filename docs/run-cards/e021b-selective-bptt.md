---
id: e021b
created: 2026-03-16
status: running
type: training-regime
base_build: b001
built_on: [e018c]
source_paper: null
rollout_coherence: null
prior_best_rc: 6.03
---

# Run Card: e021b-selective-bptt

## Goal

Test whether detaching categorical head gradients during SF steps 2+ improves rollout coherence. The 400-class action softmax produces large, noisy gradients that may destabilize the shared trunk during multi-step BPTT. By restricting autoregressive gradient flow to continuous/binary heads (smooth, low-variance signals), the trunk can learn better physics dynamics while categoricals still train on the teacher-forced step 0.

Also tests N=4 unroll, which previously regressed at N=5 (e018b) and was cancelled at N=4 (e020c). Selective BPTT may unlock longer unrolls by reducing gradient noise.

## What Changes

Two changes from E018c:
1. `self_forcing.selective_bptt: true` — detach all categorical head predictions (`*_logits` except `binary_logits`) at SF steps >= 1
2. `self_forcing.unroll_length: 4` — N=4 (was 3 in e018c)

Code change in `training/trainer.py` `_self_forcing_step()`: after model forward pass, for steps >= 1 when `sf_selective_bptt` is enabled, detach all prediction keys except `continuous_delta`, `velocity_delta`, `dynamics_pred`, and `binary_logits`.

## Data

1.9K FD top-5, 1 epoch, bs=512. K=30 context. Identical to E018c except SF config.

## Target Metrics

| Metric | E018c (N=3, full BPTT) | Target | Kill threshold |
|--------|------------------------|--------|---------------|
| **rollout_coherence** | **6.03** | **< 6.03** | >= 6.10 |
| sf_loss | 0.367 | Monitor | > 0.55 (degrading) |
| pos_mae | 0.824 | <= 0.824 | > 0.90 |

## Cost Estimate

~$5-6 Scout. Slightly longer per SF batch (4 forward passes vs 3), partially offset by fewer backward-pass parameters per step.
