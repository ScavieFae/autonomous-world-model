---
id: e018d
created: 2026-03-09
status: running
type: training-regime
base_build: b001
built_on: [e018a]
source_paper: 2508.13009
rollout_coherence: null
prior_best_rc: 6.26
---

# Run Card: e018d-horizon-weighted-loss

## Goal

Test whether weighting later AR steps more heavily in the SF loss improves rollout coherence. E018b showed truncated BPTT saturates at N=3 — this optimizes the learning signal within the 3-step window instead of extending it.

## What Changes

One change from E018a: add horizon weighting to the SF loss.

```python
# E018a (uniform):
sf_loss = torch.stack(all_losses).mean()

# E018d (linear ramp):
weights = torch.linspace(0.5, 2.0, steps=N)  # [0.5, 1.25, 2.0]
sf_loss = (losses * weights).sum() / weights.sum()
```

Step 1 (nearly TF, easy) gets 0.5× weight. Step 3 (max drift, hardest) gets 2.0× weight.

Config: `experiments/e018d-horizon-weighted.yaml` — `self_forcing.horizon_weights: true`

## Why This Matters

E018b proved that N=5 with uniform weighting regressed (SF loss doubled, RC +3% worse). The gradient signal degrades over longer horizons with truncated BPTT. Rather than going longer, we optimize within N=3:

- Step 1 is nearly identical to TF — easy for the model, low learning value
- Step 3 is where real drift happens — hardest but most informative
- Uniform weighting wastes 33% of capacity on the easy step

## Data

1.9K FD top-5, 1 epoch, bs=512. Identical to E018a.

## Target Metrics

| Metric | E018a (uniform) | Target | Kill threshold |
|--------|-----------------|--------|---------------|
| **rollout_coherence** | **6.26** | **< 6.20** | ≥ 6.26 (no improvement) |
| change_acc | 61.6% | Stable | < 50% |
| sf_loss | 0.383 | Monitor | > 0.60 (destabilized) |

## Cost Estimate

~$4 (2hr on A100 40GB). Scout tier.
