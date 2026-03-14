---
id: e018d
created: 2026-03-09
status: proposed
type: training-regime
base_build: b001
built_on: [e018a]
source_paper: 2508.13009
rollout_coherence: null
prior_best_rc: null
---

# Run Card: e018d-horizon-weighted-loss

## Goal

Test horizon-weighted loss during Self-Forcing: weight later AR steps more heavily than earlier ones. The intuition is that step 1 of an AR rollout is nearly identical to teacher forcing (only 1 frame of error), while step 5+ is where real drift happens. Weighting later steps tells the model to prioritize long-horizon stability.

**Depends on e018a (Self-Forcing).** This is a refinement — run it after baseline Self-Forcing results are in.

## What Changes

In the Self-Forcing loop, instead of uniform loss weighting:

```python
# Uniform (e018a baseline)
sf_loss = torch.stack(sf_losses).mean()

# Horizon-weighted (e018d)
weights = torch.linspace(0.5, 2.0, steps=N)  # ramp from 0.5× to 2×
sf_loss = (torch.stack(sf_losses) * weights).mean()
```

This is a one-line change on top of e018a. The rest of the experiment is identical.

## Why

Matrix-Game 2.0 doesn't discuss loss weighting by horizon, but the logic follows from their analysis: exposure bias is worst at long horizons. If you weight all horizons equally, the model optimizes for easy short-horizon predictions at the expense of hard long-horizon ones.

## Target Metrics

| Metric | E018a (uniform SF) | Target |
|--------|-------------------|--------|
| Rollout coherence at K=5 | TBD | May regress slightly (less weight on early steps) |
| Rollout coherence at K=20 | TBD | Improvement |
| Overall rollout coherence | TBD | Improvement |

## Variants to Try

- Linear ramp: `[0.5, 0.875, 1.25, 1.625, 2.0]` for N=5
- Exponential: `[1, 1.5, 2.25, 3.375, 5.0]` — aggressive weighting on later steps
- Step function: uniform weight but 2× on the final step only

Start with linear ramp. If it helps, try exponential.
