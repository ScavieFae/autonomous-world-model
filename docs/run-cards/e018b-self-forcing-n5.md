---
id: e018b
created: 2026-03-15
status: running
type: training-regime
base_build: b001
built_on: [e018a]
source_paper: 2508.13009
rollout_coherence: null
prior_best_rc: 6.26
---

# Run Card: e018b-self-forcing-n5

## Goal

Test whether longer Self-Forcing unroll (N=5 vs N=3) improves rollout coherence further. E018a established that N=3 achieves 7.5% RC improvement (6.77→6.26). This tests the horizon axis: does more drift exposure help?

## What Changes

One config change from E018a:

```yaml
self_forcing:
  unroll_length: 5      # was 3 in e018a
```

No code changes. The trainer already supports arbitrary unroll lengths.

## Why This Matters

- N=3 = 150ms of model drift exposure (at 60fps)
- N=5 = 250ms — covers a full move startup + active frame sequence
- Longer unroll = more gradient signal about the model's own error patterns
- If this helps: the improvement curve is sublinear (diminishing returns) or linear (more is better)
- If this doesn't help: saturation at N=3, pivot to SF ratio or horizon weighting (e018d)

## Data

1.9K FD top-5 (`encoded-e012-fd-top5.pt`), 1 epoch, bs=512. Identical to E018a.

## Target Metrics

| Metric | E018a (N=3) | Target | Kill threshold |
|--------|------------|--------|---------------|
| **rollout_coherence** | **6.26** | **< 6.10** | ≥ 6.26 (no improvement) |
| change_acc | 61.6% | No catastrophic drop | < 55% |
| sf_loss | 0.383 | Monitor | > 0.50 (gradient saturating) |

## Cost Estimate

~$4.20 (2hr on A100 40GB). Scout tier.
