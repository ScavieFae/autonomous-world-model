---
id: e020b
created: 2026-03-16
status: cancelled
type: training-regime
base_build: b001
built_on: [e018c]
source_paper: 2508.13009
rollout_coherence: null
prior_best_rc: 6.03
---

# Run Card: e020b-sf-ratio-30

## Goal

Test whether higher SF ratio (30% vs 20%) improves rollout coherence. More exposure to own errors = more learning signal for error recovery. Previously rejected by Director with 0 data points — running it to get actual data.

## What Changes

One config change from E018c: `self_forcing.ratio: 3` (30% SF, was `ratio: 4` = 20%).

## Data

1.9K FD top-5, 1 epoch, bs=512. K=30 context. Identical to E018c except SF ratio.

## Target Metrics

| Metric | E018c (20% SF) | Target | Kill threshold |
|--------|---------------|--------|---------------|
| **rollout_coherence** | **6.03** | **< 6.03** | ≥ 6.10 |
| change_acc | 62.3% | Expect regression (more SF) | < 55% |

## Cost Estimate

~$5 Scout.
