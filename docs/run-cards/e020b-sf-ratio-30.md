---
id: e020b
created: 2026-03-19
status: running
type: training-regime
base_build: b001
built_on: [e018c]
source_paper: 2508.13009
rollout_coherence: null
prior_best_rc: 6.03
---

# Run Card: e020b-sf-ratio-30

## Goal

Test whether higher SF ratio improves rollout coherence. More exposure to own errors = more learning signal for error recovery. E018c used ratio:4 (SF every 4th step = 25% SF). This experiment uses ratio:3 (SF every 3rd step = 33% SF).

**Director note:** ratio:3 = 1/3 = 33%, not 30% as the experiment name implies. The name is approximate. The actual SF percentage is 33%.

## What Changes

One config change from E018c: `self_forcing.ratio: 3` (was `ratio: 4`).

## Data

1.9K FD top-5, 1 epoch, bs=512. K=30 context. Identical to E018c except SF ratio.

## Target Metrics

| Metric | E018c (25% SF) | Target | Kill threshold |
|--------|---------------|--------|---------------|
| **rollout_coherence** | **6.03** | **< 6.03** | >= 6.10 |
| change_acc | 62.3% | Expect regression (more SF) | < 55% |

## Cost Estimate

~$5 Scout.
