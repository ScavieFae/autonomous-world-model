---
id: e020c
created: 2026-03-16
status: cancelled
type: training-regime
base_build: b001
built_on: [e018c]
source_paper: 2508.13009
rollout_coherence: null
prior_best_rc: 6.03
---

# Run Card: e020c-sf-n4

## Goal

Test N=4 unroll — between the working N=3 (E018a, RC 6.26) and failing N=5 (E018b, RC 6.45). Finds the exact truncated BPTT ceiling. If N=4 works, the ceiling is between 4 and 5. If it fails, the ceiling is at exactly N=3.

## What Changes

One config change from E018c: `self_forcing.unroll_length: 4` (was 3).

## Data

1.9K FD top-5, 1 epoch, bs=512. K=30 context. Identical to E018c except unroll length.

## Target Metrics

| Metric | E018c (N=3) | Target | Kill threshold |
|--------|------------|--------|---------------|
| **rollout_coherence** | **6.03** | **< 6.03** | ≥ 6.10 |
| sf_loss | 0.367 | Monitor | > 0.55 (degrading) |

## Cost Estimate

~$5 Scout. Slightly longer per SF batch (4 forward passes vs 3).
