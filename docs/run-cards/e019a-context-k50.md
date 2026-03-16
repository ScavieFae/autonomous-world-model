---
id: e019a
created: 2026-03-16
status: running
type: architectural
base_build: b001
built_on: [e018c]
source_paper: 2505.20171
rollout_coherence: null
prior_best_rc: 6.03
---

# Run Card: e019a-context-k50

## Goal

Test whether extending context from K=30 (500ms) to K=50 (833ms) continues the RC improvement trend. E018c showed K=10 to K=30 yielded a 3.7% gain (6.26 to 6.03). This tests the next point on the scaling curve.

## What Changes

Two config params from E018c: `context_len: 50`, `chunk_size: 50`. No code changes. Self-Forcing still enabled with ratio=4, unroll_length=3.

## Target Metrics

- **Primary:** rollout_coherence < 6.03 (beat E018c)
- **Guard rails:** no regression in change_acc, action accuracy, or sf_loss

## Cost Estimate

- Expected runtime: ~12,000-13,000s (~200-220min) on A100 40GB. K=50 is ~67% longer sequences than K=30, so ~50-60% longer wall time accounting for overhead.
- Estimated cost: ~$5-6

## Risk

Diminishing returns are likely as context grows. K=50 may exceed the useful temporal horizon for Melee state prediction, or the SSM may struggle to utilize the additional context effectively. If RC plateaus or regresses, that establishes the ceiling for this axis.
