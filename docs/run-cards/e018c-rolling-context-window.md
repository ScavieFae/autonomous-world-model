---
id: e018c
created: 2026-03-09
status: running
type: architectural
base_build: b001
built_on: [e018a]
source_paper: 2505.20171
rollout_coherence: null
prior_best_rc: 6.26
---

# Run Card: e018c-rolling-context-window

## Goal

Test whether longer context (K=30, 500ms) improves rollout coherence. K=10 (167ms) is less than a Melee reaction window. Mamba2 SSMs should benefit from longer temporal context. Orthogonal to Self-Forcing — tests the architecture axis.

## What Changes

Two config params from E018a:

```yaml
model:
  context_len: 30    # was 10
  chunk_size: 30     # must equal context_len for SSD scan
```

No code changes. Self-Forcing still enabled (20% SF, N=3, uniform weighting).

## Why This Matters

K=10 at 60fps is 167ms — less than a single Melee reaction window. K=30 is 500ms — enough to see a full move startup + active frame sequence. The SSM paper (2505.20171) shows longer temporal context helps SSMs build up useful recurrent state. During AR rollouts, the model can see more of its own recent predictions, potentially detecting drift patterns invisible in 10 frames.

## Data

1.9K FD top-5, 1 epoch, bs=512. Identical to E018a except context window.

## Target Metrics

| Metric | E018a (K=10) | Target | Kill threshold |
|--------|-------------|--------|---------------|
| **rollout_coherence** | **6.26** | **< 6.10** | ≥ 6.26 |
| change_acc | 61.6% | Stable | < 55% |

## Cost Estimate

~$4 (2hr on A100 40GB). Scout tier.

## wandb

https://wandb.ai/shinewave/melee-worldmodel/runs/i7bz1gm1
