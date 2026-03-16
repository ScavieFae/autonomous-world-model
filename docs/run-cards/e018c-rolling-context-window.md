---
id: e018c
created: 2026-03-09
status: kept
type: architectural
base_build: b001
built_on: [e018a]
source_paper: 2505.20171
rollout_coherence: 6.03
prior_best_rc: 6.26
---

# Run Card: e018c-rolling-context-window

## Goal

Test whether longer context (K=30, 500ms) improves rollout coherence. Orthogonal to Self-Forcing — tests the architecture axis.

## What Changes

Two config params from E018a: `context_len: 30`, `chunk_size: 30`. No code changes. SF still enabled.

## Results

**KEPT — new best.**

| Metric | E018a (K=10) | E018c (K=30) | Delta |
|--------|-------------|-------------|-------|
| **rollout_coherence** | **6.26** | **6.03** | **-3.7% (better)** |
| change_acc | 61.6% | 62.3% | +0.7pp |
| pos_mae | 0.825 | 0.824 | flat |
| sf_loss | 0.383 | 0.367 | -4.2% |
| h10_pos_mae | 5.89 | 5.77 | -2.0% |
| h10_action_acc | 73.7% | 75.5% | +1.8pp |

### Cost

- Runtime: 8910s (~149min) on A100 40GB. 41% longer than E018a due to K=30.
- Cost: ~$5.20
- wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/i7bz1gm1

## Director Evaluation

**Verdict:** KEPT

**Confidence:** HIGH — 0.23-point improvement (6.26→6.03) is well outside ±0.05 noise band. All secondary metrics improved or held flat. No trade-offs.

**Finding:** Longer context (K=30, 500ms) improved RC by 3.7% on top of Self-Forcing, with no TF metric regression. The improvements compound: SF addresses exposure bias, longer context gives the SSM more temporal information to stabilize predictions. K=10 (167ms) was undershooting — less than a Melee reaction window. K=30 covers a full move sequence.
