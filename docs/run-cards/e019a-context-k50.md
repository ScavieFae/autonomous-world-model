---
id: e019a
created: 2026-03-16
status: discarded
type: architectural
base_build: b001
built_on: [e018c]
source_paper: 2505.20171
rollout_coherence: 5.97
prior_best_rc: 6.03
---

# Run Card: e019a-context-k50

## Goal

Test whether extending context from K=30 to K=50 (833ms) continues the RC improvement seen in K=10→K=30.

## What Changes

Config only: `context_len: 50`, `chunk_size: 50`. SF enabled.

## Results

**DISCARDED — diminishing returns.**

| Metric | E018c (K=30) | E019a (K=50) | Delta |
|--------|-------------|-------------|-------|
| **rollout_coherence** | **6.03** | **5.97** | **-1.0% (marginal)** |
| change_acc | 62.3% | 61.8% | -0.5pp |
| pos_mae | 0.824 | 0.832 | +1.0% |
| sf_loss | 0.367 | 0.579 | +58% |
| h10_pos_mae | 5.77 | 5.77 | flat |
| h10_action_acc | 75.5% | 76.0% | +0.5pp |

- Runtime: 10068s (~168min), cost ~$5.90
- wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/yo1q1buy

## Director Evaluation

**Verdict:** DISCARDED

**Confidence:** HIGH — RC improvement (6.03→5.97, -1.0%) barely exceeds ±0.05 noise band. Diminishing returns: K=10→K=30 gave -3.7%, K=30→K=50 gives -1.0% (27% of prior gain). SF loss jumped 58% (model struggles more at K=50). Secondary metrics flat or slightly regressed.

**Finding:** Context window benefits saturate at K=30. K=50 overshoots useful temporal structure — Melee move sequences are ~30-40 frames (K=30). Longer windows introduce irrelevant decayed history. K=30 established as optimal for this architecture.
