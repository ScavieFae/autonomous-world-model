---
id: e026c
created: 2026-03-22
status: proposed
type: training-regime
base_build: b002
built_on: []
source_paper: null
rollout_coherence: null
prior_best_rc: 5.146
---

# Run Card: e026c-sf-curriculum

## Goal

Test progressive SF horizon curriculum: train at N=1, then N=2, then N=3 over the course of training. The curriculum learning literature predicts this should work better than jumping directly to N=3 because the model learns single-step error correction before attempting multi-step.

This also directly tests whether E018b's N=5 failure was due to "the model wasn't ready" (curriculum hypothesis) vs "N=5 is fundamentally too long for truncated BPTT" (gradient hypothesis). If curriculum improves N=3 results, the curriculum hypothesis is supported and N=4/5 become viable via longer curricula.

## Context

| Experiment | SF Approach | RC | Notes |
|---|---|---|---|
| E018a | Jump to N=3 | 6.26 | First SF, -7.5% |
| E018b | Jump to N=5 | 6.45 | Regressed — too much too fast? |
| E025a | Jump to N=3 + warmup | 5.146 | Current best |
| **E026c** | **Curriculum N=1→2→3** | **?** | **This experiment** |

## What Changes

From E025a (b002 baseline): add `self_forcing.curriculum: [1, 2, 3]` and `num_epochs: 2`.

Training divided into 3 equal stages over ~61K total batches:
- Stage 1 (~20K batches): N=1 — learn single-step error correction
- Stage 2 (~20K batches): N=2 — learn 2-step cascading errors
- Stage 3 (~20K batches): N=3 — full 3-step unroll, building on prior stages

## Target Metrics

- **Keep:** RC < 5.146 (improvement over E025a)
- **Kill:** RC > 5.5 (regression)

## Cost

~$10 on A100 with AMP (2 epochs, ~6hr). Confirm tier.

## Confounds

- **2 epochs vs 1:** E023b-epoch2 showed RC flat at epoch 2. But here epoch 2 is qualitatively different (N=3 stage, not replay of same training).
- **Stage duration:** Equal thirds is a guess. Maybe N=3 should get more time (e.g., [1, 2, 3, 3] for 4 stages with N=3 getting 50%).
- **LR schedule interaction:** Cosine with warmup spans all 2 epochs. N=3 stage trains at low LR (end of cosine). This may help (gentle fine-tuning) or hurt (insufficient learning rate for new regime).
