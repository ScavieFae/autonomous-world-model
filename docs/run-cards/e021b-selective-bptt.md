---
id: e021b
created: 2026-03-16
status: discarded
type: training-regime
base_build: b001
built_on: [e018c]
source_paper: null
rollout_coherence: 6.87
prior_best_rc: 6.03
---

# Run Card: e021b-selective-bptt

## Goal

Test whether detaching categorical head gradients during SF steps 2+ enables longer unrolls (N=4). Hypothesis: categorical gradients are noisier under truncated BPTT, detaching them recovers signal for continuous heads.

## What Changes

Two changes from E018c: `selective_bptt: true` (detach categorical heads at steps 2+), `unroll_length: 4`.

## Results

**DISCARDED — substantial regression, SF loss exploded.**

| Metric | E018c (N=3, full) | E021b (N=4, selective) | Delta |
|--------|-------------------|----------------------|-------|
| **rollout_coherence** | **6.03** | **6.87** | **+13.9% (worse)** |
| change_acc | 62.3% | 52.3% | -10pp |
| pos_mae | 0.824 | 0.826 | flat |
| sf_loss | 0.367 | 1.481 | +304% |
| total_loss | 0.416 | 0.684 | +64% |
| h10_action_acc | 75.5% | 71.7% | -3.8pp |

Runtime: 10893s (~182min), cost ~$6.40.
wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/ibg5r4ya

## Finding

Selective BPTT catastrophically destabilized training. SF loss exploded 4x (0.37→1.48), far beyond the 0.55 kill threshold. Detaching categorical heads at steps 2+ removed too much gradient signal — the model needs categorical supervision at every step to maintain coherent state predictions. The continuous heads alone don't carry enough information to guide error recovery.

This suggests the categorical heads (especially the 400-class action state) are not "noise" in the gradient — they're structurally important for the model to understand what it's doing wrong during AR unrolls. The gradient surgery hypothesis was wrong for this architecture: all heads contribute meaningfully to the learning signal, even under truncated BPTT.

Combined with E018b (N=5 full, regressed) and E018d (horizon weights, regressed), N>3 under truncated BPTT appears firmly limited regardless of which heads get gradient. The constraint is fundamental to truncated BPTT, not to specific head gradients.
