---
id: e020a
created: 2026-03-16
status: discarded
type: training-regime
base_build: b001
built_on: [e018c]
source_paper: 2508.13009
rollout_coherence: 6.62
prior_best_rc: 6.03
---

# Run Card: e020a-sf-ratio-10

## Goal

Test whether lower SF ratio (10% vs 20%) improves rollout coherence.

## What Changes

One config change from E018c: `self_forcing.ratio: 9` (10% SF, was 20%).

## Results

**DISCARDED — clear regression.**

| Metric | E018c (20% SF) | E020a (10% SF) | Delta |
|--------|---------------|---------------|-------|
| **rollout_coherence** | **6.03** | **6.62** | **+9.8% (worse)** |
| change_acc | 62.3% | 63.3% | +1.0pp (better TF) |
| pos_mae | 0.824 | 0.836 | +1.5% |
| sf_loss | 0.367 | 0.534 | +46% |
| h10_action_acc | 75.5% | 76.3% | +0.8pp |

Runtime: 7288s (~121min), cost ~$4.25.
wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/ryv7n04y

## Finding

10% SF is too little. The model gets better TF metrics (change_acc +1pp) but substantially worse AR quality (RC +9.8%). This is the core insight in action: more TF training doesn't help AR. The 20% ratio in E018a/E018c is closer to optimal than 10%. SF ratio observations: 10% regressed (1/1), 20% works (1/1), 30% untested.
