---
id: e025b
created: 2026-03-21
status: discarded
type: hyperparameter
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: 5.907
prior_best_rc: 5.775
---

# Run Card: e025b-loss-reweight

## Goal

Test whether shifting gradient budget from action to position improves rollout coherence. Position prediction (continuous head) is the primary driver of RC but has weight 1.0, while action has weight 2.0 despite TF action_acc already at 95.8%. Hypothesis: the action head is near saturation and additional gradient signal is wasted; reallocating to position will improve AR quality.

## What Changes

Two loss weight changes from E023b:
- `continuous: 1.0 -> 2.0` (double position weight)
- `action: 2.0 -> 1.0` (halve action weight)

AMP enabled (validated safe in e023b-epoch2). All other hyperparameters identical.

## Target Metrics

- **Keep:** RC < 5.775 (improvement over E023b)
- **Kill:** RC > 5.85 or action_acc drops below 93% (head undertraining)

## Model

Identical to E023b: d_model=768, d_state=64, n_layers=4, headdim=64, 15,648,882 params.

## Training

- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- AMP: enabled (float16 autocast + GradScaler)
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Loss Weights

| Weight | E023b | E025b | Rationale |
|--------|-------|-------|-----------|
| continuous | 1.0 | **2.0** | Position drives RC; currently underweighted |
| action | 2.0 | **1.0** | 95.8% TF acc — near saturation, wasted gradient |
| velocity | 0.5 | 0.5 | Unchanged |
| dynamics | 0.5 | 0.5 | Unchanged |
| binary | 1.0 | 1.0 | Unchanged |
| jumps | 0.5 | 0.5 | Unchanged |
| l_cancel | 0.3 | 0.3 | Unchanged |
| hurtbox | 0.3 | 0.3 | Unchanged |
| ground | 0.3 | 0.3 | Unchanged |
| last_attack | 0.3 | 0.3 | Unchanged |

## Cost

~$5 (A100 40GB with AMP).

## Confounds

- Changing two weights simultaneously (continuous up, action down). If RC improves, a follow-up could isolate which change mattered.
- Action accuracy may drop, which could hurt RC through action-dependent position dynamics. Watch for action_acc regression below 94%.

## Results

| Metric | E023b (baseline) | E025b (reweight) | Delta |
|--------|-----------------|------------------|-------|
| **rollout_coherence** | **5.775** | **5.907** | **+2.3%** |
| change_acc | 66.0% | 65.3% | -0.7pp |
| pos_mae | 0.823 | 0.764 | -7.2% |
| loss | — | 0.227 | — |

wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/3pmpx26d

## Director Evaluation

**Verdict: DISCARDED**

RC 5.907 is a 2.3% regression. Despite a notable improvement in pos_mae (0.823 to 0.764, -7.2%), rollout coherence worsened. The lower loss (0.227 vs typical ~0.4) suggests the model optimized for teacher-forced position accuracy at the expense of autoregressive stability.

Halving action weight degraded change_acc (-0.7pp), and the action head may need its full gradient budget to maintain the action-position coupling that matters for AR rollouts. The hypothesis that action was near saturation and wasting gradient was incorrect — the gradient signal to the action head contributes to backbone representations that benefit all heads.
