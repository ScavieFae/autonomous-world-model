---
id: e023b-epoch2
created: 2026-03-21
status: discarded
type: training-regime
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: 5.775
prior_best_rc: 5.775
---

# Run Card: e023b-epoch2

## Goal

Test whether a second training epoch improves RC at d_model=768 (15.8M params). Prior observation at d_model=384: "val metrics plateau after 1 epoch." At 3.7x more params, the model may be data-starved. Also validates AMP (mixed precision) as safe for future experiments.

## What Changes

Resume from e023b best.pt checkpoint with `num_epochs=2` and `training.amp=true`. Effective LR ~2.5e-4 (cosine schedule at step 1/2).

## Results

| Metric | E023b (epoch 1) | E023b-epoch2 (epoch 2) | Delta |
|---|---|---|---|
| **rollout_coherence** | **5.775** | **5.775** | **flat (identical to 3 decimals)** |
| change_acc | 66.0% | 77.1% | +11.1pp |
| pos_mae | 0.823 | 0.693 | -15.8% |
| sf_loss | 0.618 | 0.517 | -16.3% |

Runtime: 11202s (~3.1hr) on A100 40GB with AMP. Cost: ~$4.50.
wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/mv7i0z5m

## Director Evaluation

**Verdict: DISCARDED — RC flat despite massive TF metric gains.**

Epoch 2 dramatically improved teacher-forced metrics: change_acc +11.1pp, pos_mae -15.8%, sf_loss -16.3%. But rollout coherence is identical (5.775 to 3 decimal places). This is the strongest possible confirmation of program.md's core insight: **teacher-forced improvements do not predict autoregressive quality.**

The model is not data-starved in the TF sense — it's learning more about Melee physics from epoch 2. But that additional knowledge doesn't help when consuming its own imperfect outputs. The bottleneck is exposure bias, not data coverage.

**AMP validation:** AMP produced identical RC to the non-AMP epoch 1, confirming float16 autocast is safe. Saved ~1hr (3.1hr vs 4.1hr). Use AMP for all future A100/H100 experiments.

**Key finding:** At 15.8M params on 1.9K data, epoch 2 is pure TF overfitting with zero AR benefit. Don't run epoch 2 unless data scale changes. More data (7.7K) would be a better use of compute than more epochs.
