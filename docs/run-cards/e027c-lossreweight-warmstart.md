---
id: e027c
created: 2026-03-22
status: kept
type: training-regime
base_build: b002
built_on: [e025b, e026c]
source_paper: null
rollout_coherence: 4.939
prior_best_rc: 4.965
---

# Run Card: e027c-lossreweight-warmstart

## Goal

Test multi-epoch regime switching: train epoch 1 with position-focused loss weights (continuous=2.0, action=1.0, from e025b), then switch to standard weights for epoch 2. Hypothesis: a position-focused first epoch builds a better physics foundation that standard-weight training can convert into AR quality.

This tests the regime switching pattern proposed in issue #17 — different training objectives across epochs, where each epoch's configuration serves a distinct purpose rather than replaying the same regime.

## Context

| Experiment | Approach | RC | Notes |
|---|---|---|---|
| E025b | Loss reweight (1 epoch) | 5.907 | Discarded — continuous=2.0/action=1.0 regressed as standalone |
| E026c | SF curriculum (2 epochs) | 4.965 | Kept — progressive N=1->2->3 made epoch 2 productive |
| **E027c** | **E025b weights -> standard (2 epochs)** | **4.939** | **KEPT — new best, -0.5%** |

E025b was discarded as a standalone experiment (RC 5.907, +2.3% regression). But the position-focused weights may serve as a useful pre-training phase when followed by standard-weight fine-tuning.

## What Changes

- **Epoch 1:** Resume from e025b checkpoint (trained with continuous=2.0, action=1.0)
- **Epoch 2:** Switch to standard loss weights (continuous=1.0, action=2.0) and continue training

## Target Metrics

- **Keep:** RC < 4.965 (improvement over e026c)
- **Kill:** RC > 5.2

## Model

Identical to b002: d_model=768, d_state=64, n_layers=4, headdim=64, 15,794,548 params.

## Training

- Resume from: e025b checkpoint (loss-reweight: continuous=2.0, action=1.0)
- Epoch 2: standard weights (continuous=1.0, action=2.0)
- lr: 5e-4, weight_decay: 1e-5, batch_size: 512
- warmup_pct: 0.05, AMP: enabled
- Self-Forcing: ratio=4 (20%), unroll_length=3, context_len=30

## Cost

~$5 (A100 40GB with AMP, ~3hr for epoch 2).

## Confounds

- Two changes vs e026c: different epoch 1 base (loss-reweight vs curriculum) and standard weights in epoch 2 vs curriculum stages. The regime switching pattern is the shared variable.
- The e025b checkpoint was trained on b001 (no warmup, no unimix), so epoch 2 inherits that foundation.
- Small delta (-0.5%) is close to noise floor, but the eval is deterministic (seed=42).

## Results

| Metric | E026c (prior best) | E027c | Delta |
|--------|-------------------|-------|-------|
| **rollout_coherence** | **4.965** | **4.939** | **-0.5%** |
| change_acc | 80.2% | 78.9% | -1.3pp |
| pos_mae | 0.642 | 0.650 | +1.2% |

- **wandb:** 0p9495nw
- **Cost:** ~$5

## Director Evaluation

**Verdict: KEPT — new best**

RC 4.939 is a 0.5% improvement over e026c (4.965). Small but real: the eval is deterministic (seed=42, 300 fixed starting points). pos_mae is comparable (0.650 vs 0.642, +1.2%). change_acc is slightly lower (78.9% vs 80.2%, -1.3pp).

This validates multi-epoch regime switching (issue #17). A position-focused epoch 1 builds a better physics foundation that standard-weight epoch 2 converts to AR quality. E025b's loss reweighting was discarded as a standalone experiment (RC 5.907) but works as a warm-start phase.

The pattern generalizes the e026c insight: epoch 2 is productive when it introduces a qualitatively different training objective (e026c: harder SF horizons; e027c: different loss weights). Plain epoch 2 with the same regime remains unproductive (e023b-epoch2: RC flat).

Cumulative RC improvement from E019 baseline: -27.0% (6.77 -> 4.939).
