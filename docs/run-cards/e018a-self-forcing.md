---
id: e018a
created: 2026-03-09
status: kept
type: training-regime
base_build: b001
built_on: [e019]
source_paper: 2508.13009
rollout_coherence: 6.26
prior_best_rc: 6.77
---

# Run Card: e018a-self-forcing

## Goal

Test whether Self-Forcing (training on the model's own autoregressive outputs) improves rollout coherence. The model unrolls N AR steps during training, sees its own drift, and learns to recover from compounding errors.

## What Changes

One change: add Self-Forcing to the training loop (`training/trainer.py`).

- `_self_forcing_step()`: every 5th batch, unroll model 3 AR steps using `reconstruct_frame()` from `scripts/ar_utils.py`
- Ground-truth controller inputs at each AR step
- Loss at each step vs ground truth, averaged
- Truncated BPTT (detach between unroll steps)
- All other training identical to E019

### Parameters

| Param | Value |
|-------|-------|
| SF ratio | 1:4 (20% SF batches) |
| Unroll length N | 3 |
| BPTT | Truncated (detach between steps) |
| batch_size | 512 |
| epochs | 1 |
| Data | 1.9K FD top-5 |

Config: `experiments/e018a-sf-minimal.yaml`

## Results

| Metric | E019 baseline | E018a SF | Delta |
|--------|--------------|----------|-------|
| **rollout_coherence** | **6.77** | **6.26** | **-7.5% (better)** |
| change_acc | 78.7% | 61.6% | -17.1pp |
| pos_mae | 0.756 | 0.825 | +9.1% (worse) |
| val_loss | — | 0.231 | — |
| action_acc | — | 95.3% | — |
| tf_loss (final batch) | — | 0.164 | — |
| sf_loss (final batch) | — | 0.383 | — |

### Per-horizon rollout divergence

| Horizon | pos_mae | action_acc |
|---------|---------|------------|
| t+10 | 5.89 | 73.7% |

### Training cost

- Wall time: 6683s (~111 min) on A100 40GB
- Actual cost: ~$3.90 ($2.10/hr × 1.85hr)
- wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/mtpaj930

## Director Evaluation

**Verdict:** KEPT

**Rollout coherence:** 6.26 (prior best: 6.77, delta: -7.5%)
**Confidence:** HIGH — 0.51 improvement on a metric where E012→E019 moved only 0.07. Well outside ±0.05 noise band.

**Finding:** Self-Forcing with 20% SF ratio and N=3 unroll improved rollout coherence by 7.5% (6.77→6.26) despite -17pp change_acc regression and +9% pos_mae regression in TF metrics. SF loss (0.38) was ~2.3× TF loss (0.16), confirming the model faces substantially harder predictions from its own state. Validates program.md core insight: TF metrics don't predict AR quality.

**program.md update:** Add Self-Forcing to "Proven improvements" table. Update Current Best. Self-Forcing is now the most impactful single change discovered — larger effect than any encoding or architecture change in the E008-E019 series.

## Open Questions (for follow-up experiments)

- **N=5 or N=10**: Longer unroll should improve long-horizon stability further. Cost: ~2× current.
- **Higher SF ratio**: 30% or 50% SF. More exposure to own errors, but more TF regression.
- **Horizon-weighted loss (e018d)**: Weight later AR steps more heavily — step 1 is nearly TF, step 3 is where drift happens.
- **Full BPTT**: Gradient flowing through reconstruct_frame between steps. More memory, potentially better learning signal.
- **change_acc regression**: -17pp is substantial. Can we recover some with a dedicated action_change_weight or SF-specific loss weighting?

## Prior Art

- Matrix-Game 2.0 (2508.13009): Self-Forcing distillation for video world models
- E012b (noise SS): Random noise ≠ structured model errors — null result. SF uses actual model errors instead.
- E015 (true SS): Designed but never implemented. Superseded by this.

## Launch Command

```bash
modal run --detach scripts/modal_train.py \
    --config experiments/e018a-sf-minimal.yaml \
    --encoded-file /encoded-e012-fd-top5.pt \
    --run-name e018a-sf-minimal
```
