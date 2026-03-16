---
id: e018d
created: 2026-03-09
status: discarded
type: training-regime
base_build: b001
built_on: [e018a]
source_paper: 2508.13009
rollout_coherence: 6.81
prior_best_rc: 6.26
---

# Run Card: e018d-horizon-weighted-loss

## Goal

Test whether weighting later AR steps more heavily in the SF loss improves rollout coherence. E018b showed truncated BPTT saturates at N=3 — this optimizes the learning signal within the 3-step window instead of extending it.

## What Changes

One change from E018a: add horizon weighting to the SF loss.

```python
# E018a (uniform):
sf_loss = torch.stack(all_losses).mean()

# E018d (linear ramp):
weights = torch.linspace(0.5, 2.0, steps=N)  # [0.5, 1.25, 2.0]
sf_loss = (losses * weights).sum() / weights.sum()
```

Step 1 (nearly TF, easy) gets 0.5× weight. Step 3 (max drift, hardest) gets 2.0× weight.

## Results

**DISCARDED — substantial regression.**

| Metric | E018a (uniform) | E018d (weighted) | Delta |
|--------|-----------------|-----------------|-------|
| **rollout_coherence** | **6.26** | **6.81** | **+8.8% (worse)** |
| change_acc | 61.6% | 61.6% | flat |
| pos_mae | 0.825 | 0.830 | ~flat |
| sf_loss | 0.383 | 0.576 | +50% |
| tf_loss | 0.164 | 0.302 | +84% |
| action_acc | 95.3% | 95.3% | flat |

### Cost

- Runtime: 6884s (~115min) on A100 40GB
- Cost: ~$4.00
- wandb: https://wandb.ai/shinewave/melee-worldmodel/runs/gabdfivr

## Director Evaluation

**Verdict:** DISCARDED

**Confidence:** HIGH — RC 6.81 is worse than even the E019 baseline (6.77). The horizon weighting actively hurt training. SF loss increased 50% (0.38→0.58), and TF loss nearly doubled (0.16→0.30), indicating the ramp destabilized training across both objectives.

**Finding:** Linear ramp [0.5, 1.25, 2.0] over-weighted step 3 where truncated BPTT has the weakest gradient signal, amplifying noise rather than learning signal. The assumption that "later steps are more informative" was wrong — with truncated BPTT, later steps have worse gradients, and weighting them more makes things worse. Combined with e018b (N=5 regression), this confirms: truncated BPTT's limitation is fundamental, not fixable by reweighting within the window.

**Implication:** The SF refinement axis (within truncated BPTT) is explored. E018a's uniform N=3 is the current best configuration. Next directions should be orthogonal: longer context (e018c), data scaling, full BPTT, or SF ratio changes.
