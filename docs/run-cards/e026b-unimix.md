---
id: e026b
created: 2026-03-21
status: proposed
type: training-regime
base_build: b002
built_on: []
source_paper: "2301.04104"
rollout_coherence: null
prior_best_rc: 5.146
---

# Run Card: e026b-unimix

## Goal

Test whether 1% uniform distribution mixing (unimix) on categorical predictions improves rollout coherence. DreamerV3 uses this technique to prevent categorical heads from becoming overconfident and collapsing to near-deterministic predictions, which causes compounding errors during autoregressive rollout.

The action head has 400 classes with highly skewed distribution — most frames are "Wait" or similar idle states. Unimix ensures the model always assigns at least 0.01/num_classes probability to every class, maintaining exploration during training.

## What Changes

One change: `model.unimix_ratio: 0.01`. Applied to all 12 categorical head outputs (p0/p1 action, jumps, l_cancel, hurtbox, ground, last_attack) during training only. At inference time, raw logits pass through unchanged.

Implementation: softmax -> mix 1% uniform -> log -> CE loss. This is equivalent to label smoothing but applied in probability space rather than target space.

All other hyperparameters identical to b002.

## Target Metrics

- **Keep:** RC < 5.146 (improvement over b002)
- **Kill:** RC > 5.3 or action_acc drops > 2pp (unimix too aggressive)

## Model

Identical to b002: d_model=768, d_state=64, n_layers=4, headdim=64, 15,648,882 params.

## Training

- lr: 5e-4, weight_decay: 1e-5, batch_size: 512, 1 epoch
- warmup_pct: 0.05, AMP: true
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15
- **model.unimix_ratio: 0.01**

## Cost

~$5 (A100 40GB, ~3hr). No additional compute cost — unimix is a few softmax/log ops per forward pass.

## Confounds

- 1% is the DreamerV3 default. If null, doesn't rule out smaller (0.1%) or larger (5%) ratios.
- Interaction with Self-Forcing: SF already provides implicit regularization against overconfident categoricals. Unimix may be redundant, or the two may compound.
- Unimix affects training loss magnitude (slightly higher CE loss due to entropy floor). Comparison via RC, not raw loss.
- Only applied during training — eval/rollout uses raw logits, so the regularization effect must transfer.

## Results

*Pending — status: proposed*
