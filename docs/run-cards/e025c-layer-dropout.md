---
id: e025c
created: 2026-03-21
status: proposed
type: training-regime
base_build: b001
built_on: [e023b]
source_paper: null
rollout_coherence: null
prior_best_rc: 5.775
---

# Run Card: e025c-layer-dropout

## Goal

Test whether adding dropout after each Mamba block improves rollout coherence by reducing overfitting. At d_model=768 (15.8M params, ~8200 params/game), the model has a high parameter-to-data ratio. Currently only input dropout (0.1) provides regularization inside the backbone. Layer dropout adds regularization at every residual connection, which is standard for transformers but untested for Mamba2.

## What Changes

One model parameter added: `layer_dropout: 0.1`. Applied after each Mamba block output, before residual addition. AMP enabled (validated safe in e023b-epoch2).

All other hyperparameters identical to E023b.

## Implementation

In `models/mamba2.py`, the forward loop changes from:
```python
x = x + layer["mamba"](layer["norm"](x))
```
to:
```python
x = x + self.layer_dropout(layer["mamba"](layer["norm"](x)))
```

When `layer_dropout=0.0` (default), `nn.Identity()` is used — zero overhead for existing configs.

## Target Metrics

- **Keep:** RC < 5.775 (improvement over E023b)
- **Kill:** RC > 5.85 or significant TF metric regression (pos_mae > 0.85)

## Model

Based on E023b with one addition:
- d_model=768, d_state=64, n_layers=4, headdim=64
- input dropout: 0.1 (unchanged)
- **layer_dropout: 0.1** (new)
- 15,648,882 params (identical — dropout adds no parameters)

## Training

- lr: 0.0005, weight_decay: 1e-5, batch_size: 512, 1 epoch
- AMP: enabled (float16 autocast + GradScaler)
- Self-Forcing: ratio=4 (20%), unroll_length=3
- context_len=30, chunk_size=15

## Cost

~$7-9 Scout tier (A100 40GB). Dropout adds negligible compute.

## Confounds

- 0.1 is a single data point. If null, doesn't rule out different rates (0.05, 0.2).
- Interaction with Self-Forcing: layer dropout during SF unrolling may smooth error propagation, or may add noise that destabilizes SF gradients.
- E023b showed no overfitting signal (val loss stable). If the model isn't overfitting, regularization may hurt by reducing effective capacity.
