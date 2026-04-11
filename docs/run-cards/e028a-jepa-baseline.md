---
id: e028a
created: 2026-04-04
status: proposed
type: architectural
base_build: null
built_on: []
source_paper: "2603.19312"
rollout_coherence: null
prior_best_rc: 4.965
---

# Run Card: e028a-jepa-baseline

## Goal

First test of the JEPA paradigm (LeWorldModel) on structured Melee game state. This is a new experiment lineage — different architecture, different loss, different training regime from the Mamba2 line. Not an incremental experiment on b002.

The core question: does latent-space prediction (JEPA) work for structured game data, or does it only help when abstracting away pixel noise?

## What's New

**Architecture:** JEPA world model replacing Mamba2. MLP encoder → 192-dim latent → 6-layer Transformer predictor with AdaLN-zero action conditioning. Predicts in latent embedding space, not per-field game state.

**Loss:** MSE in latent space + SIGReg (isotropic Gaussian regularizer). Two terms, one hyperparameter (λ=0.1). Replaces the 16-head weighted loss.

**Training:** Matches LeWM defaults — AdamW 5e-5, 100 epochs, batch 128, history_size=3.

## Context

| Direction | Architecture | Params | Loss | Best RC |
|-----------|-------------|--------|------|---------|
| Mamba2 (b002) | SSM backbone, 16 pred heads | 15.8M | Weighted per-field | 4.965 |
| **JEPA (e028a)** | **MLP encoder + Transformer predictor** | **~13M** | **MSE + SIGReg** | **TBD** |

## What Changes

Everything. This is a different architecture with a different training objective.

| Component | Mamba2 (b002) | JEPA (e028a) |
|-----------|--------------|-------------|
| Backbone | Mamba-2 SSM (4 layers) | Transformer (6 layers) |
| Prediction | Per-field heads (16) | Latent space (192-dim) |
| Loss | Weighted sum (10+ terms) | MSE + SIGReg (2 terms) |
| Action conditioning | Additive after backbone | AdaLN modulation per layer |
| Context | 30 frames (500ms) | 3 frames (50ms) |
| Anti-collapse | N/A (supervised) | SIGReg regularizer |
| Training epochs | 1-2 | 100 |

## Data

Same as b002: FD (stage=32), top-5 chars, v3 encoding with state_flags, hitstun, ctrl_threshold, state_age_as_embed, multi_position.

## Model

- Encoder: MLP (288→512→512→2048→192 + BatchNorm)
- Predictor: 6-layer Transformer, 16 heads, dim_head=64, AdaLN-zero
- SIGReg: λ=0.1, 1024 projections, 17 knots
- Total: ~13M params

## Training

- AdamW, lr=5e-5, wd=1e-3
- 100 epochs, batch 128
- 5% linear warmup + cosine decay
- Gradient clip 1.0
- AMP (float16)

## Success Criteria

This is exploratory — we're testing a paradigm, not tuning a hyperparameter.

- **Promising (RC < 6.0):** Within 20% of Mamba2 best. Worth continuing the lineage.
- **Competitive (RC < 5.0):** Matches Mamba2. Serious contender.
- **Not viable (RC > 8.0):** Paradigm doesn't fit structured data. Learn what we can, close the line.

Note: RC comparison requires a decoder (not in v1). Initial evaluation will be qualitative (does SIGReg converge? does pred_loss decrease? do embeddings show structure?).

## Key Risks

1. **50ms context.** history_size=3 at 60fps = 50ms. Mamba2 uses 500ms. If this fails, increasing history_size is the first lever.
2. **Encoder capacity.** 2-layer MLP vs LeWM's 12-layer ViT. May bottleneck representation quality.
3. **No decoder yet.** Can't compute RC until decoder is trained. First run is loss-curve-only evaluation.

## References

- LeWM paper: `research/sources/2603.19312-summary.md`
- Implementation plan: `docs/jepa-implementation-plan.md`
- Architecture docs: `docs/jepa-direction.md`
- Code: `models/jepa/`, `data/jepa_dataset.py`, `training/jepa_trainer.py`, `scripts/train_jepa.py`
