# JEPA World Model Direction

An alternative architecture exploring LeWorldModel (JEPA-based latent prediction) as a replacement for the Mamba2 backbone.

## What is JEPA?

Joint Embedding Predictive Architecture — a self-supervised framework from Yann LeCun's group. Instead of predicting raw observations (pixels, game state fields), JEPA predicts in a learned **latent embedding space**. The model chooses its own representation.

**Key difference from our Mamba2 approach:** Mamba2 predicts each game state field separately (16 prediction heads, per-field losses). JEPA compresses each frame into a single 192-dim vector and predicts the next vector. No per-field decomposition during training.

## Architecture

```
Game State (288D per frame)
    → Encoder (categorical embeddings + MLP + BatchNorm)
    → Latent Embedding (192D)
    → Predictor (6-layer Transformer with AdaLN-zero)
        ↑ Controller inputs condition via Adaptive Layer Normalization
    → Predicted Next Embedding (192D)

Loss = MSE(predicted, target) + 0.1 × SIGReg(all embeddings)
```

### Components

| Component | What It Does | Reference |
|-----------|-------------|-----------|
| **Encoder** | Game state → 192-dim latent. MLP with categorical embeddings + BatchNorm projector. | Replaces LeWM's ViT (accepted divergence — no pixels) |
| **Predictor** | Causal Transformer, AdaLN-zero conditioning from controller inputs. 6 layers, 16 heads. | Direct port of LeWM's ARPredictor |
| **SIGReg** | Anti-collapse regularizer. Forces embeddings toward isotropic Gaussian via Epps-Pulley test on random projections. | Direct port of LeWM/LeJEPA |
| **AdaLN-zero** | Controller inputs modulate LayerNorm parameters (shift, scale, gate) at each Transformer layer. Zero-initialized gates so conditioning ramps up gradually. | Direct port of LeWM (DiT pattern) |
| **Decoder** | Latent → game state fields. Trained separately on frozen encoder. Not part of the world model itself. | Necessary addition for our eval pipeline |

### Parameter Count (~13.2M)

| Component | Params | Notes |
|-----------|--------|-------|
| Encoder (embeddings + trunk + projector) | ~1.5M | MLP, much lighter than LeWM's 5M ViT |
| Predictor (blocks + action encoder + projector) | ~11.7M | Matches LeWM — 6 Transformer blocks with AdaLN |
| **Total** | **~13.2M** | vs LeWM's ~15M, vs Mamba2's ~15.8M |

## Training

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimizer | AdamW | LeWM default |
| Learning rate | 5e-5 | LeWM default |
| Weight decay | 1e-3 | LeWM default |
| Gradient clip | 1.0 | LeWM default |
| Batch size | 128 | LeWM default |
| Epochs | 100 | LeWM default |
| History size | 3 frames (50ms) | LeWM default (no frameskip) |
| SIGReg λ | 0.1 | LeWM default |
| SIGReg projections | 1024 | LeWM default |
| Precision | AMP (float16) | From b002 |
| Warmup | 5% linear | From b002 |

**Loss has exactly two terms:** MSE in latent space + SIGReg. One tunable hyperparameter (λ). Compare to Mamba2's 16-head weighted loss with 10+ weight coefficients.

## Why Explore This?

1. **Dynamics decoupled from representation.** Mamba2 simultaneously learns representation and dynamics. JEPA separates them: encoder learns representation, predictor learns dynamics.
2. **Simpler loss landscape.** Two loss terms vs. 16-head weighted sum. Less hyperparameter tuning.
3. **Cross-field joint structure.** A single latent embedding captures dependencies between all game state fields natively.
4. **LeCun's world model vision.** JEPA is the architecture proposed for learning world models in "A Path Towards Autonomous Machine Intelligence" (2022).

## What's Different From LeWM

| Aspect | LeWM | Our Adaptation |
|--------|------|---------------|
| Input | Raw pixels (224×224 images) | Structured game state (288D) |
| Encoder | ViT-tiny (5M params, 12 layers) | MLP (1.5M params, 2 layers) |
| Environment | Single-agent (PushT, Reacher) | Two-player fighting game |
| Action space | Continuous robot control | Mixed continuous/binary controller |
| Frameskip | 5 (history covers 15 raw frames) | None (history covers 3 raw frames) |
| Evaluation | Planning success rate (CEM in latent space) | Rollout coherence (decoded game state MAE) |

Everything else matches: predictor architecture, SIGReg, AdaLN, training regime.

## Future Levers

Design decisions locked to LeWM defaults for the baseline experiment (e030a). Each is a potential experiment axis:

| Lever | Baseline | Range | Rationale |
|-------|----------|-------|-----------|
| History size | 3 (50ms) | 3–30 | Melee may need more temporal context than robot control |
| Embed dim | 192 | 128–512 | Capacity vs regularization trade-off |
| Encoder depth | 2 layers | 2–6 | MLP may be too shallow; LeWM's ViT is 12 layers |
| Predictor layers | 6 | 2–8 | Standard depth sweep |
| SIGReg λ | 0.1 | 0.01–1.0 | Bisection search (O(log n) per LeWM) |
| Batch size | 128 | 128–1024 | Short sequences use less memory — can we scale up? |
| Frameskip | None | 2–5 | Longer effective context via frame subsampling |
| Action tokenization | Continuous | Discrete vocab | Define Melee action primitives (robot arm analogy) |
| Multi-step prediction | 1 | 1–3 | Predict multiple steps ahead during training |
| Decoder | Separate (frozen encoder) | Joint (auxiliary loss) | Trade purity for guaranteed decodability |

## Key Research Sources

| Source | What | Location |
|--------|------|----------|
| LeWM paper | Core architecture and results | `research/sources/2603.19312-summary.md` |
| LeWM repo analysis | Code-level implementation details | `research/sources/lewm-repo-analysis.md` |
| EB-JEPA paper | Energy-based JEPA framework | `research/sources/2602.03604-summary.md` |
| EB-JEPA repo analysis | Action-conditioned world model example | `research/sources/eb-jepa-repo-analysis.md` |
| V-JEPA / I-JEPA foundations | JEPA theory and anti-collapse evolution | `research/sources/vjepa-ijepa-summary.md` |
| Adaptation notes | Mapping JEPA to Melee — design options | `research/sources/jepa-adaptation-notes.md` |
| Implementation plan | Full pseudocode and file structure | `docs/jepa-implementation-plan.md` |

## File Structure

```
models/jepa/
├── CLAUDE.md       # Implementation rules — follow LeWM, flag divergences
├── __init__.py     # Exports
├── encoder.py      # GameStateEncoder: frame → 192-dim latent
├── predictor.py    # ARPredictor: Transformer + AdaLN-zero
├── sigreg.py       # SIGReg regularizer (ported from LeWM)
└── model.py        # JEPAWorldModel: wraps pieces, training forward, rollout

data/
└── jepa_dataset.py # JEPAFrameDataset: returns frame subsequences

training/
└── jepa_trainer.py # JEPA training loop (MSE + SIGReg)

scripts/
└── train_jepa.py   # Training entry point

experiments/
└── e030a-jepa-baseline.yaml  # First experiment config
```
