# V-JEPA and I-JEPA — Foundation JEPA Papers

**V-JEPA:** Bardes et al. (2024), Meta FAIR
**I-JEPA:** Assran et al. (2023), Meta FAIR
**V-JEPA Code:** https://github.com/facebookresearch/jepa
**EB-JEPA Code:** https://github.com/facebookresearch/eb_jepa
**LeCun's Position Paper:** "A Path Towards Autonomous Machine Intelligence" (2022)

## What is JEPA?

Joint Embedding Predictive Architecture. Self-supervised learning that predicts the *representation* of missing/future content from the representation of observed content, entirely in latent space. No pixel reconstruction.

Three key properties that distinguish it from alternatives:
1. **vs Contrastive (SimCLR, CLIP):** No negative examples needed. Contrastive creates invariant representations; JEPA creates predictive representations that capture more structure.
2. **vs Generative (MAE, VideoGPT):** No pixel reconstruction. Doesn't waste capacity on low-level details.
3. **vs BYOL/DINO:** Different content between views (context vs target), not different augmentations of same content. Learns prediction, not invariance.

## Architecture Pattern

- **Context Encoder:** ViT, processes unmasked tokens
- **Target Encoder:** EMA copy of context encoder (V-JEPA) or same encoder + regularization (EB-JEPA, LeWM)
- **Predictor:** Narrower ViT/MLP that takes context representations + mask tokens → predicts target representations
- **Loss:** L1/MSE between predicted and target representations + anti-collapse mechanism

## Anti-Collapse Strategies (Evolution)

| Method | Strategy | Complexity |
|--------|----------|------------|
| I-JEPA, V-JEPA | EMA target encoder | Momentum schedule hyperparameter |
| PLDM | VICReg (variance + covariance + invariance) | 7 loss terms, 6 hyperparameters |
| EB-JEPA | VC regularization (HingeStd + Covariance) | Multiple terms, simpler than VICReg |
| **LeWM** | **SIGReg (isotropic Gaussian test)** | **1 hyperparameter (λ)** |

## LeCun's World Model Vision

From "A Path Towards Autonomous Machine Intelligence":
- World models should predict in representation space, not pixel space
- JEPA is the proposed architecture for the world model module
- Latent prediction abstracts away irrelevant visual detail
- Combined with planning (CEM, MPPI) for action selection

## Relevance to AWM/Melee

The JEPA family represents a specific bet: that learning to predict abstract representations is better than predicting raw observations. For Melee, the question is whether game state (which is already abstract/structured) benefits from further embedding into a latent space, or whether direct prediction of game state fields is more appropriate. The answer may depend on whether latent-space prediction captures cross-field dependencies that per-head prediction misses.
