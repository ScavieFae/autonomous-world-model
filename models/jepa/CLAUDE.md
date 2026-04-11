# JEPA World Model — models/jepa/

This directory implements a JEPA-based world model for Melee, exploring LeWorldModel (arXiv 2603.19312) as an alternative to the Mamba2 backbone.

## Guiding Principle

**Hew closely to LeWorldModel and Facebook's JEPA implementations.** This is an adaptation of proven architectures to a new domain (structured game state instead of pixels), not an original architecture. When in doubt, do what LeWM does.

## Implementation Rules

1. **Follow the reference implementations.** LeWM (`lucas-maes/le-wm`) is the primary reference. EB-JEPA (`facebookresearch/eb_jepa`) is the secondary reference for JEPA training patterns. Copy their patterns, naming conventions, and architectural choices unless there is a domain-specific reason to diverge.

2. **Flag divergences, don't ship them.** When you identify a place where the Melee domain forces a departure from the reference implementations — or where you see an opportunity to do something different that might work better — **stop and flag it**. Write a comment like `# DIVERGENCE: [reason] �� discuss before changing` and surface it to Mattie. We will decide together whether to pursue it.

3. **The encoder is the main divergence point.** LeWM uses a ViT on pixels. We use structured game state (~278D per frame with b002 encoding flags). The encoder necessarily differs — this is an accepted, known divergence. Everything downstream of the encoder (predictor architecture, SIGReg, AdaLN conditioning, rollout logic) should match LeWM as closely as possible.

4. **Don't import patterns from the Mamba2 side without flagging.** The existing AWM codebase has accumulated its own patterns (self-forcing, per-head losses, unimix, curriculum). Some of these might compose well with JEPA. But each one is a divergence from the reference. Flag before mixing.

5. **Use existing tools and implementations — don't hand-roll.** When a component already exists in the reference repos (LeWM, EB-JEPA) or in well-maintained libraries, port or import it rather than reimplementing from scratch. SIGReg, AdaLN, attention — these all have reference implementations. Custom reimplementations introduce subtle bugs and drift from the proven recipe. If the reference implementation doesn't fit our use case, that's a divergence — flag it.

## Reference Material

All saved in `research/sources/`:

| File | What |
|------|------|
| `2603.19312-summary.md` | LeWM paper summary — architecture, SIGReg, training, results |
| `2602.03604-summary.md` | EB-JEPA paper summary — energy-based JEPA, VC regularization |
| `vjepa-ijepa-summary.md` | Foundation JEPA papers — I-JEPA, V-JEPA, LeCun's vision |
| `lewm-repo-analysis.md` | LeWM repo code-level analysis — configs, loss implementation, components |
| `eb-jepa-repo-analysis.md` | EB-JEPA repo code-level analysis — planning, unrolling, losses |
| `jepa-adaptation-notes.md` | Adaptation notes — how JEPA maps to Melee, encoder/predictor/decoder options |

Read these before making architectural decisions. They exist so we don't re-derive things from scratch each session.

## Architecture (Target)

```
Game State (116D) → Encoder (MLP) → Latent (192D) → Predictor (Transformer + AdaLN) → Next Latent
                                                        ↑ Controller inputs condition via AdaLN
```

- **Encoder:** MLP. The one component that must differ from LeWM (no pixels). Accepted divergence.
- **Predictor:** Transformer with AdaLN-zero action conditioning. Match LeWM: 6 layers, 16 heads, causal masking.
- **SIGReg:** Isotropic Gaussian regularizer. Match LeWM: λ=0.1, M=1024 projections.
- **Decoder:** Lightweight MLP → game state fields. Trained separately on frozen encoder (LeWM doesn't decode during training). This is a necessary addition for our eval pipeline — LeWM plans in latent space and never decodes.
- **Rollout:** Autoregressive in latent space, then decode for evaluation. Match LeWM's `history_size=3` truncation pattern.

## Loss

```
L = L_pred + λ * SIGReg(Z)
```

That's it. MSE in latent space + SIGReg. No per-field heads, no weighted loss sum. Match LeWM exactly.

## What This Directory Does NOT Own

- Data loading (`data/`) — shared with Mamba2, driven by `EncodingConfig`
- Replay parsing (`data/parse.py`) — untouched
- Existing model code (`models/mamba2.py`, `models/mlp.py`) — untouched
- Existing training loop (`training/trainer.py`) — untouched
- Experiment evaluation (rollout coherence) — shared infrastructure, JEPA decodes into the same format

## Experiment Tracking

JEPA experiments use the same run card system (`docs/run-cards/`). Series starts at **e030** (e028a was taken by Mamba2 full-stack, e029 by the 7.7K scaling bet). The JEPA lineage is e030a → e030b → e030c+.
