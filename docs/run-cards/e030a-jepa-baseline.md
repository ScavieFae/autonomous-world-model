---
id: e030a
created: 2026-04-04
status: proposed
type: architectural
base_build: null
built_on: []
source_paper: "2603.19312"
rollout_coherence: null
prior_best_rc: 4.798
---

# Run Card: e030a-jepa-baseline

> **Lineage note:** First JEPA experiment. Renumbered from the original e028a slot to avoid collision with the Mamba2 `e028a-full-stack` (kept, RC 4.798 — current best). JEPA series is e030a → e030b → e030c.

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
| **JEPA (e030a)** | **MLP encoder + Transformer predictor** | **~13M** | **MSE + SIGReg** | **TBD** |

## What Changes

Everything. This is a different architecture with a different training objective.

| Component | Mamba2 (b002) | JEPA (e030a) |
|-----------|--------------|-------------|
| Backbone | Mamba-2 SSM (4 layers) | Transformer (6 layers) |
| Prediction | Per-field heads (16) | Latent space (192-dim) |
| Loss | Weighted sum (10+ terms) | MSE + SIGReg (2 terms) |
| Action conditioning | Additive after backbone | AdaLN modulation per layer |
| Context | 30 frames (500ms) | 3 frames (50ms) |
| Anti-collapse | N/A (supervised) | SIGReg regularizer |
| Training epochs | 1-2 | 100 |

## Data

**Dataset:** `encoded-e012-fd-top5.pt` — the ~2K pre-encoded file (1,988 FD top-5 games, ~11 GB, loads into RAM). This is the distinct pre-encoded 2K dataset, **not** `max_games`-capping of the 7.7K file. Same distribution as 7.7K (FD / top-5), ~4× fewer games.

**Why 2K not 7.7K:**
- **Epistemic purity.** Two open questions right now — (1) does JEPA work on structured game state at all? (2) does data scaling help JEPA? — running on 7.7K confounds them. e029a is already answering (2) for the Mamba2 line; let e030a answer (1) for the JEPA line on directly-comparable data.
- **Apples-to-apples with b002.** b002's best (E025a, RC 5.146) was trained on this exact file. Any JEPA number on the same dataset is directly comparable.
- **Grokking is a same-data-many-passes phenomenon.** If we want to honestly test the cliff hypothesis, 2K × many epochs is canonical. 7.7K × fewer epochs would show different examples each epoch and muddy the phase-transition signal.
- **Cheap and fast.** ~$5–10 on A100 vs ~$40–80 on H100 at the 50-epoch budget. Leaves room for e030b encoding ablation and e030c data scaling as follow-ups.
- **Infra.** 2K fits in RAM easily; no mmap story to debug. e029a had to disable mmap to run on H100 — we sidestep it.

**Encoding flags (match b002 contract):** FD (stage=32), top-5 chars [1,2,7,18,22], v3 encoding with state_flags, hitstun, ctrl_threshold, state_age_as_embed, multi_position. Note: this inheritance is itself a **risk** — see Key Risks #3 and the e030b follow-up.

## Lineage plan

- **e030a** (this card) — JEPA paradigm test on 2K, LeWM defaults, instrumented for the cliff. Answers "does JEPA work on structured data at our scale?"
- **e030b** — v1-minimal encoding ablation on 2K. Drop `state_flags`, `ctrl_threshold_features`, `multi_position` — test whether the inherited b002 data contract helps or hurts JEPA.
- **e030c** — data scaling on 7.7K with whatever 2K regime proved best. Structured comparison against e029a.
- **e030d+** — LR/WD sweep, history_size lever, two-player embedding structure, multi-step prediction (after unblocking `num_preds==1` assert).

**Pre-registered architectural fix — `e030-identity-fix`**: if e030a's identity diagnostics fire (`swap.ditto_cosine_sim > 0.9` or any per-player probe R² < 0.3), **before** running any other e030 experiment, run `e030-identity-fix` with a per-player shared-weight sub-encoder + cross-attention fusion. Same-weight encoder applied independently to each player, output as two tokens with explicit player-ID positional embedding, 2 layers of self-attention over the two tokens, mean-pool or CLS readout. This is the canonical two-player symmetric architecture from AlphaZero — it processes P0 and P1 with the same weights (right inductive bias for permutation symmetry of character logic) but positional embeddings + attention preserve and model their interaction. Pre-registering the fix means we don't debate the architecture after observing the failure; we have a ready replacement.

## Model

- Encoder: MLP (288→512→512→2048→192 + BatchNorm)
- Predictor: 6-layer Transformer, 16 heads, dim_head=64, AdaLN-zero
- SIGReg: λ=0.1, 1024 projections, 17 knots
- Total: ~13M params

## Training

- AdamW, lr=5e-5, wd=1e-3 (**probable levers** — LeWM defaults tuned for ViT-tiny on pixels, not our ~1.5M-param MLP encoder; b002 history converged on 5e-4 / 1e-5)
- Up to 50 epochs, batch 128 (**instrument, don't cap** — see below)
- 5% linear warmup + cosine decay
- Gradient clip 1.0
- AMP (float16)
- **GPU: A100** — 2K dataset fits in RAM, H100 is unnecessary overhead at this scale. Target budget ~$10.

## Launch command

```bash
modal run --detach scripts/modal_train_jepa.py \
    --config experiments/e030a-jepa-baseline.yaml \
    --encoded-file /encoded-e012-fd-top5.pt \
    --gpu A100 \
    --run-name e030a-jepa-baseline
```

### Epoch policy: instrument, don't cap artificially

b002 precedent was "1 epoch sufficient, epoch 2 flat" for supervised per-field prediction. JEPA is representation learning — phase transitions / grokking-style cliffs are the canonical failure mode for representation objectives, and the Mamba2 precedent doesn't necessarily transfer.

We're explicitly not assuming either "no cliff" or "cliff exists." We're setting up to observe which one is true on the first run. Evals are cheap enough (see below) that running to ~50 epochs with per-epoch logging is affordable, and that lets us actually see the curve instead of guessing at it. If there's a plateau by epoch 10, future runs cap there. If there's a cliff at epoch 30, the probe sees it coming.

## Evaluation

**Primary signals (per epoch):**
- `pred_loss`, `sigreg_loss` — training dynamics
- **RC@5, RC@10** — shortened from K=20. Fighting game state at 60fps is chaos-dominated past ~150ms; K=20 is below the noise floor for architecture discrimination. K=5 (83ms) is a pure local-dynamics test; K=10 (167ms) is near-horizon coherence. Both numbers logged, neither alone is the north star.
- **Identity diagnostics suite** — see below. Ships in `training/jepa_diagnostics.py`, called per-epoch by `JEPATrainer`.
- **Temporal straightness** — cosine similarity of consecutive latent velocity vectors. LeWM calls this out as an emergent diagnostic; free to compute; included in the diagnostic suite.

**Why shortened RC:** K=20 is hard to discriminate — recent Mamba2 kept experiments (E027c 4.939 → E028a 4.798) are at ~2.9% deltas at K=20 where the trajectory is already chaos-dominated. RC@5 is where architecture differences actually live. Shorter AR unroll also unlocks affordable per-epoch eval, which unlocks the instrument-don't-cap policy above. The broader project-wide RC shortening is a follow-up.

### Identity diagnostics (swap test, per-player probes, relational probes)

See `docs/jepa-data-flow.md` for the full trace of how P0 and P1 reach the latent and why this failure mode is structurally more likely under JEPA's loss than Mamba2's. In short: the encoder concatenates both players' features into a dedicated-slot input, but the trunk MLP is free to learn swap-symmetric features, and unlike Mamba2 the JEPA loss does not explicitly penalize player-identity collapse. We need to measure whether it actually stays identity-aware or silently collapses. All three diagnostics run on a fixed held-out batch, per epoch, from `training/jepa_diagnostics.py`:

| Diagnostic | Detects | Healthy | Collapsed |
|------------|---------|---------|-----------|
| `swap_test.mean_cosine_sim` | Encoder trunk learns swap-symmetric features | < ~0.5 | > ~0.9 |
| `swap_test.ditto_cosine_sim` | Same, on ditto matchups where character asymmetry is zero (sharpest signal) | < ~0.5 | > ~0.9 |
| `per_player_probes[p0_x, p1_x]` R² | Projector loses one player's info while keeping the other | both > 0.8 | one < 0.3 or large asymmetry |
| `per_player_probes[p0_percent, p1_percent]` R² | Same, for percent | both > 0.8 | — |
| `relational_probes[rel_x, rel_y]` R² | Latent encodes cross-player binding, not just stacked per-player slots | > 0.8 | < 0.3 |

**Reporting policy**: swap similarity (mean + ditto) and per-player probe R² are **required reported numbers** in every JEPA run card from e030a onward, alongside RC. They are **not** a kept/discarded gate yet — we need 5–10 experiments' worth of observed values before we can set a principled threshold. Until then: report, track, and flag any run where `swap.ditto_cosine_sim > 0.9` or any per-player probe has R² < 0.3 in the run card's closeout discussion.

## Success Criteria

This is exploratory — we're testing a paradigm, not tuning a hyperparameter. Go/no-go is driven by the linear probe and RC@5, not `pred_loss`.

- **Promising:** RC@5 within 30% of Mamba2 best; linear probe recovers position to <10% error. Worth continuing the lineage, probably with e030b (v1-minimal encoding ablation) and hyperparameter sweep.
- **Competitive:** RC@5 matches Mamba2. Serious contender. Push to full decoder + multi-step prediction.
- **Not viable:** Linear probe can't recover basic physics (position, percent) after the curve clearly plateaus. Paradigm doesn't fit structured data at our scale. Close the line.
- **Too early to tell:** Loss curves still descending at epoch 50. Extend if budget allows; cliff theory says this is a real possibility.

## Key Risks

1. **Player identity collapse (structural weakness vs Mamba2).** The encoder concatenates P0 and P1 into dedicated slots, so the exact CLIP bag-of-words pathology doesn't apply — but the trunk MLP is free to *learn* swap-symmetric features. Unlike Mamba2, whose per-player prediction heads compute separate MSEs for P0 and P1 at every gradient step (forcing distinct representation), JEPA's loss does not explicitly penalize player-identity collapse. SIGReg regularizes the latent distribution and MSE measures predictor self-consistency; neither cares who is who. `pred_loss` only indirectly requires identity preservation. Dittos (same character, both ports) are where this breaks first — character embeddings give no asymmetry signal and all the identity has to come from positional/kinematic slots. **Directly measured by the identity diagnostics suite above.** Pre-registered fix: `e030-identity-fix` per the Lineage plan.
2. **50ms context.** history_size=3 at 60fps = 50ms. Mamba2 uses 500ms. LeWM's 3 at frameskip=5 is effectively 15 raw frames — we're literal 3, so 5x less effective context. If this fails, increasing history_size is the first lever.
3. **Encoder capacity.** 2-layer MLP vs LeWM's 12-layer ViT. May bottleneck representation quality. No input normalization before the trunk — relative feature magnitudes drive early gradients.
4. **Data contract inherited from b002 without re-justification.** `state_flags`, `ctrl_threshold_features`, `multi_position`, `state_age_as_embed`, stage/char filters all were chosen for Mamba2 per-field heads. JEPA has no per-field heads. Directly tested by follow-up **e030b: v1-minimal encoding ablation**.
5. **LR/WD/bs are LeWM defaults, not tuned for our data.** b002 settled on 10x higher LR and 100x lower WD. Probable levers after first run.
6. **SIGReg only constrains encoder distribution.** Predictor output has no direct anti-collapse protection — watch for mode collapse via constant predictor in the probe.
7. **Two-player dynamics** inherited as "concat both players." Open question #4 in `jepa-adaptation-notes.md`. Addressed by `e030-identity-fix` if the identity diagnostics fire; otherwise still flagged for e030c or later.

## Known blocking fixes (pre-Modal)

- BatchNorm `eval()` guard in `JEPAWorldModel.rollout()` — `@torch.no_grad()` alone is not enough; both projectors end in `nn.BatchNorm1d`.
- `assert cfg.lookahead == 0 and not cfg.press_events` in `JEPAFrameDataset.__init__` — `ctrl_conditioning_dim` formula drifts from `_extract_ctrl` layout when either flag is toggled.
- Encoding-mismatch validation carried into `scripts/modal_train_jepa.py` (copy from `modal_train.py:175-184`).
- Linear probe + temporal straightness logging shipped in the same PR as first Modal run.

## References

- LeWM paper: `research/sources/2603.19312-summary.md`
- Implementation plan: `docs/jepa-implementation-plan.md`
- Architecture docs: `docs/jepa-direction.md`
- Code: `models/jepa/`, `data/jepa_dataset.py`, `training/jepa_trainer.py`, `scripts/train_jepa.py`
