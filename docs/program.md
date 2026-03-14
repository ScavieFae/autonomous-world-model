# Program: Autonomous World Model

The research direction document. This is the human's lever — everything else is agent-driven. Autoresearch agents read this before starting. Mattie writes this. Agents execute against it.

## What We're Building

A learned world model that runs onchain as an autonomous world. Trained on Super Smash Bros. Melee replay data, quantized to INT8, deployed on Solana. The model IS the world — its learned physics become ground truth.

## Current Best

| Checkpoint | Experiment | change_acc | pos_mae | val_loss | AR quality |
|-----------|-----------|-----------|---------|----------|------------|
| `e012-clean-fd-top5/best.pt` | E012 | 91.1% | 0.706 | 0.527 | Drifts by frame ~50, dies |
| `e017a-absolute-y/best.pt` | E017a | 90.7% | 0.979* | 0.507 | Ground-level but jittery |

*pos_mae inflated — absolute y has wider target distribution than delta.

**E012** is the TF metrics king. **E017a** is the AR quality king. Neither is good enough.

**Rollout coherence score**: not yet measured. Building the eval is the #1 priority (e018b).

## The Eval

**Primary metric: rollout coherence** — mean position MAE over K=20 autoregressive horizons, averaged across N=300 starting frames from the val set. Single number, lower is better. This replaces eyeballing demos.

**Secondary metrics (teacher-forced):** val_change_acc, val_pos_mae, val_loss. These are sanity checks, not targets. A model can have great TF metrics and terrible AR quality (E016 proved this — 38% better val_loss, worse demos).

**The eval script** (`scripts/eval_rollout.py`) does not exist yet. See e018b card. Must run in 30-60 seconds.

## What We Know

### Proven improvements (compound, keep in all experiments)

| Technique | Source | Effect | Status |
|-----------|--------|--------|--------|
| action_change_weight=5.0 | E010b | +9.9pp change_acc | In all configs since E010d |
| ctrl_threshold_features | E010c | +5.9pp change_acc | In all configs since E010d |
| multi_position=true | E008c | 10x training signal | In all configs since E008c |
| FD-only + top-5 characters | E012 | Cleaner physics, less noise | Current data filter |

### Promising but unfinished

| Technique | Source | Finding | Next step |
|-----------|--------|---------|-----------|
| Absolute y prediction | E017a | Trades drift for jitter — structural improvement | Combine with Self-Forcing |
| Cascaded heads | E014 | Fixed damage drift, overfit at 1.9K games | Needs 7.7K+ data |
| True scheduled sampling | E015 | Designed but not implemented in trainer | Superseded by Self-Forcing (e018a) |

### Dead ends (don't revisit)

| Technique | Source | Why it failed |
|-----------|--------|---------------|
| Noise-based SS | E012b | Random noise ≠ structured model errors. -2.8pp, null result |
| Character embedding scaling | E011 | Null result — not enough rare-character data |
| Hitbox features | E013 | Null result — model already infers from (action, state_age) |
| Absolute x position | E017c | Oscillation — x has high legitimate variance |
| Absolute velocities | E017d | Head decoupling — 165/200 frames contradicted |
| Physics loss (soft constraint) | E017b | No gradient signal in TF training — violations only happen in AR |
| focal_offset>0 | E008a-E010 | Inflated training metrics ~15pp, unclear inference benefit |

### The core insight

> Every fix that targets teacher-forced behavior doesn't help autoregressive quality. The model understands Melee physics when given perfect context (91-94% change_acc). The problem is entirely what happens when it consumes its own outputs.

This points at **training regime changes** (Self-Forcing, hybrid AR/TF) as the highest-leverage direction. Architecture and target representation have been explored extensively. The next gains come from teaching the model to handle its own errors.

## Research Directions

Ordered by expected impact. An autoresearch agent should try these roughly in order, but can skip ahead if blocked.

### 1. Self-Forcing training (e018a)

**The bet:** Train on the model's own autoregressive outputs during training. The model learns what its errors look like and how to recover.

**Why:** Matrix-Game 2.0 (2508.13009) uses Self-Forcing distillation to hit 25 FPS real-time generation. Our E015/E016/E017 series proved the problem is exposure bias, not architecture or target representation. Self-Forcing directly addresses exposure bias.

**Card:** `docs/run-cards/e018a-self-forcing.md`
**Depends on:** e018b (rollout coherence eval, so we can measure the effect)

### 2. Longer context window (e018c)

**The bet:** K=10 at 60fps is 167ms — less than a Melee reaction window. Longer context gives the model more temporal information to stabilize predictions.

**Why:** The SSM paper (2505.20171) shows SSMs benefit from longer context. Our Mamba2 backbone supports it. K=30 (500ms) would cover a full move sequence.

**Card:** `docs/run-cards/e018c-rolling-context-window.md`

### 3. Horizon-weighted Self-Forcing loss (e018d)

**The bet:** Weight later AR steps more heavily in the Self-Forcing loss. The model prioritizes long-horizon stability.

**Why:** Step 1 of AR is nearly identical to TF. Step 5+ is where drift happens. Equal weighting wastes capacity on easy short-horizon predictions.

**Card:** `docs/run-cards/e018d-horizon-weighted-loss.md`
**Depends on:** e018a (baseline Self-Forcing results)

### 4. Data scaling

**The bet:** 1.9K games may not be enough. E016 used 7.7K games and TF metrics improved dramatically. The question is whether more data also improves AR quality.

**No card yet.** Need rollout coherence eval first to measure the effect. Run E012 config on 7.7K data, compare AR metrics.

### 5. Cascaded heads + Self-Forcing

**The bet:** E014 (cascaded heads) fixed damage drift. Self-Forcing should fix position/velocity drift. Together: structurally coherent outputs AND robustness to own errors.

**No card yet.** Run after Self-Forcing baseline is established.

## Hard Constraints

- **MPS GPU or Modal H100.** Local training on M3 Max (MPS), production training on H100 via Modal. Experiments must work on both.
- **Memory budget.** M3 Max has 128GB unified memory. H100 has 80GB HBM3. Batch sizes and context lengths must fit.
- **INT8 quantization compatibility.** All architectures must remain quantizable. No operations that break INT8 determinism (no float-dependent branching, no dynamic shapes).
- **Deterministic eval.** Same checkpoint + same starting frames = same rollout coherence score. No randomness in eval.

## Taste

- **AR rollout quality over TF metrics.** If an experiment improves rollout coherence but regresses val_loss, that's a win.
- **Simpler is better if metrics are close.** A one-line training change that gets 90% of the gain beats a complex architecture change.
- **State observations with hit rates, not editorials.** "WD 0.001 improved in 3/3 experiments" not "weight decay is important."
- **One idea per experiment.** Don't combine untested changes. If Self-Forcing + longer context both help, we want to know which helped how much.
- **Kill fast.** If an experiment isn't showing signal by epoch 1, kill it. Don't throw good compute after bad.

## The Eval Protocol

For autoresearch agents:

1. Train for the time budget (~15 min on M3 Max, ~32 min on H100 for 2 epochs on 1.9K data)
2. Run rollout coherence eval (~30-60 seconds)
3. Compare to current best rollout coherence score
4. **Keep** if rollout coherence improves. Write run card with `built_on` citations and numbers.
5. **Discard** if rollout coherence regresses or stays flat. Write run card documenting the null result.
6. Either way, the run card is the permanent record. PRs are for discussion, cards are for results.

## Source Papers

Summaries in `research/sources/`:

| Paper | arXiv | Key technique | Relevance |
|-------|-------|--------------|-----------|
| Long-Context SSM Video World Models | 2505.20171 | SSM + local attention hybrid | High — architecture |
| The Matrix | 2412.03568 | Swin-DPM + SCM for real-time generation | Moderate — patterns |
| Matrix-Game 2.0 | 2508.13009 | Self-Forcing distillation | **High** — training regime |
| EgoScale | 2602.16710 | Scaling laws for cross-embodiment transfer | Conceptual — scaling |
