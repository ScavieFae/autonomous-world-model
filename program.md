# Program: Autonomous World Model

The research direction document. This is the human's lever — everything else is agent-driven. Autoresearch agents read this before starting. Mattie writes this. Agents execute against it.

## What We're Building

A learned world model that runs onchain as an autonomous world. Trained on Super Smash Bros. Melee replay data, quantized to INT8, deployed on Solana. The model IS the world — its learned physics become ground truth.

## Current Best

| Checkpoint | Experiment | change_acc | pos_mae | rollout_coherence | Notes |
|-----------|-----------|-----------|---------|-------------------|-------|
| `e018a-sf-minimal/best.pt` | E018a | 61.6% | 0.825 | **6.26** | b001 + Self-Forcing (20% SF, N=3), 1.9K data |
| `e019-baseline-1k/best.pt` | E019 | 78.7% | 0.756 | 6.77 | b001, 1.9K data, full loss suite (10 heads) |
| `e012-clean-fd-top5/best.pt` | E012 | 91.1% | 0.706 | 6.84 | Pre-migration, partial losses (4 heads) |

E018a is the new best. Self-Forcing improved rollout coherence by 7.5% (6.77→6.26) despite -17pp change_acc and +9% pos_mae regression in TF metrics. The trend from E012→E019→E018a is consistent: each step trades TF accuracy for AR quality. SF loss (0.38) was 2.3× TF loss (0.16), confirming the model faces harder predictions from its own state. E018b (N=5 unroll) is in flight testing whether longer horizon helps further.

Val metrics plateau after 1 epoch on 1.9K data (epoch 2 showed identical val performance). 1 epoch is sufficient for Scout experiments.

## The Eval

**Primary metric: rollout coherence** — mean position MAE over K=20 autoregressive horizons, averaged across N=300 starting frames from the val set. Single number, lower is better. This replaces eyeballing demos.

**Secondary metrics (teacher-forced):** val_change_acc, val_pos_mae, val_loss. These are sanity checks, not targets. A model can have great TF metrics and terrible AR quality (E016 proved this — 38% better val_loss, worse demos).

**The eval script** (`scripts/eval_rollout.py`) is built and integrated into the Trainer. Runs automatically after each epoch (~30-60s on A100). Deterministic (seed=42). Results log to wandb as `eval/summary_pos_mae` and per-horizon metrics.

**Known issue:** `vel_mae` reads zero in eval — velocity reconstruction fix is in the code but hasn't been validated on a training run yet. Position and action metrics are correct.

## What We Know

### Proven improvements (compound, keep in all experiments)

| Technique | Source | Effect | Status |
|-----------|--------|--------|--------|
| action_change_weight=5.0 | E010b | +9.9pp change_acc | In all configs since E010d |
| ctrl_threshold_features | E010c | +5.9pp change_acc | In all configs since E010d |
| multi_position=true | E008c | 10x training signal | In all configs since E008c |
| FD-only + top-5 characters | E012 | Cleaner physics, less noise | Current data filter |
| Full loss suite (10 heads) | E019 | Better rollout coherence (6.77 vs 6.84) despite -12pp change_acc | Velocity/dynamics supervision improves AR quality |
| Self-Forcing (20% SF, N=3) | E018a | -7.5% rollout coherence (6.77→6.26), -17pp change_acc | Largest single-experiment gain. Trains on own AR errors. |

### Promising but unfinished

| Technique | Source | Finding | Next step |
|-----------|--------|---------|-----------|
| Absolute y prediction | E017a | Trades drift for jitter — structural improvement | Combine with Self-Forcing |
| Cascaded heads | E014 | Fixed damage drift, overfit at 1.9K games | Needs 7.7K+ data |
| True scheduled sampling | E015 | Designed but not implemented in trainer | Superseded by Self-Forcing (e018a) |

### Dead ends (revisitable with new reasoning)

These didn't work in their original context. Agents CAN revisit if they have specific reasoning for why the context is different now (different data scale, different training regime, new interaction hypothesis). The reasoning must be specific, not hand-wavy.

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

E018a confirmed this empirically: Self-Forcing improved RC by 7.5% while regressing TF metrics by 17pp. The model with "worse" teacher-forced accuracy produces substantially better autoregressive rollouts.

## Research Directions

### What we've tested (observations, not conclusions)

| Direction | Experiments | Results | Confidence |
|-----------|------------|---------|------------|
| Self-Forcing (basic) | E018a (20% SF, N=3) | RC 6.26 (-7.5%). KEPT. | HIGH — large effect, clear mechanism |
| SF longer unroll | E018b (N=5) | RC 6.45 (+3.0%). Regressed. | 1/1 regressed — could be N=5 specifically, not all N>3 |
| SF loss weighting | E018d (linear ramp) | RC 6.81 (+8.8%). Regressed. | 1/1 regressed — one weighting scheme tested |
| Context length | E018c (K=30), E019a (K=50) | K=30: RC 6.03 (-3.7%). K=50: RC 5.97 (-1.0%, marginal). | K=30 is a clear win. K=50 is diminishing returns but not zero. |
| SF ratio (10%) | E020a (10% SF) | RC 6.62 (+9.8%). Regressed. | 1/1 regressed. 10% too little SF signal. 20% remains only proven ratio. |
| Selective BPTT | E021b (detach categoricals after step 0) | RC 6.87 (+13.9%). Regressed. | 1/1 regressed. SF loss 4× higher — destabilized training. |
| Dropout tuning | untested | — | Director rejected 0.1→0.05 proposal. 0 experiments run. |

**Important:** "Regressed in 1 experiment" ≠ "axis is closed." The SF refinement experiments (E018b, E018d) tested specific configurations, not the entire parameter space. Different unroll lengths (N=4), different weighting schemes (exponential, step-function), or different SF ratios may behave differently. Agents should maintain uncertainty and can revisit with specific reasoning.

### Engineering directions (require code changes)

#### 1. Full backpropagation through Self-Forcing steps
Currently gradients are truncated (detached) between the 3 Self-Forcing steps. This limits what the model can learn about multi-step error recovery. Full BPTT would let gradients flow through `reconstruct_frame()`, potentially enabling N=5+ and better error correction.

**Challenge:** `reconstruct_frame()` uses argmax for categorical predictions (action state, jumps), which isn't differentiable. Options: Gumbel-softmax relaxation, straight-through estimator, or gradient flow only through continuous heads.

**Expected cost:** ~$10-15 per experiment (need gradient checkpointing for memory).

#### 2. Data scaling to 7.7K games
The 7.7K dataset exists on the Modal volume but loading takes 9hr/epoch with `num_workers=0` (fork OOM with `num_workers=4`, `share_memory_()` fails on Modal). E016 showed dramatic TF metric improvement at 7.7K — unknown if AR quality scales similarly.

**Blocker:** Data loader infrastructure. Fix before running experiments.

#### 3. Cascaded heads + Self-Forcing
E014 showed cascaded heads fixed damage drift at 1.9K. Code not ported to this repo. Requires implementation work.

#### 4. Muon optimizer
The optimizer space is unexplored — vanilla AdamW since day one. Muon (Newton-Schulz orthogonalized SGD for weight matrices, AdamW for embeddings/scalars) showed strong results in Karpathy's autoresearch. Open question whether it helps for Mamba2 (different weight matrix structure than transformer Q/K/V). Low-risk: if it doesn't help, revert.

### Config directions (no code changes)

#### 5. SF parameter space (revisitable)
- **SF ratio:** 10% regressed (E020a, RC 6.62). 30%, 50% untested. 20% remains the only proven ratio.
- **Unroll N=4:** Between the working N=3 and failing N=5. Might find a sweet spot.
- **Alternative loss weighting:** Exponential, step-function, inverse-loss weighting. Linear ramp failed but other schemes untested.

#### 6. Batch size sweep
Karpathy's autoresearch found halving batch size was his single largest improvement. We've locked bs=512 without a systematic sweep. Test 256 (half) and 1024 (double), coupled with LR scaling (linear scaling rule: LR ∝ batch_size). Especially relevant given short training (1 epoch on 1.9K data).

#### 7. Architecture exploration (ref: issue #6)

The model is ~4.3M params (d_model=384, n_layers=4, d_state=64, headdim=64). This was inherited from the nojohns migration without a formal ablation. We don't know if the model is the right size or shape.

**Phase 1 — Width/Depth Grid (config-only, autonomous)**

Run against E018c baseline (SF + K=30, RC 6.03). Change ONE dimension at a time. Double or half:

| Variable | Half | Current | Double | What it teaches |
|---|---|---|---|---|
| d_model (width) | 192 (~1.1M) | 384 (~4.3M) | 768 (~17M) | Is the trunk too narrow? |
| n_layers (depth) | 2 (~2.2M) | 4 (~4.3M) | 8 (~8.5M) | Does it need more processing stages? |
| d_state (SSM memory) | 32 | 64 | 128 | Is the recurrent state forgetting game state? |
| headdim (head granularity) | 32 (→24 heads) | 64 (→12 heads) | 128 (→6 heads) | Many independent streams vs few rich ones? |
| Combined scale-up | — | — | d_model=768, n_layers=8 (~34M) | Is the model just too small? |

Also test dropout=0.0 and dropout=0.3 (current 0.1).

**Decision rules for Phase 2 (autonomous):**
- If width (d_model) improves RC more than depth (n_layers): try d_model=512 at n_layers=4.
- If depth improves more than width: try n_layers=6 at d_model=384.
- If the 8× scale-up doesn't beat the best single-axis change: model isn't capacity-limited — stop scaling, focus on training/encoding.
- If d_state=128 improves RC by >3%: extend to d_state=256.
- If any single experiment improves RC by >5%: axis is high-value — run intermediate values to find the efficient frontier.
- If ALL Tier 1 experiments are within ±2% of baseline: architecture isn't the bottleneck — deprioritize and move to training regime or data.

**Phase 3 — Structural changes (requires Mattie approval)**

These require code changes to `models/mamba2.py`. Don't attempt without sign-off:
- Player-specific intermediate layers before heads
- Cascaded heads (action → continuous) — revisit E014 with SF
- Wider heads (2-layer MLP instead of single linear)
- Hybrid SSM + attention (replace one Mamba2 layer with causal self-attention)

Run on whatever model size wins from Phases 1-2.

#### 8. Broader research
Agents should actively search for techniques from the world model, video prediction, and reinforcement learning literature that might apply. The source papers list is small (4 papers). There may be relevant work on:
- Differentiable world models and planning through learned dynamics
- Curriculum learning for autoregressive models
- Contrastive or auxiliary losses for temporal consistency
- Knowledge distillation from teacher-forced to autoregressive models

## Hard Constraints

- **Modal A100 40GB for training.** $2.10/hr. H100 only with Mattie approval.
- **Memory budget.** M3 Max has 128GB unified memory. H100 has 80GB HBM3. Batch sizes and context lengths must fit.
- **INT8 quantization compatibility.** All architectures must remain quantizable. No operations that break INT8 determinism (no float-dependent branching, no dynamic shapes).
- **Deterministic eval.** Same checkpoint + same starting frames = same rollout coherence score. No randomness in eval.

## Taste

- **AR rollout quality over TF metrics.** If an experiment improves rollout coherence but regresses val_loss, that's a win.
- **Simpler is better if metrics are close.** A one-line training change that gets 90% of the gain beats a complex architecture change.
- **State observations with hit rates, not editorials.** "WD 0.001 improved in 3/3 experiments" not "weight decay is important."
- **One idea per experiment.** Don't combine untested changes. If Self-Forcing + longer context both help, we want to know which helped how much.
- **Kill fast.** If an experiment isn't showing signal by epoch 1, kill it. Don't throw good compute after bad.
- **Double or half, not +1.** When exploring a parameter, make big moves to see directional shift. We're triangulating, not hill-climbing. Going from 4 layers to 8 is better than 4 to 5.

## The Eval Protocol

For autoresearch agents:

1. Start from `experiments/e018c-context-k30.yaml` (current best: SF + K=30). Change ONE thing.
2. Train on 1.9K data (`encoded-e012-fd-top5.pt`), 1 epoch, bs=512. ~2.5hr on A100, ~$5.
3. Rollout coherence eval runs automatically after training (integrated in Trainer).
4. Compare to baseline: **rollout_coherence = 6.03** (E018c).
5. **Keep** if rollout coherence improves. Write run card with `base_build`, `built_on` citations and numbers.
6. **Discard** if rollout coherence regresses or stays flat. Write run card documenting the null result.
7. Either way, the run card is the permanent record. Null results are data.
8. Val metrics plateau after 1 epoch on 1.9K data — don't spend $3 on epoch 2 unless Scout shows signal.

### Epistemic standards for agents

- **State hit rates, not conclusions.** "0/1 experiments improved" not "axis is closed."
- **Maintain uncertainty.** One failed experiment on an axis doesn't rule out the axis. Two failures with different configurations increases confidence. Three is strong evidence.
- **Bring in outside ideas.** Don't just recombine what's in program.md. Search for papers, techniques, and approaches we haven't considered.
- **Challenge assumptions.** If program.md says something is a dead end and you have specific reasoning for why it might work in a new context, propose it. The Director will evaluate.

## Source Papers

Summaries in `research/sources/`:

| Paper | arXiv | Key technique | Relevance |
|-------|-------|--------------|-----------|
| Long-Context SSM Video World Models | 2505.20171 | SSM + local attention hybrid | High — architecture |
| The Matrix | 2412.03568 | Swin-DPM + SCM for real-time generation | Moderate — patterns |
| Matrix-Game 2.0 | 2508.13009 | Self-Forcing distillation | **High** — training regime |
| EgoScale | 2602.16710 | Scaling laws for cross-embodiment transfer | Conceptual — scaling |
