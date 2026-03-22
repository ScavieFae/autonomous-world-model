# Program: Autonomous World Model

The research direction document. This is the human's lever — everything else is agent-driven. Autoresearch agents read this before starting. Mattie writes this. Agents execute against it.

## What We're Building

A learned world model that runs onchain as an autonomous world. Trained on Super Smash Bros. Melee replay data, quantized to INT8, deployed on Solana. The model IS the world — its learned physics become ground truth.

## Current Best

| Checkpoint | Experiment | change_acc | pos_mae | rollout_coherence | Notes |
|-----------|-----------|-----------|---------|-------------------|-------|
| `e026c-sf-curriculum/best.pt` | E026c | 80.2% | 0.642 | **4.965** | b002 + SF curriculum N=1->2->3, 2 epochs, 1.9K data |
| `e026b-unimix/best.pt` | E026b | 65.4% | 0.798 | 5.120 | b002 + 1% unimix on categoricals, 1.9K data |
| `e025a-lr-warmup/best.pt` | E025a | 65.6% | 0.814 | 5.146 | b001 + SF + K=30 + d_model=768 + warmup 5%, 1.9K data |
| `e023b-dmodel768-r3/best.pt` | E023b | 66.0% | 0.823 | 5.775 | b001 + SF + K=30 + d_model=768, 1.9K data |
| `e018c-context-k30/best.pt` | E018c | 62.3% | 0.824 | 6.03 | b001 + SF + K=30, d_model=384 |
| `e018a-sf-minimal/best.pt` | E018a | 61.6% | 0.825 | 6.26 | b001 + Self-Forcing (20% SF, N=3), 1.9K data |
| `e019-baseline-1k/best.pt` | E019 | 78.7% | 0.756 | 6.77 | b001, 1.9K data, full loss suite (10 heads) |

E026c is the new best. Progressive SF curriculum (N=1->2->3 over 2 epochs) broke below RC 5.0 for the first time: 4.965 (-3.0% vs prior best 5.120). The curriculum made epoch 2 productive unlike e023b-epoch2 (RC flat) — each stage introduces qualitatively new training signal. change_acc jumped +14.8pp (65.4%->80.2%), the largest single-experiment gain on that metric. pos_mae improved 19.5% (0.798->0.642). Cumulative improvement from E019 baseline: -26.7% (6.77->4.965).

The curriculum hypothesis is supported: E018b's N=5 failure was likely "the model wasn't ready" rather than "N=5 is fundamentally too long." This opens N=4/5 via longer curricula (e.g., [1,2,3,4,5]).

Val metrics plateau after 1 epoch on 1.9K data with fixed SF — but curriculum breaks this pattern because each stage is qualitatively different training. 2 epochs are justified when the curriculum spans them.

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
| Muon optimizer | E026a | Newton-Schulz orthogonalization doesn't suit Mamba-2 weight matrices. RC +3.8%, pos_mae +38.8%. |

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

#### 1. Full backpropagation through Self-Forcing steps — TESTED, needs different approach
E024a tested full BPTT via soft embeddings (softmax @ embed.weight): RC 8.980 (+55.5% catastrophic regression). Train/eval mismatch — model optimized for soft-embed path but eval uses hard argmax. The idea is sound but the implementation needs to avoid this mismatch: temperature annealing (start soft, anneal to hard) or straight-through estimator (hard forward, approximate backward). Revisit with curriculum approach: full BPTT at N=1 first, grow from there.

#### 2. Data scaling to 7.7K games — UNBLOCKED
Data loader fixed via `mmap=True` on `torch.load()`. 53GB encoded file loads in 0.0s, VRAM peak 8.19GB. Ready to run. Cost: ~$31/epoch on A100 with AMP. The 7.7K dataset is `encoded-v3-ranked-fd-top5.pt` (FD-only, top-5 characters — same distribution as 1.9K, just 4x more games).

#### 3. Constraint violation penalties during SF (ref: issue #18)
E015 predicted that constraint penalties would work *after* scheduled sampling is in place (violations only occur during AR, not TF). SF is now in place. The plan:

**Phase 1 — Instrument.** Add config-driven constraint spec (`configs/constraints.yaml`) and evaluator that checks SF reconstructed frames and rollout eval frames for violations (negative percent, impossible jumps, stocks increasing, position outside blast zones). Log per-constraint violation rates to wandb, broken down by AR horizon step. Ship with the next training run for free data collection.

**Phase 2 — Penalize (only if violations >5%).** Add `λ * violation_penalty` to SF loss using the same constraint spec. Differentiable for continuous constraints (relu relaxation). Discrete constraints (jumps, action transitions) may already be addressed by unimix (E026b).

**Phase 3 — Target.** Use violation data to identify which game situations produce the most errors. Hard example mining: oversample high-violation starting points during SF.

**Key uncertainty:** Violations may already be rare at RC 4.965. Measure first.

#### 3. Cascaded heads + Self-Forcing
E014 showed cascaded heads fixed damage drift at 1.9K. Code not ported to this repo. Requires implementation work.

#### 4. Muon optimizer — TESTED, dead end
Muon tested in E026a: RC 5.342 (+3.8% regression), pos_mae 1.130 (+38.8%). Newton-Schulz orthogonalization doesn't suit Mamba-2 weight matrices. The in_proj/out_proj matrices have structured SSM roles that differ from transformer Q/K/V where Muon was validated. 1/1 regressed. AdamW remains the optimizer.

### Config directions (no code changes)

#### 5. SF parameter space (revisitable)
- **SF ratio:** 10% regressed (E020a, RC 6.62). 30%, 50% untested. 20% remains the only proven ratio.
- **Unroll N=4:** Between the working N=3 and failing N=5. Might find a sweet spot.
- **Alternative loss weighting:** Exponential, step-function, inverse-loss weighting. Linear ramp failed but other schemes untested.

#### 6. Batch size sweep
Karpathy's autoresearch found halving batch size was his single largest improvement. We've locked bs=512 without a systematic sweep. Test 256 (half) and 1024 (double), coupled with LR scaling (linear scaling rule: LR ∝ batch_size). Especially relevant given short training (1 epoch on 1.9K data).

#### 7. Architecture exploration (ref: issue #6)

The model was ~4.3M params (d_model=384). E023b proved it was capacity-constrained: d_model=768 (~15.8M) improved RC 4.2% and change_acc +3.7pp. Width axis is monotonic and high-value.

**Phase 1 — Width/Depth Grid (config-only, autonomous)**

| Variable | Half | Current | Double | Results |
|---|---|---|---|---|
| d_model (width) | 192 (~1.3M) | 384 (~4.3M) | **768 (~15.8M)** | **192: RC 6.065 (underfit). 768: RC 5.775 (-4.2%, KEPT). Monotonic.** |
| n_layers (depth) | 2 (~2.2M) | 4 (~4.3M) | 8 (~30.5M at d=768) | **8: RC 7.108 (+23.1%, regressed). Depth hurts at d_model=768.** |
| d_state (SSM memory) | 32 | 64 | 128 | Untested |
| headdim (head granularity) | 32 (→24 heads) | 64 (→12 heads) | 128 (→6 heads) | Untested |

**Phase 1 status:** Width axis is the clear winner (monotonic, -4.2% RC at 768). Depth regressed 23.1% (0/1 improved). d_state and headdim are lower priority — the depth result suggests the model is already over-parameterized for 1.9K data at d_model=768, so further capacity axes are unlikely to help without more data.

**Phase 2 — now active (triggered by d_model=768 improving RC >5%):**
- Test d_model=512 at n_layers=4 to find efficient frontier (onchain weight size matters).
- If 512 ≈ 768 in RC: efficient frontier is 512. If 512 << 768: scaling law continues.

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
- **Kill fast for single-regime experiments.** If a single-regime experiment isn't showing signal by epoch 1, kill it. But multi-phase experiments (curriculum, regime switching) should be evaluated after all phases complete — E026c's epoch 1 RC was 6.262 (would have been killed) but epoch 2 hit 4.965 (new best). See issue #17.
- **Double or half, not +1.** When exploring a parameter, make big moves to see directional shift. We're triangulating, not hill-climbing. Going from 4 layers to 8 is better than 4 to 5.

## The Eval Protocol

For autoresearch agents:

1. Start from the current best config (see `.loop/state/best.json`). Change ONE thing.
2. Train on 1.9K data (`encoded-e012-fd-top5.pt`), 1 epoch, A100. Cost varies with model size.
3. Rollout coherence eval runs automatically after training (integrated in Trainer).
4. Compare to baseline RC in `.loop/state/best.json`.
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
